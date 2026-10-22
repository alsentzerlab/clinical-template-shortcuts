"""ShorT analysis: does template encoding track unfairness across gradient-scaled models?

Collects every <runs-root>/**/results.json written by train.py and, per template
pair, reports the two-sided Spearman correlation between template encoding (probe
AUROC on frozen features) and unfairness (equalized odds) — Brown et al. 2023, Figs.
4-5. A significant positive rho (more encoding -> less fair) indicates shortcut
learning.

As in the paper, replicates with unacceptable readmission performance are dropped
first. The paper used a fixed AUC cut-off (0.8, "arbitrary"); readmission AUCs here
are lower, so the default cut-off is relative: drop runs more than --max-auc-drop
below that pair's mean grad_scale=0 test AUC.

The paper also stresses ShorT is only informative if the intervention actually moves
encoding across a wide range, so the encoding range and the grad_scale -> encoding
correlation are reported too.

    python -m gradient_scaling.analyze --runs-root ~/gradient_scaling_runs
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def spearman(df: pd.DataFrame, x: str, y: str):
    d = df[[x, y]].dropna()
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return float("nan"), float("nan")
    rho, pval = spearmanr(d[x], d[y])
    return float(rho), float(pval)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--max-auc-drop", type=float, default=0.05, help="Drop runs this far below the pair's baseline (grad_scale=0) AUC")
    p.add_argument("--after-scale", type=float, default=-1.0, help="grad_scale compared against the grad_scale=0 baseline in before_after.csv")
    p.add_argument("--out", default=None, help="Directory for runs.csv / summary.csv / by_scale.csv (default: runs-root)")
    args = p.parse_args()

    root = Path(args.runs_root).expanduser()
    rows = [json.loads(f.read_text()) for f in sorted(root.rglob("results.json"))]
    if not rows:
        raise SystemExit(f"No results.json under {root}")
    runs = pd.DataFrame(rows)
    # Same definition as the main pair results: overall AUC minus the mean of the
    # within-template AUCs. >0 means ranking across templates adds AUC that no
    # within-template ranking earns, i.e. the template is doing work.
    runs["shortcut_gap"] = runs.test_readmission_auroc - runs[["test_readmission_auroc_a", "test_readmission_auroc_b"]].mean(axis=1)

    baseline = runs[runs.grad_scale == 0].groupby("pair").test_readmission_auroc.mean().rename("baseline_auroc")
    runs = runs.join(baseline, on="pair")
    missing = sorted(runs.loc[runs.baseline_auroc.isna(), "pair"].unique())
    if missing:
        print(f"WARNING: no grad_scale=0 runs for {missing}; no performance filter applied to them")
    runs["included"] = runs.baseline_auroc.isna() | (runs.test_readmission_auroc >= runs.baseline_auroc - args.max_auc_drop)

    summary = []
    for pair, df in runs.groupby("pair"):
        kept = df[df.included]
        rho, pval = spearman(kept, "probe_template_auroc", "equalized_odds")
        rho_ll, pval_ll = spearman(kept, "probe_template_logloss", "equalized_odds")
        rho_gs, pval_gs = spearman(df, "grad_scale", "probe_template_auroc")
        rho_sg, pval_sg = spearman(kept, "probe_template_auroc", "shortcut_gap")
        summary.append({
            "pair": pair,
            "n_runs": len(df),
            "n_included": len(kept),
            "baseline_auroc": df.baseline_auroc.iloc[0],
            "probe_auroc_min": kept.probe_template_auroc.min(),
            "probe_auroc_max": kept.probe_template_auroc.max(),
            "eo_min": kept.equalized_odds.min(),
            "eo_max": kept.equalized_odds.max(),
            # Main ShorT result: >0 and significant => shortcut learning
            "rho_encoding_vs_eo": rho,
            "p_encoding_vs_eo": pval,
            # Same test with probe log-loss (higher = LESS encoding, so expect rho < 0)
            "rho_logloss_vs_eo": rho_ll,
            "p_logloss_vs_eo": pval_ll,
            "shortcut_gap_min": kept.shortcut_gap.min(),
            "shortcut_gap_max": kept.shortcut_gap.max(),
            # Does encoding track the main paper's shortcut measure?
            "rho_encoding_vs_shortcut_gap": rho_sg,
            "p_encoding_vs_shortcut_gap": pval_sg,
            # Intervention check: did grad_scale actually move encoding?
            "rho_gradscale_vs_encoding": rho_gs,
            "p_gradscale_vs_encoding": pval_gs,
        })
    summary = pd.DataFrame(summary)

    by_scale = (
        runs.groupby(["pair", "grad_scale"])
        [["test_readmission_auroc", "shortcut_gap", "probe_template_auroc", "probe_template_logloss", "equalized_odds", "tpr_gap", "fpr_gap"]]
        .agg(["mean", "std", "count"])
    )

    # Before/after the intervention: baseline (grad_scale=0) vs --after-scale, seed-averaged.
    # Uses every run (no performance filter): the AUC cost of reversal is part of the result.
    cols = ["test_readmission_auroc", "test_readmission_auroc_a", "test_readmission_auroc_b", "shortcut_gap", "equalized_odds"]
    before = runs[runs.grad_scale == 0].groupby("pair")[cols].mean().add_suffix("_before")
    after_runs = runs[runs.grad_scale == args.after_scale]
    after = after_runs.groupby("pair")[cols].mean().add_suffix("_after")
    before_after = before.join(after, how="outer").join(after_runs.groupby("pair").size().rename("n_after"))
    for c in cols:
        before_after[f"{c}_delta"] = before_after[f"{c}_after"] - before_after[f"{c}_before"]

    out = Path(args.out).expanduser() if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    by_scale.to_csv(out / "by_scale.csv")
    before_after.to_csv(out / "before_after.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary.round(4).to_string(index=False))
    print(f"\nShortcut gap before (grad_scale=0) vs after (grad_scale={args.after_scale:g}):")
    gap_cols = ["shortcut_gap_before", "shortcut_gap_after", "shortcut_gap_delta", "test_readmission_auroc_before", "test_readmission_auroc_after", "n_after"]
    print(before_after[gap_cols].round(4).to_string())
    print(f"\nWrote runs.csv, summary.csv, by_scale.csv, before_after.csv to {out}")


if __name__ == "__main__":
    main()
