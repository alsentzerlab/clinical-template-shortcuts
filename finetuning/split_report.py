#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report the train/val/test composition of processed dataset directories.

`process_data.py` prints split sizes and one overall readmission rate per
split. With --match-rates the interesting quantity is finer than that: the
readmission rate PER TEMPLATE per split. That is what the rate-matched
patient-grouped assignment is solving for, and a per-template drift between
train and test is exactly the confound the split is meant to remove -- an
overall rate can look flat while one template's rate moves several points.

This reads only the `label` and `template` columns of each split, so it emits
counts and rates only; note text is never loaded.

Usage:
    uv run python finetuning/split_report.py data/pairs/206596_229657/
    uv run python finetuning/split_report.py data/pairs/*/
    uv run python finetuning/split_report.py <dirs...> --csv splits.csv
"""

import argparse
import os
import sys

import pandas as pd

SPLITS = ["train", "val", "test"]

# process_data.py writes the arm as "A"/"B". The paper refers to pairs by the
# arm letters it assigned (IM A/B/C/D, non-IM A/B/C/D), and those letters are
# NOT the same set -- so a report that only says "A" and "B" is ambiguous
# across the 12 runs. --label-map attaches the real template IDs.
def parse_label_map(spec):
    """'A=801125,B=624343' -> {'A': 'A (801125)', 'B': 'B (624343)'}"""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        arm, _, tid = part.partition("=")
        arm, tid = arm.strip(), tid.strip()
        if not arm or not tid:
            raise SystemExit(f"--label-map: cannot parse '{part}' (want ARM=ID)")
        out[arm] = f"{arm} ({tid})"
    return out


def infer_label_map(path):
    """Derive arm->ID from a dataset directory name like '206596_248528'.

    The pair parquets are named <A-id>_<B-id>[_removed], and process_data.py
    assigns A to the first template it encounters, so the filename carries the
    mapping. Returns {} when the name does not look like an ID pair.
    """
    name = os.path.basename(path.rstrip("/")).replace("_removed", "")
    parts = [p for p in name.split("_") if p.isdigit() and len(p) >= 5]
    if len(parts) == 2:
        return {"A": f"A ({parts[0]})", "B": f"B ({parts[1]})"}
    return {}



def load_split(d, split):
    p = os.path.join(d, f"{split}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p, columns=["label", "template"])


def summarize(d, label_map=None):
    """Long-form rows: one per (split, template) plus per-split totals."""
    rows = []
    for split in SPLITS:
        df = load_split(d, split)
        if df is None:
            continue
        for tid, sub in df.groupby("template"):
            rows.append({"dataset": os.path.basename(d.rstrip("/")),
                         "split": split,
                         "template": (label_map or {}).get(tid, tid), "n": len(sub),
                         "n_readmit": int(sub["label"].sum()),
                         "rate": float(sub["label"].mean())})
        rows.append({"dataset": os.path.basename(d.rstrip("/")),
                     "split": split, "template": "ALL", "n": len(df),
                     "n_readmit": int(df["label"].sum()),
                     "rate": float(df["label"].mean())})
    return pd.DataFrame(rows)


def report(long_df, name):
    print("=" * 76)
    print(name)
    print("=" * 76)
    # n and rate side by side, splits as columns.
    piv_n = long_df.pivot(index="template", columns="split", values="n")
    piv_r = long_df.pivot(index="template", columns="split", values="rate")
    piv_n = piv_n.reindex(columns=[s for s in SPLITS if s in piv_n.columns])
    piv_r = piv_r.reindex(columns=piv_n.columns)

    out = pd.DataFrame(index=piv_n.index)
    for s in piv_n.columns:
        out[f"n_{s}"] = piv_n[s].astype("Int64")
    for s in piv_r.columns:
        out[f"rate_{s}"] = piv_r[s].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    # Spread = the drift --match-rates is trying to kill.
    spread = (piv_r.max(axis=1) - piv_r.min(axis=1)) * 100
    out["spread_pp"] = spread.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    # Put the ALL row last.
    idx = [i for i in out.index if i != "ALL"] + (["ALL"] if "ALL" in out.index else [])
    print(out.loc[idx].to_string())

    per_tpl = spread.drop(index="ALL", errors="ignore")
    if len(per_tpl):
        print(f"\n  worst per-template train/val/test drift: {per_tpl.max():.2f}pp "
              f"(template {per_tpl.idxmax()})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="Processed output directories")
    ap.add_argument("--csv", default=None, help="Write the long-form table here")
    ap.add_argument("--label-map", default=None,
                    help="Rename the arms, e.g. 'A=801125,B=624343'. Applied to "
                         "every directory; omit to infer each one from its "
                         "<A-id>_<B-id> directory name.")
    args = ap.parse_args()

    all_rows, missing = [], []
    for d in args.dirs:
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "train.parquet")):
            missing.append(d)
            continue
        lm = parse_label_map(args.label_map) if args.label_map else infer_label_map(d)
        long_df = summarize(d, lm)
        all_rows.append(long_df)
        report(long_df, os.path.abspath(d))

    if missing:
        print(f"Skipped {len(missing)} path(s) with no train.parquet:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if args.csv and all_rows:
        pd.concat(all_rows).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")

    if all_rows:
        # One cross-dataset line per dataset, so 12 runs fit on a screen.
        print("=" * 76)
        print("SUMMARY — worst per-template train/val/test rate drift per dataset")
        print("=" * 76)
        for long_df in all_rows:
            piv = long_df[long_df.template != "ALL"].pivot(
                index="template", columns="split", values="rate")
            sp = (piv.max(axis=1) - piv.min(axis=1)) * 100
            name = long_df["dataset"].iloc[0]
            tot = int(long_df[(long_df.template == "ALL")].n.sum())
            print(f"  {name:<28} n={tot:>6,}  worst drift {sp.max():.2f}pp")


if __name__ == "__main__":
    main()
