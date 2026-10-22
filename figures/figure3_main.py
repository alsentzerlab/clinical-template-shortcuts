#!/usr/bin/env python3
"""Figure 3 (main results) for the 5 template pairs (pairs ordered by readmission-rate disparity).

  a  readmission rate of both templates in every pair            (full width)
  b  ROC, smallest-disparity pair (1)   c  ROC, largest-disparity pair (5)
  d  shortcut gap across pairs          e  OOD gap across pairs
  f  held-out templates, pair 1         g  held-out templates, pair 5

Shortcut gap       = overall AUC - mean(per-template AUCs)          (auc_data.py)
OOD gap            = mean(per-template AUCs) - mean(AUC on 10 held-out templates)
                     (Shortcut Learning Meeting Notes (1).pdf, "Metric 2", sign
                     flipped so a bigger drop is a taller bar)

PLACEHOLDERS still in here:
  * ROC curves (b, c) are simulated binormal curves that reproduce the reported
    AUCs; swap in real scores via roc_curve(y, score) when available.
  * Held-out points (f, g) are DIGITIZED from the meeting-notes scatter plots
    (+-0.005 AUC, +-0.3 pp). Templates plotted below AUC 0.50 there were clipped
    off the axis, so they are missing here -- which is why the points' mean sits
    above the reported held-out mean line. Replace with the per-template CSV.

Run:  cd figures && python figure3_main.py [--palette dark2] [--out DIR]
Saves figures/figure_3/figure3.{png,pdf}. --palette picks a
fig_style.ROC_PALETTES entry for the template lines in b/c/f/g.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import roc_curve

import auc_data
import fig_style

sns.set_theme(style="ticks")
fig_style.use_style()                 # AFTER set_theme: drops top + right spines

PAIRS, COLOR, INK = fig_style.PAIRS, fig_style.PAIR_COLOR, fig_style.INK
RATE, LETTER = fig_style.TEMPLATE_RATE, fig_style.TEMPLATE_LETTER
C_OVERALL = fig_style.ROC_SERIES_COLORS["overall"]
C_TA, C_TB = fig_style.ROC_SERIES_COLORS["anchor"], fig_style.ROC_SERIES_COLORS["paired"]
MUTED = "#777777"
LO, HI = PAIRS[0], PAIRS[-1]          # smallest / largest disparity pair

MODE = "mean"
BASE = auc_data.delta_df(MODE).query("condition == 'baseline'").set_index("pair")

# ---- generalization (meeting notes) -----------------------------------------
# held-out AUC mean +- std over the 10 unseen templates, and Metric 2 (OOD
# performance drop, reported negative in the notes; stored here as a positive gap)
GEN = pd.DataFrame({
    "pair":         [1,    2,    3,    4,    5],
    "heldout_mean": [0.55, 0.55, 0.60, 0.54, 0.51],
    "heldout_std":  [0.08, 0.08, 0.06, 0.10, 0.07],
    "ood_gap":      [0.13, 0.17, 0.16, 0.17, 0.18],
}).set_index("pair")

# DIGITIZED held-out points: template -> (readmission %, AUC). See docstring.
HELDOUT = {
    1: {"593537": (18.8, 0.661), "714564": (14.8, 0.585), "1327338": (20.8, 0.593),
        "1459783": (24.6, 0.589), "899007": (10.4, 0.556), "803224": (32.4, 0.572),
        "529413": (35.8, 0.567), "1137807": (32.6, 0.552), "181781": (30.9, 0.545),
        "966314": (45.9, 0.547)},
    5: {"803224": (32.2, 0.618), "966314": (45.6, 0.564), "529413": (35.6, 0.551),
        "1137807": (32.4, 0.542), "593537": (18.7, 0.544), "899007": (10.3, 0.534),
        "1459783": (24.4, 0.528), "1327338": (20.6, 0.522), "181781": (30.6, 0.516),
        "714564": (14.7, 0.497)},
}


def _templates(pair):
    return fig_style.PAIR_IDS[pair].split("_")




# ---- panels ------------------------------------------------------------------
def panel_readmit(ax):
    """a: both templates' readmission rates per pair, bars in the pair colour."""
    w, gap = 0.36, 0.04
    for i, p in enumerate(PAIRS):
        for j, tid in enumerate(_templates(p)):
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, RATE[tid], w, color=COLOR[p], edgecolor=INK, lw=0.8)
            ax.text(x, RATE[tid] + 1.2, f"{RATE[tid]:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
            # template letter just under the bar, above the pair tick label
            ax.text(x, -1.2, LETTER[tid], ha="center", va="top", fontsize=8,
                    fontweight="bold", color=INK)
    ax.set_ylim(0, 68)
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    ax.set_ylabel("30-day readmission rate (%)")
    fig_style.pair_axis(ax, length=0.46)
    ax.tick_params(axis="x", length=0, pad=16)     # room for the A-D letters


def sim(auc, n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(np.sqrt(2) * norm.ppf(auc) * y, 1)


def panel_roc(ax, pair, seeds):
    """b / c: ROC for one pair (PLACEHOLDER curves at the reported AUCs)."""
    r = BASE.loc[pair]
    ta, tb = _templates(pair)
    series = [("Overall", C_OVERALL, r.overall), (f"Template {LETTER[ta]}", C_TA, r.tA),
              (f"Template {LETTER[tb]}", C_TB, r.tB)]
    for (name, color, auc), seed in zip(series, seeds):
        fpr, tpr, _ = roc_curve(*sim(auc, seed=seed))
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUROC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.set_aspect("equal", "box")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(f"Pair {pair}", fontsize=10)
    # right-aligned, as in the earlier figure: never collides with the curves
    ax.set_title(f"\u0394 = {r.delta:+.3f}", loc="right", fontsize=10)


def _gap_bars(ax, values, ylabel, stars=None):
    x = np.arange(len(PAIRS))
    ax.bar(x, values, 0.62, color=[COLOR[p] for p in PAIRS], edgecolor=INK, lw=0.8)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.006, f"{v:+.3f}" if stars is not None else f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color=INK)
    if stars is not None:
        for xi, v, s in zip(x, values, stars):
            if s:
                ax.text(xi, v + 0.024, "*", ha="center", va="bottom", fontsize=13, color=INK)
        ax.text(0.01, 0.99, "* overall AUROC > both per-template AUROCs",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel(ylabel)
    fig_style.pair_axis(ax, positions=x, fontsize=7.5)


def panel_shortcut(ax):
    """d: shortcut gap (overall - mean per-template AUC) across pairs."""
    _gap_bars(ax, BASE.loc[PAIRS, "delta"].to_numpy(), "Shortcut gap (\u0394)",
              stars=BASE.loc[PAIRS, "exceeds_both"].to_numpy())
    ax.set_ylim(0, 0.25)


def panel_oodgap(ax):
    """e: OOD gap (mean per-template AUC - mean held-out AUC) across pairs."""
    _gap_bars(ax, GEN.loc[PAIRS, "ood_gap"].to_numpy(), "Out-of-distribution gap")
    ax.set_ylim(0, 0.25)


# Relative test-set size per held-out template, eyeballed from the marker areas
# in the meeting-notes plots (markers are sized by n). APPROXIMATE -- replace
# with the real n from template_predictions.parquet.
# Held-out templates are shown as H1-H10, numbered by readmission rate (low -> high).
# Same order as Supplementary Table S1.
HELDOUT_LABEL = {t: f"H{i}" for i, t in enumerate(
    sorted(HELDOUT[1], key=lambda t: HELDOUT[1][t][0]), start=1)}

HELDOUT_REL_N = {"1137807": 1.0, "181781": 1.0, "714564": 0.35}   # all others ~0.15


def panel_heldout(ax, pair):
    """f / g: AUC on each held-out template vs its readmission rate, styled after
    the original analysis plots: marker area ~ n,
    template IDs on the points, dashed verticals at the training templates'
    readmission rates, dotted overall AUC, dash-dot per-template AUCs."""
    r, g = BASE.loc[pair], GEN.loc[pair]
    ta, tb = _templates(pair)
    ids = list(HELDOUT[pair])
    pts = np.array([HELDOUT[pair][t] for t in ids])
    x, y = pts[:, 0] / 100, pts[:, 1]
    size = np.array([HELDOUT_REL_N.get(t, 0.15) for t in ids]) * 400 + 30
    ax.scatter(x, y, s=size, color=COLOR[pair], alpha=0.8, edgecolors="k",
               linewidths=0.5, zorder=3)
    for t, xi, yi in zip(ids, x, y):
        ax.annotate(HELDOUT_LABEL[t], (xi, yi), xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5)

    for tid, c in ((ta, C_TA), (tb, C_TB)):
        ax.axvline(RATE[tid] / 100, color=c, ls="--", lw=1.5, zorder=2,
                   label=f"Template {LETTER[tid]} ({RATE[tid]:.0f}%)")
    ax.axhline(r.overall, color=C_OVERALL, ls=":", lw=1.5, zorder=2,
               label=f"Overall AUROC ({r.overall:.3f})")
    ax.axhline(r.tA, color=C_TA, ls="-.", lw=1.5, zorder=2,
               label=f"Template {LETTER[ta]} AUROC ({r.tA:.3f})")
    ax.axhline(r.tB, color=C_TB, ls="-.", lw=1.5, zorder=2,
               label=f"Template {LETTER[tb]} AUROC ({r.tB:.3f})")

    ax.set(xlabel="Readmission rate (per template)", ylabel="AUROC",
           xlim=(0.0, 0.58), ylim=(0.0, 1.0))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.52, 0.02), fontsize=7.5, framealpha=0.9)


# ---- figure ------------------------------------------------------------------
def make_fig2(out=None):
    out = Path(out) if out else Path(__file__).resolve().parent / "figure_3"
    out.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(11, 17), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 1.15, 0.95, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_d, ax_e = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    ax_f, ax_g = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])

    panel_readmit(ax_a)
    panel_roc(ax_b, LO, [1, 2, 3])
    panel_roc(ax_c, HI, [4, 5, 6])
    panel_shortcut(ax_d)
    panel_oodgap(ax_e)
    panel_heldout(ax_f, LO)
    panel_heldout(ax_g, HI)

    for a, letter in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g], "abcdefg"):
        a.set_title(letter, loc="left", fontweight="bold", fontsize=13)
    fig_style.align_disparity(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure3.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out / 'figure3.png'}")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default=fig_style.ROC_PALETTE, choices=fig_style.ROC_PALETTES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bar-color", default=None, help="single fill for the bar panels")
    args = ap.parse_args()
    pal = fig_style.ROC_PALETTES[args.palette]
    C_OVERALL, C_TA, C_TB = pal["overall"], pal["anchor"], pal["paired"]
    if args.bar_color:
        COLOR = {p: args.bar_color for p in PAIRS}
    make_fig2(args.out)
