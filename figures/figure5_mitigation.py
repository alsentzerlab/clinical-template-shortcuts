#!/usr/bin/env python3
"""Figure 5: mitigation for the 5 template pairs: shortcut gap before -> after.

  a  balanced training (per-template readmission rates equalized in training)
  b  TRACE (templated contents removed)
  c  gradient scaling (ShorT), every lambda, on the middle pair (pair 3)
  d  gradient scaling at lambda = -1 (gradient reversal), every pair

!! PLACEHOLDER !!  c and d are MADE UP -- the gradient-scaling sweep
(gradient_scaling) is still running. The fake values mimic the
expected result (no reliable reduction in the shortcut gap). They are drawn
dashed / hollow with a red "placeholder" note. When the runs land, add
`gradscale_-1` rows to auc_data.py and fill GS_SWEEP from analyze.py; panel d
picks real auc_data rows up automatically.

Style: black filled dot =
original, open marker = after mitigation (circle = balanced, triangle = TRACE),
green / red connector = shortcut gap decreased / increased. The IM / non-IM
background bands are gone -- pairs are simply ordered by readmission-rate
disparity, labelled on the shared y axis.

Numbers: auc_data.py (Figures (6).pdf), shortcut gap = overall - mean(per-template).

Run:  cd figures && python figure5_mitigation.py
Saves figures/figure_5/figure5.{png,pdf}.
"""
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import auc_data
import fig_style

sns.set_theme(style="ticks")
fig_style.use_style()                 # AFTER set_theme: drops top + right spines

PAIRS, INK = fig_style.PAIRS, fig_style.INK
MODE = "mean"
DOWN_C, UP_C = "green", "red"        # direction of change in the shortcut gap
FAKE_C = "#b2182b"                   # placeholder note

GS_PAIR = 3                          # middle pair for the lambda sweep
# MADE UP (see docstring). lambda -> shortcut gap, mean and SD over 3 seeds.
GS_SWEEP = {
    -1: (0.094, 0.011), -0.1: (0.091, 0.009), -0.01: (0.087, 0.008), -0.001: (0.090, 0.007),
     0: (0.089, 0.006),
     0.001: (0.088, 0.007), 0.01: (0.092, 0.008), 0.1: (0.095, 0.009), 1: (0.101, 0.010),
}
# MADE UP (see docstring). pair -> shortcut gap at lambda = -1.
GS_FAKE_REVERSAL = {1: 0.041, 2: 0.049, 3: 0.094, 4: 0.131, 5: 0.188}
# MADE UP. pair -> overall AUROC LOST at lambda = -1 (small: reversal costs a bit
# of accuracy without shrinking the gap). The real pilot on pair 3 lost more
# (0.831 -> 0.718); replace with the sweep output.
GS_FAKE_AUC_DROP = {1: 0.022, 2: 0.035, 3: 0.041, 4: 0.028, 5: 0.037}

# strategy -> (auc_data condition, marker, colour, label)
STRATEGIES = [("balanced", "o", "#1E88E5", "Balanced"),
              ("trace", "^", "#FFC107", "TRACE"),
              ("gradscale_-1", "s", "#D81B60", "Gradient reversal (λ = −1)")]


def _after(cond):
    """pair -> shortcut gap for an 'after' condition; real rows win over placeholders."""
    real = auc_data.deltas(cond, MODE)
    if all(p in real for p in PAIRS):
        return real, False
    return GS_FAKE_REVERSAL, True


def _fake_note(ax):
    ax.text(0.99, 0.02, "placeholder values (runs in progress)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=FAKE_C)


def dumbbell(ax, after_cond, after_marker, pair_axis=False, pair_labels=False):
    base = auc_data.deltas("baseline", MODE)
    after, fake = _after(after_cond)

    for p in PAIRS:
        ax.plot([base[p], after[p]], [p, p],
                color=DOWN_C if after[p] < base[p] else UP_C, lw=2.5, zorder=1)
    ax.scatter([base[p] for p in PAIRS], PAIRS, facecolor="black", edgecolor="black",
               s=70, zorder=3)
    ax.scatter([after[p] for p in PAIRS], PAIRS, marker=after_marker,
               facecolor="white", edgecolor="black", s=70, zorder=3)
    if fake:
        _fake_note(ax)

    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Shortcut gap (Δ)")
    if pair_axis:
        fig_style.pair_axis(ax, orient="y", positions=PAIRS,
                            offset=-0.20, thickness=0.028, labelpad=34)
    elif pair_labels:                  # pair names, but no second arrow
        fig_style.pair_axis(ax, orient="y", positions=PAIRS, arrow=False, label=None)
    else:
        ax.set_yticks(PAIRS)
        ax.set_ylabel("")


def gs_sweep(ax):
    """c: shortcut gap vs gradient-scaling lambda for one pair (PLACEHOLDER)."""
    lam = np.array(sorted(GS_SWEEP))
    mu, sd = np.array([GS_SWEEP[l] for l in lam]).T
    base = auc_data.deltas("baseline", MODE)[GS_PAIR]
    ax.axhline(base, color=INK, ls=":", lw=1.2, label=f"Original ({base:+.3f})")
    ax.errorbar(lam, mu, yerr=sd, color="black", ls="--", lw=1.4, marker="s", ms=7,
                mfc="white", mec="black", capsize=3, label="Gradient scaling (mean ± SD, 3 seeds)")
    ax.axvline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(-1.7, 1.7)
    ax.set_xticks(lam)
    ax.set_xticklabels([f"{l:g}" for l in lam], fontsize=8)
    ax.set_xlabel("Gradient-scaling λ  (< 0 reverses, > 0 amplifies)")
    ax.set_ylabel("Shortcut gap (Δ)")
    ax.set_ylim(0, 0.16)
    ax.set_title(f"Pair {GS_PAIR}  (Δ {fig_style.READMIT_GAP[GS_PAIR]:.0f} pp)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _fake_note(ax)


def _drops(cond):
    """pair -> (shortcut-gap reduction, AUROC drop), both before - after, + flag
    for placeholder values."""
    base = auc_data.AUC.query("condition == 'baseline'").set_index("pair")
    gap_b = auc_data.deltas("baseline", MODE)
    gap_a, fake = _after(cond)
    if fake:
        auc_drop = GS_FAKE_AUC_DROP
    else:
        aft = auc_data.AUC.query("condition == @cond").set_index("pair")
        auc_drop = {q: base.loc[q, "overall"] - aft.loc[q, "overall"] for q in PAIRS}
    return {q: (gap_b[q] - gap_a[q], auc_drop[q]) for q in PAIRS}, fake


def tradeoff(ax):
    """e: what each strategy costs (AUROC drop) vs what it buys (shortcut-gap
    reduction), one point per pair. Ideal = far right, near zero."""
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    any_fake = False
    for cond, marker, color, label in STRATEGIES:
        d, fake = _drops(cond)
        any_fake |= fake
        xs, ys = zip(*(d[q] for q in PAIRS))
        ax.scatter(xs, ys, marker=marker, s=80, color=color, edgecolor=INK, lw=0.8,
                   zorder=3, label=label + (" (placeholder)" if fake else ""))
        for q, x, y in zip(PAIRS, xs, ys):
            ax.annotate(str(q), (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=INK)
    ax.set_xlabel("Shortcut-gap reduction  (Δ before − Δ after)")
    ax.set_ylabel("AUROC drop  (before − after)")
    ax.set_xlim(-0.04, 0.16)
    ax.set_ylim(-0.03, 0.21)
    ax.text(0.98, 0.04, "ideal: gap removed,\nno AUROC lost", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#777777", style="italic")
    ax.legend(loc="upper right", fontsize=8, frameon=False, title="numbers = pair",
              title_fontsize=8)


def _dot(marker, face, label):
    return Line2D([], [], marker=marker, ls="", markerfacecolor=face,
                  markeredgecolor="black", markersize=8, label=label)


LEGEND = [_dot("o", "black", "Original"),
          _dot("o", "white", "Balanced"),
          _dot("^", "white", "TRACE applied"),
          _dot("s", "white", "Gradient scaling (λ = −1)"),
          Line2D([], [], color=DOWN_C, lw=2.5, label="Shortcut gap decreased"),
          Line2D([], [], color=UP_C, lw=2.5, label="Shortcut gap increased")]


def make_fig5_2x2():
    """Previous 2x2 layout (a balanced, b TRACE, c sweep, d reversal). Kept for reference."""
    out = Path(__file__).resolve().parent / "figure_5"
    out.mkdir(exist_ok=True)
    fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "d"]], figsize=(11.5, 9.2),
                                  constrained_layout=True)
    dumbbell(axd["a"], "balanced", "o", pair_axis=True)
    dumbbell(axd["b"], "trace", "^")
    gs_sweep(axd["c"])
    dumbbell(axd["d"], "gradscale_-1", "s", pair_labels=True)
    axd["b"].sharey(axd["a"])
    axd["b"].tick_params(labelleft=False)
    for k in "abd":
        axd[k].set_xlim(-0.01, 0.21)
        axd[k].set_ylim(len(PAIRS) + 0.5, 0.5)     # pair 1 on top
    for letter, a in axd.items():
        a.set_title(letter, loc="left", fontweight="bold", fontsize=13)
    fig.legend(handles=LEGEND, loc="outside lower center", ncol=len(LEGEND),
               fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure5.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out / 'figure5.png'}")
    for cond in ("balanced", "trace", "gradscale_-1"):
        b, a = auc_data.deltas("baseline", MODE), _after(cond)[0] if cond.startswith("grad") else auc_data.deltas(cond, MODE)
        print(f"  {cond:9s}", "  ".join(f"P{p} {b[p]:+.3f}->{a[p]:+.3f}" for p in PAIRS))
    return fig


def make_fig5(name="figure5_alt"):
    """Experimental layout: the three dumbbells across the top (a balanced,
    b TRACE, c gradient reversal), then the lambda sweep (d) and the
    cost-vs-benefit scatter (e) underneath."""
    out = Path(__file__).resolve().parent / "figure_5"
    out.mkdir(exist_ok=True)
    fig, axd = plt.subplot_mosaic([["a", "a", "b", "b", "c", "c"],
                                   ["d", "d", "d", "e", "e", "e"]],
                                  figsize=(15, 9.6), constrained_layout=True)
    dumbbell(axd["a"], "balanced", "o", pair_axis=True)
    dumbbell(axd["b"], "trace", "^")
    dumbbell(axd["c"], "gradscale_-1", "s")
    for k in "bc":
        axd[k].tick_params(labelleft=False)
    for k in "abc":
        axd[k].set_xlim(-0.01, 0.21)
        axd[k].set_ylim(len(PAIRS) + 0.5, 0.5)     # pair 1 on top
    gs_sweep(axd["d"])
    axd["d"].set_title("")
    tradeoff(axd["e"])
    for letter, a in axd.items():
        a.set_title(letter, loc="left", fontweight="bold", fontsize=13)
    fig.legend(handles=LEGEND, loc="outside upper center", ncol=len(LEGEND),
               fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out / (name + '.png')}")
    for cond, *_ in STRATEGIES:
        d, fake = _drops(cond)
        print(f"  {cond:13s}", "  ".join(f"P{q} gap-{d[q][0]:+.3f} auc-{d[q][1]:+.3f}" for q in PAIRS),
              "(placeholder)" if fake else "")
    return fig


if __name__ == "__main__":
    make_fig5()
