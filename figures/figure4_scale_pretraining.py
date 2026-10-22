#!/usr/bin/env python3
"""Figure 4: scale + pretraining for the 5 template pairs (pairs on the x axis).

  a  shortcut gap vs pair, one line per Qwen-3 size (0.6B / 1.7B / 4B)
  b  shortcut gap vs pair, general (Llama-2 13B) vs clinical (MeLlama 13B) pretraining

All series come from auc_data.py: 0.6B (baseline) and 1.7B from Figures (6).pdf,
4B / Llama-2 / MeLlama from Figures (7).pdf.

Run:  cd figures && python figure4_scale_pretraining.py
Saves figures/figure_4/figure4.{png,pdf}.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import auc_data
import fig_style

sns.set_theme(style="ticks")
fig_style.use_style()                 # AFTER set_theme: drops top + right spines

PAIRS, INK = fig_style.PAIRS, fig_style.INK
MODE = "mean"

# one light -> dark blue ramp shared by both panels (more capacity / more
# clinical pretraining = darker)
BLUES = ["#deebf7", "#9ecae1", "#3182bd"]   # ColorBrewer Blues 3-class
SCALE = [("baseline", "0.6B", BLUES[0]), ("scale_1.7B", "1.7B", BLUES[1]),
         ("scale_4B", "4B", BLUES[2])]
PRETRAIN = [("pretrain_llama2", "Llama-2 (general)", BLUES[1]),
            ("pretrain_mellama", "MeLlama (clinical)", BLUES[2])]
YLIM = (-0.01, 0.27)


def _lines(ax, spec, title):
    x = np.arange(len(PAIRS))
    for cond, name, color in spec:
        d = auc_data.deltas(cond, MODE)
        ax.plot(x, [d[p] for p in PAIRS], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=color, markeredgewidth=1.6, label=name)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Shortcut gap (Δ)")
    ax.legend(title=title, loc="upper left", fontsize=9, title_fontsize=9.5,
              frameon=False)
    fig_style.pair_axis(ax, positions=x, fontsize=8)


def make_fig3():
    out = Path(__file__).resolve().parent / "figure_4"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True,
                           constrained_layout=True)
    _lines(ax[0], SCALE, "Qwen-3 size")
    _lines(ax[1], PRETRAIN, "Pretraining (13B)")
    ax[1].tick_params(labelleft=True)
    for a, letter in zip(ax, "ab"):
        a.set_title(letter, loc="left", fontweight="bold", fontsize=13)
    fig_style.align_disparity(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure4.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out / 'figure4.png'}")
    return fig


if __name__ == "__main__":
    make_fig3()
