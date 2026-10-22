"""Shared styling for the shortcut-learning figures.

Importing this module sets the global look:
  * only the left + bottom spines are drawn (no boxed-in axes)

It also provides the shared pair axis: instead of labelling the template pairs
"Pair 1 ... Pair 5", we label them by the readmission-rate gap between the two
templates in the pair and draw a colour-ramped arrow reading
"Increasing readmission rate disparity".

NOTE: the pairs are ordered by that gap, so the arrow direction is meaningful.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as _Path
from matplotlib.patches import PathPatch

# ---------------------------------------------------------------- palette ----
# The paper's 5 template pairs, ordered by readmission-rate disparity. No
# per-pair colour ramp any more: pair identity is carried by the tick label, and
# every pair-level mark (bars, held-out points) is one blue fill.
PAIRS      = [1, 2, 3, 4, 5]
PAIR_FILL  = "#1E88E5"                 # blue
PAIR_RAMP  = [PAIR_FILL] * len(PAIRS)
PAIR_COLOR = {p: PAIR_FILL for p in PAIRS}
ARROW_GREY = "#636363"                 # Greys, dark -- the disparity arrow
PAIR_IDS   = {1: "248528_229657", 2: "801125_229657", 3: "206596_801125",
              4: "206596_248528", 5: "206596_229657"}
INK = "#222222"

# Tick labels wear ink, not the series colour.
TICK_COLOR = {p: INK for p in PAIRS}

ARROW_CMAP = LinearSegmentedColormap.from_list("arrow", [ARROW_GREY, ARROW_GREY])

# ------------------------------------------------------- readmission rates ----
# 30-day readmission rate (%) per training template.
TEMPLATE_RATE = {"206596": 53.5, "801125": 14.5, "248528": 8.7, "229657": 3.6}
# One letter per training template, assigned in order of first appearance
# along the pair axis (pair 1 = A/B). Used for every template label in every
# figure, so a letter always means the same note type.
TEMPLATE_LETTER = {"248528": "A", "229657": "B", "801125": "C", "206596": "D"}
# Absolute gap (pp) between the two templates of each pair: 5.1/10.9/39.0/44.8/49.9
READMIT_GAP = {p: round(abs(TEMPLATE_RATE[a] - TEMPLATE_RATE[b]), 1)
               for p, (a, b) in ((p, PAIR_IDS[p].split("_")) for p in PAIRS)}

DISPARITY_LABEL = "Increasing readmission rate disparity"


def use_style():
    """Drop the top and right spines everywhere (call once per notebook)."""
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


use_style()


def gap_ticklabels(unit=True, show_pair=True):
    """Tick labels for the pair axis: 'Pair N' over the readmission gap.

    The gap is written "Δ 8" (percentage points implied) rather than "8 pp".
    """
    gap = [f"Δ {READMIT_GAP[p]:.0f} pp" if unit else f"{READMIT_GAP[p]:.0f}" for p in PAIRS]
    if not show_pair:
        return gap
    return [f"Pair {p}\n{g}" for p, g in zip(PAIRS, gap)]


def _arrow_patch(ax_inset, orient):
    """An arrow-shaped clip path filling the inset axes (data coords 0-1)."""
    lo, hi, head = 0.30, 0.70, 0.86           # shaft edges, where the head starts
    if orient == "x":                          # points right
        verts = [(0, lo), (head, lo), (head, 0.0), (1.0, 0.5),
                 (head, 1.0), (head, hi), (0, hi), (0, lo)]
    else:                                      # points down (pair 1 at top)
        verts = [(lo, 1.0), (lo, 1 - head), (0.0, 1 - head), (0.5, 0.0),
                 (1.0, 1 - head), (hi, 1 - head), (hi, 1.0), (lo, 1.0)]
    patch = PathPatch(_Path(verts), transform=ax_inset.transData,
                      facecolor="none", edgecolor="none")
    ax_inset.add_patch(patch)
    return patch


def disparity_arrow(ax, orient="x", offset=-0.14, thickness=0.028, length=0.72):
    """Draw the solid 'increasing disparity' arrow beside `ax`.

    offset/thickness are in axes-fraction units, so they may need nudging when
    the panel is unusually short or tall. `length` is the fraction of the axis
    the arrow spans (centred).
    """
    lead = (1.0 - length) / 2
    if orient == "x":
        rect, grad = [lead, offset, length, thickness], np.linspace(0, 1, 256).reshape(1, -1)
    else:
        rect, grad = [offset, lead, thickness, length], np.linspace(0, 1, 256).reshape(-1, 1)

    cax = ax.inset_axes(rect, transform=ax.transAxes, zorder=5)
    cax.set_axis_off()
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    im = cax.imshow(grad, cmap=ARROW_CMAP, aspect="auto", extent=[0, 1, 0, 1],
                    interpolation="bilinear")
    im.set_clip_path(_arrow_patch(cax, orient))
    return cax


def pair_axis(ax, orient="x", positions=None, arrow=True, label=DISPARITY_LABEL,
              labelpad=32, fontsize=8, label_fontsize=9, unit=True, show_pair=True,
              **arrow_kw):
    """Re-label a template-pair axis by 'Pair N' + readmission gap, and add the
    disparity arrow.

    positions: tick locations for pairs 1..5. Defaults to 0..4 (categorical
    seaborn axes); pass [0.5 .. 4.5] for heatmaps or [1 .. 5] for figure 4.
    """
    if positions is None:
        positions = range(len(PAIRS))
    labels = gap_ticklabels(unit, show_pair)
    # two-line labels need the arrow pushed further off the axis
    if show_pair:
        arrow_kw.setdefault("offset", -0.185 if orient == "x" else -0.185)

    setter = (ax.set_xticks, ax.set_xticklabels) if orient == "x" else (ax.set_yticks, ax.set_yticklabels)
    setter[0](list(positions))
    setter[1](labels, fontsize=fontsize)

    ticklabels = ax.get_xticklabels() if orient == "x" else ax.get_yticklabels()
    for p, tl in zip(PAIRS, ticklabels):
        tl.set_color(TICK_COLOR[p])
        tl.set_fontweight("bold")
        if orient == "y":
            tl.set_rotation(0)

    if arrow:
        cax = disparity_arrow(ax, orient=orient, **arrow_kw)
        # remembered so align_disparity() can re-place it once the layout is final
        ax._disparity = dict(cax=cax, orient=orient, kw=dict(arrow_kw))
    if label:
        if orient == "x":
            ax.set_xlabel(label, labelpad=labelpad, fontsize=label_fontsize)
        else:
            ax.set_ylabel(label, labelpad=labelpad, fontsize=label_fontsize)
    return ax


def align_disparity(fig, gap=0.13, thickness=0.13, label_gap=-0.01):
    """Second pass over a finished figure: put every horizontal disparity arrow
    a fixed PHYSICAL distance (inches) below its tick labels, and its axis label
    the same distance below the arrow.

    `disparity_arrow` works in axes fractions, so panels of different heights
    would otherwise end up with visibly different spacing. Call once after all
    panels are drawn and before saving.
    """
    fig.canvas.draw()                       # constrained_layout must be settled
    for ax in fig.axes:
        d = getattr(ax, "_disparity", None)
        if d is None or d["orient"] != "x":
            continue
        box = ax.get_window_extent()
        labels = [t for t in ax.get_xticklabels() if t.get_text()]
        if not labels:
            continue
        depth = (box.y0 - min(t.get_window_extent().y0 for t in labels)) / fig.dpi
        h_in = box.height / fig.dpi

        kw = dict(d["kw"])
        kw["offset"] = -(depth + gap) / h_in
        kw["thickness"] = thickness / h_in
        d["cax"].remove()
        d["cax"] = disparity_arrow(ax, orient="x", **kw)
        ax.xaxis.labelpad = (gap + thickness + label_gap) * 72


# ------------------------------------------------------------ ROC series ----
# Overall / first template of the pair / second template. All candidates are
# colourblind-safe ColorBrewer schemes (no red-vs-green pairing).
ROC_PALETTES = {
    "dark2":  {"overall": "#1b9e77", "anchor": "#d95f02", "paired": "#7570b3"},  # Dark2
    "rdbu":   {"overall": "#252525", "anchor": "#0571b0", "paired": "#ca0020"},  # RdBu ends + black
    "puor":   {"overall": "#252525", "anchor": "#e66101", "paired": "#5e3c99"},  # PuOr ends + black
    "brbg":   {"overall": "#252525", "anchor": "#a6611a", "paired": "#018571"},  # BrBG ends + black
    "rdylbu": {"overall": "#252525", "anchor": "#fc8d59", "paired": "#91bfdb"},  # RdYlBu 3-class + black
    "cvd4":   {"overall": "#004D40", "anchor": "#1E88E5", "paired": "#D81B60"},  # green/blue/crimson (+ amber bars)
    "cvd4_y_for_blue":    {"overall": "#004D40", "anchor": "#FFC107", "paired": "#D81B60"},  # blue bars
    "cvd4_y_for_crimson": {"overall": "#004D40", "anchor": "#1E88E5", "paired": "#FFC107"},  # crimson bars
    "cvd4_y_for_green":   {"overall": "#FFC107", "anchor": "#1E88E5", "paired": "#D81B60"},  # green bars
}
ROC_PALETTE = "cvd4_y_for_blue"
ROC_SERIES_COLORS = ROC_PALETTES[ROC_PALETTE]
