#!/usr/bin/env python3
"""Figure 2: synthetic-data bias sweep: what the model learns as p moves away from 0.5.

p is the training template-outcome coupling. p = 0.5 is the unbiased case;
moving toward p = 0 or p = 1 makes the template an increasingly strong shortcut.

  a  overall readmission AUC vs p
  b  overall vs within-template AUC; the wedge between them IS the shortcut gap
  c  the shortcut gap on its own

Numbers are the paper's sweep, as written by finetuning/auc_decomposition.py
(results/auc_decomposition.csv) and copied into SWEEP below. To plot a new
sweep, pass its CSV:  python figure2_synthetic.py --csv path/to/auc_decomposition.csv

Two gap definitions are kept. `overall - AUC_within` is the better one:
the unweighted mean over-weights whichever template has few positives, which at
extreme p is the noisy one. `overall - mean(auc_A, auc_B)` is kept because the
manuscript text currently quotes it. They agree at p = 0, 0.5 and 1.

Run:  cd figures && python figure2_synthetic.py
Saves figures/figure_2/synthetic_bias_sweep_{ab,within}.png (+ .pdf) and the
final 2-panel figure figure_2/figure2.{png,pdf}: a = pipeline schematic,
b = overall vs within-template AUC.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import fig_style

sns.set_theme(style="ticks")
fig_style.use_style()

# ---- DATA ------------------------------------------------------------------
# The paper's sweep (Qwen-3 0.6B, 3,000 paired notes), from auc_decomposition.py.
# auc_a is nan at p = 0 and auc_b at p = 1: only one template has positives
# there, so both gap definitions collapse to overall - that template's AUC.
nan = float("nan")
SWEEP = pd.DataFrame([
    # p, overall, auc_a, auc_b, auc_within, auc_cross, overall-within, overall-mean(a,b)
    (0.0, 0.8023, nan, 0.5248, 0.5248, 0.9906, +0.2775, +0.2775),
    (0.1, 0.7354, 0.6425, 0.5496, 0.5615, 0.8709, +0.1739, +0.1394),
    (0.2, 0.6586, 0.6041, 0.5300, 0.5473, 0.7550, +0.1113, +0.0916),
    (0.3, 0.6330, 0.6143, 0.5715, 0.5860, 0.6773, +0.0470, +0.0401),
    (0.4, 0.5899, 0.6118, 0.5502, 0.5761, 0.6036, +0.0138, +0.0089),
    (0.5, 0.6033, 0.6300, 0.5734, 0.6017, 0.6049, +0.0016, +0.0016),
    (0.6, 0.5828, 0.5619, 0.6140, 0.5838, 0.5818, -0.0010, -0.0051),
    (0.7, 0.5804, 0.5540, 0.5905, 0.5663, 0.5937, +0.0141, +0.0082),
    (0.8, 0.6353, 0.5726, 0.5619, 0.5701, 0.6918, +0.0652, +0.0681),
    (0.9, 0.7113, 0.5697, 0.5346, 0.5652, 0.8250, +0.1460, +0.1591),
    (1.0, 0.8323, 0.6086, nan, 0.6086, 0.9842, +0.2238, +0.2238),
], columns=["p_val", "overall_auc", "auc_a", "auc_b", "auc_within", "auc_cross",
            "overall_minus_within", "overall_minus_mean_ab"])

_ap = argparse.ArgumentParser()
_ap.add_argument("--csv", default=None, help="auc_decomposition.csv from a new sweep")
CSV = _ap.parse_args().csv
if CSV:
    df = pd.read_csv(CSV)
    df["p_val"] = df["p"].str.lstrip("p").astype(int) / 100.0
else:
    df = SWEEP.copy()
df = df.sort_values("p_val").reset_index(drop=True)

p = df["p_val"].to_numpy()
overall = df["overall_auc"].to_numpy()
within = df["auc_within"].to_numpy()
cross = df["auc_cross"].to_numpy()
gap_within = df["overall_minus_within"].to_numpy()
gap_mean = df["overall_minus_mean_ab"].to_numpy()

# ---- palette ---------------------------------------------------------------
# Validated with the dataviz palette checker (light surface): passes the
# lightness band, chroma floor, CVD separation, normal-vision floor and contrast
# checks. Hues follow the paper's convention -- overall AUC green, gap red.
C_OVERALL, C_STRAT, C_GAP = "#004D40", "#1E88E5", "#D81B60"

GRID, INK, MUTED = "#e6e6e6", "#222222", "#777777"

MARK = dict(marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.2, lw=2)


def _p_axis(ax, ylo, yhi):
    """Shared x-axis furniture: the unbiased reference line at p = 0.5."""
    ax.axvline(0.5, color=MUTED, ls="--", lw=1, zorder=0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("$p$")
    ax.set_ylim(ylo, yhi)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _bias_arrows(ax, y, label_dy):
    """'increasing bias' runs outward in both directions from p = 0.5."""
    for x_to in (0.02, 0.98):
        ax.annotate("", xy=(x_to, y), xytext=(0.5, y),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    for x_mid in (0.26, 0.74):
        ax.text(x_mid, y + label_dy, "increasing bias", color=MUTED,
                fontsize=7.5, ha="center")


def _bias_arrows_below(ax, label_pt=44, arrow_pt=52, title_pt=62):
    """Below the x axis, top to bottom: the axis label p, outward 'stronger
    association' arrows from p = 0.5, then 'Template-outcome association'.
    p itself is not the association (that grows in BOTH directions from 0.5),
    so the two are labelled separately. Offsets are points below the axis."""
    from matplotlib.transforms import offset_copy
    base = ax.get_xaxis_transform()
    tr = lambda pt: offset_copy(base, fig=ax.figure, y=-pt, units="points")
    for x_to in (0.0, 1.0):
        ax.annotate("", xy=(x_to, 0), xytext=(0.5, 0), xycoords=tr(arrow_pt),
                    textcoords=tr(arrow_pt), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, mutation_scale=10))
    for x_mid in (0.25, 0.75):
        ax.text(x_mid, 0, "stronger association", color=MUTED, fontsize=8,
                ha="center", va="bottom", transform=tr(label_pt))
    ax.text(0.5, 0, "Template–outcome association", fontsize=11, color=INK,
            ha="center", va="top", transform=tr(title_pt))
    ax.xaxis.labelpad = 4


def panel_overall(ax):
    """a: the overall readmission AUC is U-shaped in p."""
    ax.plot(p, overall, color=C_OVERALL, **MARK)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=0)
    ax.text(-0.04, 0.503, "chance", color=MUTED, fontsize=7.5, ha="left", va="bottom")

    _p_axis(ax, 0.48, 0.90)
    ax.set_ylabel("Overall readmission AUC")
    _bias_arrows(ax, 0.855, 0.012)


def panel_within(ax):
    """b: overall vs within-template AUC. Within-template AUC is the model's
    genuine discriminative skill and is flat across the sweep; the shaded wedge
    above it -- the shortcut gap -- is what template identity buys."""
    ax.fill_between(p, within, overall, color=C_GAP, alpha=0.15, lw=0, zorder=0,
                    label="Shortcut gap (\u0394)")
    ax.plot(p, overall, color=C_OVERALL, label="Overall AUROC", **MARK)
    ax.plot(p, within, color=C_STRAT, label="Within-template AUROC", **MARK)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1, zorder=0)

    _p_axis(ax, 0.0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("AUROC")
    _bias_arrows_below(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=8, frameon=False)


def panel_gap(ax, within_ref=True):
    """c: the shortcut gap on its own. within_ref picks the reference the gap is
    measured against -- keep it matched to whatever panel b draws."""
    y = gap_within if within_ref else gap_mean
    label = ("AUC gap  (overall $-$ within)" if within_ref
             else "AUC gap  (overall $-$ mean(A, B))")
    ax.plot(p, y, color=C_GAP, **MARK)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)

    _p_axis(ax, -0.03, 0.32)
    ax.set_ylabel(label)
    _bias_arrows(ax, 0.295, 0.009)


# ---- schematic (panel a of the final figure) ---------------------------------
from matplotlib.patches import FancyBboxPatch

C_TA, C_TB = "#e08214", "#35978f"   # Template A / B (off the green/blue/red data hues)
LINE = "#bdbdbd"


def _card(ax, x, y, w, h, headers, color=None, face="white"):
    """A note card: coloured section headers each followed by grey text lines.
    `headers` = number of text lines under each section."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                lw=1.1, edgecolor=INK, facecolor=face, zorder=2))
    n_rows = sum(1 + k for k in headers)
    step = (h - 0.24) / n_rows
    yy = y + h - 0.12 - step / 2
    for k in headers:
        ax.plot([x + 0.12, x + 0.12 + w * 0.38], [yy, yy], color=color or INK,
                lw=2.4, solid_capstyle="round", zorder=3)
        yy -= step
        for _ in range(k):
            ax.plot([x + 0.18, x + w - 0.14], [yy, yy], color=LINE, lw=1.3,
                    solid_capstyle="round", zorder=3)
            yy -= step


def _free_note(ax, x, y, w, h, seed):
    """A raw note in its own free-text format: ragged grey lines, no structure."""
    rng = np.random.default_rng(seed)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                lw=1.1, edgecolor=INK, facecolor="white", zorder=2))
    for yy in np.linspace(y + h - 0.18, y + 0.18, 7):
        ax.plot([x + 0.12, x + 0.12 + (w - 0.24) * rng.uniform(0.45, 1.0)], [yy, yy],
                color=LINE, lw=1.3, solid_capstyle="round", zorder=3)


def _arrow(ax, xy_from, xy_to, text=None, text_xy=None, **kw):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=12,
                                shrinkA=0, shrinkB=0, **kw))
    if text:
        ax.text(*text_xy, text, ha="center", va="bottom", fontsize=8, color=INK)


def panel_schematic(ax):
    """a: reformat once with an LLM, then re-key deterministically into two
    templates with identical content; bias the template-outcome link with p."""
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.2)
    ax.set_aspect("equal"); ax.axis("off")

    # 1  real notes, varied formats
    for i, (dx, dy) in enumerate([(0.3, 0.3), (0.15, 0.15), (0, 0)]):
        _free_note(ax, 0.2 + dx, 2.2 + dy, 1.5, 1.9, seed=i)
    ax.text(1.1, 1.85, "Real discharge notes\n(n = 3,000;\nvaried formats)",
            ha="center", va="top", fontsize=8, color=INK)

    # 2  LLM reformat -> standardized
    _arrow(ax, (2.1, 3.25), (3.5, 3.25), "LLM reformat", (2.8, 3.35))
    ax.text(2.8, 3.1, "GPT-5.5\n+ repair", ha="center", va="top", fontsize=7, color=MUTED)
    _card(ax, 3.6, 2.2, 1.5, 2.1, headers=[2, 2, 2, 1])
    ax.text(4.35, 1.85, "Standardized note\n(62 content fields)",
            ha="center", va="top", fontsize=8, color=INK)

    # 3  deterministic re-keying -> A / B
    _arrow(ax, (5.2, 3.6), (6.55, 4.55))
    _arrow(ax, (5.2, 2.9), (6.55, 1.95))
    ax.text(5.8, 3.25, "re-key\n(no LLM)", ha="center", va="center", fontsize=7.5,
            color=INK)
    _card(ax, 6.65, 3.7, 1.5, 2.1, headers=[1] * 7, color=C_TA)
    _card(ax, 6.65, 0.4, 1.5, 2.1, headers=[3, 3, 2], color=C_TB)
    ax.text(7.4, 5.95, "Template A", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=C_TA)
    ax.text(7.4, 0.25, "Template B", ha="center", va="top", fontsize=9,
            fontweight="bold", color=C_TB)
    ax.text(7.4, 3.1, "same content,\ndifferent structure", ha="center",
            va="center", fontsize=7.5, style="italic", color=MUTED)

    # 4  biased assignment in training
    ax.add_patch(FancyBboxPatch((8.85, 1.35), 3.4, 3.5,
                                boxstyle="round,pad=0,rounding_size=0.12", lw=1.0,
                                edgecolor=MUTED, facecolor="#f7f7f7", zorder=1))
    _arrow(ax, (8.25, 4.75), (8.8, 3.6))
    _arrow(ax, (8.25, 1.45), (8.8, 2.6))
    ax.text(10.55, 4.6, "Training split", ha="center", va="top", fontsize=9,
            fontweight="bold", color=INK)
    ax.text(10.55, 4.05, "each readmitted note\nis shown as", ha="center", va="top",
            fontsize=8, color=INK)
    ax.text(10.55, 3.2, "Template A", ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=C_TA)
    ax.text(10.55, 2.85, "with probability $p$", ha="center", va="top", fontsize=8,
            color=INK)
    ax.text(10.55, 2.25, "$p$ = 0.5: no shortcut\n$p$ → 0 or 1: template\npredicts readmission",
            ha="center", va="top", fontsize=7.5, color=MUTED)


PIPELINE_SVG = Path(__file__).resolve().parent / "figure_2" / "pipeline_v2.svg"   # figure2_pipeline_schematic.py
PIPELINE_PNG = PIPELINE_SVG.with_suffix(".png")    # 4x raster of the SVG, cached


def panel_pipeline(ax):
    """a: the reformatting pipeline, from figure_2/pipeline_v2.svg (written by
    figure2_pipeline_schematic.py). Re-rasterised at 4x when cairosvg and the
    cairo library are available; otherwise the committed pipeline_v2.png is used."""
    try:
        import cairosvg
        # cairo's Helvetica has no "→": set any line containing it in Arial
        import re
        svg = re.sub(r'(<text[^>]*font-family=")[^"]*("[^>]*>[^<]*→)', r"\1Arial\2",
                     PIPELINE_SVG.read_text())
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(PIPELINE_PNG), scale=4)
    except (ImportError, OSError):
        pass
    ax.imshow(plt.imread(PIPELINE_PNG), interpolation="lanczos")
    ax.axis("off")


def main():
    out = Path(__file__).resolve().parent / "figure_2"
    out.mkdir(exist_ok=True)

    # compact version: the illusion, then the metric
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 4.0),
                                     constrained_layout=True)
    panel_overall(ax_a)
    panel_gap(ax_b, within_ref=False)
    for a, letter in zip((ax_a, ax_b), "ab"):
        a.set_title(letter, loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"synthetic_bias_sweep_ab.{ext}", dpi=200, bbox_inches="tight")

    # full version: illusion (a), mechanism (b), metric (c). b and c share the
    # within-template reference, so the wedge in b is exactly the curve in c.
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    panel_overall(axes[0])
    panel_within(axes[1])
    panel_gap(axes[2], within_ref=True)
    for a, letter in zip(axes, "abc"):
        a.set_title(letter, loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"synthetic_bias_sweep_within.{ext}", dpi=200, bbox_inches="tight")

    # FINAL figure: reformatting schematic (a) + overall vs within-template AUC (b)
    # a = two-row schematic across the top; b = AUC plot centred underneath
    # row heights sized so the schematic (1084 x 734) fills the full width
    fig3 = plt.figure(figsize=(10.5, 12.7), constrained_layout=True)
    gs = fig3.add_gridspec(2, 1, height_ratios=[6.75, 5.5])
    ax_a, ax_b = fig3.add_subplot(gs[0]), fig3.add_subplot(gs[1])
    panel_pipeline(ax_a)
    panel_within(ax_b)
    for a, letter in zip((ax_a, ax_b), "ab"):
        a.set_title(letter, loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig3.savefig(out / f"figure2.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out/'figure2.png'}")

    print(f"read {CSV or 'SWEEP table'}  ({len(df)} levels, p={p.min():.1f}-{p.max():.1f})")
    print(f"saved {out/'synthetic_bias_sweep_ab.png'}")
    print(f"saved {out/'synthetic_bias_sweep_within.png'}")
    mid = (p >= 0.4) & (p <= 0.7)
    print(f"  within AUC  : {np.nanmin(within):.4f}-{np.nanmax(within):.4f} "
          f"(excl. endpoints {np.nanmin(within[1:-1]):.4f}-{np.nanmax(within[1:-1]):.4f})")
    print(f"  overall AUC : {np.nanmin(overall):.4f}-{np.nanmax(overall):.4f}")
    print(f"  noise floor : |gap_within| <= {np.nanmax(np.abs(gap_within[mid])):.4f}, "
          f"|gap_mean| <= {np.nanmax(np.abs(gap_mean[mid])):.4f}  (p=0.4-0.7)")


if __name__ == "__main__":
    main()
