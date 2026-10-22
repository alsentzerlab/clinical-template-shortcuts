#!/usr/bin/env python3
"""Paper Fig 2 panel a (synthetic arm): two-row schematic, written as SVG.

  Step 1  Reformatting               real notes -> LLM extractor -> JSON -> Template A / B
  Step 2  Training Data Construction paired notes -> patient-level split ->
                                     one template per note (bias p) -> example split

Step 2 follows finetuning/process_data.py exactly (_assign_paired_train,
run with --group-column pat_id --match-rates --match-eval-bias, 70/15/15 split):
notes are split by patient BEFORE any template is chosen; within a split each
note is used once, as A or B; a fraction p of readmitted notes go to A, and
non-readmitted notes fill A and B to equal size; val/test use the same p.

Palette and type follow the Lucid-made figure_2/pipeline.svg.

Run:  cd figures && python3 figure2_pipeline_schematic.py   -> figure_2/pipeline_v2.svg
"""
from pathlib import Path

INK, SLATE, LIGHT, BG = "#1E2630", "#5E6B78", "#B8C0C8", "#F6F8FA"
CA, CB, TEAL = "#D9772B", "#2F6FB0", "#2A8C80"
CA_T, CB_T, TEAL_T = "#FBEBDD", "#E1ECF7", "#DCEFEC"
READ, READ_T = "#D81B60", "#FADBE6"   # readmitted: crimson dot on a pink note
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Courier New, monospace"
W, H = 1084, 694

out = []
add = out.append


def text(x, y, s, size=11, color=INK, weight="normal", anchor="middle", style="normal", font=FONT):
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def rect(x, y, w, h, fill="#FFFFFF", stroke=INK, sw=1.2, rx=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def bar(x, y, w, color=LIGHT, h=3):
    add(f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="1" fill="{color}"/>')


def arrow(x1, y1, x2, y2, color=SLATE, sw=1.3):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#arr)"/>')


def dot(cx, cy, r, fill):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')


def note(x, y, w, h, color=INK, head=None, lines=(1, .6, .9, .5, .8, .7, .95, .55), fill="#FFFFFF"):
    """A note card; `head` = coloured header strip (template identity)."""
    rect(x, y, w, h, fill=fill, stroke=color)
    top = y + 10
    if head:
        add(f'<rect x="{x}" y="{y}" width="{w}" height="7" rx="2" fill="{head}"/>')
        top = y + 16
    step = (h - (top - y) - 8) / len(lines)
    for i, f in enumerate(lines):
        bar(x + 8, top + i * step, (w - 16) * f, color=LIGHT if not head else LIGHT)


def template_card(x, y, w, h, color, sections):
    """Template card: coloured section headers each followed by grey body lines."""
    rect(x, y, w, h, stroke=color)
    add(f'<rect x="{x}" y="{y}" width="{w}" height="7" rx="2" fill="{color}"/>')
    rows = sum(1 + k for k in sections)
    step = (h - 26) / rows
    yy = y + 17
    for k in sections:
        bar(x + 8, yy, w * 0.35, color=color, h=3.5)
        yy += step
        for _ in range(k):
            bar(x + 14, yy, w - 26)
            yy += step


def mini(x, y, color, readmit, w=22, h=28):
    """Mini note icon for the example split; filled dot = readmitted."""
    rect(x, y, w, h, fill=READ_T if readmit else "#FFFFFF", stroke=color, sw=1)
    add(f'<rect x="{x}" y="{y}" width="{w}" height="4" rx="1.5" fill="{color}"/>')
    for i in range(3):
        bar(x + 4, y + 9 + i * 5, w - 8, h=2)
    if readmit:
        dot(x + w - 4, y + h - 4, 3.6, READ)


def dropped(x, y, color, readmit, w=22, h=26):
    """A version that is NOT used: faded, with a cross through it."""
    add('<g opacity="0.25">'); mini(x, y, color, readmit, w=w, h=h); add('</g>')
    for (x1, y1, x2, y2) in ((x - 2, y - 2, x + w + 2, y + h + 2), (x + w + 2, y - 2, x - 2, y + h + 2)):
        add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{SLATE}" stroke-width="1.4"/>')


def step_header(x, y, n, title):
    dot(x + 11, y - 4, 11, INK)
    text(x + 11, y, str(n), 12, "#FFFFFF", "bold")
    text(x + 30, y, title, 15, INK, "bold", anchor="start")


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# =============================== STEP 1 =====================================
step_header(24, 34, 1, "Reformatting")
cy = 185

# real notes: 3 sheets, evenly spaced ragged lines
for i, (dx, dy) in enumerate([(20, 0), (10, 10), (0, 20)]):
    note(40 + dx, 112 + dy, 96, 124,
         lines=[(1, .6, .9, .5, .8, .6, .95, .5), (.7, 1, .4, .8, .7, 1, .4, .8),
                (1, .5, .8, .9, .6, 1, .5, .8)][i])
text(98, 290, "Real discharge notes", 11.5, INK, "bold")

arrow(168, cy, 222, cy)

# LLM extractor
rect(226, cy - 52, 128, 104, fill=TEAL_T, stroke=TEAL, sw=1.3, rx=10)
nodes = [[(256, cy - 22), (256, cy), (256, cy + 22)], [(290, cy - 26), (290, cy - 8), (290, cy + 10), (290, cy + 28)],
         [(324, cy - 12), (324, cy + 12)]]
for a, b in zip(nodes, nodes[1:]):
    for p in a:
        for q in b:
            add(f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" stroke="#8CC4BC" stroke-width="0.9"/>')
for layer in nodes:
    for (x, y) in layer:
        dot(x, y, 4.2, TEAL)
text(290, cy - 66, "LLM extractor", 12.5, INK, "bold")

arrow(354, cy, 408, cy)

# JSON
jx, jy, jw, jh = 412, 92, 236, 186
rect(jx, jy, jw, jh, fill="#FFFFFF", stroke=INK, rx=6)
add(f'<rect x="{jx}" y="{jy}" width="{jw}" height="22" rx="6" fill="{INK}"/>')
add(f'<rect x="{jx}" y="{jy + 12}" width="{jw}" height="10" fill="{INK}"/>')
text(jx + 12, jy + 15, "note.json", 10.5, "#FFFFFF", "bold", anchor="start", font=MONO)
json_lines = [("{", INK), ('  "chief_complaint": "…",', None), ('  "hospital_course": "…",', None),
              ('  "discharge_meds": [ … ],', None), ('  "follow_up": "…",', None),
              ("  …", SLATE), ("}", INK)]
for i, (s, c) in enumerate(json_lines):
    if c is None:
        key, val = s.split(":", 1)
        add(f'<text x="{jx + 12}" y="{jy + 44 + i * 19}" font-family="{MONO}" font-size="10.5" '
            f'text-anchor="start" xml:space="preserve"><tspan fill="{TEAL}">{key}</tspan>'
            f'<tspan fill="{INK}">:{val}</tspan></text>')
    else:
        text(jx + 12, jy + 44 + i * 19, s, 10.5, c, anchor="start", font=MONO)
text(jx + jw / 2, 290, "Structured JSON", 11.5, INK, "bold")

# JSON -> two templates
arrow(648, cy - 30, 744, 104)
arrow(648, cy + 30, 744, 266)

template_card(748, 44, 104, 120, CA, [1, 1, 1, 1, 1])
template_card(748, 206, 104, 120, CB, [3, 3])
text(800, 36, "Template A", 11.5, CA, "bold")
text(800, 342, "Template B", 11.5, CB, "bold")

# brace + note to the right
add(f'<path d="M872,48 Q884,48 884,60 L884,173 Q884,185 894,185 Q884,185 884,197 L884,310 '
    f'Q884,322 872,322" fill="none" stroke="{SLATE}" stroke-width="1.2"/>')
text(906, cy - 4, "Every note rendered", 12.5, INK, "bold", anchor="start")
text(906, cy + 14, "into both templates", 12.5, INK, "bold", anchor="start")

# divider
add(f'<line x1="24" y1="372" x2="{W - 24}" y2="372" stroke="#C5CDD6" stroke-width="1" stroke-dasharray="4 4"/>')

# =============================== STEP 2 =====================================
step_header(24, 412, 2, "Training Data Construction")
TOP, BH = 448, 226
r2 = TOP + BH / 2     # vertical centre of row 2
GAP = 36              # space between boxes (holds the arrow)

# (i) paired notes: same note in both versions; readmission belongs to the note
bx, bw = 32, 250
rect(bx, TOP, bw, BH, fill=BG, stroke="#C5CDD6", rx=10)
text(bx + bw / 2, TOP + 24, "Paired notes", 12, INK, "bold")
text(bx + 103, TOP + 48, "A", 11.5, CA, "bold")
text(bx + 139, TOP + 48, "B", 11.5, CB, "bold")
for k, r in enumerate([0, 1, 0, 0, 1]):
    yy = TOP + 58 + k * 32
    if r:
        add(f'<rect x="{bx + 10}" y="{yy - 4}" width="{bw - 20}" height="31" rx="6" fill="{READ_T}"/>')
    text(bx + 20, yy + 16, f"Note {k + 1}", 10.5, INK, "bold", anchor="start")
    mini(bx + 92, yy, CA, r, w=22, h=23)
    mini(bx + 128, yy, CB, r, w=22, h=23)
    if r:
        text(bx + 162, yy + 15, "readmitted", 9.5, READ, "bold", anchor="start")

arrow(bx + bw + 2, r2, bx + bw + GAP - 4, r2)

# (ii) keep one version of each note
ax_ = bx + bw + GAP
aw = 380
rect(ax_, TOP, aw, BH, fill=BG, stroke="#C5CDD6", rx=10)
text(ax_ + aw / 2, TOP + 24, "Keep one version of each note", 12, INK, "bold")
add('<g transform="translate(0,6)">')
text(ax_ + 46, TOP + 62, "Readmitted", 10.5, INK, "bold")
mini(ax_ + 20, TOP + 70, CA, True, h=26); mini(ax_ + 50, TOP + 70, CB, True, h=26)
arrow(ax_ + 82, TOP + 80, ax_ + 150, TOP + 58)
arrow(ax_ + 82, TOP + 88, ax_ + 150, TOP + 112)
text(ax_ + 112, TOP + 59, "p", 12, CA, "bold", style="italic")
text(ax_ + 114, TOP + 123, "1 − p", 12, CB, "bold")
mini(ax_ + 158, TOP + 42, CA, True, h=26); dropped(ax_ + 188, TOP + 42, CB, True)
text(ax_ + 224, TOP + 60, "keep the A version", 10.5, CA, "bold", anchor="start")
dropped(ax_ + 158, TOP + 98, CA, True); mini(ax_ + 188, TOP + 98, CB, True, h=26)
text(ax_ + 224, TOP + 116, "keep the B version", 10.5, CB, "bold", anchor="start")
text(ax_ + 46, TOP + 152, "Not readmitted", 10.5, INK, "bold")
mini(ax_ + 20, TOP + 160, CA, False, h=26); mini(ax_ + 50, TOP + 160, CB, False, h=26)
arrow(ax_ + 82, TOP + 173, ax_ + 150, TOP + 173)
text(ax_ + 158, TOP + 169, "split between A and B so each", 10.5, SLATE, anchor="start")
text(ax_ + 158, TOP + 184, "template has the same number of notes", 10.5, SLATE, anchor="start")
add("</g>")

arrow(ax_ + aw + 2, r2, ax_ + aw + GAP - 4, r2)

# (iii) example training set
ex = ax_ + aw + GAP
ew = 330
rect(ex, TOP, ew, BH, fill=BG, stroke="#C5CDD6", rx=10)
text(ex + ew / 2, TOP + 24, "Resulting training set (example p = 0.8)", 12, INK, "bold")
ca_x, cb_x = ex + ew * 0.27, ex + ew * 0.73
add('<g transform="translate(0,30)">')   # centre the example below the title
text(ca_x, TOP + 52, "Template A", 11, CA, "bold")
text(cb_x, TOP + 52, "Template B", 11, CB, "bold")
a_read = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]     # 4 of 10 readmitted
b_read = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]     # 1 of 10  -> p = 4/5 = 0.8
for i, r in enumerate(a_read):
    mini(ca_x - 60 + (i % 5) * 24, TOP + 64 + (i // 5) * 34, CA, r)
for i, r in enumerate(b_read):
    mini(cb_x - 60 + (i % 5) * 24, TOP + 64 + (i // 5) * 34, CB, r)
text(ca_x, TOP + 154, "4 / 10 readmitted", 10.5, INK)
text(cb_x, TOP + 154, "1 / 10 readmitted", 10.5, INK)
add("</g>")

add("</svg>")
dst = Path(__file__).resolve().parent / "figure_2" / "pipeline_v2.svg"
dst.parent.mkdir(exist_ok=True)
dst.write_text("\n".join(out))
print(f"saved {dst}")
