#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swap one template in a sampled set for another, recomputing similarity columns.

A hand review of the browser will sometimes reject a single pick -- two shells
that look alike, or a shell whose content is wrong for the study. Re-running the
sampler with a different seed would redraw all twelve and throw away the picks
that were already approved, so this replaces just the one and leaves the rest
byte-identical.

Usage:
    uv run python swap_template.py --sample heldout_templates.csv \
        --drop 970744 --add 714564
"""

import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SHELL_DIRS = ["template_pool_txt", "nonim_pair_templates_txt",
              "generalization_templates_txt"]
IN_USE_IDS = [206596, 248528, 229657, 624343, 801125, 249614, 1337772]


def read_shell(tid):
    for d in SHELL_DIRS:
        p = os.path.join(d, f"template_{int(tid)}.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="heldout_templates.csv")
    ap.add_argument("--drop", type=int, required=True)
    ap.add_argument("--add", type=int, required=True)
    ap.add_argument("--frame", default="template_sampling_frame.csv")
    ap.add_argument("--classifications", default="template_pool_classifications.csv")
    ap.add_argument("--out", default=None, help="Default: overwrite --sample")
    args = ap.parse_args()

    df = pd.read_csv(args.sample)
    if args.drop not in set(df.template_id):
        raise SystemExit(f"{args.drop} is not in {args.sample}")
    if args.add in set(df.template_id):
        raise SystemExit(f"{args.add} is already in {args.sample}")
    if args.add in IN_USE_IDS:
        raise SystemExit(f"{args.add} is a TRAINING template; it cannot be held out")
    if read_shell(args.add) is None:
        raise SystemExit(f"no shell on disk for {args.add}")

    frame = pd.read_csv(args.frame)
    row = frame[frame.template_id == args.add]
    if row.empty:
        raise SystemExit(f"{args.add} is not in {args.frame}")
    row = row.iloc[0]

    new = {c: None for c in df.columns}
    new.update({"template_id": args.add, "n_notes": int(row.n_notes),
                "readmission_rate": float(row.readmission_rate)})
    if "shell_chars" in df.columns:
        new["shell_chars"] = int(row.get("shell_chars", len(read_shell(args.add))))
    if os.path.exists(args.classifications):
        cls = pd.read_csv(args.classifications)
        c = cls[cls.template_id == args.add]
        if not c.empty:
            for k in ("classification", "confidence", "specialty_detail"):
                if k in df.columns and k in c.columns:
                    new[k] = c.iloc[0][k]

    out = pd.concat([df[df.template_id != args.drop], pd.DataFrame([new])],
                    ignore_index=True).sort_values("readmission_rate")

    # Recompute pick-vs-pick and pick-vs-in-use similarity for the whole set.
    inuse = [t for t in IN_USE_IDS if read_shell(t) is not None]
    ids = list(out.template_id) + inuse
    S = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9,
                                          sublinear_tf=True)
                          .fit_transform([read_shell(t) for t in ids]))
    n = len(out)
    if "max_sim_to_pick" in out.columns:
        out["max_sim_to_pick"] = [round(max(S[i, j] for j in range(n) if j != i), 3)
                                  for i in range(n)]
    if "max_sim_to_in_use" in out.columns and inuse:
        out["max_sim_to_in_use"] = S[:n, n:].max(axis=1).round(3)
    if "stratum" in out.columns:
        out["stratum"] = range(n)

    dest = args.out or args.sample
    out.to_csv(dest, index=False)
    r = out.readmission_rate.values
    print(f"swapped {args.drop} -> {args.add}; wrote {dest}\n")
    print(out[[c for c in ("template_id", "n_notes", "readmission_rate",
                           "classification", "max_sim_to_pick") if c in out]]
          .to_string(index=False))
    print(f"\n{len(out)} templates  {int(out.n_notes.sum()):,} notes  "
          f"rate {100 * r.min():.1f}%-{100 * r.max():.1f}% "
          f"({100 * (r.max() - r.min()):.1f}pp)  "
          f"max pairwise sim {out.max_sim_to_pick.max():.3f}")


if __name__ == "__main__":
    main()
