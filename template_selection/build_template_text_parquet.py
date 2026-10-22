#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a two-column parquet of every unique template and its full shell text.

    template_id  int64
    template     string   the SmartPhrase shell (section headers, boilerplate,
                          @placeholders@), RTF-stripped; null if the pickle has
                          no shell for that id

The template set defaults to the templates that actually appear in
notes_with_templates.parquet, so this file joins 1:1 against the exploded
template_ids column of that parquet. Pass --all-mapped to instead take every
template in the id map, including those that only appear on notes outside the
discharge-summary set.

Shell text is the SmartPhrase DEFINITION, not patient text -- but it is read
from a PHI-adjacent pickle, so nothing is printed to stdout beyond counts.

Loaders are imported from template_pool so the shell text
matches template_pool_txt/ byte for byte.

Usage:
    uv run python build_template_text_parquet.py --out templates_with_text.parquet
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_pool import (  # noqa: E402
    build_template_texts,
    load_pickle,
)


def ids_from_notes(notes_parquet):
    """Distinct template ids appearing in the note-level parquet."""
    df = pd.read_parquet(os.path.expanduser(notes_parquet), columns=["template_ids"])
    s = df["template_ids"].explode().dropna()
    return sorted({int(i) for i in s})


def ids_from_map(template_ids_pkl):
    t = load_pickle(template_ids_pkl)
    out = set()
    for ids in t["smartphrases_id"]:
        if ids is not None:
            out.update(int(i) for i in ids)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-parquet", default="notes_with_templates.parquet",
                    help="Note-level parquet from build_note_template_parquet.py")
    ap.add_argument("--template-ids-pkl", default="data/raw/template_ids.pkl")
    ap.add_argument("--smartphrases-pkl", default="data/raw/smartphrases.pkl")
    ap.add_argument("--out", default="templates_with_text.parquet")
    ap.add_argument("--all-mapped", action="store_true",
                    help="Use every template in the id map instead of only those "
                         "appearing in --notes-parquet")
    ap.add_argument("--only-with-text", action="store_true",
                    help="Drop templates that have no shell in the pickle instead "
                         "of emitting them with a null 'template'")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    if args.all_mapped:
        print("Collecting template ids from the id map...")
        ids = ids_from_map(args.template_ids_pkl)
    else:
        print(f"Collecting template ids from {args.notes_parquet}...")
        ids = ids_from_notes(args.notes_parquet)
    print(f"  {len(ids)} unique templates")

    print("Loading smartphrase shells (this reads a ~4 GB pickle)...")
    texts = build_template_texts(load_pickle(args.smartphrases_pkl))
    print(f"  {len(texts)} templates have shell text in the pickle")

    rows = [{"template_id": tid, "template": texts.get(tid)} for tid in ids]
    df = pd.DataFrame(rows)

    n_text = int(df["template"].notna().sum())
    print(f"  {n_text} of {len(df)} requested templates matched a shell "
          f"({len(df) - n_text} missing)")

    if args.only_with_text:
        df = df[df["template"].notna()].reset_index(drop=True)
        print(f"  --only-with-text keeps {len(df)}")

    out = os.path.expanduser(args.out)
    df.to_parquet(out, index=False, compression=args.compression)
    size_mb = os.path.getsize(out) / 1e6
    print(f"\n{len(df)} templates -> {out} ({size_mb:.1f} MB)")

    lens = df["template"].dropna().str.len()
    if len(lens):
        print(f"  shell length: median {int(lens.median())} chars, "
              f"min {int(lens.min())}, max {int(lens.max())}")


if __name__ == "__main__":
    main()
