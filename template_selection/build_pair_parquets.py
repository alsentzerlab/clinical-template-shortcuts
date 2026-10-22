#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the per-pair note parquets that finetuning/split_pairs.sh splits.

One parquet per pair, named <A-id>_<B-id>.parquet, holding every note of both
templates (no subsampling):

    pat_enc_csn_id  note_id  pat_id  full_note  readmit_binary  template  template_id

`template` is 'A' for the first ID in the pair name and 'B' for the second (the
convention split_report.infer_label_map assumes). pat_id is included so
process_data.py --group-column pat_id needs no separate group map.

Notes carrying BOTH templates of a pair are dropped (their arm is ambiguous);
notes carrying one pair template plus unrelated templates are kept.

*** PHI WARNING ***
The outputs contain full clinical notes and identifiers. They are PHI, must stay
in approved storage, and are never printed -- only counts and rates go to stdout.
Outputs land in data/pairs/, which .gitignore excludes.

Usage:
    uv run python build_pair_parquets.py \
        --reference 801125_229657=path/to/existing_801125_229657.parquet
"""
import argparse
import os

import pandas as pd
import pyarrow.parquet as pq

from template_pool import load_pickle

PAPER_PAIRS = ["248528_229657", "801125_229657", "206596_801125", "206596_248528", "206596_229657"]
OUT_COLS = ["pat_enc_csn_id", "note_id", "pat_id", "full_note", "readmit_binary", "template", "template_id"]


def load_template_notes(notes_path, ids, batch_size):
    """Every note carrying at least one of `ids`, with the matching ids as a frozenset."""
    pf = pq.ParquetFile(notes_path)
    cols = ["pat_enc_csn_id", "note_id", "pat_id", "full_note", "template_ids"]
    keep = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        hits = df["template_ids"].map(
            lambda x: frozenset(ids.intersection(int(i) for i in x)) if x is not None and len(x) else frozenset())
        m = hits.map(len) > 0
        if m.any():
            sub = df[m].drop(columns="template_ids").copy()
            sub["hits"] = hits[m]
            keep.append(sub)
    return pd.concat(keep, ignore_index=True)


def build_pair(notes, a, b):
    has_a = notes["hits"].map(lambda h: a in h)
    has_b = notes["hits"].map(lambda h: b in h)
    both = int((has_a & has_b).sum())
    df = notes[has_a ^ has_b].copy()
    df["template_id"] = df["hits"].map(lambda h: a if a in h else b).astype("int64")
    df["template"] = df["template_id"].map({a: "A", b: "B"})
    return df[OUT_COLS].reset_index(drop=True), both


def compare_to_reference(ours, ref_path):
    ref = pd.read_parquet(os.path.expanduser(ref_path), columns=["note_id", "readmit_binary", "template"])
    ref["note_id"] = ref["note_id"].astype("int64")
    o = ours[["note_id", "readmit_binary", "template"]]
    merged = o.merge(ref, on="note_id", suffixes=("", "_ref"))
    print(f"    reference: {len(ref)} rows | ours: {len(o)} | shared note_ids: {len(merged)}")
    print(f"    label agreement on shared: {(merged.readmit_binary == merged.readmit_binary_ref.astype(int)).mean():.4f}")
    print("    template (ours x reference):")
    print(pd.crosstab(merged.template, merged.template_ref.astype(str)).to_string().replace("\n", "\n      "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=PAPER_PAIRS, help="Pair names <A-id>_<B-id>")
    ap.add_argument("--notes", default="notes_with_templates.parquet",
                    help="Note-level parquet WITH text (build_note_template_parquet.py --with-text)")
    ap.add_argument("--readmission-pkl",
                    default="data/raw/readmission.pkl")
    ap.add_argument("--out-dir", default="data/pairs")
    ap.add_argument("--reference", nargs="*", default=[],
                    help="PAIR=PATH of an existing pair parquet to check against (counts only)")
    ap.add_argument("--batch-size", type=int, default=20000)
    args = ap.parse_args()

    pairs = [tuple(int(x) for x in p.split("_")) for p in args.pairs]
    ids = {i for pair in pairs for i in pair}
    refs = dict(r.split("=", 1) for r in args.reference)

    print("Loading readmission outcomes...")
    r = load_pickle(os.path.expanduser(args.readmission_pkl))
    outcome = (r["readmission"] > 0).astype("int64")
    outcome.index = r["pat_enc_csn_id"]
    outcome = outcome[~outcome.index.duplicated()]

    print(f"Streaming {args.notes} for templates {sorted(ids)}...")
    notes = load_template_notes(os.path.expanduser(args.notes), ids, args.batch_size)
    notes["note_id"] = notes["note_id"].astype("int64")
    notes["readmit_binary"] = notes["pat_enc_csn_id"].map(outcome)
    n_no_outcome = int(notes["readmit_binary"].isna().sum())
    notes = notes[notes["readmit_binary"].notna()].copy()
    notes["readmit_binary"] = notes["readmit_binary"].astype("int64")
    print(f"  {len(notes)} notes with an outcome ({n_no_outcome} dropped for no outcome row)")
    for t in sorted(ids):
        sel = notes["hits"].map(lambda h: t in h)
        print(f"  {t}: n={int(sel.sum())}, readmit={notes.loc[sel, 'readmit_binary'].mean():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, (a, b) in zip(args.pairs, pairs):
        df, both = build_pair(notes, a, b)
        dest = os.path.join(args.out_dir, f"{name}.parquet")
        df.to_parquet(dest, index=False)
        ra = df.loc[df.template == "A", "readmit_binary"].mean()
        rb = df.loc[df.template == "B", "readmit_binary"].mean()
        print(f"\n{name}: {len(df)} notes -> {dest}")
        print(f"    A={a} n={int((df.template == 'A').sum())} readmit={ra:.4f} | "
              f"B={b} n={int((df.template == 'B').sum())} readmit={rb:.4f} | "
              f"gap={abs(ra - rb) * 100:.1f}pp | dropped (both templates): {both} | "
              f"patients: {df.pat_id.nunique()}")
        if name in refs:
            compare_to_reference(df, refs[name])


if __name__ == "__main__":
    main()
