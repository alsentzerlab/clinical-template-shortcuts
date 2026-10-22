#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the note-level parquet for the held-out generalization templates.

By default the templates are the paper's H1-H10 (heldout_templates.py); pass
--sample <csv with a template_id column> to build a different set, e.g. a fresh
draw from sample_heldout_templates.py.

Output schema:

    note_id  template_id  readmission_binary  readmission_rate
    note_csn_id  full_note

*** PHI WARNING ***
The output contains full clinical notes and identifiers. It is PHI, must stay
in approved storage, and is never printed by this script -- only aggregates
(counts and rates) go to stdout.

Differences from the original notebook pipeline (make_12_templates_file.ipynb
+ Untitled8.ipynb), all deliberate:

  * readmission_rate is mean(readmission > 0), the BINARY rate. The
    notebook stored mean(readmission), a mean of a COUNT: `readmission` runs
    0-9, 2.3% of encounters readmit more than once, and that inflated every
    stored rate by ~18% relative (corpus-wide 0.1739 vs 0.1479). It also made
    the column incomparable to readmission_binary in the same file, and to
    template_sampling_frame.csv, which is what the templates were selected on.

  * Notes are deduplicated on template ids before assignment, so a template
    used twice in one discharge summary counts once.

Like the original, readmission_rate is computed over EVERY note carrying
the template, while each note is assigned to only one template_id -- so for a
note carrying two of the twelve, both templates' rates see it but the note
appears once.

Usage:
    uv run python template_selection/build_heldout_parquet.py --with-pat-id
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_pool import load_pickle  # noqa: E402
from heldout_templates import HELDOUT_IDS  # noqa: E402

SCHEMA_COLS = ["note_id", "template_id", "readmission_binary",
               "readmission_rate", "note_csn_id", "full_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=None,
                    help="CSV with a template_id column (default: the paper's H1-H10)")
    ap.add_argument("--notes", default="notes_with_templates.parquet",
                    help="Note-level parquet WITH text (from build_note_template_parquet.py --with-text)")
    ap.add_argument("--readmission-pkl",
                    default="data/raw/readmission.pkl")
    ap.add_argument("--out", default="data/heldout/heldout_notes.parquet")
    ap.add_argument("--with-pat-id", action="store_true",
                    help="Also emit pat_id and pat_enc_csn_id. STRONGLY "
                         "recommended: patient-grouped splitting needs pat_id, "
                         "and without it notes from one patient can land in both train and test.")
    ap.add_argument("--drop-multi", action="store_true",
                    help="Drop notes carrying more than one of the sampled "
                         "templates instead of assigning the lowest id.")
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    if args.sample:
        ids = pd.read_csv(args.sample)["template_id"].astype("int64").tolist()
    else:
        ids = list(HELDOUT_IDS)
    idset = set(ids)
    print(f"{len(ids)} templates from {args.sample or 'heldout_templates.py'}")

    print("Loading readmission outcomes...")
    r = load_pickle(os.path.expanduser(args.readmission_pkl))
    r["readmission_binary"] = (r["readmission"] > 0).astype("int64")
    outcome = r.set_index("pat_enc_csn_id")["readmission_binary"]
    outcome = outcome[~outcome.index.duplicated()]

    src = os.path.expanduser(args.notes)
    pf = pq.ParquetFile(src)
    have = set(pf.schema_arrow.names)
    if "full_note" not in have:
        sys.exit(f"{src} has no full_note column; rebuild it with --with-text")
    cols = ["note_id", "pat_enc_csn_id", "pat_id", "note_csn_id",
            "template_ids", "full_note"]
    cols = [c for c in cols if c in have]

    print(f"Streaming {pf.metadata.num_rows} notes from {os.path.basename(src)}...")
    keep, n_multi, seen = [], 0, 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        seen += len(df)
        hits = df["template_ids"].map(
            lambda x: sorted(idset.intersection(int(i) for i in x))
            if x is not None and len(x) else [])
        m = hits.map(len) > 0
        if not m.any():
            continue
        sub = df[m].copy()
        h = hits[m]
        n_multi += int((h.map(len) > 1).sum())
        if args.drop_multi:
            ok = h.map(len) == 1
            sub, h = sub[ok], h[ok]
        sub["template_id"] = h.map(lambda v: v[0]).astype("int64")
        sub["all_matches"] = h.map(len)
        keep.append(sub)
    print(f"  scanned {seen}, matched {sum(len(k) for k in keep)} notes")

    df = pd.concat(keep, ignore_index=True)
    df["note_id"] = df["note_id"].astype("int64")

    before = len(df)
    df["readmission_binary"] = df["pat_enc_csn_id"].map(outcome)
    df = df[df["readmission_binary"].notna()].copy()
    df["readmission_binary"] = df["readmission_binary"].astype("int64")
    print(f"  {before - len(df)} notes dropped for having no outcome row")

    # Rate over every note carrying the template, not just the assigned ones.
    rate = {}
    for t in ids:
        sel = df["template_id"] == t
        rate[t] = float(df.loc[sel, "readmission_binary"].mean()) if sel.any() else float("nan")
    df["readmission_rate"] = df["template_id"].map(rate).astype("float64")

    out_cols = list(SCHEMA_COLS)
    if args.with_pat_id:
        out_cols += ["pat_id", "pat_enc_csn_id"]
    out = df[out_cols]

    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dest,
                   compression=args.compression)

    print(f"\nWrote {dest}  ({len(out)} rows, {len(out_cols)} cols)")
    if n_multi:
        print(f"  {n_multi} notes carried >1 sampled template "
              f"({'dropped' if args.drop_multi else 'assigned the lowest id'})")
    g = out.groupby("template_id").agg(
        n_notes=("note_id", "size"),
        readmit=("readmission_binary", "mean"),
        rate=("readmission_rate", "first")).round(4)
    print(f"\n{g.to_string()}")
    print(f"\ntotal {len(out)} notes, overall readmission "
          f"{out.readmission_binary.mean():.4f}")


if __name__ == "__main__":
    main()
