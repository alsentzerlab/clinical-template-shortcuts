#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a shareable parquet of discharge summaries with their template assignment.

One row per note. Templates are kept as a LIST column, because 127,582 notes
(29% of the corpus) carry more than one SmartPhrase -- collapsing to a single
id would silently drop the others.

Columns:
    note_id, pat_enc_csn_id, pat_id, note_csn_id
    full_note        the discharge summary text        (only with --with-text)
    template_ids     list[int64], null when the note has no template
    n_templates      int, 0 when the note has no template

*** PHI WARNING ***
With --with-text the output contains full clinical notes and patient
identifiers. It is PHI, carries the same restrictions as the source parquet,
and must stay in approved storage. The default (ids only) is the safe artifact
to hand to a collaborator who already has their own copy of the notes.

The note set comes from --notes (filtered_notes.parquet, 392,340 notes),
because that is the only source of note TEXT. The template-id map covers a
wider 434,503 notes; the extra 42,163 have no text available and are reported
as unmatched rather than emitted as textless rows.

Output is streamed row group by row group, so memory stays flat regardless of
how big the note file is.

Usage:
    # safe to share: ids only, no note text
    uv run python build_note_template_parquet.py --out notes_with_templates_ids.parquet

    # full PHI version
    uv run python build_note_template_parquet.py --with-text \
        --out notes_with_templates.parquet
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_pool import load_pickle  # noqa: E402

ID_COLS = ["note_id", "pat_enc_csn_id", "pat_id", "note_csn_id"]


def template_map(template_ids_pkl):
    """{note_id: sorted list of distinct template ids} for every mapped note."""
    t = load_pickle(template_ids_pkl)
    t["note_id"] = t["note_id"].astype(int)
    out = {}
    for note_id, ids in zip(t["note_id"], t["smartphrases_id"]):
        if ids is None:
            continue
        distinct = sorted({int(i) for i in ids})
        if distinct:
            out[note_id] = distinct
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="data/raw/filtered_notes.parquet")
    ap.add_argument("--template-ids-pkl", default="data/raw/template_ids.pkl")
    ap.add_argument("--out", default="notes_with_templates.parquet")
    ap.add_argument("--with-text", action="store_true",
                    help="Include full_note. Makes the output PHI.")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    print("Loading note -> template map...")
    tmap = template_map(args.template_ids_pkl)
    print(f"  {len(tmap)} notes carry at least one template")

    src = os.path.expanduser(args.notes)
    pf = pq.ParquetFile(src)
    available = set(pf.schema_arrow.names)
    cols = [c for c in ID_COLS if c in available]
    if args.with_text:
        if "full_note" not in available:
            sys.exit("--with-text requested but 'full_note' is not in the note file")
        cols.append("full_note")
    print(f"Streaming {pf.metadata.num_rows} notes from {os.path.basename(src)} "
          f"({pf.metadata.num_row_groups} row groups)")
    if args.with_text:
        print("  *** output will contain full clinical notes -- treat as PHI ***")

    writer = None
    seen = matched = 0
    n_multi = 0
    out_path = os.path.expanduser(args.out)
    try:
        for rg in range(pf.metadata.num_row_groups):
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            df["note_id"] = df["note_id"].astype("int64")
            lists = [tmap.get(nid) for nid in df["note_id"]]
            df["template_ids"] = lists
            df["n_templates"] = [0 if v is None else len(v) for v in lists]

            seen += len(df)
            matched += int((df["n_templates"] > 0).sum())
            n_multi += int((df["n_templates"] > 1).sum())

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema,
                                          compression=args.compression)
            writer.write_table(table)
            if (rg + 1) % 20 == 0 or rg + 1 == pf.metadata.num_row_groups:
                print(f"  row group {rg + 1}/{pf.metadata.num_row_groups} "
                      f"({seen} notes)")
    finally:
        if writer is not None:
            writer.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n{seen} notes -> {out_path} ({size_mb:.1f} MB)")
    print(f"  with a template   : {matched} ({matched / seen:.1%})")
    print(f"  no template (null): {seen - matched} ({(seen - matched) / seen:.1%})")
    print(f"  more than one     : {n_multi}")

    only_in_map = len(set(tmap) - set(pd.read_parquet(src, columns=["note_id"])
                                     ["note_id"].astype(int)))
    if only_in_map:
        print(f"  NOTE: {only_in_map} templated notes in the id map have no text in "
              f"{os.path.basename(src)} and are not in this file")


if __name__ == "__main__":
    main()
