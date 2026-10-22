#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract EVERY template used by any discharge summary, with no filtering.

This is the unfiltered counterpart to template_pool.py
--frame-out, which drops templates under a note-count floor (MIN_NOTES = 100)
and drops the in-use pair templates. Here nothing is dropped: every SmartPhrase
id that appears on at least one note in the template-id map gets a row, whether
or not it has shell text, outcome data, or enough volume for a meaningful
readmission rate.

Readmission rate and n_notes are still emitted so downstream code can impose its
own floor, but a rate computed on a handful of notes is noise -- use n_notes.

SCOPE. "No filtering" applies to the templates, not to the notes. By default this
counts templates over filtered_notes.parquet (392,340 notes, the filtered
discharge summaries) -> 11,580 templates. template_ids.pkl is a
strict superset of that (434,503 notes); counting over all of it gives 11,908
templates, where the extra 328 appear ONLY on the 42,163 notes that are not
discharge summaries. Pass --all-note-types for that wider set.

Note that filterednotes.parquet carries no note_type column -- the discharge
summary filter was applied upstream, during cohort extraction.

Loaders (unpickling, RTF stripping, DataFrame-vs-dict shell lookup) are imported
from template_pool so the shell text matches the existing pool
byte for byte.

Emits aggregates only (ids, counts, rates, shell lengths) to stdout. Shell text
goes to disk when --dump-shells / --with-text is given; it is never printed.

Usage:
    uv run python extract_all_templates.py \
        --out all_templates_full.parquet \
        --dump-shells all_templates_txt

    # ids + counts + rates only, skipping the 4 GB shell pickle:
    uv run python extract_all_templates.py --no-shells --out all_templates_full.parquet
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


def note_counts(template_ids_pkl, scope_note_ids=None):
    """Per-template note count over the in-scope notes.

    Dedupes ids within a note first: a template used twice in one discharge
    summary is one note, matching compute_stats() in the filtered pipeline.

    scope_note_ids restricts to a note set (the discharge summaries). The
    template-id map is a strict SUPERSET of the filtered discharge summaries --
    434,503 notes vs 392,340 -- so leaving it None counts templates that only
    ever appear on the 42,163 notes that are not discharge summaries.
    """
    t = load_pickle(template_ids_pkl)
    t["note_id"] = t["note_id"].astype(int)
    exploded = t[["note_id", "smartphrases_id"]].copy()
    if scope_note_ids is not None:
        exploded = exploded[exploded["note_id"].isin(scope_note_ids)]
    exploded["smartphrases_id"] = exploded["smartphrases_id"].apply(
        lambda x: list(set(x)) if x is not None else [])
    exploded = exploded.explode("smartphrases_id").dropna(subset=["smartphrases_id"])
    exploded["smartphrases_id"] = exploded["smartphrases_id"].astype(int)
    counts = exploded.groupby("smartphrases_id")["note_id"].nunique()
    return counts.rename("n_notes"), exploded


def readmission_rates(exploded, readmission_pkl, filtered_notes):
    """Per-template readmission rate, for whatever subset has outcome data.

    Mirrors compute_stats() in template_pool, but left-joined
    so templates without outcomes survive with a null rate instead of vanishing.
    """
    readmission = load_pickle(readmission_pkl)
    readmission["readmit_binary"] = readmission["readmission"] > 0

    notes = pd.read_parquet(os.path.expanduser(filtered_notes),
                            columns=["note_id", "pat_enc_csn_id"])
    notes["note_id"] = notes["note_id"].astype(int)

    df = exploded.merge(notes, on="note_id", how="inner").merge(
        readmission[["pat_enc_csn_id", "readmit_binary"]],
        on="pat_enc_csn_id", how="inner")
    stats = df.groupby("smartphrases_id")["readmit_binary"].agg(["mean", "count"])
    return stats.rename(columns={"mean": "readmission_rate",
                                 "count": "n_notes_with_outcome"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-ids-pkl", default="data/raw/template_ids.pkl")
    ap.add_argument("--smartphrases-pkl", default="data/raw/smartphrases.pkl")
    ap.add_argument("--readmission-pkl",
                    default="data/raw/readmission.pkl")
    ap.add_argument("--filtered-notes", default="data/raw/filtered_notes.parquet")
    ap.add_argument("--out", default="all_templates_full.parquet",
                    help="Output table, one row per template (.parquet or .csv)")
    ap.add_argument("--dump-shells", default=None,
                    help="Write each template's shell to <DIR>/template_<id>.txt")
    ap.add_argument("--with-text", action="store_true",
                    help="Include the shell text as a 'template' column in --out. "
                         "Off by default so the table stays shareable.")
    ap.add_argument("--no-shells", action="store_true",
                    help="Skip the 4 GB smartphrases pickle entirely; emit ids, "
                         "counts and rates only")
    ap.add_argument("--no-outcomes", action="store_true",
                    help="Skip the readmission join; emit ids + counts only")
    ap.add_argument("--all-note-types", action="store_true",
                    help="Count templates over EVERY note in the template-id map, "
                         "not just the filtered discharge summaries. The map covers "
                         "434,503 notes vs 392,340 discharge summaries, so this adds "
                         "templates that never appear on a discharge summary.")
    args = ap.parse_args()

    scope = None
    if not args.all_note_types:
        print("Scoping to the filtered discharge summaries...")
        ids = pd.read_parquet(os.path.expanduser(args.filtered_notes),
                              columns=["note_id"])["note_id"].astype(int)
        scope = set(ids)
        print(f"  {len(scope)} discharge summaries in scope")

    print("Counting notes per template (no floor)...")
    counts, exploded = note_counts(args.template_ids_pkl, scope_note_ids=scope)
    frame = counts.to_frame()
    frame.index.name = "template_id"
    noun = "notes" if args.all_note_types else "discharge summaries"
    print(f"  {len(frame)} distinct templates across "
          f"{exploded['note_id'].nunique()} {noun} that carry a template")

    if not args.no_outcomes:
        print("Joining readmission outcomes...")
        stats = readmission_rates(exploded, args.readmission_pkl, args.filtered_notes)
        frame = frame.join(stats, how="left")
        n_rate = int(frame["readmission_rate"].notna().sum())
        print(f"  {n_rate} templates have outcome data "
              f"({len(frame) - n_rate} left null)")

    if not args.no_shells:
        print("Loading smartphrase shells (this reads a ~4 GB pickle)...")
        texts = build_template_texts(load_pickle(args.smartphrases_pkl))
        print(f"  {len(texts)} templates with shell text in the pickle")
        frame["shell_chars"] = [len(texts.get(tid, "")) for tid in frame.index]
        frame["has_shell"] = [tid in texts for tid in frame.index]
        print(f"  {int(frame['has_shell'].sum())} of {len(frame)} templates in "
              f"use have shell text; {int((~frame['has_shell']).sum())} do not")

        if args.dump_shells:
            os.makedirs(args.dump_shells, exist_ok=True)
            written = 0
            for tid in frame.index:
                txt = texts.get(tid)
                if not txt:
                    continue
                with open(os.path.join(args.dump_shells, f"template_{tid}.txt"), "w") as f:
                    f.write(txt)
                written += 1
            print(f"  {written} shells written to {args.dump_shells}/")

        if args.with_text:
            frame["template"] = [texts.get(tid, "") for tid in frame.index]

    frame = frame.reset_index().sort_values("n_notes", ascending=False)
    out = os.path.expanduser(args.out)
    if out.endswith(".csv"):
        frame.to_csv(out, index=False)
    else:
        frame.to_parquet(out, index=False)
    print(f"\n{len(frame)} templates -> {out}")

    print("\nCoverage by note-count floor (for reference, nothing was dropped):")
    for k in (1, 10, 50, 100, 200, 400, 1000):
        print(f"  >= {k:5d} notes: {int((frame['n_notes'] >= k).sum()):6d} templates")


if __name__ == "__main__":
    main()
