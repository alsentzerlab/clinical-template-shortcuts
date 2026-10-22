#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure pairwise textual similarity among template shells.

A generalization test wants DIVERSE documentation styles; two near-duplicate
shells (e.g. the same base template cloned across sites/teams) add no signal.
This scores every pair of shells by TF-IDF cosine over word 1-2 grams, so we can
(a) see how redundant a candidate set is and (b) drive a diversity constraint in
the sampler.

Operates on shell text already on disk (template_pool_txt/, nonim_pair_..._txt/).
Emits aggregates only: template_id pairs + similarity scores. No shell text is
printed.

Usage:
    # similarity within a chosen set (e.g. the current 12)
    uv run python template_similarity.py --ids-csv generalization_sample_12.csv

    # full NxN matrix over the whole pool -> CSV (for the sampler)
    uv run python template_similarity.py --all --out template_pairwise_sim.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SHELL_DIRS = ["template_pool_txt", "nonim_pair_templates_txt"]


def read_shell(tid):
    for d in SHELL_DIRS:
        p = os.path.join(d, f"template_{tid}.txt")
        if os.path.exists(p):
            with open(p) as f:
                return f.read()
    return None


def load_texts(ids):
    texts, kept = [], []
    for tid in ids:
        t = read_shell(int(tid))
        if t:
            texts.append(t)
            kept.append(int(tid))
    return kept, texts


def sim_matrix(texts):
    # Word 1-2 grams, drop boilerplate that every discharge summary shares by
    # capping document frequency, so similarity reflects distinctive content.
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                          min_df=1, max_df=0.9, sublinear_tf=True)
    X = vec.fit_transform(texts)
    return cosine_similarity(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-csv", default=None,
                    help="CSV with a template_id column (e.g. the sampled set)")
    ap.add_argument("--all", action="store_true",
                    help="Score the entire pool instead of a chosen set")
    ap.add_argument("--out", default=None, help="Write long-form pair CSV here")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Flag pairs at or above this cosine similarity")
    args = ap.parse_args()

    if args.all:
        ids = [int(os.path.basename(p).split("_")[1].split(".")[0])
               for p in glob.glob("template_pool_txt/template_*.txt")]
    elif args.ids_csv:
        ids = pd.read_csv(args.ids_csv)["template_id"].astype(int).tolist()
    else:
        ap.error("pass --ids-csv <file> or --all")

    ids, texts = load_texts(ids)
    print(f"Loaded {len(ids)} shells")
    S = sim_matrix(texts)

    # Long-form upper triangle.
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            rows.append({"a": ids[i], "b": ids[j], "cosine": round(float(S[i, j]), 3)})
    pairs = pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)

    if args.out:
        pairs.to_csv(args.out, index=False)
        print(f"Wrote {len(pairs)} pairs to {args.out}")

    print(f"\nSimilarity distribution over {len(pairs)} pairs:")
    print(pairs["cosine"].describe()[["mean", "50%", "75%", "max"]].round(3).to_string())

    flag = pairs[pairs["cosine"] >= args.threshold]
    print(f"\n{len(flag)} pair(s) >= {args.threshold} cosine (potential near-duplicates):")
    if len(flag):
        print(flag.head(25).to_string(index=False))

    if not args.all:
        # For a chosen set, report each template's nearest neighbour in the set.
        print("\nEach template's most-similar peer in this set:")
        near = []
        for i, tid in enumerate(ids):
            row = S[i].copy()
            row[i] = -1
            j = int(np.argmax(row))
            near.append({"template_id": tid, "nearest": ids[j],
                         "cosine": round(float(row[j]), 3)})
        nd = pd.DataFrame(near).sort_values("cosine", ascending=False)
        print(nd.to_string(index=False))


if __name__ == "__main__":
    main()
