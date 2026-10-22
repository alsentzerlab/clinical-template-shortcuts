#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample 12 held-out generalization templates spanning the readmission
range, with NO specialty quotas.

Selection is a STRATIFIED RANDOM DRAW, not an optimization. The eligible pool is
cut into --n-picks equal-count quantile strata by readmission rate and one
template is drawn uniformly from each. That guarantees coverage of the range
while keeping the draw random and seed-reproducible.

Quality gates on the eligible pool:
    >= --min-notes notes                    (default 100)
    >= --min-shell-chars characters         (default 1200, drops fragments)
    first 250 chars free of STUB_RE         (addenda / attestations)
    >= --min-sections distinct sections     (default 2, via structural_ok)
    pairwise TF-IDF cosine <= --dedup-sim   (default 0.90, collapses clones)
    cosine to any IN-USE template <= --max-inuse-sim   (default 0.35)
    cosine between any two PICKS <= --max-sim          (default 0.35)

Draws are rejected and retried until the similarity and --min-span constraints
hold, or --max-attempts is exhausted.

This documents how held-out candidates were screened. It does NOT regenerate
the paper's set: the shipped draw was made with settings that were not recorded
(one pair of its templates has cosine 0.43, above --max-sim), and two of its
twelve picks were later dropped. The paper's ten are fixed in
heldout_templates.py, and build_heldout_parquet.py uses them by default.

Usage:
    uv run python template_selection/sample_heldout_templates.py --out heldout_candidates.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A real discharge-summary template has multiple narrative sections. Attestation
# blocks ("I have personally reviewed...", teaching-physician statements) and
# other stubs do not, so we require a minimum number of section headers and
# reject shells that are mostly attestation language.
SECTION_RE = re.compile(
    r"hospital course|discharge (medication|diagnos|instruct|condition)|"
    r"history of present illness|reason for (admission|hospitalization)|"
    r"assessment and plan|physical exam|chief complaint|follow.?up|"
    r"pending results|discharge summary", re.I)
ATTEST_RE = re.compile(
    r"attest|i have personally|teaching physician|was present for|"
    r"i performed the|resident.{0,20}supervis", re.I)


def structural_ok(text: str, min_sections: int) -> bool:
    n_sec = len(set(m.group(0).lower() for m in SECTION_RE.finditer(text)))
    if n_sec < min_sections:
        return False
    # Reject shells dominated by attestation language with little else.
    if ATTEST_RE.search(text) and n_sec <= 1:
        return False
    return True


# A shell that merely appends to an existing note -- an addendum, an
# attestation, a late signature block -- is not a documentation style a model
# can generalize to. Such shells announce themselves in their opening lines, so
# reject any whose first HEAD_CHARS characters lead with one of these markers.
HEAD_CHARS = 250
STUB_RE = re.compile(
    r"\baddend|\bamendment\b|late entry|i have personally|teaching physician|"
    r"attestation|was present for", re.I)

# Every template used in training during the study. A generalization set must
# not contain anything a model trained on.
IN_USE_IDS = [206596, 248528, 229657, 624343, 801125, 249614, 1337772]

SHELL_DIRS = ["template_pool_txt", "nonim_pair_templates_txt",
              "generalization_templates_txt"]


def read_shell(tid):
    for d in SHELL_DIRS:
        p = os.path.join(d, f"template_{int(tid)}.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def load_pool(args):
    """Frame + classifications, with every quality gate applied in order."""
    frame = pd.read_csv(args.frame)
    pool = frame.copy()
    print(f"sampling frame: {len(pool)} templates")

    if os.path.exists(args.classifications):
        cls = pd.read_csv(args.classifications)
        keep = [c for c in ("template_id", "classification", "confidence",
                            "specialty_detail") if c in cls.columns]
        pool = pool.merge(cls[keep], on="template_id", how="left")
        if args.min_confidence != "any":
            order = {"low": 0, "medium": 1, "high": 2}
            floor = order[args.min_confidence]
            before = len(pool)
            pool = pool[pool["confidence"].map(order).fillna(-1) >= floor]
            print(f"  --min-confidence {args.min_confidence} drops {before - len(pool)}")
    else:
        print(f"  NOTE: {args.classifications} missing; no specialty labels attached")

    before = len(pool)
    pool = pool[pool["n_notes"] >= args.min_notes]
    print(f"  --min-notes {args.min_notes} drops {before - len(pool)}")

    excl = set(IN_USE_IDS)
    if args.exclude_ids:
        excl |= {int(x) for x in args.exclude_ids.replace(",", " ").split()}
    before = len(pool)
    pool = pool[~pool["template_id"].isin(excl)]
    print(f"  in-use / --exclude-ids drops {before - len(pool)} "
          f"({len(excl)} ids excluded)")

    # Shell-text gates.
    shells, keep = {}, []
    for tid in pool["template_id"]:
        txt = read_shell(tid)
        shells[tid] = txt
        keep.append(txt is not None)
    before = len(pool)
    pool = pool[pd.Series(keep, index=pool.index)]
    print(f"  no shell on disk drops {before - len(pool)}")

    for label, fn in (
        (f"--min-shell-chars {args.min_shell_chars}",
         lambda t: len(shells[t]) >= args.min_shell_chars),
        # Attestation LANGUAGE is fine -- plenty of real discharge-summary
        # shells carry an attestation block. What must be excluded is a shell
        # that is ONLY an attestation. So the head marker alone is not
        # disqualifying: it disqualifies only a shell that also fails to show
        # real discharge-summary structure.
        ("attestation-only filter",
         lambda t: not (STUB_RE.search(shells[t][:HEAD_CHARS])
                        and not structural_ok(shells[t], args.min_sections))),
        (f"--min-sections {args.min_sections}",
         lambda t: structural_ok(shells[t], args.min_sections)),
    ):
        before = len(pool)
        pool = pool[[fn(t) for t in pool["template_id"]]]
        print(f"  {label} drops {before - len(pool)}")

    return pool.reset_index(drop=True), shells


def similarity(pool, shells, args):
    """TF-IDF cosine for pool-vs-pool and pool-vs-in-use, fit together."""
    inuse = [t for t in IN_USE_IDS if read_shell(t) is not None]
    missing = sorted(set(IN_USE_IDS) - set(inuse))
    if missing:
        print(f"  NOTE: no shell on disk for in-use template(s) {missing}; "
              f"they are NOT covered by the --max-inuse-sim check")
    ids = list(pool["template_id"]) + inuse
    texts = [shells.get(t) or read_shell(t) for t in ids]
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9,
                        sublinear_tf=True).fit_transform(texts)
    S = cosine_similarity(X)
    n = len(pool)
    inuse_max = S[:n, n:].max(axis=1) if inuse else np.zeros(n)

    pool = pool.copy()
    pool["max_sim_to_in_use"] = inuse_max.round(3)
    before = len(pool)
    mask = inuse_max <= args.max_inuse_sim
    pool, S_pool = pool[mask].reset_index(drop=True), S[:n, :n][np.ix_(mask, mask)]
    S = S_pool
    print(f"  --max-inuse-sim {args.max_inuse_sim} drops {before - len(pool)}")

    # Collapse near-duplicate shells, keeping the most-used of each cluster.
    # Paused with --no-dedup: the --max-sim gate still guarantees no two PICKS
    # are near-identical, so switching this off changes only the draw weights
    # (a cluster of k clones gets k chances instead of one), not the validity
    # of the resulting set.
    if args.no_dedup:
        print("  --no-dedup: near-duplicate collapse DISABLED")
        return pool, S[:n, :n][np.ix_(mask, mask)] if False else S_pool
    order = pool.sort_values("n_notes", ascending=False).index.tolist()
    kept = []
    for i in order:
        if all(S[i, j] <= args.dedup_sim for j in kept):
            kept.append(i)
    kept = sorted(kept)
    print(f"  --dedup-sim {args.dedup_sim} drops {len(pool) - len(kept)} near-duplicates")
    return pool.loc[kept].reset_index(drop=True), S[np.ix_(kept, kept)]


def strata_indices(pool, n, mode, borrow_k=5):
    """Index groups, one per pick slot.

    quantile: equal-COUNT strata. A representative sample of the pool, but the
        pool is right-skewed (half the templates sit below 14%), so half the
        picks land there and the top of the range is thinly covered.
    width: equal-WIDTH strata over the observed rate range, which spreads the
        picks evenly across low->high. Bins can be empty -- the 26.9-29.8% band
        holds no eligible template -- so an empty bin borrows candidates from
        the nearest non-empty one.
    """
    r = pool["readmission_rate"]
    if mode == "quantile":
        strata = pd.qcut(r.rank(method="first"), n, labels=False)
        groups = [np.where(strata == k)[0] for k in range(n)]
        if any(len(g) == 0 for g in groups):
            sys.exit(f"pool of {len(pool)} is too small for {n} strata")
        return groups

    edges = np.linspace(r.min(), r.max(), n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    groups = []
    for k in range(n):
        lo, hi = edges[k], edges[k + 1]
        g = np.where((r >= lo) & ((r < hi) if k < n - 1 else (r <= hi)))[0]
        if len(g) == 0:
            # Offer the borrow_k nearest templates, not just the single
            # closest: the sparse top of the range has bins holding one
            # template, so a one-element borrow deadlocks against the
            # neighbouring bin that already claimed it.
            g = (r - centers[k]).abs().values.argsort()[:borrow_k]
            print(f"  bin {k + 1} ({100 * lo:.1f}-{100 * hi:.1f}%) is empty; "
                  f"borrowing from the {borrow_k} nearest templates")
        groups.append(g)
    return groups


def draw(pool, S, args, rng):
    """One template drawn uniformly at random from each rate stratum."""
    n = args.n_picks
    groups = strata_indices(pool, n, args.strata, args.borrow_k)

    for attempt in range(1, args.max_attempts + 1):
        pick, used = [], set()
        for g in groups:
            free = [i for i in g if i not in used]
            if not free:                      # borrowed bins can collide
                break
            i = int(rng.choice(free))
            pick.append(i)
            used.add(i)
        if len(pick) < n:
            continue
        sims = [S[a, b] for i, a in enumerate(pick) for b in pick[i + 1:]]
        rates = pool.loc[pick, "readmission_rate"].values
        span = rates.max() - rates.min()
        if max(sims) <= args.max_sim and span >= args.min_span:
            print(f"\naccepted on attempt {attempt}: max pairwise sim "
                  f"{max(sims):.3f}, span {100 * span:.1f}pp")
            return pool.loc[pick].assign(
                stratum=range(n),
                max_sim_to_pick=[round(max(S[a, b] for b in pick if b != a), 3)
                                 for a in pick]).sort_values("readmission_rate")
    sys.exit(f"no draw satisfied the constraints in {args.max_attempts} attempts; "
             f"loosen --max-sim / --min-span or re-seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="template_sampling_frame.csv")
    ap.add_argument("--classifications", default="template_pool_classifications.csv")
    ap.add_argument("--out", default="heldout_candidates.csv")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--n-picks", type=int, default=12)
    ap.add_argument("--min-notes", type=int, default=100)
    ap.add_argument("--min-shell-chars", type=int, default=1200)
    ap.add_argument("--min-sections", type=int, default=2)
    ap.add_argument("--min-confidence", default="any",
                    choices=["any", "low", "medium", "high"],
                    help="Specialty-label confidence floor. Default 'any': the "
                         "new framing does not use the specialty labels, so "
                         "there is no reason to discard templates for having an "
                         "uncertain one.")
    ap.add_argument("--max-sim", type=float, default=0.35,
                    help="Max TF-IDF cosine between any two PICKS")
    ap.add_argument("--max-inuse-sim", type=float, default=0.35,
                    help="Max cosine to any template used in training")
    ap.add_argument("--dedup-sim", type=float, default=0.90)
    ap.add_argument("--no-dedup", action="store_true",
                    help="Pause the near-duplicate collapse. --max-sim still "
                         "keeps the 12 picks mutually distinct.")
    ap.add_argument("--min-span", type=float, default=0.30,
                    help="Reject draws whose readmission range is narrower than "
                         "this (default 0.30 = 30pp). The structurally-valid pool only\n"
                         "reaches ~35%% readmission, so 0.40 is unsatisfiable.")
    ap.add_argument("--strata", default="width", choices=["width", "quantile"],
                    help="width: equal-width rate bins, spreads picks evenly "
                         "low->high (default). quantile: equal-count strata, a "
                         "representative sample of a right-skewed pool.")
    ap.add_argument("--borrow-k", type=int, default=5,
                    help="How many nearby templates an empty equal-width bin "
                         "may draw from")
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--exclude-ids", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pool, shells = load_pool(args)
    pool, S = similarity(pool, shells, args)
    print(f"\neligible pool: {len(pool)} templates, "
          f"readmission {100 * pool.readmission_rate.min():.1f}%"
          f"-{100 * pool.readmission_rate.max():.1f}%")

    picks = draw(pool, S, args, rng)
    cols = [c for c in ("template_id", "n_notes", "readmission_rate", "stratum",
                        "classification", "confidence", "specialty_detail",
                        "shell_chars", "max_sim_to_pick", "max_sim_to_in_use")
            if c in picks.columns]
    picks[cols].to_csv(args.out, index=False)

    print(f"\n{len(picks)} templates -> {args.out}\n")
    show = picks[[c for c in ("template_id", "n_notes", "readmission_rate",
                              "classification", "max_sim_to_pick") if c in picks]]
    print(show.to_string(index=False))
    r = picks["readmission_rate"].values
    print(f"\ntotal notes {int(picks.n_notes.sum())}  "
          f"rate {100 * r.min():.1f}%-{100 * r.max():.1f}% "
          f"({100 * (r.max() - r.min()):.1f}pp)  "
          f"min gap {100 * np.diff(np.sort(r)).min():.2f}pp")


if __name__ == "__main__":
    main()
