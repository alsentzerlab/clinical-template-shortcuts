#!/usr/bin/env python3
"""
Patient-grouped train/val/test splitting that also matches per-template outcome rates.

Why this exists
---------------
For the shortcut-learning study two things must hold simultaneously:

1. No patient leakage — every note from one patient lands in exactly one split.
2. Each template's readmission rate is the same in train, val and test (and so is
   the overall rate, and each template's share of each split).

(2) is not a nicety here: the effect under study on the Dec9 pair is a 2.8pp gap
between templates (A 14.5% vs B 17.3%). GroupShuffleSplit and StratifiedGroupKFold
both leave a 2-3pp spread in per-template rate across splits, i.e. noise the size of
the signal. StratifiedGroupKFold does not fix it because it greedily balances folds
one at a time; it never optimises the three splits jointly against explicit targets.

Approach
--------
Treat it as a constrained assignment problem over patients, not over rows.

Each patient is an indivisible item carrying a count vector over (template x label)
cells. Assigning patients to splits is a multi-dimensional number-partitioning
problem: choose an assignment minimising

    J = w_rate * sum_{t,s} (rate[t,s] - rate[t])^2
      + w_size * sum_{t,s} ((n[t,s] - r_s * N_t) / N_t)^2

where t indexes templates, s indexes splits, r_s is the target ratio, rate[t,s] is
the achieved readmission rate of template t in split s, and rate[t] is its global
rate. The first term is exactly "per-template rates match"; the second keeps each
template's share of each split on target (which also pins the overall rate and the
70/15/15 ratios).

It is solved in two stages, with random restarts:

  Seed. Most patients have all their notes in a single (template, label) cell — 87%
  of Dec9 patients have one note at all. Those are placed by exact deficit filling:
  within each cell, patients go (largest first) to whichever split is furthest below
  its target count for that cell, which fills every cell to its target up to patient
  granularity. The minority of patients spanning two or more cells are placed first,
  by whichever split least worsens J, so the deficit fill absorbs their imbalance.

  Polish. First-improvement local search over single-patient moves until no move
  reduces J.

This is fast (seconds), deterministic given a seed, and gets the per-template spread
down to the quantisation floor — one patient is the smallest unit you can move.

Nothing here reads or emits note text; it operates on ID/label/template columns only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _objective(pos: np.ndarray, tot: np.ndarray, target_rate: np.ndarray,
               target_tot: np.ndarray, n_per_template: np.ndarray,
               bucket: np.ndarray, target_bucket: np.ndarray, n_per_bucket: np.ndarray,
               w_rate: float, w_size: float, w_mix: float) -> float:
    """J for the current (template x split) count state.

    pos/tot: (T, S) positives and totals. target_rate: (T,). target_tot: (T, S).
    bucket: (B, S) rows held by patients of each encounter-count bucket.
    Empty (t, s) cells are charged the full squared rate so the optimiser never
    empties a template out of a split.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.where(tot > 0, pos / np.maximum(tot, 1), np.nan)
    dev = np.where(np.isnan(rate), target_rate[:, None], rate - target_rate[:, None])
    j_rate = float((dev ** 2).sum())
    scale = np.maximum(n_per_template, 1)[:, None]
    j_size = float((((tot - target_tot) / scale) ** 2).sum())
    # Case-mix: without this the optimiser parks every multi-encounter patient in
    # train (they are lumpy and easiest to absorb there), leaving val/test almost
    # purely single-encounter — matched rates but an unrepresentative population,
    # since repeat patients readmit ~3x as often.
    bscale = np.maximum(n_per_bucket, 1)[:, None]
    j_mix = float((((bucket - target_bucket) / bscale) ** 2).sum())
    return w_rate * j_rate + w_size * j_size + w_mix * j_mix


def balanced_group_split(
    df: pd.DataFrame,
    label_column: str,
    group_column: str,
    template_column: str | None = None,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
    n_restarts: int = 8,
    max_passes: int = 30,
    w_rate: float = 1.0,
    w_size: float = 1.0,
    w_mix: float = 1.0,
    verbose: bool = True,
):
    """Split df into train/val/test by group, matching per-template label rates.

    Returns (train_df, val_df, test_df). Every row sharing a `group_column` value
    lands in one split; the caller's row order is preserved within each split.

    If `template_column` is None (or absent from df) the whole dataset is treated
    as a single template, which reduces to "patient-grouped split with the overall
    label rate matched" — still strictly better than GroupShuffleSplit.
    """
    ratios = np.asarray(ratios, dtype=float)
    if abs(ratios.sum() - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios.sum()}")

    work = df[[group_column, label_column]].copy()
    work[label_column] = work[label_column].astype(int)
    if template_column and template_column in df.columns:
        work['_tpl'] = df[template_column].astype(str).values
    else:
        work['_tpl'] = '_all'

    templates = np.sort(work['_tpl'].unique())
    t_index = {t: i for i, t in enumerate(templates)}
    T, S = len(templates), 3

    # --- per-group count vectors over (template x label) ---
    work['_ti'] = work['_tpl'].map(t_index).astype(int)
    grouped = work.groupby(group_column, sort=True)
    group_keys = np.array(list(grouped.groups.keys()), dtype=object)
    G = len(group_keys)

    g_pos = np.zeros((G, T), dtype=np.int64)
    g_tot = np.zeros((G, T), dtype=np.int64)
    key_to_row = {k: i for i, k in enumerate(group_keys)}
    agg = work.groupby([group_column, '_ti'])[label_column].agg(['size', 'sum'])
    for (gkey, ti), row in agg.iterrows():
        gi = key_to_row[gkey]
        g_tot[gi, ti] = row['size']
        g_pos[gi, ti] = row['sum']

    n_per_template = g_tot.sum(axis=0).astype(float)          # (T,)
    pos_per_template = g_pos.sum(axis=0).astype(float)
    target_rate = np.where(n_per_template > 0, pos_per_template / np.maximum(n_per_template, 1), 0.0)
    target_tot = n_per_template[:, None] * ratios[None, :]     # (T, S)

    group_size = g_tot.sum(axis=1)

    # Cell view: C = 2T cells indexed c = template * 2 + label. A patient is "pure"
    # if all of its notes fall in one cell (the common case) and "mixed" otherwise.
    C = 2 * T
    g_cell = np.zeros((G, C), dtype=np.int64)
    g_cell[:, 0::2] = g_tot - g_pos      # negatives
    g_cell[:, 1::2] = g_pos              # positives
    cell_total = g_cell.sum(axis=0).astype(float)
    n_cells_touched = (g_cell > 0).sum(axis=1)
    mixed_idx = np.flatnonzero(n_cells_touched > 1)
    n_mixed = len(mixed_idx)
    pure_cell = np.where(n_cells_touched == 1, g_cell.argmax(axis=1), -1)
    pure_members = [np.flatnonzero(pure_cell == c) for c in range(C)]

    # Case-mix buckets: groups with 1, 2, 3, 4, 5, 6+ rows. Rows (not groups) are
    # counted, so the target is "each split holds r_s of the rows contributed by
    # groups of each size". Coarser buckets (e.g. just 1/2/3+) let a split fill up
    # on 3-row groups while another takes the 9-row ones.
    g_bucket = np.clip(group_size, 1, 6).astype(int) - 1
    B = 6
    n_per_bucket = np.array([group_size[g_bucket == b].sum() for b in range(B)], dtype=float)
    target_bucket = n_per_bucket[:, None] * ratios[None, :]      # (B, S)
    if verbose:
        print(f"  {G} groups over {C} (template x label) cells: "
              f"{G - n_mixed} single-cell, {n_mixed} spanning >1 cell")

    rng_master = np.random.RandomState(random_seed)

    best_assign, best_j = None, np.inf
    for restart in range(max(1, n_restarts)):
        rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))

        assign = np.full(G, -1, dtype=int)
        pos = np.zeros((T, S)); tot = np.zeros((T, S)); buck = np.zeros((B, S))

        def J():
            return _objective(pos, tot, target_rate, target_tot, n_per_template,
                              buck, target_bucket, n_per_bucket, w_rate, w_size, w_mix)

        def place(gi, s, sign=1):
            pos[:, s] += sign * g_pos[gi]
            tot[:, s] += sign * g_tot[gi]
            buck[g_bucket[gi], s] += sign * group_size[gi]

        # Seed, stage 1: groups spanning >1 cell, biggest first, each to the split
        # that least worsens J. Tie-broken randomly so restarts explore.
        for gi in mixed_idx[np.lexsort((rng.rand(n_mixed), -group_size[mixed_idx]))]:
            best_s, best_local = 0, np.inf
            for s in range(S):
                place(gi, s)
                j = J()
                place(gi, s, -1)
                if j < best_local:
                    best_local, best_s = j, s
            assign[gi] = best_s
            place(gi, best_s)

        # Seed, stage 2: single-cell groups fill each cell's remaining deficit.
        # Largest groups first, each to the split furthest below target for that
        # cell, so every cell ends up split near-exactly r_s. `placed` accounts
        # for what stage 1 already put down.
        placed = np.zeros((C, S))
        for gi in mixed_idx:
            placed[:, assign[gi]] += g_cell[gi]
        for c in range(C):
            members = pure_members[c]
            if len(members) == 0:
                continue
            deficit = cell_total[c] * ratios - placed[c]
            for gi in members[np.lexsort((rng.rand(len(members)), -group_size[members]))]:
                s = int(np.argmax(deficit))
                assign[gi] = s
                deficit[s] -= g_cell[gi, c]
                place(gi, s)

        # Polish: first-improvement local search over single-group moves.
        j_cur = J()
        for _ in range(max_passes):
            improved = False
            for gi in rng.permutation(G):
                s_from = assign[gi]
                place(gi, s_from, -1)
                best_s, best_local = s_from, np.inf
                for s in range(S):
                    place(gi, s)
                    j = J()
                    place(gi, s, -1)
                    if j < best_local - 1e-15:
                        best_local, best_s = j, s
                if best_s != s_from and best_local < j_cur - 1e-12:
                    assign[gi] = best_s
                    j_cur = best_local
                    improved = True
                else:
                    best_s = s_from
                place(gi, best_s)
            if not improved:
                break

        if j_cur < best_j:
            best_j, best_assign = j_cur, assign.copy()
        if verbose:
            print(f"  restart {restart}: J={j_cur:.3e}" + ("  <- best" if j_cur == best_j else ""))

    split_of_group = pd.Series(best_assign, index=pd.Index(group_keys, name=group_column))
    row_split = df[group_column].map(split_of_group).values

    parts = [df[row_split == s] for s in range(S)]
    if verbose:
        print(f"  final J={best_j:.3e}")
        print(split_report(parts, label_column, template_column, group_column, ratios))

    # Hard guarantee, same contract as the rest of process_data.py.
    sets = [set(p[group_column]) for p in parts]
    names = ['train', 'val', 'test']
    for i in range(S):
        for k in range(i + 1, S):
            overlap = sets[i] & sets[k]
            if overlap:
                raise AssertionError(
                    f"{len(overlap)} '{group_column}' values appear in both {names[i]} and {names[k]}")
    return parts[0], parts[1], parts[2]


def split_report(parts, label_column, template_column, group_column, ratios=(0.70, 0.15, 0.15)) -> str:
    """Aggregate-only summary: sizes, per-template rates, worst spread. No row data."""
    names = ['train', 'val', 'test']
    n_total = sum(len(p) for p in parts)
    lines = []
    lines.append(f"  {'split':<6} {'rows':>6} {'frac':>7} {'target':>7} {'groups':>8} "
                 f"{'rows/grp':>9} {'rate':>7}")
    for name, part, r in zip(names, parts, ratios):
        n_grp = part[group_column].nunique()
        lines.append(f"  {name:<6} {len(part):>6} {len(part)/n_total:>7.4f} {r:>7.2f} "
                     f"{n_grp:>8} {len(part)/max(n_grp,1):>9.3f} {part[label_column].mean():>7.4f}")

    if template_column and template_column in parts[0].columns:
        lines.append(f"\n  {'template':<12} " + " ".join(f"{n:>16}" for n in names) + f"{'spread':>9}")
        worst = 0.0
        all_tpl = sorted(set().union(*[set(p[template_column].astype(str)) for p in parts]))
        for tpl in all_tpl:
            cells = []
            for part in parts:
                sub = part[part[template_column].astype(str) == tpl]
                cells.append((len(sub), sub[label_column].mean() if len(sub) else float('nan')))
            rates = [c[1] for c in cells]
            spread = max(rates) - min(rates)
            worst = max(worst, spread)
            lines.append(f"  {tpl:<12} " + " ".join(f"{n:>6} {rt:>9.4f}" for n, rt in cells)
                         + f"{spread:>9.4f}")
        overall = [p[label_column].mean() for p in parts]
        lines.append(f"  {'OVERALL':<12} " + " ".join(f"{len(p):>6} {r:>9.4f}" for p, r in zip(parts, overall))
                     + f"{max(overall)-min(overall):>9.4f}")
        lines.append(f"\n  worst per-template rate spread across splits: {worst:.4f} ({worst*100:.2f}pp)")
    return "\n".join(lines)


def feasibility_report(df, label_column, group_column, template_column=None,
                       ratios=(0.70, 0.15, 0.15)) -> str:
    """How well *could* any grouped split do on this dataset, before running one?

    Two things bound the achievable per-template rate match:

      * Quantisation — the smallest split of a template holds n rows, so its rate
        moves in steps of 1/n. A template with 60 test rows can only match to ~1.7pp.
      * Group lumpiness — if the largest patient in a cell holds k rows, a single
        placement can overshoot by up to k rows.

    `~limit` combines the two as a rough order-of-magnitude guide, not a bound the
    solver actually hits: with many single-note patients to fine-tune with, the
    achieved spread is typically well below it (Dec9: ~limit 1.0pp, achieved 0.07pp).
    Treat it as a red flag, not a prediction — if `~limit` for some template is
    already bigger than the rate gap you are studying, that template is too small
    to split three ways and you need more data for it or a merge.
    """
    ratios = np.asarray(ratios, dtype=float)
    lines = [f"  {'template':<12} {'rows':>7} {'pos':>6} {'rate':>7} {'min-split':>10} "
             f"{'step':>8} {'max grp':>8} {'~limit':>8}"]
    tpl = (df[template_column].astype(str) if (template_column and template_column in df.columns)
           else pd.Series('_all', index=df.index))
    worst = 0.0
    for t, sub in df.groupby(tpl):
        n = len(sub)
        n_min = ratios.min() * n
        step = 1.0 / max(n_min, 1)
        max_grp = sub.groupby(sub[group_column]).size().max()
        floor = step * max(1, max_grp) / 2
        worst = max(worst, floor)
        lines.append(f"  {str(t):<12} {n:>7} {int(sub[label_column].sum()):>6} "
                     f"{sub[label_column].mean():>7.4f} {n_min:>10.0f} {step:>8.4f} "
                     f"{max_grp:>8} {floor:>8.4f}")
    lines.append(f"\n  worst ~limit is {worst:.4f} ({worst*100:.2f}pp) — a rough ceiling on the "
                 f"difficulty,\n  not a prediction; compare it to the rate gap you are studying.")
    return "\n".join(lines)


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Patient-grouped train/val/test split with per-template rate matching. "
                    "Reads only the ID/label/template columns unless it has to write splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input-path', required=True, help='parquet/csv with the rows to split')
    ap.add_argument('--group-column', required=True, help="e.g. pat_id — never spans two splits")
    ap.add_argument('--label-column', default='readmit_binary')
    ap.add_argument('--template-column', default='template',
                    help="set to '' if the data has no template column")
    ap.add_argument('--group-map-path', default=None,
                    help='file to join --group-column from when the input lacks it')
    ap.add_argument('--group-map-key', default='pat_enc_csn_id', help='join key for --group-map-path')
    ap.add_argument('--train-ratio', type=float, default=0.70)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--test-ratio', type=float, default=0.15)
    ap.add_argument('--random-seed', type=int, default=42)
    ap.add_argument('--n-restarts', type=int, default=8)
    ap.add_argument('--w-rate', type=float, default=1.0, help='weight on per-template rate matching')
    ap.add_argument('--w-size', type=float, default=1.0, help='weight on split-size / template-share targets')
    ap.add_argument('--w-mix', type=float, default=1.0, help='weight on matching the rows-per-group case mix')
    ap.add_argument('--output-dir', default=None,
                    help='write train/val/test parquet here (omit to only audit)')
    ap.add_argument('--assignment-out', default=None,
                    help='write a group_id -> split CSV here (contains IDs; treat as PHI)')
    ap.add_argument('--audit-only', action='store_true',
                    help='report feasibility and the achieved split, write nothing')
    args = ap.parse_args()

    template_column = args.template_column or None
    read_cols = None
    if args.audit_only or not args.output_dir:
        # No need to pull note text into memory just to score a split.
        read_cols = [c for c in {args.group_column, args.label_column, template_column,
                                 args.group_map_key} if c]

    path = args.input_path
    if path.lower().endswith(('.csv', '.tsv')):
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_parquet(path, columns=read_cols)
        except Exception:
            df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")

    if args.group_column not in df.columns:
        if not args.group_map_path:
            raise SystemExit(f"'{args.group_column}' not in the input; pass --group-map-path")
        gmap = pd.read_parquet(args.group_map_path, columns=[args.group_map_key, args.group_column])
        gmap = gmap.drop_duplicates(args.group_map_key)
        df = df.merge(gmap, on=args.group_map_key, how='left')
        n_missing = int(df[args.group_column].isna().sum())
        print(f"  joined '{args.group_column}': {len(df) - n_missing}/{len(df)} matched")
        if n_missing:
            raise SystemExit(f"{n_missing} rows have no '{args.group_column}'; fix the mapping first")

    df = df.dropna(subset=[args.label_column])
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    print("\n=== feasibility (what any grouped splitter could achieve here) ===")
    print(feasibility_report(df, args.label_column, args.group_column, template_column, ratios))

    print("\n=== solving ===")
    parts = balanced_group_split(
        df, args.label_column, args.group_column, template_column, ratios=ratios,
        random_seed=args.random_seed, n_restarts=args.n_restarts,
        w_rate=args.w_rate, w_size=args.w_size, w_mix=args.w_mix, verbose=True)

    if args.assignment_out and not args.audit_only:
        rows = pd.concat([p[[args.group_column]].assign(split=name)
                          for p, name in zip(parts, ['train', 'val', 'test'])])
        rows.drop_duplicates().to_csv(args.assignment_out, index=False)
        print(f"\nWrote group assignment to {args.assignment_out}")

    if args.output_dir and not args.audit_only:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        for part, name in zip(parts, ['train', 'val', 'test']):
            out = os.path.join(args.output_dir, f'{name}.parquet')
            part.to_parquet(out, index=False)
            print(f"Wrote {out} ({len(part)} rows)")


if __name__ == '__main__':
    _main()
