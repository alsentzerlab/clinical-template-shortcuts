#!/usr/bin/env python3
"""
Decompose overall readmission AUC into within-template and between-template parts.

Why: overall AUC is computed over every (positive, negative) pair in the test set.
Some of those pairs sit inside one template; the rest straddle the two. When template
correlates with outcome, the straddling pairs are easy for the wrong reason — the model
can rank them using template identity alone. So

    overall AUC  =  w_within * AUC_within  +  w_cross * AUC_cross

where the weights are just the share of pairs of each kind. AUC_cross is the shortcut
term: at p=0.5 (no template->outcome correlation) it should sit near 0.5, and it climbs
toward 1.0 as the correlation strengthens.

`overall - mean(auc_A, auc_B)` — the quick version — is reported too, but prefer
`overall - AUC_within`: the unweighted mean over-weights whichever template has few
positives, which at extreme p is the noisy one.

Usage:
    python finetuning/auc_decomposition.py results/synthetic_sweep
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def _pair_auc(pos_scores, neg_scores):
    """P(pos ranked above neg), ties at 0.5. Returns (auc, n_pairs)."""
    n = pos_scores.size * neg_scores.size
    if n == 0:
        return np.nan, 0
    d = pos_scores[:, None] - neg_scores[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean()), n


def decompose(df, score_col='baseline_pred_prob'):
    y = df['true_label'].values.astype(int)
    s = df[score_col].values.astype(float)
    t = df['template'].values
    # template may be 0/1 or 'A'/'B'
    is_a = (t == 0) if np.issubdtype(np.asarray(t).dtype, np.number) else (t == 'A')

    out = {'overall_auc': _auc(y, s)}

    pos_a, neg_a = s[is_a & (y == 1)], s[is_a & (y == 0)]
    pos_b, neg_b = s[~is_a & (y == 1)], s[~is_a & (y == 0)]
    out['n_pos_a'], out['n_neg_a'] = pos_a.size, neg_a.size
    out['n_pos_b'], out['n_neg_b'] = pos_b.size, neg_b.size

    auc_a, n_aa = _pair_auc(pos_a, neg_a)
    auc_b, n_bb = _pair_auc(pos_b, neg_b)
    out['auc_a'], out['auc_b'] = auc_a, auc_b

    # cross-template pairs: positive from one template vs negative from the other
    auc_ab, n_ab = _pair_auc(pos_a, neg_b)
    auc_ba, n_ba = _pair_auc(pos_b, neg_a)

    n_within = n_aa + n_bb
    n_cross = n_ab + n_ba
    within = np.nansum([np.nan_to_num(auc_a) * n_aa, np.nan_to_num(auc_b) * n_bb])
    within = within / n_within if n_within else np.nan
    cross = np.nansum([np.nan_to_num(auc_ab) * n_ab, np.nan_to_num(auc_ba) * n_ba])
    cross = cross / n_cross if n_cross else np.nan

    out['auc_within'] = within
    out['auc_cross'] = cross
    out['w_within'] = n_within / (n_within + n_cross)
    out['w_cross'] = n_cross / (n_within + n_cross)
    # the two "overall minus average" flavours
    # np.nanmean of an all-NaN slice warns; that case is real (a template with zero
    # positives at p=0 / p=1), so report NaN deliberately rather than warn.
    present = [v for v in (auc_a, auc_b) if v == v]
    out['overall_minus_mean_ab'] = (out['overall_auc'] - float(np.mean(present))
                                    if present else np.nan)
    out['overall_minus_within'] = out['overall_auc'] - within
    return out


def main(results_dir):
    rows = []
    pattern = os.path.join(os.path.expanduser(results_dir), 'p*_predictions.parquet')
    for f in sorted(glob.glob(pattern)):
        tag = os.path.basename(f).split('_')[0]
        r = decompose(pd.read_parquet(f))
        r['p'] = tag
        rows.append(r)
    if not rows:
        print(f"No *_predictions.parquet found in {results_dir}")
        return
    df = pd.DataFrame(rows).sort_values('p', key=lambda c: c.str[1:].astype(int))

    print(f"\n{'p':>6}{'overall':>9}{'auc_A':>8}{'auc_B':>8}{'within':>9}{'cross':>8}"
          f"{'ovr-mean(A,B)':>15}{'ovr-within':>12}")
    print('-' * 75)
    for _, r in df.iterrows():
        def f(v, w=8, d=4):
            return f"{v:>{w}.{d}f}" if isinstance(v, float) and v == v else f"{'nan':>{w}}"
        print(f"{r['p']:>6}{f(r['overall_auc'], 9)}{f(r['auc_a'])}{f(r['auc_b'])}"
              f"{f(r['auc_within'], 9)}{f(r['auc_cross'])}"
              f"{f(r['overall_minus_mean_ab'], 15)}{f(r['overall_minus_within'], 12)}")

    print(f"\n  within/cross pair weights: {df.w_within.mean():.3f} / {df.w_cross.mean():.3f}")
    print("  auc_cross is the shortcut term — near 0.5 means template carries no outcome")
    print("  information; toward 1.0 means the model can rank across templates by template alone.")
    df.to_csv(os.path.join(os.path.expanduser(results_dir), 'auc_decomposition.csv'), index=False)
    print(f"\n  wrote {os.path.join(results_dir, 'auc_decomposition.csv')}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/synthetic_sweep')
