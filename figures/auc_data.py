"""Master table of per-run ROC AUCs for the 5 paper template pairs, and delta helpers.

Source: the internal results deck `Figures (6).pdf` (Sep 29 2026), transcribed by hand from the
ROC legends. That deck plots 7 pairs (0-6); the paper keeps 5 of them:

  pair  templates (tA_tB)   readmit tA vs tB     deck panel  deck letters
  1     248528_229657        8.7 vs  3.6  (5.1pp)   1         C, D
  2     801125_229657       14.5 vs  3.6 (10.9pp)   3         B, D
  3     206596_801125       53.5 vs 14.5 (39.0pp)   4         E, B
  4     206596_248528       53.5 vs  8.7 (44.8pp)   5         E, "A"  (legend says A; rates say 248528)
  5     206596_229657       53.5 vs  3.6 (49.9pp)   6         E, D

tA = first template in the pair name, tB = second. "trace" = the deck's
"Templated contents removed" runs.

scale_4B / pretrain_mellama / pretrain_llama2 come from the internal results deck `Figures (7).pdf`
(Oct 5 2026, pages 8-10), which re-ran them on the 5 paper pairs and already
uses the paper's letters (A=248528, B=229657, C=801125, D=206596). Its baseline /
1.7B / balanced / TRACE pages match the rows above exactly.

Delta modes relative to the two per-template (stratified) AUCs:
  mode="mean" -> overall - mean(tA, tB)      (original)
  mode="max"  -> overall - max(tA, tB)       (how far overall beats the BETTER template)
  mode="min"  -> overall - min(tA, tB)       (how far overall beats the WORSE template)
"""
import numpy as np
import pandas as pd

PAIR_IDS = {1: "248528_229657", 2: "801125_229657", 3: "206596_801125",
            4: "206596_248528", 5: "206596_229657"}

_rows = [
    # condition,          pair, overall,   tA,      tB
    ("baseline",          1, 0.7212, 0.6348, 0.7364),
    ("baseline",          2, 0.7746, 0.7035, 0.7365),
    ("baseline",          3, 0.8491, 0.8354, 0.6844),
    ("baseline",          4, 0.8452, 0.7701, 0.6433),
    ("baseline",          5, 0.8866, 0.7675, 0.6146),

    ("scale_1.7B",        1, 0.7430, 0.6708, 0.7454),
    ("scale_1.7B",        2, 0.7865, 0.7109, 0.7741),
    ("scale_1.7B",        3, 0.8419, 0.8240, 0.6698),
    ("scale_1.7B",        4, 0.8306, 0.7375, 0.6099),
    ("scale_1.7B",        5, 0.8937, 0.7898, 0.6857),

    ("balanced",          1, 0.6963, 0.5978, 0.7324),
    ("balanced",          2, 0.5855, 0.5703, 0.5999),
    ("balanced",          3, 0.7550, 0.8028, 0.6256),
    ("balanced",          4, 0.7169, 0.7830, 0.6460),
    ("balanced",          5, 0.7690, 0.7804, 0.5375),

    ("trace",             1, 0.7147, 0.6534, 0.6989),
    ("trace",             2, 0.7805, 0.7156, 0.7441),
    ("trace",             3, 0.8347, 0.8179, 0.6315),
    ("trace",             4, 0.8544, 0.7824, 0.6592),
    ("trace",             5, 0.8900, 0.7845, 0.7142),

    ("scale_4B",          1, 0.6561, 0.5805, 0.6127),
    ("scale_4B",          2, 0.7328, 0.6016, 0.6778),
    ("scale_4B",          3, 0.8594, 0.8426, 0.7265),
    ("scale_4B",          4, 0.8490, 0.7620, 0.6509),
    ("scale_4B",          5, 0.8996, 0.8013, 0.6604),

    ("pretrain_mellama",  1, 0.6916, 0.5917, 0.7249),
    ("pretrain_mellama",  2, 0.7156, 0.5812, 0.6012),
    ("pretrain_mellama",  3, 0.7981, 0.7401, 0.5969),
    ("pretrain_mellama",  4, 0.8061, 0.6553, 0.5798),
    ("pretrain_mellama",  5, 0.8674, 0.7115, 0.6148),

    ("pretrain_llama2",   1, 0.6731, 0.5679, 0.7144),
    ("pretrain_llama2",   2, 0.7150, 0.5725, 0.6307),
    ("pretrain_llama2",   3, 0.7989, 0.7371, 0.6269),
    ("pretrain_llama2",   4, 0.8064, 0.6481, 0.6319),
    ("pretrain_llama2",   5, 0.8572, 0.6913, 0.5281),
]
AUC = pd.DataFrame(_rows, columns=["condition", "pair", "overall", "tA", "tB"])

def _ref(tA, tB, mode):
    if mode == "mean": return (tA + tB) / 2
    if mode == "max":  return np.maximum(tA, tB)
    if mode == "min":  return np.minimum(tA, tB)
    raise ValueError(f"mode must be mean/max/min, got {mode!r}")

def delta_val(overall, tA, tB, mode):
    """Scalar Δ for one row."""
    return overall - _ref(tA, tB, mode)

def delta_df(mode):
    """Full table with a `delta` column and an `exceeds_both` flag (mode-independent)."""
    d = AUC.copy()
    d["delta"] = d["overall"] - _ref(d["tA"], d["tB"], mode)
    d["exceeds_both"] = d["overall"] > np.maximum(d["tA"], d["tB"])
    return d

def deltas(condition, mode):
    """dict pair -> Δ for a condition."""
    d = delta_df(mode)
    return d[d.condition == condition].set_index("pair")["delta"].to_dict()

_LABEL = {
    "mean": "Shortcut gap",                          # overall − mean(tA, tB)
    "max":  "Shortcut gap (vs. better template)",    # overall − max(tA, tB)
    "min":  "Shortcut gap (vs. worse template)",     # overall − min(tA, tB)
}

def label(mode):
    """Axis label. 'Shortcut gap' is the term defined in the paper."""
    return _LABEL[mode]
