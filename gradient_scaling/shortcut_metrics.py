"""The three per-model quantities ShorT needs (Brown et al. 2023, Methods):

1. performance     — readmission AUROC on the test set
2. fairness        — equalized odds between templates (the paper's metric for a
                     binary attribute, used in its CheXpert/race analysis)
3. encoding        — "attribute transfer": freeze the feature extractor, fit a fresh
                     linear classifier for the template on train features, report
                     its test AUROC
"""

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present (e.g. a template with no positives)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Operating point maximising TPR - FPR. Chosen on validation, applied to test.

    A fixed 0.5 cut on a ~16%-prevalence outcome can predict almost all negatives,
    which makes every TPR/FPR ~0 and the equalized-odds gap trivially ~0.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # roc_curve's first threshold is +inf (predict nothing positive). A model no better
    # than chance can have its max J there, which would make EO trivially 0 — skip it.
    finite = np.isfinite(thr)
    return float(thr[finite][np.argmax((tpr - fpr)[finite])])


def _rates(y_true: np.ndarray, y_pred: np.ndarray):
    pos, neg = y_true == 1, y_true == 0
    tpr = float(y_pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(y_pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def equalized_odds(y_true: np.ndarray, y_score: np.ndarray, template: np.ndarray, threshold: float) -> Dict[str, float]:
    """Equalized-odds difference = max(|TPR_A - TPR_B|, |FPR_A - FPR_B|) (fairlearn convention).

    template: 0 = A, 1 = B. Lower is fairer.
    """
    y_pred = (y_score >= threshold).astype(int)
    tpr_a, fpr_a = _rates(y_true[template == 0], y_pred[template == 0])
    tpr_b, fpr_b = _rates(y_true[template == 1], y_pred[template == 1])
    tpr_gap, fpr_gap = abs(tpr_a - tpr_b), abs(fpr_a - fpr_b)
    return {
        "tpr_a": tpr_a,
        "tpr_b": tpr_b,
        "fpr_a": fpr_a,
        "fpr_b": fpr_b,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        # Undefined (NaN) if either template lacks positives or negatives in test.
        # Python's max() is order-dependent with NaN, so propagate explicitly.
        "equalized_odds": float("nan") if np.isnan([tpr_gap, fpr_gap]).any() else float(max(tpr_gap, fpr_gap)),
    }


def template_probe(train_x: np.ndarray, train_t: np.ndarray, test_x: np.ndarray, test_t: np.ndarray, seed: int) -> Dict[str, float]:
    """Attribute transfer: linear template classifier on frozen features.

    Log-loss is reported alongside AUROC because template identity is often close to
    perfectly separable (AUROC pinned at 1.0); log-loss still moves in that regime.
    """
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    probe.fit(train_x, train_t)
    p = probe.predict_proba(test_x)[:, 1]
    return {
        "probe_template_auroc": safe_auc(test_t, p),
        "probe_template_logloss": float(log_loss(test_t, p, labels=[0, 1])),
        "probe_template_accuracy": float(((p >= 0.5).astype(int) == test_t).mean()),
    }
