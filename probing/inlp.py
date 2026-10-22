"""
Iterative Nullspace Projection (INLP) for removing linear template information
from encoder representations.

Reference: Ravfogel et al., "Null it Out: Guarding Protected Attributes
by Iterative Nullspace Projection" (ACL 2020).

The algorithm:
  1. Fit a logistic regression to predict template from representations.
  2. Extract the weight vector (the template direction in representation space).
  3. Project out that direction: P = I - (w wᵀ) / (wᵀ w)
  4. Repeat on projected representations until template AUC < threshold.

This guarantees removal of *linear* template information — unlike gradient
reversal, it cannot "flip" to encode the opposite signal.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import List, Tuple


def _fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 0.1,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, float]:
    """
    Fit a logistic regression probe on X → y.

    Returns:
        w: weight vector, shape (hidden_size,)
        auc: AUROC on the training data (used to decide whether to keep projecting)
    """
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", n_jobs=1)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    return clf.coef_[0], auc


def _projection_matrix(w: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection matrix that removes the w direction.
    P = I - (w wᵀ) / ||w||²
    """
    w = w / (np.linalg.norm(w) + 1e-12)  # unit vector
    return np.eye(len(w)) - np.outer(w, w)


def run_inlp(
    X: np.ndarray,
    template_labels: np.ndarray,
    *,
    n_iters: int = 40,
    auc_threshold: float = 0.55,
    probe_C: float = 0.1,
    probe_max_iter: int = 1000,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run INLP and return the accumulated projection matrix.

    Args:
        X: Representations, shape (N, hidden_size). float64 recommended.
        template_labels: Binary template labels (0/1), shape (N,).
        n_iters: Maximum projection iterations.
        auc_threshold: Stop when template AUC drops below this value.
            0.55 is a sensible default — anything close to 0.5 means the
            linear probe can barely beat chance.
        probe_C: Logistic regression regularization (smaller = more regularized).
        probe_max_iter: Solver iteration limit for logistic regression.
        verbose: Print AUC at each iteration.

    Returns:
        P: Accumulated projection matrix, shape (hidden_size, hidden_size).
           Apply to new representations as: X_new @ P.T
        aucs: List of template AUC values measured before each projection step.
    """
    hidden_size = X.shape[1]
    P = np.eye(hidden_size, dtype=np.float64)
    X_proj = X.astype(np.float64).copy()
    aucs: List[float] = []

    for i in range(n_iters):
        w, auc = _fit_probe(X_proj, template_labels, C=probe_C, max_iter=probe_max_iter)
        aucs.append(auc)

        if verbose:
            print(f"  INLP iter {i + 1:2d}: template AUC = {auc:.4f}")

        if auc < auc_threshold:
            if verbose:
                print(f"  Converged (AUC {auc:.4f} < threshold {auc_threshold}). Done.")
            break

        P_i = _projection_matrix(w)
        P = P_i @ P               # accumulate: P is product of all projections so far
        X_proj = X_proj @ P_i.T   # project current representations

    else:
        if verbose:
            print(f"  Reached max iterations ({n_iters}) without converging.")

    return P, aucs


def apply_projection(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply projection matrix to representations. X: (N, H), P: (H, H)."""
    return X.astype(np.float64) @ P.T
