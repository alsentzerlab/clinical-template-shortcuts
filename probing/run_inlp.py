"""
End-to-end INLP evaluation script.

Workflow:
  1. Load a trained encoder checkpoint (GradientReversalClassifier or plain AutoModel).
  2. Extract mean-pooled representations from train / val / test splits.
  3. Run INLP on the *training* representations (template labels only).
  4. Apply the resulting projection matrix to all splits.
  5. Train a logistic regression readmission head on projected train reps.
  6. Evaluate on the test set — before and after projection — reporting:
       - Template AUC (should drop to ~0.5 after projection)
       - Overall readmission AUC
       - Per-template readmission AUC (A and B separately)
       - Equalized odds violation: |TPR_A - TPR_B| + |FPR_A - FPR_B|

Usage:
  python -m probing.run_inlp \\
    --model-path path/to/checkpoint \\
    --train-path data/train.parquet \\
    --val-path   data/val.parquet \\
    --test-path  data/test.parquet \\
    --output     results/inlp_results.json

Note on "forcibly encode more" (the other direction):
  Use the existing GradientReversalClassifier training with lambda_grl < 0.
  When lambda_grl = -1.0, the backward pass sends +1 × grad through the
  template pathway — equivalent to standard multi-task learning, which
  *encourages* the encoder to represent template information.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from probing.inlp import apply_projection, run_inlp


# ---------------------------------------------------------------------------
# Per-iteration INLP step (used when --track-per-iter is enabled)
# ---------------------------------------------------------------------------

def _one_inlp_step(
    X_proj: np.ndarray,
    template_labels: np.ndarray,
    probe_C: float,
) -> Tuple[np.ndarray, float]:
    """
    Run a single INLP projection step on already-projected representations.
    Returns (P_i, train_template_auc).
    """
    from sklearn.linear_model import LogisticRegression as _LR
    clf = _LR(C=probe_C, max_iter=1000, solver="lbfgs")
    clf.fit(X_proj, template_labels)
    w = clf.coef_[0]
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    P_i = np.eye(len(w_unit), dtype=np.float64) - np.outer(w_unit, w_unit)
    auc = roc_auc_score(template_labels, clf.predict_proba(X_proj)[:, 1])
    return P_i, float(auc)


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states, ignoring padding."""
    mask_f = mask.to(hidden.dtype).unsqueeze(-1)      # [B, T, 1]
    summed = (hidden * mask_f).sum(dim=1)              # [B, H]
    denom = mask_f.sum(dim=1).clamp(min=1.0)          # [B, 1]
    return summed / denom                              # [B, H]


def extract_representations(
    encoder: torch.nn.Module,
    tokenizer,
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run the encoder over df[text_col] and return mean-pooled representations.

    Returns:
        reps: float32 numpy array, shape (N, hidden_size)
    """
    encoder.eval()
    encoder.to(device)

    all_reps = []
    texts = df[text_col].tolist()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = _mean_pool(out.last_hidden_state, attention_mask)
            all_reps.append(pooled.cpu().float().numpy())

    return np.concatenate(all_reps, axis=0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(model_path: str, device: str) -> Tuple[torch.nn.Module, object]:
    """
    Load an encoder from a checkpoint: a PEFT (LoRA) adapter directory, or
    failing that a plain AutoModel.

    Returns (encoder, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        from peft import PeftModel, PeftConfig
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base = AutoModel.from_pretrained(peft_cfg.base_model_name_or_path, trust_remote_code=True)
        encoder = PeftModel.from_pretrained(base, model_path)
        print(f"Loaded PEFT model from {model_path}.")
    except Exception:
        encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loaded AutoModel from {model_path}.")

    return encoder, tokenizer


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

@dataclass
class FairnessMetrics:
    readmission_auc: float
    template_auc: float          # measured by a fresh probe on the reps
    auc_template_a: float
    auc_template_b: float
    auc_gap: float               # |auc_a - auc_b|
    tpr_a: float
    tpr_b: float
    fpr_a: float
    fpr_b: float
    equalized_odds_violation: float  # |TPR_A-TPR_B| + |FPR_A-FPR_B|


def _per_template_auc(y_true, y_score, template_labels) -> Tuple[float, float]:
    mask_a = template_labels == 0
    mask_b = template_labels == 1
    auc_a = roc_auc_score(y_true[mask_a], y_score[mask_a]) if mask_a.sum() > 1 else float("nan")
    auc_b = roc_auc_score(y_true[mask_b], y_score[mask_b]) if mask_b.sum() > 1 else float("nan")
    return auc_a, auc_b


def _tpr_fpr(y_true, y_pred) -> Tuple[float, float]:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr), float(fpr)


def compute_fairness_metrics(
    X_test: np.ndarray,
    readmission_labels: np.ndarray,
    template_labels: np.ndarray,
    readmission_clf: LogisticRegression,
    probe_C: float = 0.1,
) -> FairnessMetrics:
    """
    Compute all fairness metrics given test representations and a trained
    readmission classifier. A fresh template probe is fit on X_test to
    measure how much template information remains.
    """
    # Readmission predictions
    readmission_probs = readmission_clf.predict_proba(X_test)[:, 1]
    readmission_preds = readmission_clf.predict(X_test)

    overall_auc = roc_auc_score(readmission_labels, readmission_probs)
    auc_a, auc_b = _per_template_auc(readmission_labels, readmission_probs, template_labels)

    mask_a = template_labels == 0
    mask_b = template_labels == 1
    tpr_a, fpr_a = _tpr_fpr(readmission_labels[mask_a], readmission_preds[mask_a])
    tpr_b, fpr_b = _tpr_fpr(readmission_labels[mask_b], readmission_preds[mask_b])

    # Fresh template probe on test representations — the key verification step.
    # If INLP worked, this should be ~0.5 regardless of what the training probe saw.
    template_clf = LogisticRegression(C=probe_C, max_iter=1000, solver="lbfgs")
    template_clf.fit(X_test, template_labels)
    template_probs = template_clf.predict_proba(X_test)[:, 1]
    template_auc = roc_auc_score(template_labels, template_probs)

    return FairnessMetrics(
        readmission_auc=float(overall_auc),
        template_auc=float(template_auc),
        auc_template_a=float(auc_a),
        auc_template_b=float(auc_b),
        auc_gap=float(abs(auc_a - auc_b)),
        tpr_a=tpr_a,
        tpr_b=tpr_b,
        fpr_a=fpr_a,
        fpr_b=fpr_b,
        equalized_odds_violation=float(abs(tpr_a - tpr_b) + abs(fpr_a - fpr_b)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run INLP to remove template information.")
    parser.add_argument("--model-path", required=True,
                        help="Path to trained encoder checkpoint (GRL or AutoModel).")
    parser.add_argument("--train-path", required=True, help="Path to train.parquet.")
    parser.add_argument("--val-path", required=True, help="Path to val.parquet.")
    parser.add_argument("--test-path", required=True, help="Path to test.parquet.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--readmission-col", default="label")
    parser.add_argument("--template-col", default="template")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--inlp-iters", type=int, default=40)
    parser.add_argument("--auc-threshold", type=float, default=0.55)
    parser.add_argument("--probe-C", type=float, default=0.1,
                        help="Logistic regression regularization for INLP probes.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="inlp_results.json",
                        help="Path to write JSON results.")
    parser.add_argument("--save-predictions", default=None,
                        help="If set, save per-sample test predictions to this parquet path.")
    parser.add_argument("--track-per-iter", action="store_true",
                        help="If set, evaluate fresh probe fairness metrics at every INLP iteration.")
    args = parser.parse_args()

    # --- Load data ---
    print("Loading data...")
    train_df = pd.read_parquet(args.train_path).dropna(
        subset=[args.text_col, args.readmission_col, args.template_col]
    )
    val_df = pd.read_parquet(args.val_path).dropna(
        subset=[args.text_col, args.readmission_col, args.template_col]
    )
    test_df = pd.read_parquet(args.test_path).dropna(
        subset=[args.text_col, args.readmission_col, args.template_col]
    )

    # Map template A→0, B→1
    template_map = {"A": 0, "B": 1}
    for df in [train_df, val_df, test_df]:
        df[args.template_col] = df[args.template_col].map(template_map)
    train_df = train_df.dropna(subset=[args.template_col])
    val_df = val_df.dropna(subset=[args.template_col])
    test_df = test_df.dropna(subset=[args.template_col])

    train_t = train_df[args.template_col].astype(int).values
    train_y = train_df[args.readmission_col].astype(int).values
    test_t = test_df[args.template_col].astype(int).values
    test_y = test_df[args.readmission_col].astype(int).values

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Train template dist: A={( train_t==0).sum()}, B={(train_t==1).sum()}")

    # --- Load encoder ---
    print("\nLoading encoder...")
    encoder, tokenizer = load_encoder(args.model_path, args.device)

    # --- Extract representations ---
    print("\nExtracting representations...")
    X_train = extract_representations(
        encoder, tokenizer, train_df,
        text_col=args.text_col, batch_size=args.batch_size,
        max_length=args.max_length, device=args.device,
    )
    X_test = extract_representations(
        encoder, tokenizer, test_df,
        text_col=args.text_col, batch_size=args.batch_size,
        max_length=args.max_length, device=args.device,
    )
    print(f"  Train reps: {X_train.shape}, Test reps: {X_test.shape}")

    # --- Baseline: evaluate before INLP ---
    print("\n--- Baseline (no INLP) ---")
    baseline_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    baseline_clf.fit(X_train, train_y)
    baseline_metrics = compute_fairness_metrics(
        X_test, test_y, test_t, baseline_clf, probe_C=args.probe_C
    )
    print(f"  Readmission AUC:           {baseline_metrics.readmission_auc:.4f}")
    print(f"  Template AUC (probe):      {baseline_metrics.template_auc:.4f}")
    print(f"  AUC Template A:            {baseline_metrics.auc_template_a:.4f}")
    print(f"  AUC Template B:            {baseline_metrics.auc_template_b:.4f}")
    print(f"  AUC gap |A-B|:             {baseline_metrics.auc_gap:.4f}")
    print(f"  Equalized odds violation:  {baseline_metrics.equalized_odds_violation:.4f}")

    # --- Run INLP ---
    print(f"\n--- Running INLP (max {args.inlp_iters} iters, threshold {args.auc_threshold}) ---")

    per_iter_trace = []

    if args.track_per_iter:
        # Run one step at a time, evaluating fresh probe metrics at each iteration.
        hidden_size = X_train.shape[1]
        P = np.eye(hidden_size, dtype=np.float64)
        X_train_proj = X_train.astype(np.float64).copy()
        inlp_aucs = []

        for i in range(args.inlp_iters):
            P_i, train_auc = _one_inlp_step(X_train_proj, train_t, args.probe_C)
            inlp_aucs.append(train_auc)

            # Fresh probe and fairness metrics on test set at this iteration
            X_test_proj_i = apply_projection(X_test, P)
            iter_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            iter_clf.fit(X_train_proj, train_y)
            iter_metrics = compute_fairness_metrics(
                X_test_proj_i, test_y, test_t, iter_clf, probe_C=args.probe_C
            )
            per_iter_trace.append({
                "iter": i,
                "train_template_auc": train_auc,
                **asdict(iter_metrics),
            })
            print(f"  iter {i+1:2d}: train_tmpl={train_auc:.4f} | "
                  f"test_tmpl={iter_metrics.template_auc:.4f} | "
                  f"readmit={iter_metrics.readmission_auc:.4f} | "
                  f"eq_odds={iter_metrics.equalized_odds_violation:.4f}")

            if train_auc < args.auc_threshold:
                print(f"  Converged. Done.")
                break

            # Accumulate projection
            P = P_i @ P
            X_train_proj = X_train_proj @ P_i.T

        X_train_proj = apply_projection(X_train, P)
        X_test_proj = apply_projection(X_test, P)
    else:
        P, inlp_aucs = run_inlp(
            X_train, train_t,
            n_iters=args.inlp_iters,
            auc_threshold=args.auc_threshold,
            probe_C=args.probe_C,
            verbose=True,
        )
        X_train_proj = apply_projection(X_train, P)
        X_test_proj = apply_projection(X_test, P)

    # --- Train readmission head on projected reps ---
    print("\n--- After INLP ---")
    inlp_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    inlp_clf.fit(X_train_proj, train_y)
    inlp_metrics = compute_fairness_metrics(
        X_test_proj, test_y, test_t, inlp_clf, probe_C=args.probe_C
    )
    print(f"  Readmission AUC:           {inlp_metrics.readmission_auc:.4f}")
    print(f"  Template AUC (fresh probe):{inlp_metrics.template_auc:.4f}  ← should be ~0.5")
    print(f"  AUC Template A:            {inlp_metrics.auc_template_a:.4f}")
    print(f"  AUC Template B:            {inlp_metrics.auc_template_b:.4f}")
    print(f"  AUC gap |A-B|:             {inlp_metrics.auc_gap:.4f}")
    print(f"  Equalized odds violation:  {inlp_metrics.equalized_odds_violation:.4f}")

    # --- Save results ---
    results = {
        "model_path": args.model_path,
        "inlp_iterations": len(inlp_aucs),
        "inlp_auc_trace": inlp_aucs,
        "baseline": asdict(baseline_metrics),
        "after_inlp": asdict(inlp_metrics),
        "per_iter_trace": per_iter_trace,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # --- Save per-sample predictions ---
    if args.save_predictions is not None:
        preds_df = pd.DataFrame({
            "true_label": test_y,
            "template": test_t,
            "baseline_pred_prob": baseline_clf.predict_proba(X_test)[:, 1],
            "after_inlp_pred_prob": inlp_clf.predict_proba(X_test_proj)[:, 1],
        })
        os.makedirs(os.path.dirname(os.path.abspath(args.save_predictions)), exist_ok=True)
        preds_df.to_parquet(args.save_predictions, index=False)
        print(f"Per-sample predictions saved to {args.save_predictions}")


if __name__ == "__main__":
    main()
