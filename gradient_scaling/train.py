"""Train ONE gradient-scaled multitask model and compute its ShorT quantities.

One run = one (template pair, grad_scale, seed). A ShorT analysis is a sweep of these
runs, then analyze.py correlates template encoding with unfairness across them
(Brown et al. 2023). Training settings default to finetuning/train.py
(Qwen3-0.6B, LoRA r16/a32, LR 2e-4, effective batch 4, 4 epochs, best epoch by val
readmission AUC with patience 2, full notes). Everything that does not have to differ
is copied from finetuning/train.py, so a grad_scale = 0 run reproduces a normal
finetuning run on the same split and seed; see train_matches_finetuning below.

Run as a module from the repo root:

    python -m gradient_scaling.train \
        --data-dir data/pairs/206596_229657 \
        --grad-scale -0.1 --seed 42 --output-dir runs/206596_229657/gs-0.1_s42

Writes <output-dir>/results.json and <output-dir>/test_predictions.npz (labels,
template, scores only — no text or IDs), used by analyze.py.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from gradient_scaling.data import DataColumns, load_dataset_from_splits, tokenize_dataset
from gradient_scaling.gradient_scaling_classifier import GradientScalingClassifier
from gradient_scaling.shortcut_metrics import equalized_odds, safe_auc, template_probe, youden_threshold

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def readmission_prob(logits: np.ndarray) -> np.ndarray:
    """P(readmit) from the 2 readmission logits, computed as in finetuning/train.py."""
    two = logits[:, :2]
    exp = np.exp(two - np.max(two, axis=1, keepdims=True))
    return (exp / np.sum(exp, axis=1, keepdims=True))[:, 1]


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    return {
        "readmission_auroc": safe_auc(labels[:, 0], readmission_prob(logits)),
        "template_head_auroc": safe_auc(labels[:, 1], logits[:, 2]),
    }


class BestEpochCallback(TrainerCallback):
    """In-memory copy of finetuning's load_best_model_at_end + EarlyStoppingCallback.

    Same rules as transformers 4.57: the best epoch is any epoch whose val AUC beats
    the previous best; the patience counter only resets when it beats it by more than
    `threshold` (finetuning uses 0.001). Only the trainable params (LoRA + heads, a few
    MB) are snapshotted, so nothing is written to disk.
    """

    def __init__(self, metric: str = "eval_readmission_auroc", patience: int = 2, threshold: float = 0.001):
        self.metric, self.patience, self.threshold = metric, patience, threshold
        self.best, self.best_epoch, self.best_state, self.bad = None, None, None, 0

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        score = metrics.get(self.metric, float("nan"))
        # EarlyStoppingCallback.check_metric_value, run before the best is updated
        if self.best is None or (score > self.best and abs(score - self.best) > self.threshold):
            self.bad = 0
        else:
            self.bad += 1
        # Trainer._determine_best_metric
        if self.best is None or score > self.best:
            self.best, self.best_epoch = score, state.epoch
            self.best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        if self.bad >= self.patience:
            control.should_training_stop = True

    def restore(self, model):
        if self.best_state is None:
            return
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in self.best_state.items():
                params[n].copy_(v.to(params[n].device))


class SplitGradClipCallback(TrainerCallback):
    """Clip the template head's gradients separately from everything else.

    The Trainer clips the norm of ALL gradients together, which would let the template
    head's gradients change how much the readmission update is scaled down. Clipping
    the finetuning params (LoRA + score head) as their own group keeps that update
    identical to finetuning at grad_scale = 0. Used with max_grad_norm=0 so the
    Trainer doesn't also clip.
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        main, template = [], []
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                (template if n.startswith("template_head.") else main).append(p)
        torch.nn.utils.clip_grad_norm_(main, self.max_norm)
        torch.nn.utils.clip_grad_norm_(template, self.max_norm)


@torch.no_grad()
def extract_features(model, dataset, collator, batch_size):
    """Pooled features f(X) with the encoder frozen (eval mode, no grad)."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset.remove_columns("labels"), batch_size=batch_size, collate_fn=collator)
    feats = [
        model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy()
        for batch in loader
    ]
    return np.concatenate(feats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory with train/val/test.parquet (text, label, template A/B)")
    p.add_argument("--grad-scale", type=float, required=True, help=">0 encourages template encoding, <0 reverses it, 0 = baseline")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--template-column", default="template")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4, help="EFFECTIVE batch size (per_device x grad_accum)")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-seq", type=int, default=None, help="Truncate notes to this many tokens (default: full notes)")
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    cols = DataColumns(text=args.text_column, readmission=args.label_column, template=args.template_column)
    data_dir = Path(args.data_dir)
    dataset = load_dataset_from_splits(
        train_path=str(data_dir / "train.parquet"),
        validation_path=str(data_dir / "val.parquet"),
        test_path=str(data_dir / "test.parquet"),
        columns=cols,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=args.max_seq)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = GradientScalingClassifier(
        args.model_name,
        grad_scale=args.grad_scale,
        pad_token_id=tokenizer.pad_token_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if args.gradient_checkpointing:
        model.enable_input_require_grads()  # as in finetuning; LoRA freezes the base weights

    grad_accum = max(1, args.batch_size // max(1, args.per_device_batch_size))
    training_args = TrainingArguments(
        output_dir=str(out),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        max_grad_norm=0,  # clipped per group by SplitGradClipCallback instead
        seed=args.seed,
    )
    best = BestEpochCallback(patience=args.early_stopping_patience)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[best, SplitGradClipCallback(max_norm=1.0)],
    )
    trainer.train()
    best.restore(model)

    # --- Performance + fairness on test (threshold picked on validation) ---
    val_pred = trainer.predict(tokenized["validation"])
    test_pred = trainer.predict(tokenized["test"])
    y_val, s_val = val_pred.label_ids[:, 0], readmission_prob(val_pred.predictions)
    y, t = test_pred.label_ids[:, 0].astype(int), test_pred.label_ids[:, 1].astype(int)
    s, s_template = readmission_prob(test_pred.predictions), test_pred.predictions[:, 2]
    threshold = youden_threshold(y_val, s_val)

    results = {
        "data_dir": str(data_dir),
        "pair": data_dir.name,
        "grad_scale": args.grad_scale,
        "seed": args.seed,
        "best_epoch": best.best_epoch,
        "val_readmission_auroc": best.best,
        "test_readmission_auroc": safe_auc(y, s),
        "test_readmission_auroc_a": safe_auc(y[t == 0], s[t == 0]),
        "test_readmission_auroc_b": safe_auc(y[t == 1], s[t == 1]),
        "test_template_head_auroc": safe_auc(t, s_template),
        "threshold": threshold,
        **equalized_odds(y, s, t, threshold),
        "n_train": len(tokenized["train"]),
        "n_test": len(y),
        "n_test_a": int((t == 0).sum()),
        "n_test_b": int((t == 1).sum()),
    }

    # --- Template encoding: attribute transfer on frozen features ---
    train_x = extract_features(model, tokenized["train"], collator, args.per_device_batch_size)
    test_x = extract_features(model, tokenized["test"], collator, args.per_device_batch_size)
    train_t = np.array([lab[1] for lab in tokenized["train"]["labels"]])
    results.update(template_probe(train_x, train_t, test_x, t, seed=args.seed))

    np.savez(out / "test_predictions.npz", label=y, template=t, score=s)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
