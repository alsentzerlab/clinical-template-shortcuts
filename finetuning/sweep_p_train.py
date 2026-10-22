"""
p_train sweep: process data → train → probe for each p_train value.

For each p_train value, this script:
  1. Runs process_data.py to create train/val/test splits with the given bias
  2. Fine-tunes a LoRA model on the processed data
  3. Runs a fresh template probe (run_inlp.py) to measure template AUC and fairness
  4. Saves a summary JSON across all p_train values

Usage:
  python -m finetuning.sweep_p_train \\
    --template-a-path  data/synthetic_templates/template_a.parquet \\
    --template-b-path  data/synthetic_templates/template_b.parquet \\
    --p-trains         0.5 0.6 0.7 0.8 0.9 \\
    --max-samples-per-template 2000 \\
    --output-dir       models/p_train_sweep \\
    --results-dir      results/p_train_sweep

Results:
  {results_dir}/p{value}.json       — full probe output per p_train
  {results_dir}/summary.json        — collated table for plotting
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def _p_tag(p: float) -> str:
    """Turn a p_train float into a filesystem-safe tag, e.g. 0.8 -> 'p08'."""
    return "p" + f"{p:.2f}".replace("0.", "").replace(".", "")


def process_one(
    p: float,
    template_a_path: str,
    template_b_path: str,
    max_samples_per_template: int,
    processed_dir: str,
    skip_if_exists: bool,
    group_column: str | None = None,
    group_map_path: str | None = None,
    group_map_key: str = "pat_enc_csn_id",
    stratify_groups: bool = False,
    match_rates: bool = False,
    match_eval_bias: bool = False,
) -> str:
    """Run process_data.py for one p_train value. Returns output_dir path."""
    tag = _p_tag(p)
    out_dir = os.path.join(processed_dir, tag)

    if skip_if_exists and os.path.exists(os.path.join(out_dir, "train.parquet")):
        print(f"  [skip] Processed data already exists at {out_dir}")
        return out_dir

    print(f"\n{'='*60}")
    print(f"  Processing data for p_train={p}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "finetuning.process_data",
        "--template-a-path", template_a_path,
        "--template-b-path", template_b_path,
        "--p-train", str(p),
        "--output-dir", out_dir,
    ]
    # Only cap per-template size when explicitly asked. Capping above the available
    # supply used to starve one template of negatives (spurious template<->label
    # correlation); the uncapped path uses all notes and stays balanced.
    if max_samples_per_template is not None:
        cmd += ["--max-samples-per-template", str(max_samples_per_template)]
    # Patient-safe splitting: keep every note from one patient in a single split.
    if group_column:
        cmd += ["--group-column", group_column, "--group-map-key", group_map_key]
        if group_map_path:
            cmd += ["--group-map-path", group_map_path]
    # Hold the readmission rate constant across splits. --match-rates solves the
    # assignment directly and holds it far tighter than --stratify-groups.
    if match_rates:
        cmd += ["--match-rates"]
    elif stratify_groups:
        cmd += ["--stratify-groups"]
    # Bias val/test at the same p as train, so each template's readmission rate
    # matches train-vs-test (a persistent shortcut rather than a neutral test).
    if match_eval_bias:
        cmd += ["--p-eval", str(p)]
    subprocess.run(cmd, check=True)
    return out_dir


def train_one(
    p: float,
    processed_dir: str,
    model_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq: int,
    skip_if_exists: bool,
    model_name: str | None = None,
    per_device_batch_size: int = 1,
    gradient_checkpointing: bool = False,
) -> str:
    """Fine-tune a LoRA model for one p_train value. Returns model path."""
    tag = _p_tag(p)
    model_path = os.path.join(model_dir, tag)

    if skip_if_exists and os.path.exists(os.path.join(model_path, "config.json")):
        print(f"  [skip] Model already exists at {model_path}")
        return model_path

    print(f"\n{'='*60}")
    print(f"  Training model for p_train={p}")
    print(f"{'='*60}")

    data_dir = os.path.join(processed_dir, tag)
    cmd = [
        sys.executable, "-m", "finetuning.train",
        "--train-data", os.path.join(data_dir, "train.parquet"),
        "--val-data", os.path.join(data_dir, "val.parquet"),
        "--test-data", os.path.join(data_dir, "test.parquet"),
        "--output-dir", model_path,
        "--epochs", str(num_epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--per-device-batch-size", str(per_device_batch_size),
    ]
    # Only cap sequence length if asked; otherwise train.py reads full notes (max_seq=None).
    if max_seq is not None:
        cmd += ["--max-seq", str(max_seq)]
    # Override the model (else train.py's default Qwen/Qwen3-0.6B is used).
    if model_name:
        cmd += ["--model-name", model_name]
    # Trade compute for memory: needed to fit full notes on a 40GB GPU.
    if gradient_checkpointing:
        cmd += ["--gradient-checkpointing"]
    subprocess.run(cmd, check=True)
    return model_path


def probe_one(
    p: float,
    model_path: str,
    processed_dir: str,
    results_dir: str,
    skip_if_exists: bool,
) -> dict:
    """Run a fresh template probe on one model. Returns results dict."""
    tag = _p_tag(p)
    results_path = os.path.join(results_dir, f"{tag}.json")
    predictions_path = os.path.join(results_dir, f"{tag}_predictions.parquet")

    if skip_if_exists and os.path.exists(results_path):
        print(f"  [skip] Probe results already exist at {results_path}")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n  Probing model for p_train={p} ...")
    data_dir = os.path.join(processed_dir, tag)
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "probing.run_inlp",
        "--model-path", model_path,
        "--train-path", os.path.join(data_dir, "train.parquet"),
        "--val-path", os.path.join(data_dir, "val.parquet"),
        "--test-path", os.path.join(data_dir, "test.parquet"),
        "--output", results_path,
        "--inlp-iters", "1",
        "--save-predictions", predictions_path,
    ]
    subprocess.run(cmd, check=True)

    with open(results_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep p_train values.")
    parser.add_argument("--template-a-path", required=True)
    parser.add_argument("--template-b-path", required=True)
    parser.add_argument(
        "--p-trains", nargs="+", type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--max-samples-per-template", type=int, default=None,
                        help="Cap per-template size. Default None = use all notes (balanced, "
                             "constant train size across p). Only set this if you need a fixed "
                             "cap AND have enough negatives to fill both templates.")
    parser.add_argument("--output-dir", default="models/p_train_sweep",
                        help="Root directory for model checkpoints.")
    parser.add_argument("--results-dir", default="results/p_train_sweep",
                        help="Root directory for probe results.")
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq", type=int, default=None,
                        help="Token truncation length. Default None = NO truncation (read every "
                             "note in full; longest ~14.5k tokens). Set an int (e.g. 9000) to cap "
                             "if you hit GPU-memory limits.")
    parser.add_argument("--model-name", default=None,
                        help="HF model to fine-tune (default: train.py's Qwen/Qwen3-0.6B). "
                             "Set e.g. Qwen/Qwen3-1.7B for scaling; adjust --learning-rate too.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Pass --gradient-checkpointing to train.py: recompute activations "
                             "to fit full notes on a 40GB GPU. ~30%% slower; does NOT change the "
                             "data, so results stay comparable to runs without it.")
    parser.add_argument("--per-device-batch-size", type=int, default=1,
                        help="Notes per forward pass. Raise (memory permitting) to speed up "
                             "without changing AUC (effective batch stays --batch-size).")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip steps that already have output on disk.")
    parser.add_argument("--delete-models", action="store_true",
                        help="Delete model checkpoints after probing to save disk space.")
    parser.add_argument("--group-column", default=None,
                        help="Split by this column (e.g. 'pat_id') so no patient spans splits. "
                             "Passed through to process_data.py for every p_train.")
    parser.add_argument("--group-map-path", default=None,
                        help="File to join --group-column from, if the template parquets don't carry it.")
    parser.add_argument("--group-map-key", default="pat_enc_csn_id",
                        help="Join key linking the template parquets to --group-map-path.")
    parser.add_argument("--stratify-groups", action="store_true",
                        help="Hold the overall readmission rate constant across train/val/test "
                             "(StratifiedGroupKFold). Recommended with --group-column.")
    parser.add_argument("--match-rates", action="store_true",
                        help="Hold each template's readmission rate (and the overall rate, split "
                             "ratios and case mix) constant across train/val/test by solving the "
                             "patient assignment. Preferred over --stratify-groups.")
    parser.add_argument("--match-eval-bias", action="store_true",
                        help="Bias val/test at the same p as train (--p-eval=p per value), so each "
                             "template's readmission rate matches train-vs-test. Measure the shortcut "
                             "via the INLP probe, not a raw test-accuracy drop.")
    args = parser.parse_args()

    processed_dir = os.path.join(args.output_dir, "processed")
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    summary = []

    for p in args.p_trains:
        tag = _p_tag(p)

        # 1. Process data
        process_one(
            p=p,
            template_a_path=args.template_a_path,
            template_b_path=args.template_b_path,
            max_samples_per_template=args.max_samples_per_template,
            processed_dir=processed_dir,
            skip_if_exists=args.skip_if_exists,
            group_column=args.group_column,
            group_map_path=args.group_map_path,
            group_map_key=args.group_map_key,
            stratify_groups=args.stratify_groups,
            match_rates=args.match_rates,
            match_eval_bias=args.match_eval_bias,
        )

        # 2. Train
        model_path = train_one(
            p=p,
            processed_dir=processed_dir,
            model_dir=model_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq=args.max_seq,
            skip_if_exists=args.skip_if_exists,
            model_name=args.model_name,
            per_device_batch_size=args.per_device_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        # 3. Probe (training script saves to final_model/ subdirectory)
        results = probe_one(
            p=p,
            model_path=os.path.join(model_path, "final_model"),
            processed_dir=processed_dir,
            results_dir=args.results_dir,
            skip_if_exists=args.skip_if_exists,
        )

        # 4. Optionally delete model checkpoint
        if args.delete_models:
            print(f"  Deleting checkpoint {model_path} to free disk space ...")
            shutil.rmtree(model_path, ignore_errors=True)

        # 5. Collect summary row
        baseline = results["baseline"]
        row = {
            "p_train": p,
            "template_auc": baseline["template_auc"],
            "readmission_auc": baseline["readmission_auc"],
            "auc_gap": baseline["auc_gap"],
            "equalized_odds": baseline["equalized_odds_violation"],
        }
        print(f"\n  p_train={p} | template AUC={baseline['template_auc']:.4f} "
              f"| readmission AUC={baseline['readmission_auc']:.4f} "
              f"| eq-odds={baseline['equalized_odds_violation']:.4f}")
        summary.append(row)

    # Save summary
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Summary saved to {summary_path}")
    print(f"{'='*60}")
    print(f"\n{'p_train':>10} {'template AUC':>14} {'readmit AUC':>13} {'eq-odds':>10}")
    print("-" * 52)
    for row in summary:
        print(f"{row['p_train']:>10.2f} {row['template_auc']:>14.4f} "
              f"{row['readmission_auc']:>13.4f} {row['equalized_odds']:>10.4f}")


if __name__ == "__main__":
    main()
