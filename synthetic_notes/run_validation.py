"""
Run VeriFact validation on already-reformatted rows from an ablation results CSV.

Usage:
  python run_validation.py \
    --reformat_csv synthetic_notes/ablation_results.csv \
    --gcs_parquet gs://<bucket>/synthetic_source_notes.parquet \
    --output synthetic_notes/validation_results.csv \
    --max_workers 2
"""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient, load_keys, validate_keys
from validate import validate
from confounder_utils import load_spans, prepare_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gcs_parquet(gcs_path: str) -> pd.DataFrame:
    logger.info(f"Streaming {gcs_path} from GCS...")
    result = subprocess.run(["gsutil", "cat", gcs_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr.decode()}")
    return pd.read_parquet(io.BytesIO(result.stdout))


def load_checkpoint(output_path: str) -> set[tuple]:
    if not os.path.exists(output_path):
        return set()
    try:
        done = pd.read_csv(output_path)
        return set(zip(done["note_csn_id"].astype(str), done["condition"]))
    except Exception:
        return set()


def validate_row(row: pd.Series, original_note: str, client: LLMClient,
                 spans: list[dict] | None = None) -> dict:
    note_csn_id = str(row["note_csn_id"])
    condition = row["condition"]
    reformatted = json.loads(row["reformatted_note"])

    logger.info(f"  Validating [{condition}] note {note_csn_id}...")
    report = validate(original_note=original_note, reformatted_note=reformatted,
                      client=client, spans=spans)
    s = report.summary()
    fwd = s["forward"]
    bwd = s["backward"]
    return {
        "note_csn_id": note_csn_id,
        "condition": condition,
        "faithfulness": s["faithfulness"],
        "completeness": s["completeness"],
        "f1": s["f1"],
        "content_completeness": s.get("content_completeness"),
        "content_n": s.get("content_n"),
        "template_completeness": s.get("template_completeness"),
        "template_n": s.get("template_n"),
        "copyforward_completeness": s.get("copyforward_completeness"),
        "copyforward_n": s.get("copyforward_n"),
        "forward_n_claims": fwd["n"],
        "forward_supported_pct": fwd["supported_pct"],
        "forward_contradicted_pct": fwd["contradicted_pct"],
        "forward_not_addressed_pct": fwd["not_addressed_pct"],
        "backward_n_claims": bwd["n"],
        "backward_supported_pct": bwd["supported_pct"],
        "backward_contradicted_pct": bwd["contradicted_pct"],
        "backward_not_addressed_pct": bwd["not_addressed_pct"],
        "llm_summary": s.get("llm_summary"),
        "validation_details": json.dumps(s),
    }


def main():
    parser = argparse.ArgumentParser(description="Run VeriFact validation on ablation results")
    parser.add_argument("--reformat_csv", required=True, help="Ablation results CSV with reformatted_note column")
    parser.add_argument("--gcs_parquet",  required=True, help="GCS path to original notes parquet")
    parser.add_argument("--confounders",  default=None, help="Confounder spans CSV (required for stripped validation)")
    parser.add_argument("--output",       required=True, help="Output CSV for validation results")
    parser.add_argument("--api_key",      default=None)
    parser.add_argument("--env_file",     default=None, help="Path to .env holding LLM_API_KEY_* keys")
    parser.add_argument("--check_keys",   action="store_true", help="Ping each key at startup and drop any that 401")
    parser.add_argument("--model",        default="gpt-5", choices=["gpt-5", "gpt-5-5", "gpt-5-mini", "gpt-5-nano"])
    parser.add_argument("--reasoning_effort", default="low", choices=["minimal", "low", "medium", "high"],
                        help="reasoning_effort for VeriFact validation calls (default: low)")
    parser.add_argument("--max_workers",  type=int, default=2)
    parser.add_argument("--conditions",   nargs="+", default=None, help="Filter to specific conditions")
    args = parser.parse_args()

    api_key = args.api_key or load_keys(args.env_file)
    if not api_key:
        sys.exit("No API keys found. Set LLM_API_KEY_1/LLM_API_KEY_2, pass --api_key, or use --env_file.")
    if args.check_keys:
        good = validate_keys([k.strip() for k in api_key.split(",") if k.strip()], model=args.model)
        if not good:
            sys.exit("All API keys failed validation (401). Refresh keys before running.")
        api_key = ",".join(good)
        logger.info(f"{len(good)} key(s) passed validation")
    client = LLMClient(api_key=api_key, model=args.model, reasoning_effort=args.reasoning_effort)

    results_df = pd.read_csv(args.reformat_csv)
    successes = results_df[results_df["reformat_success"] == True].copy()
    if args.conditions:
        successes = successes[successes["condition"].isin(args.conditions)]
    logger.info(f"Found {len(successes)} successful rows to validate")

    gcs_df = load_gcs_parquet(args.gcs_parquet)
    note_map = dict(zip(gcs_df["note_csn_id"].astype(str), gcs_df["full_note"]))

    confounder_df = pd.read_csv(args.confounders) if args.confounders else None

    done = load_checkpoint(args.output)
    logger.info(f"Skipping {len(done)} already-validated rows")
    jobs = [row for _, row in successes.iterrows()
            if (str(row["note_csn_id"]), row["condition"]) not in done]
    logger.info(f"Running validation on {len(jobs)} rows")

    write_lock = threading.Lock()
    write_header = (not os.path.exists(args.output)) or os.path.getsize(args.output) == 0

    def _reference_note(row):
        note_csn_id = str(row["note_csn_id"])
        original = note_map[note_csn_id]
        if row["condition"] == "stripped" and confounder_df is not None:
            spans = load_spans(confounder_df, note_csn_id)
            stripped, _ = prepare_input(original, spans, "stripped")
            return stripped
        return original

    def _spans_for(row):
        # Pass spans so VeriFact splits completeness into content vs template/copyforward
        # (yields content_completeness). Reference note for 'stripped' already has spans removed.
        if confounder_df is not None and row["condition"] != "stripped":
            return load_spans(confounder_df, str(row["note_csn_id"]))
        return None

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(validate_row, row, _reference_note(row), client, _spans_for(row)): row
            for row in jobs if str(row["note_csn_id"]) in note_map
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                with write_lock:
                    pd.DataFrame([result]).to_csv(args.output, mode="a", header=write_header, index=False)
                    write_header = False
            except Exception as e:
                row = futures[future]
                logger.error(f"Validation failed for {row['note_csn_id']}/{row['condition']}: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
