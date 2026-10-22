"""Shared step-1 helpers for the synthetic-note pipeline: load the standardized
schema, stream the source notes, and reformat (and optionally validate) one note.

Used by run_pool_reformat.py. `condition` selects what the extractor sees: the
paper's pool used "baseline" (the note as-is); the other conditions mask or tag
templated spans using confounder_utils.
"""
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient  # noqa: E402
from reformat import reformat  # noqa: E402
from validate import validate  # noqa: E402
from confounder_utils import prepare_input  # noqa: E402

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "standardized_template.json"


# Canonical column order for the output CSV. Every row is reindexed to this before
# append, so reformat-failure rows (fewer keys) and validation-exception rows
# (validation_error key) never misalign the columns of an existing file.
CSV_COLUMNS = [
    "note_csn_id", "condition", "reformat_success", "reformat_attempts", "reformat_errors",
    "context_length_exceeded", "reformatted_note",
    "reformat_prompt_tokens", "reformat_completion_tokens", "reformat_api_calls",
    "validate_prompt_tokens", "validate_completion_tokens", "validate_api_calls",
    "faithfulness", "completeness", "f1",
    "content_completeness", "content_n", "template_completeness", "template_n",
    "copyforward_completeness", "copyforward_n",
    "forward_error_verdicts", "backward_error_verdicts",
    "forward_n_claims", "forward_supported_pct", "forward_contradicted_pct", "forward_not_addressed_pct",
    "backward_n_claims", "backward_supported_pct", "backward_contradicted_pct", "backward_not_addressed_pct",
    "llm_summary", "validation_details", "validation_error",
]


def load_schema(path=TEMPLATE_PATH) -> dict:
    # accept a bare filename (resolve under templates/), a relative, or an absolute path
    p = Path(path)
    if not p.exists():
        alt = Path(__file__).parent / "templates" / path
        if alt.exists():
            p = alt
    with open(p) as f:
        return json.load(f)


def load_gcs_parquet(gcs_path: str) -> pd.DataFrame:
    """Stream parquet from GCS without writing PHI to disk."""
    logger.info(f"Streaming {gcs_path} from GCS...")
    result = subprocess.run(["gsutil", "cat", gcs_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gsutil failed: {result.stderr.decode()}")
    return pd.read_parquet(io.BytesIO(result.stdout))


def process_one(
    row: pd.Series,
    condition: str,
    spans: list[dict],
    schema: dict,
    api_key: str,
    reformat_model: str,
    validate_model: str | None,
    validate_reasoning_effort: str | None = None,
    repair: bool = False,
    max_tokens: int = 16000,
) -> dict:
    note_csn_id = str(row["note_csn_id"])
    original_note = row["full_note"]
    source_template = row.get("template", None)

    if condition == "template":
        note_input, template_input = original_note, source_template
    else:
        note_input, template_input = prepare_input(original_note, spans, condition)

    rf_client = LLMClient(api_key=api_key, model=reformat_model)
    logger.info(f"  [{condition}] note {note_csn_id}...")
    reformat_result = reformat(
        note=note_input,
        schema=schema,
        client=rf_client,
        source_template=template_input,
        repair=repair,
        max_tokens=max_tokens,
    )
    rf_tokens = rf_client.get_tokens()

    row_out = {
        "note_csn_id": note_csn_id,
        "condition": condition,
        "reformat_success": reformat_result["success"],
        "reformat_attempts": reformat_result["attempts"],
        "reformat_errors": json.dumps(reformat_result["errors"]),
        "context_length_exceeded": "context_length_exceeded" in reformat_result["errors"],
        "reformatted_note": json.dumps(reformat_result["result"]) if reformat_result["result"] else None,
        "reformat_prompt_tokens": rf_tokens["prompt"],
        "reformat_completion_tokens": rf_tokens["completion"],
        "reformat_api_calls": rf_tokens["calls"],
        "validate_prompt_tokens": 0,
        "validate_completion_tokens": 0,
        "validate_api_calls": 0,
    }

    if validate_model and reformat_result["success"]:
        v_client = LLMClient(api_key=api_key, model=validate_model,
                             reasoning_effort=validate_reasoning_effort)
        try:
            report = validate(
                original_note=original_note,
                reformatted_note=reformat_result["result"],
                client=v_client,
                spans=spans,
            )
            summary = report.summary()
            row_out.update(_validation_columns(summary))
        except Exception as e:
            logger.error(f"  Validation failed for {note_csn_id}/{condition}: {e}")
            row_out["validation_error"] = str(e)
        v_tokens = v_client.get_tokens()
        row_out["validate_prompt_tokens"] = v_tokens["prompt"]
        row_out["validate_completion_tokens"] = v_tokens["completion"]
        row_out["validate_api_calls"] = v_tokens["calls"]

    return row_out


def _validation_columns(summary: dict) -> dict:
    """Flatten a validate.summary() dict into CSV-friendly columns."""
    fwd = summary["forward"]
    bwd = summary["backward"]
    return {
        "faithfulness": summary["faithfulness"],
        "completeness": summary["completeness"],
        "f1": summary["f1"],
        "content_completeness": summary.get("content_completeness"),
        "content_n": summary.get("content_n"),
        "template_completeness": summary.get("template_completeness"),
        "template_n": summary.get("template_n"),
        "copyforward_completeness": summary.get("copyforward_completeness"),
        "copyforward_n": summary.get("copyforward_n"),
        "forward_error_verdicts": summary.get("forward_error_verdicts", 0),
        "backward_error_verdicts": summary.get("backward_error_verdicts", 0),
        "forward_n_claims": fwd["n"],
        "forward_supported_pct": fwd["supported_pct"],
        "forward_contradicted_pct": fwd["contradicted_pct"],
        "forward_not_addressed_pct": fwd["not_addressed_pct"],
        "backward_n_claims": bwd["n"],
        "backward_supported_pct": bwd["supported_pct"],
        "backward_contradicted_pct": bwd["contradicted_pct"],
        "backward_not_addressed_pct": bwd["not_addressed_pct"],
        "llm_summary": summary.get("llm_summary"),
        "validation_details": json.dumps(summary),
    }
