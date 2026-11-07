#!/usr/bin/env python3
"""Bridge: turn the real exact-template A/B reformats into process_data.py inputs.

Takes the two per-arm CSVs produced by synthetic_notes/make_exact_arms.py
(columns: note_csn_id, condition, reformat_success, reformatted_note=exact-template
JSON) and:
  1. renders each arm's structured JSON into a plain-text discharge summary
     (schema display-names + field order + static boilerplate = the surface form
     the classifier sees; A and B differ ONLY in structure, content is identical),
  2. joins readmit_binary (from the note parquet) and pat_id (from filterednotes),
  3. writes template_a.parquet / template_b.parquet keyed by note_id (=note_csn_id),
     each carrying full_note, readmit_binary, note_id, pat_id, note_csn_id.

The outputs plug straight into process_data.py paired-bias mode:
  python -m finetuning.process_data \\
    --template-a-path template_a.parquet --template-b-path template_b.parquet \\
    --p-train 0.9 --group-column pat_id --output-dir processed/p09

Because pat_id is a column already, the grouped (patient-safe) split needs no map file.

No PHI is printed; only shapes/counts.

Usage:
  python synthetic_notes/build_paired_pool.py \\
    --arm-a-csv synthetic_notes/pool_exact_armA_input.csv \\
    --arm-b-csv synthetic_notes/pool_exact_armB_input.csv \\
    --label-parquet data/raw/synthetic_source_notes.parquet \\
    --patid-parquet data/raw/filtered_notes.parquet \\
    --output-dir data/paired_pool
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Reuse the exact same value-flattening the restructurer uses, for consistency.
NR = Path(__file__).resolve().parent
sys.path.insert(0, str(NR))
from restructure import _render_value, _pretty  # noqa: E402

csv.field_size_limit(sys.maxsize)
TEMPLATES = NR / "templates"


def _load_gcs_or_local_parquet(path: str, columns=None) -> pd.DataFrame:
    """Read a parquet from GCS (stream) or local disk, optionally a column subset."""
    if os.path.exists(path):
        return pd.read_parquet(path, columns=columns)
    buf = subprocess.run(["gsutil", "cat", path], capture_output=True)
    if buf.returncode != 0:
        raise RuntimeError(f"gsutil failed for {path}: {buf.stderr.decode()[:300]}")
    df = pd.read_parquet(io.BytesIO(buf.stdout))
    return df[columns] if columns else df


def render_note(rendered_json: dict, schema: dict) -> str:
    """Flatten one exact-template JSON into readable discharge-summary text.

    Sections appear in schema order under their display-name headers. Single-field
    sections (composed free-text, or a lone value/static block) render the value
    directly; multi-field sections render 'Field Label: value' per field; static
    boilerplate always renders raw. Only the STRUCTURE (headers/order/grouping/
    boilerplate) differs across arms — the note-derived content is identical.
    """
    segments = []
    for section, body in rendered_json.items():
        if section.startswith("_") or not isinstance(body, dict):
            continue
        sch = schema.get(section, {})
        header = (sch.get("display_name") or _pretty(section)).upper()
        fields_def = sch.get("fields", {}) or {}
        multi = len(body) > 1
        lines = [header]
        for field, val in body.items():
            fdef = fields_def.get(field, {})
            is_static = isinstance(fdef, dict) and "static" in fdef
            txt = _render_value(val)
            if not txt:
                continue
            if is_static or not multi:
                lines.append(txt)
            else:
                lines.append(f"{_pretty(field)}: {txt}")
        if len(lines) > 1:  # header had at least one non-empty field
            segments.append("\n".join(lines))
    return "\n\n".join(segments)


def _render_arm(csv_path: str, schema: dict) -> dict:
    """note_csn_id -> rendered text for every successfully-reformatted row."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        if str(r.get("reformat_success")).lower() not in ("true", "1"):
            continue
        raw = r.get("reformatted_note")
        if not raw:
            continue
        try:
            rj = json.loads(raw)
        except Exception:
            continue
        out[str(r["note_csn_id"])] = render_note(rj, schema)
    return out


def main():
    ap = argparse.ArgumentParser(description="Build paired template_a/template_b parquets from exact-template reformats.")
    ap.add_argument("--arm-a-csv", required=True)
    ap.add_argument("--arm-b-csv", required=True)
    ap.add_argument("--label-parquet", required=True, help="Note parquet with note_csn_id + readmit_binary")
    ap.add_argument("--patid-parquet", required=True, help="filterednotes parquet with note_csn_id + pat_id")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-column", default="full_note")
    ap.add_argument("--label-column", default="readmit_binary")
    args = ap.parse_args()

    schema_a = json.loads((TEMPLATES / "template_A_exact.json").read_text())
    schema_b = json.loads((TEMPLATES / "template_B_exact.json").read_text())

    text_a = _render_arm(args.arm_a_csv, schema_a)
    text_b = _render_arm(args.arm_b_csv, schema_b)
    common = sorted(set(text_a) & set(text_b))
    print(f"Arm A rendered: {len(text_a)} | Arm B rendered: {len(text_b)} | paired (both arms): {len(common)}")

    labels = _load_gcs_or_local_parquet(args.label_parquet, columns=["note_csn_id", args.label_column])
    label_map = dict(zip(labels["note_csn_id"].astype(str), labels[args.label_column]))
    patids = _load_gcs_or_local_parquet(args.patid_parquet, columns=["note_csn_id", "pat_id"])
    patid_map = dict(zip(patids["note_csn_id"].astype(str), patids["pat_id"]))

    rows_a, rows_b, dropped = [], [], {"no_label": 0, "no_patid": 0}
    for ncsn in common:
        if ncsn not in label_map or pd.isna(label_map[ncsn]):
            dropped["no_label"] += 1
            continue
        if ncsn not in patid_map or pd.isna(patid_map[ncsn]):
            dropped["no_patid"] += 1
            continue
        base = {"note_id": ncsn, "note_csn_id": ncsn,
                args.label_column: int(label_map[ncsn]), "pat_id": patid_map[ncsn]}
        rows_a.append({**base, args.text_column: text_a[ncsn]})
        rows_b.append({**base, args.text_column: text_b[ncsn]})

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    os.makedirs(args.output_dir, exist_ok=True)
    path_a = os.path.join(args.output_dir, "template_a.parquet")
    path_b = os.path.join(args.output_dir, "template_b.parquet")
    df_a.to_parquet(path_a, index=False)
    df_b.to_parquet(path_b, index=False)

    n = len(df_a)
    print(f"Dropped: {dropped['no_label']} missing label, {dropped['no_patid']} missing pat_id")
    if n:
        print(f"Wrote {n} paired notes/arm -> {path_a}, {path_b}")
        print(f"  readmit rate: {df_a[args.label_column].mean():.4f} | positives: {int(df_a[args.label_column].sum())}")
        print(f"  unique patients (pat_id): {df_a['pat_id'].nunique()} "
              f"({n / max(df_a['pat_id'].nunique(), 1):.2f} notes/patient)")
    else:
        print("No paired notes written — check inputs.")


if __name__ == "__main__":
    main()
