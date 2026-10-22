#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label every template in the generalization sampling frame with a single
specialty class, via an LLM gateway (gpt-5).

This folds the original two-stage pipeline (step 1: IM vs other-specialty;
step 2: general IM vs disease-specific/subspecialty IM) into one call, and adds
explicit classes for the three non-IM departments used by the cross-specialty
pairs (Heme/Onc = 206596, Cardiac Surgery = 248528, Orthopedic Surgery = 229657)
so the keyword department guesses can be replaced with real labels.

Input:  a directory of template_<id>.txt shells (from
        template_pool.py --dump-shells).
Output: a CSV with one row per template. Template text is read by this script
        and sent to the LLM gateway; it is never printed to stdout.

Usage:
    uv run python classify_template_pool.py \
        --input-dir template_pool_txt \
        --output template_pool_classifications.csv \
        --model gpt-5 --reasoning-effort low --workers 12
"""

import argparse
import glob
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Import the LLM gateway client from ../synthetic_notes/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic_notes"))
from llm_client import LLMClient, load_keys  # noqa: E402


CLASSES = [
    "internal-medicine-general",
    "internal-medicine-specific",
    "hematology-oncology",
    "cardiac-surgery",
    "orthopedic-surgery",
    "other-specialty",
    "ambiguous",
]

PROMPT_TEMPLATE = """Analyze the following medical note template (an Epic SmartPhrase discharge-summary shell) and assign it exactly one specialty class.

Context: we are selecting held-out discharge-summary templates for a study of how documentation templates act as confounders in readmission prediction. We need each template's originating service.

Classes:
1. "internal-medicine-general" — general adult inpatient medicine / hospitalist. Multiple organ systems, broad differential, medication reconciliation and chronic disease management, no single disease or subspecialty focus. Terms like "general medicine", "hospitalist", "medical service".
2. "internal-medicine-specific" — internal medicine but narrowed to one disease or subspecialty (e.g. a heart-failure template, a cirrhosis template, a dedicated cardiology/GI/nephrology/pulmonary/ID service template).
3. "hematology-oncology" — cancer care, chemotherapy, hematologic malignancy, BMT, or a heme/onc service. Any oncology content puts a template here, not in internal medicine.
4. "cardiac-surgery" — cardiothoracic/cardiac surgery: CABG, valve replacement, cardiopulmonary bypass, sternotomy, epicardial wires, chest tubes post-op.
5. "orthopedic-surgery" — orthopedics/orthopaedics: arthroplasty, fracture fixation, spine or joint surgery, arthritis service.
6. "other-specialty" — any other service (general surgery, neurosurgery, OB/GYN, psychiatry, pediatrics, neurology, transplant, vascular, urology, ENT, rehab, etc.).
7. "ambiguous" — genuinely could belong to several services, or too sparse to tell.

IMPORTANT:
- Judge the CONTENT and STRUCTURE of the template, not incidental keywords. A general medicine template that merely mentions "follow up with oncology" is still internal-medicine-general, NOT hematology-oncology.
- The distinguishing question for classes 3-5 is: which service WROTE and OWNS this template?
- Distinguish class 4 (surgical, operative cardiac care) from a medical cardiology template (class 2).
- If a template is general adult inpatient care, prefer class 1.

Template:
{template}

Respond in JSON format with:
1. "classification": one of "internal-medicine-general", "internal-medicine-specific", "hematology-oncology", "cardiac-surgery", "orthopedic-surgery", "other-specialty", "ambiguous"
2. "confidence": "high", "medium", or "low"
3. "specialty_detail": the specific service/department this template belongs to, in a few words (e.g. "hospitalist medicine", "cardiothoracic surgery", "total joint arthroplasty"); "N/A" if ambiguous
4. "reasoning": brief explanation, 1-2 sentences
"""


def template_id_from_path(path: str) -> str:
    m = re.search(r"template_(\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def classify_one(client, path: str, max_chars: int) -> dict:
    tid = template_id_from_path(path)
    with open(path) as f:
        text = f.read()
    prompt = PROMPT_TEMPLATE.format(template=text[:max_chars])
    try:
        result = client.generate_json([{"role": "user", "content": prompt}])
        cls = result.get("classification", "unknown")
        if cls not in CLASSES:
            cls = "unknown"
        return {
            "template_id": tid,
            "template_length": len(text),
            "classification": cls,
            "confidence": result.get("confidence", "unknown"),
            "specialty_detail": result.get("specialty_detail", "N/A"),
            "reasoning": result.get("reasoning", ""),
            "success": True,
        }
    except Exception as e:
        return {
            "template_id": tid, "template_length": len(text),
            "classification": "error", "confidence": "unknown",
            "specialty_detail": "N/A", "reasoning": str(e), "success": False,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="template_pool_txt")
    ap.add_argument("--output", default="template_pool_classifications.csv")
    ap.add_argument("--model", default="gpt-5",
                    choices=["gpt-5", "gpt-5-5", "gpt-5-mini", "gpt-5-nano"])
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None,
                    help="Classify only the first N unprocessed templates (cost check)")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "template_*.txt")))
    if not paths:
        sys.exit(f"No template_*.txt files found in {args.input_dir}/")

    # Resume support: skip ids already in the output CSV.
    rows, done = [], set()
    if os.path.exists(args.output):
        prev = pd.read_csv(args.output, dtype={"template_id": str})
        prev = prev[prev["success"] != False]  # noqa: E712 — retry past failures
        rows = prev.to_dict("records")
        done = set(prev["template_id"].astype(str))
        print(f"Resuming: {len(done)} already classified.")

    todo = [p for p in paths if template_id_from_path(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        sys.exit("Nothing to do — every template in the input dir is already classified.")

    keys = load_keys(args.env_file)
    if not keys:
        sys.exit("No API keys found. Set LLM_API_KEY_1/LLM_API_KEY_2 or pass --env-file.")
    client = LLMClient(api_key=keys, model=args.model,
                       reasoning_effort=args.reasoning_effort, check_keys=True)

    print(f"Classifying {len(todo)} templates via the LLM gateway ({args.model}, "
          f"{args.workers} workers)...\n")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_one, client, p, args.max_chars): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            with lock:
                rows.append(row)
                if i % 10 == 0 or i == len(todo):
                    pd.DataFrame(rows).to_csv(args.output, index=False)  # incremental save
            # stdout: labels only, no template content
            print(f"  [{i}/{len(todo)}] template_{row['template_id']}: "
                  f"{row['classification']:<26} ({row['confidence']})")

    df = pd.DataFrame(rows).sort_values("template_id")
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 64)
    print(f"Saved {len(df)} rows to {args.output}\n")
    print("Class breakdown:")
    print(df["classification"].value_counts().to_string())
    tok = client.get_tokens()
    print(f"\nLLM tokens: prompt={tok['prompt']} completion={tok['completion']} "
          f"calls={tok['calls']}")


if __name__ == "__main__":
    main()
