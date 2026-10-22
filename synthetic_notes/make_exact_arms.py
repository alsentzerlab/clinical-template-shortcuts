"""Deterministic Step 2 for the PDF-exact templates, emitting VeriFact-ready CSVs.

Reads a standardized Step-1 reformat CSV (columns: note_csn_id, condition,
reformat_success, reformatted_note=standardized JSON) and re-keys each note into
template_A_exact and template_B_exact. Writes ONE CSV per arm with a single
`reformatted_note` column (the exact-template JSON) so run_validation.py can
validate each arm against the original note.

No LLM. Content across arms is identical by construction (see restructure.py
--selftest-exact).

Usage:
    python make_exact_arms.py --input <standardized>.csv --out_prefix exact
    # -> exact_armA_input.csv, exact_armB_input.csv
"""
import argparse, csv, json, sys
from pathlib import Path

import restructure as R

csv.field_size_limit(sys.maxsize)
TEMPLATES = Path(__file__).parent / "templates"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Standardized Step-1 CSV")
    ap.add_argument("--out_prefix", default="exact", help="Output prefix")
    args = ap.parse_args()

    R.assert_content_parity_exact()  # fail fast if templates drifted
    schema_a = json.loads((TEMPLATES / "template_A_exact.json").read_text())
    schema_b = json.loads((TEMPLATES / "template_B_exact.json").read_text())

    rows_in = list(csv.DictReader(open(args.input)))
    fields = ["note_csn_id", "condition", "reformat_success", "reformatted_note"]
    outs = {"A": open(f"{args.out_prefix}_armA_input.csv", "w", newline=""),
            "B": open(f"{args.out_prefix}_armB_input.csv", "w", newline="")}
    writers = {k: csv.DictWriter(v, fieldnames=fields) for k, v in outs.items()}
    for w in writers.values():
        w.writeheader()

    n = 0
    for r in rows_in:
        raw = r.get("reformatted_note")
        if not raw:
            continue
        try:
            std = json.loads(raw)
        except Exception:
            continue
        base = {"note_csn_id": r.get("note_csn_id"),
                "condition": r.get("condition"),
                "reformat_success": True}
        writers["A"].writerow({**base, "reformatted_note": json.dumps(R.restructure(std, "A_exact", schema_a))})
        writers["B"].writerow({**base, "reformatted_note": json.dumps(R.restructure(std, "B_exact", schema_b))})
        n += 1

    for v in outs.values():
        v.close()
    print(f"wrote {n} notes/arm -> {args.out_prefix}_armA_input.csv, {args.out_prefix}_armB_input.csv")


if __name__ == "__main__":
    main()
