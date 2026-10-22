#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find candidate held-out generalization templates from the SAME departments as
the non-IM (cross-specialty) pair templates: Hematology/Oncology, Cardiac
Surgery, and Orthopedic Surgery.

Replicates the note-count + readmission-rate computation from
complete_template_analysis.ipynb, then tags each template's department by
scanning its SmartPhrase shell for Epic RCC markers and service-line keywords.

Emits aggregates only (template_id, department, n_notes, readmission_rate,
matched_marker) to CSV + stdout. Template shell text is read by the script and
used for matching, but never printed.

Usage:
    uv run python template_pool.py \
        --smartphrases-pkl data/raw/smartphrases.pkl \
        --readmission-pkl data/raw/readmission.pkl \
        --template-ids-pkl data/raw/template_ids.pkl \
        --filtered-notes data/raw/filtered_notes.parquet \
        --out same_department_candidates.csv
"""

import argparse
import io
import os
import pickle
import re

import pandas as pd

try:
    from striprtf.striprtf import rtf_to_text
except ModuleNotFoundError:
    rtf_to_text = None

# Templates already used — exclude from candidates.
NONIM_PAIR_IDS = {206596, 248528, 229657, 624343}
IM_PAIR_IDS = {801125, 249614, 1337772}
IM_GENERALIZATION_IDS = {
    175596, 573433, 573763, 624343, 853611, 1019919,
    1026076, 1337517, 1357123, 1387007, 1438077, 1446346,
}
EXCLUDE_IDS = NONIM_PAIR_IDS | IM_GENERALIZATION_IDS

# The sampling frame excludes only templates a model is actually TRAINED on
# (the IM and non-IM pair templates). The previously hand-picked generalization
# templates stay eligible, so the new 12 can be drawn at random over the whole
# held-out space rather than around them.
IN_USE_IDS = NONIM_PAIR_IDS | IM_PAIR_IDS

MIN_NOTES = 100

# Department detection. Order matters: surgical cardiac beats medical cardiology.
# Each entry: (department, list of lowercase substrings). First department with a
# hit wins, except we check Cardiac Surgery before medical Cardiology.
DEPT_RULES = [
    ("Heme/Onc", [
        "@rcconcology@", "hematology/oncology", "hematology-oncology",
        "hematology / oncology", "med 9", "oncology", "hematology",
    ]),
    ("Cardiac Surgery", [
        "cardiopulmonary bypass", "coronary artery bypass", "cabg",
        "epicardial", "sternotom", "cardiac surgery", "cardiothoracic",
        "aortic valve", "mitral valve", "cardiac surgeon",
    ]),
    ("Orthopedic Surgery", [
        "@rccortho@", "orthopaedic", "orthopedic surgery", "orthopedics",
        "arthritis service", "arthroplasty",
    ]),
    # Medical cardiology surfaced separately (RCC tag without surgical terms).
    ("Cardiology (medical)", ["@rcccardiology@"]),
]


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path):
    with open(os.path.expanduser(path), "rb") as f:
        return NumpyUnpickler(f).load()


def extract_text(field_value):
    if hasattr(field_value, "iloc"):
        field_value = field_value.iloc[0]
    elif isinstance(field_value, (list, tuple)) and len(field_value) > 0:
        field_value = field_value[0]
    if not isinstance(field_value, str):
        field_value = str(field_value)
    if field_value.strip().startswith("{\\rtf") and rtf_to_text is not None:
        try:
            return rtf_to_text(field_value)
        except Exception:
            return field_value
    return field_value


def build_template_texts(smartphrases_data) -> dict:
    """Return {int template_id: shell_text} for every template."""
    out = {}
    if hasattr(smartphrases_data, "iterrows"):
        df = smartphrases_data
        id_col = next((c for c in ("smartphrase_id", "smartphrases_id", "template_id")
                       if c in df.columns), None)
        text_col = "full_note" if "full_note" in df.columns else (
            "template" if "template" in df.columns else None)
        for idx, row in df.iterrows():
            tid = row.get(id_col) if id_col else idx
            txt = extract_text(row[text_col]) if text_col else ""
            try:
                out[int(tid)] = txt
            except (ValueError, TypeError):
                continue
    elif isinstance(smartphrases_data, dict):
        for key, value in smartphrases_data.items():
            if isinstance(value, dict):
                raw = value.get("full_note", value.get("template", ""))
            else:
                raw = value
            try:
                out[int(key)] = extract_text(raw)
            except (ValueError, TypeError):
                continue
    return out


def detect_department(text: str):
    low = text.lower()
    # Cardiac Surgery must beat medical Cardiology, so check its terms first.
    for dept, markers in DEPT_RULES:
        for m in markers:
            if m in low:
                return dept, m
    return None, None


def compute_stats(readmission_pkl, template_ids_pkl, filtered_notes):
    """Per-template n_notes + readmission_rate (mirrors the notebook join)."""
    readmission = load_pickle(readmission_pkl)
    readmission["readmit_binary"] = readmission["readmission"] > 0

    template_ids = load_pickle(template_ids_pkl)
    template_ids["note_id"] = template_ids["note_id"].astype(int)

    notes = pd.read_parquet(os.path.expanduser(filtered_notes))
    dc = notes.merge(template_ids, on="note_id", how="inner").merge(
        readmission, on="pat_enc_csn_id", how="inner")
    dc["smartphrases_id"] = dc["smartphrases_id"].apply(lambda x: list(set(x)))

    df = dc[["pat_enc_csn_id", "note_id", "smartphrases_id", "readmit_binary"]].explode(
        "smartphrases_id")
    stats = df.groupby("smartphrases_id")["readmit_binary"].agg(["mean", "count"])
    stats = stats.rename(columns={"mean": "readmission_rate", "count": "n_notes"})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartphrases-pkl", default="data/raw/smartphrases.pkl")
    ap.add_argument("--readmission-pkl",
                    default="data/raw/readmission.pkl")
    ap.add_argument("--template-ids-pkl", default="data/raw/template_ids.pkl")
    ap.add_argument("--filtered-notes", default="data/raw/filtered_notes.parquet")
    ap.add_argument("--out", default="same_department_candidates.csv")
    ap.add_argument("--min-notes", type=int, default=MIN_NOTES)
    ap.add_argument("--frame-out", default=None,
                    help="Also emit the FULL sampling frame (every template with "
                         ">= --min-notes notes, regardless of department) to this CSV")
    ap.add_argument("--dump-shells", default=None,
                    help="Also write each sampling-frame template's shell to "
                         "<DIR>/template_<id>.txt for the LLM specialty pass")
    args = ap.parse_args()

    print("Loading smartphrase shells...")
    texts = build_template_texts(load_pickle(args.smartphrases_pkl))
    print(f"  {len(texts)} templates with shell text")

    print("Computing note counts + readmission rates...")
    stats = compute_stats(args.readmission_pkl, args.template_ids_pkl, args.filtered_notes)
    print(f"  {len(stats)} templates with outcome data")

    # Full sampling frame: every template meeting the note-count floor. This is
    # the frame the generalization set is randomly sampled from, so it keeps the
    # in-use templates out but makes no department judgement of its own.
    if args.frame_out or args.dump_shells:
        if args.dump_shells:
            os.makedirs(args.dump_shells, exist_ok=True)
        frame = []
        for tid, txt in texts.items():
            if tid in IN_USE_IDS or tid not in stats.index:
                continue
            n = int(stats.loc[tid, "n_notes"])
            if n < args.min_notes:
                continue
            dept, marker = detect_department(txt)
            frame.append({
                "template_id": tid,
                "n_notes": n,
                "readmission_rate": round(float(stats.loc[tid, "readmission_rate"]), 4),
                "keyword_dept": dept or "",
                "matched_marker": marker or "",
                "shell_chars": len(txt),
                "prior_im_generalization": tid in IM_GENERALIZATION_IDS,
            })
            if args.dump_shells:
                with open(os.path.join(args.dump_shells, f"template_{tid}.txt"), "w") as f:
                    f.write(txt)
        frame_df = pd.DataFrame(frame).sort_values("readmission_rate").reset_index(drop=True)
        if args.frame_out:
            frame_df.to_csv(args.frame_out, index=False)
            print(f"  sampling frame: {len(frame_df)} templates -> {args.frame_out}")
        if args.dump_shells:
            print(f"  shells written to {args.dump_shells}/")
        print(f"  frame readmission range: {frame_df.readmission_rate.min():.1%}"
              f"-{frame_df.readmission_rate.max():.1%}")

    target_depts = {"Heme/Onc", "Cardiac Surgery", "Orthopedic Surgery", "Cardiology (medical)"}
    rows = []
    for tid, txt in texts.items():
        if tid in EXCLUDE_IDS:
            continue
        if tid not in stats.index:
            continue
        n = int(stats.loc[tid, "n_notes"])
        if n < args.min_notes:
            continue
        dept, marker = detect_department(txt)
        if dept not in target_depts:
            continue
        rows.append({
            "template_id": tid,
            "department": dept,
            "n_notes": n,
            "readmission_rate": round(float(stats.loc[tid, "readmission_rate"]), 4),
            "matched_marker": marker,
        })

    out = pd.DataFrame(rows).sort_values(
        ["department", "readmission_rate"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print(f"\nSaved {len(out)} candidate templates to {args.out}\n")
    print("=" * 64)
    for dept in ["Heme/Onc", "Cardiac Surgery", "Orthopedic Surgery", "Cardiology (medical)"]:
        sub = out[out["department"] == dept]
        if not len(sub):
            print(f"{dept}: 0 candidates")
            continue
        print(f"\n{dept}: {len(sub)} candidates "
              f"(readmission {sub.readmission_rate.min():.1%}-{sub.readmission_rate.max():.1%}, "
              f"n_notes {sub.n_notes.min()}-{sub.n_notes.max()})")
        print(sub[["template_id", "n_notes", "readmission_rate", "matched_marker"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
