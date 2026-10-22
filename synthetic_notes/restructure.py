"""
Deterministic reformat-once -> restructure-into-A/B for the 2-template bias pipeline.

Reformat each note ONCE into the standardized schema (standardized_template.json),
then re-key that single JSON into Template A and Template B structure
with NO second LLM pass. Because both templates pull from the SAME standardized
fields, the two arms carry byte-identical clinical content and differ ONLY in
structure/naming — the clean template confounder with zero content leakage.

Self-test (no PHI):        python restructure.py --selftest
Batch (user runs, in-proc): python restructure.py --input <standardized_reformat.csv> --output <pairs.csv>
  input CSV needs columns: note_csn_id, reformatted_note (JSON string of standardized output)
  output CSV columns:      note_csn_id, reformatted_A, reformatted_B  (JSON strings)
"""
import argparse
import json
import re
import sys
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"

# Each template field -> the standardized_template.json "section.field" it draws from.
# A and B intentionally use the SAME standardized sources (renamed/regrouped), which
# guarantees identical content across arms. Structure is taken from template_{A,B}.json.
A_MAP = {
    "hospitalization_details": {
        "attending_physician": "hospital_identification.attending_physician",
        "discharging_service": "hospital_identification.primary_team",
        "resident_fellow": "hospital_identification.resident_fellow",
        "facility": "hospital_identification.facility",
        "admission_date": "patient_demographics.admission_date",
        "discharge_date": "patient_demographics.discharge_date",
    },
    "identifier": {
        "patient_name": "patient_demographics.name",
        "mrn": "patient_demographics.mrn",
        "date_of_birth": "patient_demographics.dob",
        "age": "patient_demographics.age",
        "sex": "patient_demographics.sex",
        "length_of_stay": "patient_demographics.length_of_stay",
    },
    "reason_for_hospitalization": {
        "chief_complaint": "admission_information.chief_complaint",
        "admitting_diagnosis": "admission_information.admitting_diagnosis",
        "history_of_present_illness": "admission_information.history_of_present_illness",
    },
    "past_medical_history": {
        "medical_conditions": "past_medical_history.medical_conditions",
        "surgical_history": "past_medical_history.surgical_history",
        "prior_medications": "past_medical_history.medications_prior",
        "social_history": "past_medical_history.social_history",
        "family_history": "past_medical_history.family_history",
    },
    "summary_of_hospital_course": {
        "summary": "hospital_course.summary",
        "response_to_treatment": "hospital_course.response_to_treatment",
    },
    "discharge_plan_by_issue": {
        "principal_diagnosis": "discharge_diagnosis.primary_diagnosis",
        "secondary_diagnoses": "discharge_diagnosis.secondary_diagnoses",
        "plan_by_issue": "hospital_course.key_events",
        "consultations": "consultations.consultants",
    },
    "discharge_exam": {
        "overall_condition": "discharge_condition.overall_condition",
        "vital_signs": "discharge_condition.vital_signs",
        "functional_status": "discharge_condition.functional_status",
        "mental_status": "discharge_condition.mental_status",
        "pain_level": "discharge_condition.pain_level",
    },
    "laboratory_studies": {
        "laboratory": "diagnostic_results.laboratory",
    },
    "imaging": {
        "imaging": "diagnostic_results.imaging",
        "pathology": "diagnostic_results.pathology",
        "other_studies": "diagnostic_results.other",
    },
    "procedures": {
        "procedures": "procedures_and_interventions.procedures",
    },
    "complications": {
        "complications": "hospital_course.complications",
    },
    "discharge_medications": {
        "medications": "discharge_medications.medications",
        "started": "discharge_medications.started",
        "stopped": "discharge_medications.stopped",
        "changed": "discharge_medications.changed",
        "continued": "discharge_medications.continued",
    },
    "pending_labs": {
        "pending": "pending_results.pending",
    },
    "discharge_orders": {
        "activity": "discharge_instructions.activity",
        "diet": "discharge_instructions.diet",
        "wound_care": "discharge_instructions.wound_care",
        "monitoring": "discharge_instructions.monitoring",
        "equipment": "discharge_instructions.equipment",
        "other": "discharge_instructions.other",
        "warning_signs": "patient_education.warning_signs",
        "education_provided": "patient_education.education_provided",
        "understanding": "patient_education.understanding",
    },
    "follow_up": {
        "appointments": "follow_up.appointments",
        "recommendations": "follow_up.recommendations",
    },
    "best_practice_information": {
        "code_status": "additional_information.code_status",
        "advance_directives": "additional_information.advance_directives",
        "disposition": "discharge_disposition.disposition",
        "support_services": "discharge_disposition.support_services",
        "transportation": "discharge_disposition.transportation",
        "barriers_to_care": "additional_information.barriers_to_care",
        "allergies": "past_medical_history.allergies",
    },
    "additional_information": {
        "clinical_details": "additional_information.clinical_details",
    },
}

B_MAP = {
    "patient": {
        "patients_name": "patient_demographics.name",
        "mrn": "patient_demographics.mrn",
        "dob": "patient_demographics.dob",
        "age": "patient_demographics.age",
        "sex": "patient_demographics.sex",
        "attending_at_discharge": "hospital_identification.attending_physician",
        "discharging_service": "hospital_identification.primary_team",
        "resident": "hospital_identification.resident_fellow",
        "facility": "hospital_identification.facility",
        "date_of_admission": "patient_demographics.admission_date",
        "date_of_discharge": "patient_demographics.discharge_date",
        "los": "patient_demographics.length_of_stay",
    },
    "reason_for_admission": {
        "chief_complaint": "admission_information.chief_complaint",
        "admitting_diagnosis": "admission_information.admitting_diagnosis",
        "brief_hpi": "admission_information.history_of_present_illness",
    },
    "discharge_diagnoses": {
        "principal": "discharge_diagnosis.primary_diagnosis",
        "secondary": "discharge_diagnosis.secondary_diagnoses",
    },
    "summary_of_hospital_course": {
        "narrative_summary": "hospital_course.summary",
        "course_by_problem": "hospital_course.key_events",
        "response": "hospital_course.response_to_treatment",
        "complications": "hospital_course.complications",
    },
    "past_medical_history": {
        "medical_conditions": "past_medical_history.medical_conditions",
        "surgical_history": "past_medical_history.surgical_history",
        "prior_medications": "past_medical_history.medications_prior",
        "allergies": "past_medical_history.allergies",
        "social_history": "past_medical_history.social_history",
        "family_history": "past_medical_history.family_history",
    },
    "disposition": {
        "disposition": "discharge_disposition.disposition",
        "support_services": "discharge_disposition.support_services",
        "transportation": "discharge_disposition.transportation",
        "code_status": "additional_information.code_status",
        "advance_directives": "additional_information.advance_directives",
        "barriers_to_care": "additional_information.barriers_to_care",
    },
    "follow_up_plan": {
        "appointments": "follow_up.appointments",
        "recommendations": "follow_up.recommendations",
    },
    "significant_findings_studies_procedures": {
        "labs": "diagnostic_results.laboratory",
        "imaging": "diagnostic_results.imaging",
        "pathology": "diagnostic_results.pathology",
        "other_studies": "diagnostic_results.other",
        "procedures": "procedures_and_interventions.procedures",
        "consultants": "consultations.consultants",
        "pending_labs": "pending_results.pending",
    },
    "medications_on_discharge": {
        "medications": "discharge_medications.medications",
        "started": "discharge_medications.started",
        "stopped": "discharge_medications.stopped",
        "changed": "discharge_medications.changed",
        "continued": "discharge_medications.continued",
    },
    "additional_discharge_instructions": {
        "activity": "discharge_instructions.activity",
        "diet": "discharge_instructions.diet",
        "wound_care": "discharge_instructions.wound_care",
        "monitoring": "discharge_instructions.monitoring",
        "equipment": "discharge_instructions.equipment",
        "other": "discharge_instructions.other",
        "warning_signs": "patient_education.warning_signs",
        "education_provided": "patient_education.education_provided",
        "understanding": "patient_education.understanding",
    },
    "condition_and_exam_at_discharge": {
        "vital_signs": "discharge_condition.vital_signs",
        "overall_condition": "discharge_condition.overall_condition",
        "functional_status": "discharge_condition.functional_status",
        "cognitive_status": "discharge_condition.mental_status",
        "pain_level": "discharge_condition.pain_level",
    },
    "additional_information": {
        "clinical_details": "additional_information.clinical_details",
    },
}

# ---------------------------------------------------------------------------
# "Exact" variants: sections mirror the real Epic shells one-to-one, and every
# standardized field with no native section is emitted into a trailing catch-all
# (see the catch_all section in template_{A,B}_exact.json). Native ∪ catch-all
# equals the full standardized field set in both arms => identical content.
# ---------------------------------------------------------------------------
# Exact maps hold ONLY the structured (labeled) fields the shell itemizes. Free-text
# '***' blanks are 'composed' fields whose source paths live in the template JSON
# ("compose": [...]); the catch-all absorbs everything else. See restructure().
A_EXACT_MAP = {
    "hospitalization_details": {
        "attending_physician": "hospital_identification.attending_physician",
        "discharging_service": "hospital_identification.primary_team",
        "admission_date": "patient_demographics.admission_date",
        "discharge_date": "patient_demographics.discharge_date",
    },
    "identifier": {
        "patient_id": "patient_demographics.mrn",
    },
    "laboratory_studies": {
        "laboratory": "diagnostic_results.laboratory",
    },
    "pending_labs": {
        "pending": "pending_results.pending",
    },
    "discharge_medications": {
        "medications": "discharge_medications.medications",
        "started": "discharge_medications.started",
        "stopped": "discharge_medications.stopped",
        "changed": "discharge_medications.changed",
        "continued": "discharge_medications.continued",
    },
    "follow_up": {
        "appointments": "follow_up.appointments",
        "recommendations": "follow_up.recommendations",
    },
    "best_practice_information": {
        "code_status": "additional_information.code_status",
        "advance_directives": "additional_information.advance_directives",
        "disposition": "discharge_disposition.disposition",
        "support_services": "discharge_disposition.support_services",
        "transportation": "discharge_disposition.transportation",
        "barriers_to_care": "additional_information.barriers_to_care",
        "allergies": "past_medical_history.allergies",
    },
    # composed free-text sections (reason, summary, plan, exam, imaging, procedures,
    # complications, orders) draw their paths from template_A_exact.json.
    # additional_clinical_information: catch-all (filled dynamically)
}

B_EXACT_MAP = {
    "patient": {
        "patients_name": "patient_demographics.name",
        "mrn": "patient_demographics.mrn",
        "attending_at_discharge": "hospital_identification.attending_physician",
        "date_of_admission": "patient_demographics.admission_date",
        "date_of_discharge": "patient_demographics.discharge_date",
    },
    "medications_on_discharge": {
        "medications": "discharge_medications.medications",
        "started": "discharge_medications.started",
        "stopped": "discharge_medications.stopped",
        "changed": "discharge_medications.changed",
        "continued": "discharge_medications.continued",
    },
    "condition_and_exam_at_discharge": {
        "vital_signs": "discharge_condition.vital_signs",
    },
    # composed free-text sections (reason, diagnoses, summary, dispo, follow_up,
    # findings, instructions, exam) draw their paths from template_B_exact.json.
    # additional_clinical_information: catch-all (filled dynamically)
}

MAPS = {"A": A_MAP, "B": B_MAP, "A_exact": A_EXACT_MAP, "B_exact": B_EXACT_MAP}


def _load_universe():
    """All note-derived standardized field paths ('section.field') — the full content
    set every template must cover (natively or via catch-all) to preserve parity."""
    std = json.loads((TEMPLATES / "standardized_template.json").read_text())
    return [f"{sec}.{fld}" for sec, body in std.items()
            if not sec.startswith("_") for fld in body.get("fields", {})]


UNIVERSE = _load_universe()


def _pretty(s: str) -> str:
    return s.replace("_", " ").title()


def _render_value(val) -> str:
    """Flatten any standardized value (str / list / dict of labeled pairs / list of
    objects) into readable text, for composing free-text '***' blanks."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        parts = [_render_value(v) for v in val]
        return "; ".join(p for p in parts if p)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            rv = _render_value(v)
            if not rv:
                continue
            # prettify snake_case keys (e.g. key_recommendations); leave already-readable
            # labels alone (e.g. "WBC (2024-05-05)", "CXR", "EKG")
            label = _pretty(k) if ("_" in k or k.islower()) else k
            parts.append(f"{label}: {rv}")
        return ", ".join(parts)
    return str(val)


def _compose(paths, standardized, used):
    """Stitch the given standardized paths into one free-text blank. Records the paths
    in `used` (so the catch-all won't repeat them) and returns joined text, or None."""
    parts = []
    for path in paths:
        std_sec, std_field = path.split(".")
        used.add(path)
        txt = _render_value((standardized.get(std_sec) or {}).get(std_field))
        if txt:
            parts.append(txt if txt.endswith((".", ";", "!", "?")) else txt + ".")
    return " ".join(parts) if parts else None


# Template A groups labs into named panels. Ordered by priority; the first alias that matches
# an analyte name (the text before any "(" date or "," ) assigns its panel. Unmatched
# analytes fall into "Other". This is a deterministic display regrouping of the SAME
# diagnostic_results.laboratory values used (flat) by template B — no content changes.
_LAB_PANELS = [
    ("CBC", {"wbc", "white blood cell", "hgb", "hemoglobin", "hct", "hematocrit", "plt",
             "platelet", "platelets", "rbc", "mcv", "mch", "mchc", "rdw", "neutrophil",
             "neutrophils", "lymphocyte", "lymphocytes", "anc", "band", "bands"}),
    ("Electrolytes", {"na", "sodium", "k", "potassium", "cl", "chloride", "co2", "bicarb",
                      "bicarbonate", "bun", "cr", "creatinine", "ca", "calcium", "mg",
                      "magnesium", "phos", "phosphorus", "phosphate", "anion gap", "egfr", "gfr"}),
    ("Glucose", {"glu", "glucose", "a1c", "hba1c", "hemoglobin a1c"}),
    ("Coagulation", {"pt", "ptt", "aptt", "inr", "fibrinogen", "d-dimer", "ddimer"}),
    ("LFTs", {"ast", "alt", "alkp", "alk phos", "alkaline phosphatase", "alp", "tbil",
              "total bilirubin", "bilirubin", "dbil", "direct bilirubin", "alb", "albumin",
              "ggt", "total protein", "protein"}),
    ("Cardiac", {"tni", "tnt", "troponin", "ckmb", "ck-mb", "ck", "bnp", "nt-probnp",
                 "probnp", "pro-bnp"}),
]


def _lab_analyte(key: str) -> str:
    """Normalized analyte name: text before a '(' date or ',' , lowercased and stripped."""
    head = key.split("(", 1)[0].split(",", 1)[0]
    return head.strip().lower()


def _panel_for(key: str) -> str:
    name = _lab_analyte(key)
    tokens = set(name.replace("-", " ").split())
    for panel, aliases in _LAB_PANELS:
        for a in aliases:
            if name == a or name.startswith(a + " ") or a in tokens:
                return panel
        # multi-word aliases (e.g. "total bilirubin") won't be single tokens
        if any(" " in a and a in name for a in aliases):
            return panel
    return "Other"


def _bucket_labs(labs):
    """Regroup a flat {analyte: value} lab object into Template A's panels (CBC, Electrolytes, ...).
    Returns a dict of non-empty panels in canonical order. Non-dict input (null, or a
    selftest sentinel string) is passed through under a single 'laboratory' key so the
    underlying value is preserved for content-parity checks."""
    if not isinstance(labs, dict):
        return {} if labs is None else {"laboratory": labs}
    order = [p for p, _ in _LAB_PANELS] + ["Other"]
    grouped = {p: {} for p in order}
    for k, v in labs.items():
        grouped[_panel_for(k)][k] = v
    return {p: grouped[p] for p in order if grouped[p]}


def _std_paths(mapp):
    return [p for sec in mapp.values() for p in sec.values()]


def assert_content_parity():
    """Guarantee A and B draw from the SAME set of standardized fields (no leakage)."""
    a, b = set(_std_paths(A_MAP)), set(_std_paths(B_MAP))
    if a != b:
        raise AssertionError(
            f"Content leakage risk — A-only: {sorted(a - b)}  B-only: {sorted(b - a)}"
        )
    # each standardized field used at most once per template (no accidental duplication)
    for name, mapp in MAPS.items():
        paths = _std_paths(mapp)
        dupes = {p for p in paths if paths.count(p) > 1}
        if dupes:
            raise AssertionError(f"Template {name} maps a standardized field more than once: {sorted(dupes)}")
    return a


def restructure(standardized: dict, template_id: str, template_schema: dict = None) -> dict:
    """Re-key a standardized reformatted JSON into template A or B structure.
    Missing standardized fields become null so every template note has a fixed skeleton
    (constant structure per arm = clean template signal; identical values across arms)."""
    if template_id not in MAPS:
        raise ValueError(f"template_id must be 'A' or 'B', got {template_id!r}")
    mapp = MAPS[template_id]
    schema = template_schema or json.loads((TEMPLATES / f"template_{template_id}.json").read_text())
    out = {}
    used = set()
    catch_all_sections = []
    for section, body in schema.items():
        if section.startswith("_"):
            continue
        if body.get("catch_all"):
            catch_all_sections.append(section)
            out[section] = {}
            continue
        fields = body.get("fields", {})
        sec_out = {}
        for field, fdef in fields.items():
            if isinstance(fdef, dict) and "static" in fdef:
                sec_out[field] = fdef["static"]  # fixed boilerplate; not drawn from the note
                continue
            if isinstance(fdef, dict) and "compose" in fdef:
                sec_out[field] = _compose(fdef["compose"], standardized, used)  # free-text blank
                continue
            std_path = mapp[section][field]  # KeyError => unmapped non-static field (build error, surfaces loudly)
            std_sec, std_field = std_path.split(".")
            sec_out[field] = (standardized.get(std_sec) or {}).get(std_field)
            used.add(std_path)
        out[section] = sec_out

    # Template A (A / A_exact) regroups its single laboratory object into named panels.
    if template_id in ("A", "A_exact") and "laboratory_studies" in out:
        out["laboratory_studies"] = _bucket_labs(out["laboratory_studies"].get("laboratory"))

    # Fill any catch-all section with every standardized field not natively placed,
    # so native ∪ catch-all == the full field set (identical content across arms).
    for section in catch_all_sections:
        leftovers = {}
        for path in UNIVERSE:
            if path in used:
                continue
            std_sec, std_field = path.split(".")
            val = (standardized.get(std_sec) or {}).get(std_field)
            if val is None or val == "" or val == [] or val == {}:
                continue
            label = _pretty(std_field)
            if label in leftovers:  # disambiguate rare label collisions
                label = f"{_pretty(std_sec)} — {label}"
            leftovers[label] = val
        out[section] = leftovers

    return out


def _selftest():
    """Prove content is identical across A and B on synthetic (non-PHI) input."""
    paths = assert_content_parity()
    print(f"content-parity assertion PASSED — A and B both draw from {len(paths)} standardized fields")

    # synthetic standardized JSON: each leaf value is a unique sentinel = its own path
    std = {}
    for p in paths:
        sec, fld = p.split(".")
        std.setdefault(sec, {})[fld] = f"<<{p}>>"

    a = restructure(std, "A")
    b = restructure(std, "B")

    def leaf_values(obj):
        # Only note-derived sentinels (<<...>>) participate in the parity check;
        # static boilerplate is intentionally template-specific and excluded.
        return sorted(
            v for sec in obj.values() for v in sec.values()
            if isinstance(v, str) and v.startswith("<<")
        )

    va, vb = leaf_values(a), leaf_values(b)
    assert va == vb, "content mismatch between A and B!"
    assert set(va) == {f"<<{p}>>" for p in paths}, "some standardized field was dropped"

    # structure differs (that's the confounder), content identical (that's no-leakage)
    print(f"template A: {len(a)} sections, {sum(len(s) for s in a.values())} fields")
    print(f"template B: {len(b)} sections, {sum(len(s) for s in b.values())} fields")
    print(f"identical content values across arms: {va == vb}  ({len(va)} values each)")
    print("SELFTEST PASSED — same content, different structure.")


def _exact_declared_paths(tid):
    """All standardized paths a native (non-catch-all) field consumes: structured fields
    from the map, composed fields from the template JSON's 'compose' list."""
    schema = json.loads((TEMPLATES / f"template_{tid}.json").read_text())
    mapp = MAPS[tid]
    paths = []
    for section, body in schema.items():
        if section.startswith("_") or body.get("catch_all"):
            continue
        for field, fdef in body.get("fields", {}).items():
            if "static" in fdef:
                continue
            if "compose" in fdef:
                paths.extend(fdef["compose"])
            else:
                paths.append(mapp[section][field])
    return paths


def assert_content_parity_exact():
    """Guarantee each exact arm consumes each standardized field at most once natively,
    so native ∪ catch-all == the FULL field set and content is identical across arms."""
    universe = set(UNIVERSE)
    for tid in ("A_exact", "B_exact"):
        paths = _exact_declared_paths(tid)
        dupes = {p for p in paths if paths.count(p) > 1}
        if dupes:
            raise AssertionError(f"{tid} consumes a standardized field more than once: {sorted(dupes)}")
        unknown = set(paths) - universe
        if unknown:
            raise AssertionError(f"{tid} references unknown standardized paths: {sorted(unknown)}")
    return universe


_SENTINEL = re.compile(r"<<[^>]+>>")


def _find_sentinels(obj):
    """Recursively collect every <<section.field>> sentinel across strings, lists, dicts
    (composed blanks embed several sentinels in one string)."""
    found = []
    if isinstance(obj, str):
        found.extend(_SENTINEL.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_find_sentinels(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_sentinels(v))
    return found


def _selftest_exact():
    """Prove content is identical across A_exact and B_exact (native + catch-all)."""
    universe = assert_content_parity_exact()
    std = {}
    for p in universe:
        sec, fld = p.split(".")
        std.setdefault(sec, {})[fld] = f"<<{p}>>"

    a = restructure(std, "A_exact")
    b = restructure(std, "B_exact")

    want = {f"<<{p}>>" for p in universe}
    va, vb = set(_find_sentinels(a)), set(_find_sentinels(b))
    assert va == want, f"A_exact missing: {sorted(want - va)}  extra: {sorted(va - want)}"
    assert vb == want, f"B_exact missing: {sorted(want - vb)}  extra: {sorted(vb - want)}"

    ca = len(a["additional_clinical_information"])
    cb = len(b["additional_clinical_information"])
    print(f"EXACT content-parity PASSED — A_exact and B_exact each carry all {len(universe)} standardized fields")
    print(f"template A_exact: {len(a)} sections; catch-all holds {ca} fields")
    print(f"template B_exact: {len(b)} sections; catch-all holds {cb} fields")
    print(f"identical content across arms: {va == vb}")
    print("EXACT SELFTEST PASSED — same content, PDF-faithful structure (one blank per section), catch-all for the rest.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="Run non-PHI correctness check (styled A/B) and exit")
    ap.add_argument("--selftest-exact", action="store_true", help="Run non-PHI correctness check (exact A/B + catch-all) and exit")
    ap.add_argument("--exact", action="store_true", help="Use the PDF-exact templates (template_{A,B}_exact.json)")
    ap.add_argument("--input", help="CSV with note_csn_id + reformatted_note (standardized JSON string)")
    ap.add_argument("--output", help="Output CSV: note_csn_id, reformatted_A, reformatted_B")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.selftest_exact:
        _selftest_exact()
        return

    if not (args.input and args.output):
        ap.error("provide --input and --output, or --selftest / --selftest-exact")

    import csv
    csv.field_size_limit(sys.maxsize)
    id_a, id_b = ("A_exact", "B_exact") if args.exact else ("A", "B")
    if args.exact:
        assert_content_parity_exact()
    else:
        assert_content_parity()
    schema_a = json.loads((TEMPLATES / f"template_{id_a}.json").read_text())
    schema_b = json.loads((TEMPLATES / f"template_{id_b}.json").read_text())

    rows_in = list(csv.DictReader(open(args.input)))
    n_ok = 0
    with open(args.output, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["note_csn_id", "reformatted_A", "reformatted_B"])
        w.writeheader()
        for r in rows_in:
            raw = r.get("reformatted_note")
            if not raw:
                continue
            try:
                std = json.loads(raw)
            except Exception:
                continue
            w.writerow({
                "note_csn_id": r.get("note_csn_id"),
                "reformatted_A": json.dumps(restructure(std, id_a, schema_a)),
                "reformatted_B": json.dumps(restructure(std, id_b, schema_b)),
            })
            n_ok += 1
    print(f"wrote {n_ok} paired A/B notes -> {args.output}  (templates: {id_a} / {id_b})")


if __name__ == "__main__":
    main()
