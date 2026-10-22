import json
import logging
from typing import Optional

from llm_client import LLMClient
from preprocess import preprocess

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert medical documentation specialist. "
    "You reformat EHR discharge notes into a standardized JSON schema. "
    "Respond only with valid JSON matching the schema exactly.\n\n"
    "CRITICAL RULES:\n"
    "1. Never infer, assume, or derive information not explicitly stated in the note. "
    "If a field's value is not directly written in the note, use null.\n"
    "2. When uncertain, prefer null over best guesses. It is better to omit data than to fabricate it.\n"
    "3. Only include top-level sections that have content from the note. "
    "Omit entire sections if nothing in the note maps to them.\n"
    "4. Within included sections, use null for sub-fields that are not documented. "
    "Never write 'Not provided', 'Not documented', or 'NA' as a string — use JSON null.\n"
    "5. Preserve clinical detail verbatim: dates, lab values, provider names, medication doses.\n"
    "6. For conditional or qualified statements (e.g. 'if taking excessive X', 'probably', 'consider'), "
    "copy the exact language. Do not paraphrase, broaden, or simplify.\n"
    "7. Specialty-specific information without a dedicated field goes in "
    "additional_information.clinical_details as labeled key-value pairs.\n"
    "8. COMPLETENESS: every clinical fact stated in the note must appear somewhere in the output. "
    "Before finishing, check that no diagnosis, medication, lab value, imaging finding, procedure, "
    "consult, or instruction has been dropped. If a genuine clinical fact has no dedicated field, "
    "put it in additional_information.clinical_details rather than omitting it. Do not drop content "
    "to be concise — only template boilerplate and copy-forward filler may be left out."
)


# Second-pass audit that re-inserts clinical facts the first pass dropped. Additive only:
# it must never remove, reword, or alter content already present, and never invent.
_REPAIR_SYSTEM = (
    "You are auditing a reformatted discharge note for COMPLETENESS. You are given the ORIGINAL "
    "note and a REFORMATTED JSON produced from it. Find clinical facts stated in the original that "
    "are missing from the JSON and add them.\n\n"
    "RULES:\n"
    "1. Work only from the original note. Never invent, infer, or fabricate.\n"
    "2. Add each missing fact to the most appropriate existing field. If none fits, add it to "
    "additional_information.clinical_details as a labeled key-value pair.\n"
    "3. Preserve every value, date, name, and dose exactly as written in the note.\n"
    "4. Do NOT remove, reword, reorder, or alter anything already present and correct. Only ADD.\n"
    "5. Do not add template boilerplate, section headers, or copy-forward filler — only genuine "
    "clinical content.\n"
    "6. Keep the exact same schema and section names. Return the complete corrected JSON object."
)


def _build_repair_prompt(note: str, current_json: dict) -> str:
    return (
        f"ORIGINAL NOTE:\n{note}\n\n"
        f"CURRENT REFORMATTED JSON:\n{json.dumps(current_json, indent=2)}\n\n"
        "Return the corrected, more complete JSON object. Add any dropped clinical facts; change "
        "nothing that is already there."
    )


def _repair_pass(
    note: str,
    current_json: dict,
    schema: dict,
    client: LLMClient,
    max_tokens: int = 12000,
) -> dict:
    """Additive completeness repair. Returns the repaired dict, or the input unchanged if the
    repair output is unparseable or structurally invalid (never regress)."""
    messages = [
        {"role": "system", "content": _REPAIR_SYSTEM},
        {"role": "user", "content": _build_repair_prompt(note, current_json)},
    ]
    try:
        raw = client.generate(messages, max_tokens=max_tokens)
        parsed = _parse_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Repair pass parse failure — keeping pre-repair JSON: {e}")
        return current_json
    except Exception as e:
        logger.warning(f"Repair pass error — keeping pre-repair JSON: {e}")
        return current_json

    if _structural_errors(parsed, schema):
        logger.warning("Repair pass produced structural errors — keeping pre-repair JSON.")
        return current_json
    return parsed


def _flatten_schema(schema: dict) -> dict:
    """Strip display_name/description/fields wrappers so the LLM sees only field names and
    descriptions. Static boilerplate fields (those with a 'static' key) are deterministic
    template text, not extraction targets, so they are omitted; sections left with no
    extractable fields are dropped entirely."""
    out = {}
    for section, body in schema.items():
        if section.startswith("_"):
            continue
        fields = body.get("fields", body)
        if isinstance(fields, dict):
            fields = {
                name: fdef for name, fdef in fields.items()
                if not (isinstance(fdef, dict) and "static" in fdef)
            }
            if not fields:
                continue
        out[section] = fields
    return out


def _build_prompt(note: str, schema: dict, source_template: Optional[str]) -> str:
    flat_schema = _flatten_schema(schema)
    template_section = ""
    if source_template:
        template_section = f"\nSOURCE TEMPLATE (structure of the input note):\n{source_template[:1000]}\n"

    return f"""Reformat the following discharge note into the target JSON schema.

TARGET SCHEMA:
{json.dumps(flat_schema, indent=2)}
{template_section}
DISCHARGE NOTE:
{note}

Return a single JSON object. Only include top-level sections that have content from the note.
Omit sections entirely if nothing in the note maps to them. Do not infer or fabricate."""


def _structural_errors(result: dict, schema: dict) -> list[str]:
    errors = []
    valid_keys = set(_flatten_schema(schema).keys())
    unexpected = set(result.keys()) - valid_keys
    if unexpected:
        errors.append(f"Unexpected top-level sections (not in schema): {sorted(unexpected)}")
    return errors


def reformat(
    note: str,
    schema: dict,
    client: LLMClient,
    source_template: Optional[str] = None,
    is_rtf: bool = False,
    max_tokens: int = 12000,
    max_retries: int = 2,
    repair: bool = False,
) -> dict:
    """
    Reformat a discharge note into the standardized JSON schema.

    Returns a dict with keys:
        success (bool), result (dict|None), errors (list[str]), attempts (int)
    """
    note = preprocess(note, is_rtf)
    prompt = _build_prompt(note, schema, source_template)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_raw = None
    last_errors = []

    for attempt in range(1 + max_retries):
        try:
            if attempt == 0:
                current_messages = messages
            else:
                feedback = _build_feedback_prompt(last_raw, last_errors)
                current_messages = messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content": feedback},
                ]
                logger.info(f"Retry {attempt}/{max_retries} with error feedback: {last_errors}")

            raw = client.generate(current_messages, max_tokens=max_tokens)
            last_raw = raw

            parsed = _parse_json(raw)
            structural = _structural_errors(parsed, schema)

            if not structural:
                if repair:
                    parsed = _repair_pass(note, parsed, schema, client, max_tokens=max_tokens)
                return {"success": True, "result": parsed, "errors": [],
                        "attempts": attempt + 1, "repaired": repair}

            last_errors = structural
            logger.warning(f"Attempt {attempt+1} structural errors: {structural}")

        except (json.JSONDecodeError, ValueError) as e:
            last_errors = [f"JSON parse error: {e}"]
            logger.warning(f"Attempt {attempt+1} parse failure: {e}")
        except Exception as e:
            msg = str(e).lower()
            if "context_length" in msg or "maximum context" in msg or "context length" in msg:
                logger.error(f"Note exceeds context window — skipping retries.")
                return {"success": False, "result": None, "errors": ["context_length_exceeded"], "attempts": attempt + 1}
            last_errors = [f"Unexpected error: {e}"]
            logger.error(f"Attempt {attempt+1} unexpected error: {e}")
            break

    return {"success": False, "result": None, "errors": last_errors, "attempts": max_retries + 1}


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > 0:
            return json.loads(text[start:end])
        raise ValueError(f"No JSON object found in response")


def _build_feedback_prompt(previous_output: str, errors: list[str]) -> str:
    error_str = "\n".join(f"- {e}" for e in errors)
    return f"""Your previous response had the following issues:
{error_str}

Your previous response:
{previous_output}

Please fix only these issues and return the corrected JSON. Do not change any clinical content."""
