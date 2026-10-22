"""
Utilities for preprocessing notes using confounder span annotations.

Spans identify templated and copyforward (boilerplate) text within notes.
Each span is {start, end, type} where note[start:end] is the boilerplate text.
"""

import pandas as pd


def load_spans(confounder_df: pd.DataFrame, note_csn_id: int | str) -> list[dict]:
    """Return sorted, merged spans for a given note."""
    rows = confounder_df[confounder_df["note_csn_id"] == int(note_csn_id)]
    spans = rows[["start", "end", "type"]].to_dict("records")
    return _merge_overlapping(sorted(spans, key=lambda s: s["start"]))


def _merge_overlapping(spans: list[dict]) -> list[dict]:
    """Merge overlapping or adjacent spans (ignoring type for merging purposes)."""
    if not spans:
        return []
    merged = [spans[0].copy()]
    for s in spans[1:]:
        if s["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s.copy())
    return merged


def annotate_note(note: str, spans: list[dict]) -> str:
    """
    Wrap boilerplate spans with type tags so the LLM knows what is templated.
    Processes right-to-left to preserve index validity.

    Example: s[10:30] becomes [TEMPLATE]s[10:30][/TEMPLATE]
    """
    result = note
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        start, end = span["start"], span["end"]
        tag = span.get("type", "boilerplate").upper()
        result = result[:start] + f"[{tag}]" + result[start:end] + f"[/{tag}]" + result[end:]
    return result


def extract_template(note: str, spans: list[dict]) -> str:
    """
    Extract template-typed spans to reconstruct the source template structure.
    Used to pass as source_template to the reformatter.
    """
    template_spans = [s for s in spans if s.get("type") == "template"]
    return "\n".join(note[s["start"]:s["end"]] for s in sorted(template_spans, key=lambda s: s["start"]))


def strip_note(note: str, spans: list[dict]) -> str:
    """
    Remove TEMPLATE spans from the note, leaving copy-forward and clinical text intact.
    Processes right-to-left to preserve index validity.
    Collapses resulting extra whitespace.
    """
    import re
    template_spans = [s for s in spans if s.get("type") == "template"]
    result = note
    for span in sorted(template_spans, key=lambda s: s["start"], reverse=True):
        result = result[:span["start"]] + " " + result[span["end"]:]
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def prepare_input(note: str, spans: list[dict], condition: str) -> tuple[str, str | None]:
    """
    Single source of truth for preparing reformatter input per condition.
    Returns (note_input, source_template_input).

    Used by both pipeline.process_one (for reformatting) and run_validation
    (to derive the comparison reference) to guarantee they operate on identical text.
    """
    if condition == "baseline":
        return note, None
    if condition == "annotated":
        return annotate_note(note, spans), None
    if condition == "stripped":
        return strip_note(note, spans), None
    if condition == "template_from_spans":
        return note, extract_template(note, spans)
    raise ValueError(f"Unknown condition: {condition}")
