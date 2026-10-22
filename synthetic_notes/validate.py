"""
Bidirectional VeriFact-inspired validation for reformatted notes.

Forward pass  (faithfulness):  claims in reformatted note → supported by original?
Backward pass (completeness):  claims in original note    → present in reformatted?

Adapted from verifact2.0.ipynb.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from llm_client import LLMClient

logger = logging.getLogger(__name__)

# Within a single note, claim verification is ~300 independent API calls. Run them
# concurrently (order-preserved) instead of serially. Bounded so total in-flight load
# stays sane when combined with note-level workers. Tune via CLAIM_WORKERS env var.
_CLAIM_WORKERS = max(1, int(os.environ.get("CLAIM_WORKERS", "1")))


def _verify_claims(claims: list["Claim"], reference: str, client: LLMClient) -> list["ClaimResult"]:
    """Verify claims against a reference. Parallel when CLAIM_WORKERS>1; order preserved."""
    if _CLAIM_WORKERS <= 1 or len(claims) <= 1:
        return [_verify_claim(c, reference, client) for c in claims]
    with ThreadPoolExecutor(max_workers=_CLAIM_WORKERS) as ex:
        return list(ex.map(lambda c: _verify_claim(c, reference, client), claims))


class Verdict(Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    NOT_ADDRESSED = "Not Addressed"


@dataclass
class Claim:
    text: str
    index: int
    region: Optional[str] = None  # 'content' | 'template' | 'copyforward' (set when validating with spans)


@dataclass
class ClaimResult:
    claim: Claim
    verdict: Verdict
    explanation: str


@dataclass
class ValidationReport:
    forward_results: list[ClaimResult]   # reformatted → original
    backward_results: list[ClaimResult]  # original → reformatted
    llm_summary: Optional[str] = None

    @property
    def faithfulness(self) -> float:
        """Fraction of reformatted claims supported by original (no hallucination)."""
        if not self.forward_results:
            return 0.0
        return sum(1 for r in self.forward_results if r.verdict == Verdict.SUPPORTED) / len(self.forward_results)

    @property
    def completeness(self) -> float:
        """Fraction of original claims present in reformatted (no omission)."""
        if not self.backward_results:
            return 0.0
        return sum(1 for r in self.backward_results if r.verdict == Verdict.SUPPORTED) / len(self.backward_results)

    @property
    def f1(self) -> float:
        f = self.faithfulness
        c = self.completeness
        if f + c == 0:
            return 0.0
        return 2 * f * c / (f + c)

    def completeness_by_region(self) -> dict:
        """Completeness split by the source region of each ORIGINAL claim. Only meaningful
        when validate() was called with spans (else everything is region=None).
        content_completeness = clinical info preserved (want HIGH);
        template/copyforward_completeness = boilerplate preserved (want LOW)."""
        regions: dict = {}
        for r in self.backward_results:
            reg = r.claim.region or "unlabeled"
            d = regions.setdefault(reg, {"supported": 0, "total": 0})
            d["total"] += 1
            if r.verdict == Verdict.SUPPORTED:
                d["supported"] += 1
        return {
            reg: {"n": d["total"], "supported": d["supported"],
                  "completeness": round(d["supported"] / d["total"], 3) if d["total"] else None}
            for reg, d in regions.items()
        }

    def _bucket_counts(self, results: list[ClaimResult]) -> dict:
        n = len(results)
        c_supported = sum(1 for r in results if r.verdict == Verdict.SUPPORTED)
        c_contradicted = sum(1 for r in results if r.verdict == Verdict.CONTRADICTED)
        c_not_addressed = sum(1 for r in results if r.verdict == Verdict.NOT_ADDRESSED)
        pct = lambda x: round(x / n, 3) if n else 0.0
        return {
            "n": n,
            "supported": c_supported,
            "contradicted": c_contradicted,
            "not_addressed": c_not_addressed,
            "supported_pct": pct(c_supported),
            "contradicted_pct": pct(c_contradicted),
            "not_addressed_pct": pct(c_not_addressed),
        }

    def summary(self) -> dict:
        contradicted_forward = [
            {"claim": r.claim.text, "explanation": r.explanation}
            for r in self.forward_results
            if r.verdict == Verdict.CONTRADICTED
        ]
        not_addressed_forward = [
            {"claim": r.claim.text, "explanation": r.explanation}
            for r in self.forward_results
            if r.verdict == Verdict.NOT_ADDRESSED
        ]
        contradicted_backward = [
            {"claim": r.claim.text, "explanation": r.explanation}
            for r in self.backward_results
            if r.verdict == Verdict.CONTRADICTED
        ]
        not_addressed_backward = [
            {"claim": r.claim.text, "explanation": r.explanation}
            for r in self.backward_results
            if r.verdict == Verdict.NOT_ADDRESSED
        ]
        cbr = self.completeness_by_region()
        return {
            "faithfulness": round(self.faithfulness, 3),
            "completeness": round(self.completeness, 3),
            "f1": round(self.f1, 3),
            # region-split completeness (populated only when validated with spans):
            # content = clinical info preserved (HIGH good); template/copyforward = boilerplate kept (LOW good)
            "content_completeness": cbr.get("content", {}).get("completeness"),
            "content_n": cbr.get("content", {}).get("n"),
            "template_completeness": cbr.get("template", {}).get("completeness"),
            "template_n": cbr.get("template", {}).get("n"),
            "copyforward_completeness": cbr.get("copyforward", {}).get("completeness"),
            "copyforward_n": cbr.get("copyforward", {}).get("n"),
            "completeness_by_region": cbr,
            # remaining verdicts that came from caught network errors after re-verify
            # (>0 means this row may be slightly tainted — filter it from reported numbers)
            "forward_error_verdicts": sum(1 for r in self.forward_results if str(r.explanation).strip().lower().startswith("error")),
            "backward_error_verdicts": sum(1 for r in self.backward_results if str(r.explanation).strip().lower().startswith("error")),
            "forward": self._bucket_counts(self.forward_results),
            "backward": self._bucket_counts(self.backward_results),
            "forward_contradicted_claims": contradicted_forward,
            "forward_not_addressed_claims": not_addressed_forward,
            "backward_contradicted_claims": contradicted_backward,
            "backward_not_addressed_claims": not_addressed_backward,
            # legacy aliases (kept for backward compatibility)
            "forward_n_claims": len(self.forward_results),
            "backward_n_claims": len(self.backward_results),
            "hallucinated_claims": contradicted_forward,
            "omitted_claims": contradicted_backward + not_addressed_backward,
            "llm_summary": self.llm_summary,
        }


_DECOMPOSE_SYSTEM = (
    "You are a physician extracting atomic claims from clinical text. "
    "Respond only with valid JSON."
)

_DECOMPOSE_PROMPT = """Extract atomic claims from the clinical note below.

An atomic claim makes a single assertion with a subject, predicate, and object.
Claims must be self-contained — avoid pronouns; spell out any references.
Be comprehensive: capture all clinical facts, dates, values, and decisions.
Do not add prior knowledge.

Return JSON: {{"claims": ["claim 1", "claim 2", ...]}}

Note:
{text}"""

_VERIFY_SYSTEM = """You are verifying whether a clinical claim is supported by a reference document.

Reference document:
{reference}

For each claim, return one of:
- "Supported": the claim is fully supported by the reference
- "Contradicted": the claim is addressed but not supported (includes partial support)
- "Not Addressed": the reference does not address this claim at all

Respond with JSON: {{"verdict": "Supported"|"Contradicted"|"Not Addressed", "reason": "..."}}"""

_VERIFY_PROMPT = """Claim to verify:
{claim}

Your verdict:"""

_SUMMARIZE_SYSTEM = "You are a physician reviewing clinical note quality. Be concise and clinically precise."

_SUMMARIZE_PROMPT = """A clinical note was reformatted from one template to a standardized format.
Validation distinguishes two failure modes per direction. Contradicted claims are clinically worse than Not Addressed claims (one is wrong, the other is just missing/extra).

FORWARD — claims in REFORMATTED note that are not supported by original:
  CONTRADICTED (reformatted asserts something the original disagrees with — fabrication / wrong info):
  {forward_contradicted}

  NOT ADDRESSED (reformatted asserts something the original does not mention at all — added detail):
  {forward_not_addressed}

BACKWARD — claims in ORIGINAL note that are not present in reformatted:
  CONTRADICTED (reformatted says something different than original — got the fact wrong):
  {backward_contradicted}

  NOT ADDRESSED (original info that was simply omitted from reformatted):
  {backward_not_addressed}

Write 3–4 sentences summarizing: (1) what was actively fabricated or contradicted (worst), (2) what extra detail was added that wasn't in the source, (3) what original info was contradicted (got wrong), and (4) what was omitted. If a category is empty, say so briefly. Be specific about clinical content."""


def _summarize_issues(
    forward_contradicted: list[dict],
    forward_not_addressed: list[dict],
    backward_contradicted: list[dict],
    backward_not_addressed: list[dict],
    client: LLMClient,
) -> str:
    def fmt(items): return "\n".join(f"  - {x['claim']}" for x in items) if items else "  None"
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": _SUMMARIZE_PROMPT.format(
            forward_contradicted=fmt(forward_contradicted),
            forward_not_addressed=fmt(forward_not_addressed),
            backward_contradicted=fmt(backward_contradicted),
            backward_not_addressed=fmt(backward_not_addressed),
        )},
    ]
    return client.generate(messages, max_tokens=512, temperature=0.0)


def _decompose(text: str, client: LLMClient) -> list[Claim]:
    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM},
        {"role": "user", "content": _DECOMPOSE_PROMPT.format(text=text)},
    ]
    result = client.generate_json(messages, max_tokens=4000, temperature=0.0)

    if isinstance(result, list):
        claims_list = result
    elif "claims" in result:
        claims_list = result["claims"]
    elif "propositions" in result:
        claims_list = result["propositions"]
    else:
        raise ValueError(f"Unexpected decompose response format: {result}")

    return [Claim(text=c, index=i) for i, c in enumerate(claims_list)]


def _verify_claim(claim: Claim, reference: str, client: LLMClient) -> ClaimResult:
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM.format(reference=reference)},
        {"role": "user", "content": _VERIFY_PROMPT.format(claim=claim.text)},
    ]
    try:
        result = client.generate_json(messages, max_tokens=512, temperature=0.0)
        verdict_str = result["verdict"]
        verdict = Verdict[verdict_str.replace(" ", "_").upper()]
        explanation = result.get("reason") or result.get("explanation", "")
        return ClaimResult(claim=claim, verdict=verdict, explanation=explanation)
    except Exception as e:
        logger.warning(f"Verification failed for claim '{claim.text[:60]}': {e}")
        return ClaimResult(claim=claim, verdict=Verdict.NOT_ADDRESSED, explanation=f"Error: {e}")


def _is_error_verdict(r: ClaimResult) -> bool:
    """A verdict that came from a caught exception (transient network/API failure),
    not a genuine 'Not Addressed'."""
    return r.explanation.strip().lower().startswith("error")


def _reverify_errors(results: list[ClaimResult], reference: str, client: LLMClient) -> int:
    """Safeguard: re-verify any claim whose verdict was a network error, so a transient
    blip isn't silently counted as an omission/contradiction. Returns # successfully fixed."""
    fixed = 0
    for i, r in enumerate(results):
        if _is_error_verdict(r):
            retry = _verify_claim(r.claim, reference, client)
            if not _is_error_verdict(retry):
                results[i] = retry
                fixed += 1
    return fixed


def _region_texts(note: str, spans: list[dict]) -> dict:
    """Partition a note into clinical-content / template / copyforward text using spans.
    content = note with ALL annotated spans removed; the source region becomes the claim's label."""
    tmpl = " ".join(note[s["start"]:s["end"]] for s in spans if s.get("type") == "template")
    cf = " ".join(note[s["start"]:s["end"]] for s in spans if s.get("type") == "copyforward")
    content = note
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        content = content[:s["start"]] + " " + content[s["end"]:]
    return {"content": content, "template": tmpl, "copyforward": cf}


def _decompose_backward(original_note: str, spans: Optional[list[dict]], client: LLMClient) -> list[Claim]:
    """Backward claims. With spans, decompose each region separately and tag the source region
    (so completeness can be split into content vs boilerplate); without spans, decompose the whole note."""
    if not spans:
        return _decompose(original_note, client)
    claims: list[Claim] = []
    for region, text in _region_texts(original_note, spans).items():
        if not text.strip():
            continue
        for c in _decompose(text, client):
            c.region = region
            claims.append(c)
    for i, c in enumerate(claims):
        c.index = i
    return claims


def validate(
    original_note: str,
    reformatted_note: str,
    client: LLMClient,
    spans: Optional[list[dict]] = None,
) -> ValidationReport:
    """
    Run bidirectional VeriFact validation between original and reformatted notes.

    Args:
        original_note:    The source note (ground truth).
        reformatted_note: The reformatted note to evaluate. Pass as a string —
                          if your reformatter returns a dict, use json.dumps(result).
        client:           LLMClient instance.
        spans:            Optional confounder spans. When given, the backward/completeness
                          pass is split by source region so the report carries
                          content_completeness (clinical info) vs template/copyforward
                          completeness (boilerplate) separately.

    Returns:
        ValidationReport with faithfulness, completeness, and per-claim verdicts.
    """
    if isinstance(reformatted_note, dict):
        reformatted_note = json.dumps(reformatted_note, indent=2)

    logger.info("Decomposing reformatted note into claims (forward pass)...")
    forward_claims = _decompose(reformatted_note, client)
    logger.info(f"  {len(forward_claims)} claims extracted")

    logger.info("Decomposing original note into claims (backward pass)...")
    backward_claims = _decompose_backward(original_note, spans, client)
    if spans:
        by_region = {}
        for c in backward_claims:
            by_region[c.region] = by_region.get(c.region, 0) + 1
        logger.info(f"  {len(backward_claims)} claims extracted (by region: {by_region})")
    else:
        logger.info(f"  {len(backward_claims)} claims extracted")

    logger.info("Verifying forward claims (faithfulness: reformatted → original)...")
    forward_results = _verify_claims(forward_claims, original_note, client)

    logger.info("Verifying backward claims (completeness: original → reformatted)...")
    backward_results = _verify_claims(backward_claims, reformatted_note, client)

    # Safeguard: re-verify any claim whose verdict was a transient network error so a
    # connection blip can't masquerade as a real omission and distort reported metrics.
    n_fixed = _reverify_errors(forward_results, original_note, client)
    n_fixed += _reverify_errors(backward_results, reformatted_note, client)
    if n_fixed:
        logger.info(f"Re-verified {n_fixed} claim(s) that had transient errors")

    report = ValidationReport(forward_results=forward_results, backward_results=backward_results)

    logger.info("Generating LLM summary of hallucinations and omissions...")
    s = report.summary()
    try:
        report.llm_summary = _summarize_issues(
            s["forward_contradicted_claims"],
            s["forward_not_addressed_claims"],
            s["backward_contradicted_claims"],
            s["backward_not_addressed_claims"],
            client,
        )
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        report.llm_summary = None

    return report
