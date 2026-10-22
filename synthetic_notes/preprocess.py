import re
import logging

logger = logging.getLogger(__name__)


def strip_rtf(rtf: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(rtf)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception as e:
        logger.warning(f"RTF stripping failed, using raw text: {e}")
        return rtf


def preprocess(note: str, is_rtf: bool = False) -> str:
    if is_rtf:
        note = strip_rtf(note)
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub(r"\n{3,}", "\n\n", note)
    return note.strip()
