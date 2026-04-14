from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader


class ExtractionError(Exception):
    pass


def extract_text_from_file(path: Path, content_type: str | None, ext: str) -> tuple[str, float]:
    """
    Returns (text, extraction_cost_usd_estimate).
    Simple flat-rate estimate for rules/LLM extraction.
    """
    if not path or str(path).strip() in {"", "."} or not path.is_file():
        raise ExtractionError(f"File to parse not found at '{path}'")

    ext = ext.lower()
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".pdf":
        text = _pdf_to_text(path)
    elif ext == ".docx":
        text = _docx_to_text(path)
    else:
        raise ExtractionError(f"Unsupported extension: {ext}")

    normalized = _normalize_extracted_text(text).strip()
    if not normalized:
        raise ExtractionError("No text could be extracted (empty or scanned PDF without OCR)")
    return normalized, 0.001


def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _docx_to_text(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _normalize_extracted_text(text: str) -> str:
    """
    Repair common PDF extraction artifacts where letters/digits are spaced:
    - "A F N A N" -> "AFNAN"
    - "2 0 2 5" -> "2025"
    """
    if not text:
        return text

    def _collapse_spaced_letters(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    # Collapse repeated single-letter/digit tokens only when they use single spaces.
    # This avoids merging separate words that are split by 2+ spaces.
    text = re.sub(r"\b(?:[A-Za-z](?: [A-Za-z]){1,})\b", _collapse_spaced_letters, text)
    text = re.sub(r"\b(?:\d(?: \d){2,})\b", _collapse_spaced_letters, text)

    # Keep line structure, but normalize excessive inline whitespace.
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(lines)

