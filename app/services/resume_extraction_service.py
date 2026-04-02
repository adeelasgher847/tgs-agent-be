from pathlib import Path

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

    normalized = text.strip()
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

