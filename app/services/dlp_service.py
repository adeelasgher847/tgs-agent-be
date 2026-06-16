"""
Google Cloud DLP service — PHI redaction for HIPAA-flagged call flows.

Only called when a call flow has hipaa_compliance=True.
DLP is priced per character; never apply globally.

Pipeline per message:
  1. Local regex pass (_HIPAA_PATTERNS) — free, catches healthcare codes DLP misses
  2. Cloud DLP inspect_content — catches person names, phone numbers, emails, DOB, MRNs

Supported DLP infoTypes (per ticket spec):
  PHONE_NUMBER, EMAIL_ADDRESS, PERSON_NAME, DATE_OF_BIRTH, MEDICAL_RECORD_NUMBER
"""

from __future__ import annotations

import re

from app.core.config import settings
from app.core.logger import logger

_DLP_INFO_TYPES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "PERSON_NAME",
    "DATE_OF_BIRTH",
    "MEDICAL_RECORD_NUMBER",
]

_REDACTION_TOKEN = "[REDACTED]"

# ---------------------------------------------------------------------------
# Local regex patterns for healthcare-specific identifiers that Cloud DLP
# infoTypes either miss or classify with low confidence.
# Each tuple: (name, compiled_pattern, replacement_label)
# Applied in order; later patterns operate on already-redacted text.
# ---------------------------------------------------------------------------
_HIPAA_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Social Security Number  — 123-45-6789 or 123 45 6789
    (
        "ssn",
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        "[REDACTED-SSN]",
    ),
    # Medical Record Number with contextual prefix
    (
        "mrn",
        re.compile(
            r"\b(?:MRN|Medical\s+Record(?:\s+Number)?|Patient\s+(?:ID|Number))"
            r"[:\s#]*\d{4,12}\b",
            re.IGNORECASE,
        ),
        "[REDACTED-MRN]",
    ),
    # ICD-10 code following a clinical context word  (e.g. "diagnosis: J45.20")
    (
        "icd10_with_context",
        re.compile(
            r"\b(?:diagnosis|icd(?:[-\s]?10)?(?:\s*code)?|dx)[:\s]+"
            r"[A-Z]\d{2}\.?\d{0,4}[A-Z0-9]?\b",
            re.IGNORECASE,
        ),
        "[REDACTED-DIAGNOSIS]",
    ),
    # Standalone ICD-10 code with decimal (e.g. J45.20, E11.65)
    # Requires decimal to avoid matching generic alphanumeric codes.
    (
        "icd10_standalone",
        re.compile(r"\b[A-Z]\d{2}\.\d{1,4}[A-Z0-9]?\b"),
        "[REDACTED-ICD10]",
    ),
    # CPT procedure code with contextual prefix  (e.g. "CPT: 99213")
    (
        "cpt_code",
        re.compile(
            r"\b(?:CPT|procedure\s+code)[:\s]*\d{5}\b",
            re.IGNORECASE,
        ),
        "[REDACTED-CPT]",
    ),
    # National Provider Identifier  (e.g. "NPI: 1234567890")
    (
        "npi",
        re.compile(r"\bNPI[:\s]*\d{10}\b", re.IGNORECASE),
        "[REDACTED-NPI]",
    ),
    # Insurance member / policy ID with contextual prefix
    (
        "insurance_id",
        re.compile(
            r"\b(?:member\s+(?:id|number)|insurance\s+(?:id|number)|"
            r"policy\s+(?:number|#|no))[:\s#]*[A-Z0-9]{6,20}\b",
            re.IGNORECASE,
        ),
        "[REDACTED-INSURANCE-ID]",
    ),
    # Medication with dosage following a prescription verb
    # e.g. "prescribed metformin 500mg", "taking lisinopril 10 mg"
    (
        "medication_dosage",
        re.compile(
            r"\b(?:prescribed|taking|dose(?:age)?|medication|rx)[:\s]+"
            r"[A-Za-z][A-Za-z0-9\-\s]{1,40}"
            r"\d+(?:\.\d+)?\s*(?:mg|mcg|ml|mL|µg|units?|tabs?|capsules?)\b",
            re.IGNORECASE,
        ),
        "[REDACTED-MEDICATION]",
    ),
]


def _redact_local_patterns(text: str) -> str:
    """
    Apply _HIPAA_PATTERNS regex substitutions in sequence.
    Cheap first-pass before Cloud DLP; catches clinical codes DLP misses.
    """
    for _name, pattern, label in _HIPAA_PATTERNS:
        text = pattern.sub(label, text)
    return text


def redact_phi(text: str) -> str:
    """
    Full PHI redaction pipeline:
      1. Local regex patterns (_HIPAA_PATTERNS) — free, covers healthcare codes
      2. Cloud DLP inspect_content — covers person names, phone, email, DOB, MRNs

    Returns the original text unchanged on any error so that logging is never
    blocked by a DLP failure (fail-open policy).

    Requires GCP_PROJECT_ID to be set in settings; local-pattern pass always runs.
    """
    if not text:
        return text

    # Step 1: local patterns (always run — no API cost)
    text = _redact_local_patterns(text)

    # Step 2: Cloud DLP (skipped if GCP_PROJECT_ID not configured)
    project_id = settings.GCP_PROJECT_ID
    if not project_id:
        logger.warning("DLP redaction skipped: GCP_PROJECT_ID not configured")
        return text

    try:
        from google.cloud import dlp_v2  # type: ignore

        client = dlp_v2.DlpServiceClient()
        parent = f"projects/{project_id}"

        inspect_config = {
            "info_types": [{"name": t} for t in _DLP_INFO_TYPES],
            "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
        }
        item = {"value": text}

        response = client.inspect_content(
            request={
                "parent": parent,
                "inspect_config": inspect_config,
                "item": item,
            }
        )

        if not response.result.findings:
            return text

        # Replace findings from right to left to keep byte offsets valid
        findings = sorted(
            response.result.findings,
            key=lambda f: f.location.byte_range.start,
            reverse=True,
        )

        result = text
        for finding in findings:
            quote = finding.quote
            if not quote:
                continue
            start = finding.location.byte_range.start
            end = finding.location.byte_range.end
            result = result[:start] + _REDACTION_TOKEN + result[end:]

        return result

    except Exception as exc:
        logger.error("DLP redaction failed, returning original text: %s", exc)
        return text


def redact_phi_if_hipaa(text: str, *, hipaa_enabled: bool) -> str:
    """Convenience wrapper — only calls the full pipeline when hipaa_enabled is True."""
    if not hipaa_enabled:
        return text
    return redact_phi(text)
