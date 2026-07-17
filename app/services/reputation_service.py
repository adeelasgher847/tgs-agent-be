"""
Outbound number reputation checks (First Orion primary, Hiya fallback).

check_number_reputation() is the single entry point used by:
  - POST /api/v2/telephony/check-reputation
  - the daily ARQ cron (app.workers.batch_call_worker.check_all_phone_numbers_reputation)
  - the batch-call rotation flow (app.services.batch_call_service)

In development/local, or whenever REPUTATION_API_KEY is not configured, a
deterministic mock check is used instead of calling a real carrier API so the
feature is exercisable without third-party credentials.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.core.secret_manager import get_reputation_api_key
from app.models.phone_number import PhoneNumber
from app.models.phone_number_reputation import PhoneNumberReputation

FIRST_ORION_URL = "https://api.firstorion.com/v1/reputation/lookup"
HIYA_URL = "https://api.hiya.com/v1/reputation/lookup"

SPAM_THRESHOLD = 50
REQUEST_TIMEOUT_SECONDS = 10.0


def _use_mock() -> bool:
    env = settings.ENVIRONMENT.lower()
    if env not in ("staging", "production"):
        return True
    return not get_reputation_api_key()


def _mock_check(phone_number: str) -> tuple[int, str]:
    """
    Deterministic stub: derive a stable 0-100 score from the phone number so
    repeated checks against the same number are consistent within a test run.
    """
    digest = hashlib.sha256(phone_number.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) % 101
    checked_by = "mock"
    return score, checked_by


async def _call_first_orion(api_key: str, phone_number: str) -> tuple[int, str]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(
            FIRST_ORION_URL,
            params={"number": phone_number},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return int(data["reputation_score"]), "first_orion"


async def _call_hiya(api_key: str, phone_number: str) -> tuple[int, str]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(
            HIYA_URL,
            params={"number": phone_number},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return int(data["reputation_score"]), "hiya"


async def _fetch_score(phone_number: str) -> tuple[int, str]:
    """Return (reputation_score, checked_by). Falls back to Hiya, then the mock."""
    if _use_mock():
        return _mock_check(phone_number)

    api_key = get_reputation_api_key()
    try:
        return await _call_first_orion(api_key, phone_number)
    except Exception as exc:
        logger.warning(
            "First Orion reputation lookup failed for %s: %s — falling back to Hiya",
            phone_number,
            exc,
        )

    try:
        return await _call_hiya(api_key, phone_number)
    except Exception as exc:
        logger.error(
            "Hiya reputation lookup also failed for %s: %s — falling back to mock",
            phone_number,
            exc,
        )
        return _mock_check(phone_number)


async def check_number_reputation(db: Session, phone_number_obj: PhoneNumber) -> dict:
    """
    Query the reputation provider for phone_number_obj, upsert the
    PhoneNumberReputation row, and return the result.

    Returns {spam_flagged: bool, reputation_score: int, flagged_reason: Optional[str]}.
    """
    score, checked_by = await _fetch_score(phone_number_obj.phone_number)
    score = max(0, min(100, score))
    spam_flagged = score < SPAM_THRESHOLD
    flagged_reason: Optional[str] = (
        f"Reputation score {score} below threshold {SPAM_THRESHOLD} ({checked_by})"
        if spam_flagged
        else None
    )
    now = datetime.now(timezone.utc)

    row = (
        db.query(PhoneNumberReputation)
        .filter(PhoneNumberReputation.phone_number_id == phone_number_obj.id)
        .first()
    )
    if row is None:
        row = PhoneNumberReputation(phone_number_id=phone_number_obj.id)
        db.add(row)

    row.reputation_score = score
    row.spam_flagged = spam_flagged
    row.last_checked_at = now
    row.checked_by = checked_by
    row.flagged_reason = flagged_reason

    try:
        db.commit()
    except IntegrityError:
        # Concurrent caller (cron + rotation check, or two rotation checks)
        # inserted the row first — fall back to updating the existing one.
        db.rollback()
        row = (
            db.query(PhoneNumberReputation)
            .filter(PhoneNumberReputation.phone_number_id == phone_number_obj.id)
            .first()
        )
        row.reputation_score = score
        row.spam_flagged = spam_flagged
        row.last_checked_at = now
        row.checked_by = checked_by
        row.flagged_reason = flagged_reason
        db.commit()

    logger.info(
        "Reputation check for %s: score=%d flagged=%s checked_by=%s",
        phone_number_obj.phone_number,
        score,
        spam_flagged,
        checked_by,
    )

    return {
        "spam_flagged": spam_flagged,
        "reputation_score": score,
        "flagged_reason": flagged_reason,
    }
