#!/usr/bin/env python3
"""
Ensure gemini-2.5-flash exists in the model catalog (Google provider).

Uses GOOGLE_APPLICATION_CREDENTIALS at runtime — no api_key stored on the row.

Usage:
    python scripts/seed_gemini_25_flash_model.py
"""
from __future__ import annotations

import sys
import uuid

from sqlalchemy import or_

from app.db.session import SessionLocal
from app.models.model import Model
from app.models.provider import Provider


def main() -> int:
    db = SessionLocal()
    try:
        provider = (
            db.query(Provider)
            .filter(
                or_(
                    Provider.name.ilike("%google%"),
                    Provider.name.ilike("%gemini%"),
                )
            )
            .order_by(Provider.created_at.asc())
            .first()
        )
        if not provider:
            provider = Provider(name="Google", is_active=True, api_key=None)
            db.add(provider)
            db.flush()
            print(f"Created provider: {provider.name} ({provider.id})")

        existing = (
            db.query(Model)
            .filter(Model.model_name == "gemini-2.5-flash", Model.archive == False)  # noqa: E712
            .first()
        )
        if existing:
            print(f"Active model already exists: gemini-2.5-flash ({existing.id})")
            return 0

        row = Model(
            id=uuid.uuid4(),
            provider_id=provider.id,
            model_name="gemini-2.5-flash",
            api_key=None,
            description="Gemini 2.5 Flash via Vertex AI (GOOGLE_APPLICATION_CREDENTIALS)",
            archive=False,
        )
        db.add(row)
        db.commit()
        print(f"Inserted model gemini-2.5-flash ({row.id}) under provider {provider.name}")
        return 0
    except Exception as exc:
        db.rollback()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
