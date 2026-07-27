"""
Decode encrypted API keys stored in the model table.

Usage:
  python -m scripts.decode_model_api_keys
  python -m scripts.decode_model_api_keys --model-id <uuid>
  python -m scripts.decode_model_api_keys --provider-id <uuid>
"""

from __future__ import annotations

import argparse
import uuid
from typing import Optional

from app.core.security import decrypt_api_key, is_api_key_encrypted
from app.db.session import SessionLocal
from app.models.model import Model
from app.models.provider import Provider


def _parse_uuid(value: Optional[str], field_name: str) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name} UUID: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode encrypted model.api_key values from database."
    )
    parser.add_argument("--model-id", default=None, help="Filter by model UUID")
    parser.add_argument("--provider-id", default=None, help="Filter by provider UUID")
    args = parser.parse_args()

    model_id = _parse_uuid(args.model_id, "model-id")
    provider_id = _parse_uuid(args.provider_id, "provider-id")

    db = SessionLocal()
    try:
        query = (
            db.query(Model, Provider.name)
            .join(Provider, Provider.id == Model.provider_id)
            .order_by(Model.created_at.desc())
        )

        if model_id:
            query = query.filter(Model.id == model_id)
        if provider_id:
            query = query.filter(Model.provider_id == provider_id)

        rows = query.all()
        if not rows:
            print("[INFO] No models found for given filter.")
            return

        print(f"[INFO] Found {len(rows)} model(s)")
        for model, provider_name in rows:
            encrypted = model.api_key
            if not encrypted:
                decrypted_value = None
                status = "empty"
            elif is_api_key_encrypted(encrypted):
                try:
                    decrypted_value = decrypt_api_key(encrypted)
                    status = "decrypted"
                except ValueError as exc:
                    decrypted_value = None
                    status = f"decrypt_failed: {exc}"
            else:
                # Already plaintext in DB (legacy/manual insert case).
                decrypted_value = encrypted
                status = "plaintext"

            print("-" * 80)
            print(f"model_id     : {model.id}")
            print(f"model_name   : {model.model_name}")
            print(f"provider_id  : {model.provider_id}")
            print(f"provider_name: {provider_name}")
            print(f"status       : {status}")
            print(f"decoded_key  : {decrypted_value}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
