from __future__ import annotations

from typing import Optional
import uuid

from sqlalchemy.orm import Session

from app.core.secret_manager import get_rime_api_key
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.utils.tts_adapter import get_tts_adapter_for_provider


class TTSCatalogService:
    @staticmethod
    def verify_rime_api_key_configured() -> None:
        """Fail fast when Rime is enabled but RIME_API_KEY is missing or invalid."""
        get_rime_api_key()

    def ensure_default_provider(self, db: Session) -> TTSProvider:
        providers_to_seed = [
            {
                "slug": "elevenlabs",
                "display_name": "ElevenLabs",
                "is_active": True,
                "supports_streaming": True,
                "supports_ssml": True,
            },
            {
                "slug": "google",
                "display_name": "Google Cloud TTS",
                "is_active": True,
                "supports_streaming": True,
                "supports_ssml": True,
            },
            {
                "slug": "rime",
                "display_name": "Rime Labs",
                "is_active": True,
                "supports_streaming": True,
                "supports_ssml": False,
            },
        ]

        selected = None
        changed = False
        for spec in providers_to_seed:
            provider = db.query(TTSProvider).filter(TTSProvider.slug == spec["slug"]).first()
            if provider is None:
                provider = TTSProvider(**spec)
                db.add(provider)
                changed = True
            if spec["slug"] == "elevenlabs":
                selected = provider

        if changed:
            db.commit()
            if selected is not None:
                db.refresh(selected)

        if selected is None:
            selected = db.query(TTSProvider).filter(TTSProvider.slug == "elevenlabs").first()

        # Rime is always seeded as active — catch misconfiguration before the first call.
        self.verify_rime_api_key_configured()
        return selected

    def list_providers(self, db: Session, active_only: bool = True) -> list[TTSProvider]:
        self.ensure_default_provider(db)
        query = db.query(TTSProvider)
        if active_only:
            query = query.filter(TTSProvider.is_active == True)  # noqa: E712
        return query.order_by(TTSProvider.display_name.asc()).all()

    def list_voices(
        self,
        db: Session,
        provider_id: uuid.UUID,
        language_code: Optional[str] = None,
        active_only: bool = True,
    ) -> list[TTSVoice]:
        query = db.query(TTSVoice).filter(TTSVoice.provider_id == provider_id)
        if active_only:
            query = query.filter(TTSVoice.is_active == True)  # noqa: E712
        if language_code:
            query = query.filter(TTSVoice.language_code == language_code)
        return query.order_by(TTSVoice.display_name.asc()).all()

    def get_provider_by_id(self, db: Session, provider_id: uuid.UUID) -> Optional[TTSProvider]:
        return db.query(TTSProvider).filter(TTSProvider.id == provider_id).first()

    def get_provider_by_slug(self, db: Session, slug: str) -> Optional[TTSProvider]:
        return db.query(TTSProvider).filter(TTSProvider.slug == slug).first()

    def get_voice_by_id(self, db: Session, voice_id: uuid.UUID) -> Optional[TTSVoice]:
        return db.query(TTSVoice).filter(TTSVoice.id == voice_id).first()

    def sync_provider_voices(self, db: Session, provider_slug: str) -> dict[str, int]:
        provider = self.get_provider_by_slug(db, provider_slug)
        if not provider:
            raise ValueError(f"TTS provider '{provider_slug}' not found.")
        if not provider.is_active:
            raise ValueError(f"TTS provider '{provider_slug}' is inactive.")

        adapter = get_tts_adapter_for_provider(provider)
        raw_voices = adapter.list_voices()

        touched_ids: set[uuid.UUID] = set()
        created = 0
        updated = 0

        for payload in raw_voices:
            normalized = adapter.normalize_voice_payload(payload)
            external_voice_id = (normalized.get("external_voice_id") or "").strip()
            if not external_voice_id:
                continue

            existing = (
                db.query(TTSVoice)
                .filter(
                    TTSVoice.provider_id == provider.id,
                    TTSVoice.external_voice_id == external_voice_id,
                )
                .first()
            )
            if existing:
                existing.display_name = normalized.get("display_name") or existing.display_name
                existing.language_code = normalized.get("language_code")
                existing.gender = normalized.get("gender")
                existing.accent = normalized.get("accent")
                existing.description = normalized.get("description")
                existing.preview_audio_url = normalized.get("preview_audio_url")
                existing.sample_rate_hz = normalized.get("sample_rate_hz")
                existing.metadata_json = normalized.get("metadata_json")
                existing.is_active = True
                touched_ids.add(existing.id)
                updated += 1
                continue

            new_voice = TTSVoice(
                provider_id=provider.id,
                external_voice_id=external_voice_id,
                display_name=normalized.get("display_name") or external_voice_id,
                language_code=normalized.get("language_code"),
                gender=normalized.get("gender"),
                accent=normalized.get("accent"),
                description=normalized.get("description"),
                preview_audio_url=normalized.get("preview_audio_url"),
                sample_rate_hz=normalized.get("sample_rate_hz"),
                metadata_json=normalized.get("metadata_json"),
                is_active=True,
            )
            db.add(new_voice)
            db.flush()
            touched_ids.add(new_voice.id)
            created += 1

        if touched_ids:
            db.query(TTSVoice).filter(
                TTSVoice.provider_id == provider.id,
                TTSVoice.id.notin_(touched_ids),
            ).update({"is_active": False}, synchronize_session=False)

        db.commit()
        return {"created": created, "updated": updated, "fetched": len(raw_voices)}


tts_catalog_service = TTSCatalogService()
