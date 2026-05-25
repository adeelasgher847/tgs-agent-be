"""
Phone Number Service — manages provisioned (Twilio) and BYO/SIP (external) numbers.

agent_id vs assistant_id:
  The DB column is `assistant_id` (legacy). All new telephony endpoints expose it as
  `agent_id`. This service accepts/returns `agent_id` in its public interface and maps
  internally to `assistant_id` on the ORM model.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.security import encrypt_api_key
from app.models.agent import Agent
from app.models.phone_number import NumberConfiguration, PhoneNumber
from app.schemas.phone_number import PhoneNumberCreate, PhoneNumberUpdate


class PhoneNumberService:

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_by_id(
        self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Optional[PhoneNumber]:
        stmt = select(PhoneNumber).where(
            PhoneNumber.id == phone_number_id,
            PhoneNumber.tenant_id == tenant_id,
        )
        return db.execute(stmt).scalar_one_or_none()

    def _require_number(
        self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> PhoneNumber:
        from fastapi import HTTPException

        pn = self._get_by_id(db, phone_number_id, tenant_id)
        if pn is None:
            raise HTTPException(status_code=404, detail="Phone number not found")
        return pn

    @staticmethod
    def _attach_default_configuration(db: Session, pn: PhoneNumber) -> None:
        """Create default per-number config row if missing (numberconfiguration table)."""
        existing = db.execute(
            select(NumberConfiguration).where(
                NumberConfiguration.phone_number_id == pn.id
            )
        ).scalar_one_or_none()
        if existing is None:
            db.add(NumberConfiguration(phone_number_id=pn.id))

    # ------------------------------------------------------------------
    # Legacy CRUD (backward compat — existing router uses these)
    # ------------------------------------------------------------------

    def create_phone_number(self, db: Session, phone_number_data: PhoneNumberCreate) -> PhoneNumber:
        """Register a number already in the Twilio account (env credentials)."""
        existing = db.execute(
            select(PhoneNumber).where(PhoneNumber.phone_number == phone_number_data.phone_number)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(
                f"Phone number {phone_number_data.phone_number} is already assigned to another tenant"
            )

        from app.core.config import settings
        from app.services.twilio_service import twilio_service

        encrypted_account_sid = None
        encrypted_auth_token = None
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
            encrypted_account_sid = encrypt_api_key(settings.TWILIO_ACCOUNT_SID)
            encrypted_auth_token = encrypt_api_key(settings.TWILIO_AUTH_TOKEN)

        phone_number = PhoneNumber(
            phone_number=phone_number_data.phone_number,
            label=phone_number_data.label,
            tenant_id=phone_number_data.tenant_id,
            assistant_id=phone_number_data.assistant_id,
            status="active",
            provider="twilio",
            twilio_account_sid=encrypted_account_sid,
            twilio_auth_token=encrypted_auth_token,
        )
        db.add(phone_number)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ValueError(
                f"Phone number {phone_number_data.phone_number} is already assigned to another tenant"
            )
        db.refresh(phone_number)

        if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
            raise ValueError("Twilio account credentials are required to configure inbound webhooks")

        client = twilio_service.get_client()
        owned = client.incoming_phone_numbers.list(
            phone_number=phone_number.phone_number, limit=1
        )
        if not owned:
            raise ValueError(
                f"Phone number {phone_number.phone_number} was not found in configured Twilio account"
            )

        owned_number = owned[0]
        capabilities = getattr(owned_number, "capabilities", {}) or {}
        if not capabilities.get("voice", False):
            raise ValueError(f"Phone number {phone_number.phone_number} does not support voice")

        phone_number.twilio_phone_number_sid = owned_number.sid
        twilio_service.update_number_configuration(
            phone_number_sid=owned_number.sid,
            webhook_url=f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/incoming",
            status_callback_url=f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/call-events",
        )
        db.commit()
        db.refresh(phone_number)
        return phone_number

    def get_phone_numbers(self, db: Session, tenant_id: uuid.UUID) -> List[PhoneNumber]:
        stmt = select(PhoneNumber).where(PhoneNumber.tenant_id == tenant_id)
        return list(db.execute(stmt).scalars().all())

    def get_phone_number_by_id(
        self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Optional[PhoneNumber]:
        return self._get_by_id(db, phone_number_id, tenant_id)

    def update_phone_number(
        self,
        db: Session,
        phone_number_id: uuid.UUID,
        tenant_id: uuid.UUID,
        update_data: PhoneNumberUpdate,
    ) -> Optional[PhoneNumber]:
        pn = self._get_by_id(db, phone_number_id, tenant_id)
        if not pn:
            return None
        for field, value in update_data.model_dump(exclude_unset=True).items():
            setattr(pn, field, value)
        pn.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(pn)
        return pn

    def delete_phone_number(
        self, db: Session, phone_number_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        pn = self._get_by_id(db, phone_number_id, tenant_id)
        if not pn:
            return False
        db.delete(pn)
        db.commit()
        return True

    def import_twilio_phone_number(
        self,
        db: Session,
        phone_number: str,
        label: Optional[str],
        tenant_id: uuid.UUID,
        twilio_account_sid: str,
        twilio_auth_token: str,
    ) -> PhoneNumber:
        """Import a BYO Twilio number with custom per-number credentials."""
        existing = db.execute(
            select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(
                f"Phone number {phone_number} is already assigned to another tenant"
            )

        from app.core.config import settings
        from app.services.twilio_service import twilio_service

        client = twilio_service.get_client_with_credentials(twilio_account_sid, twilio_auth_token)
        owned = client.incoming_phone_numbers.list(phone_number=phone_number, limit=1)
        if not owned:
            raise ValueError(
                f"Phone number {phone_number} was not found in the provided Twilio account"
            )

        owned_number = owned[0]
        capabilities = getattr(owned_number, "capabilities", {}) or {}
        if not capabilities.get("voice", False):
            raise ValueError(f"Phone number {phone_number} does not support voice capability")

        twilio_sid = owned_number.sid
        inbound_webhook_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/incoming"
        status_callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/call-events"

        try:
            twilio_service.update_number_configuration_with_credentials(
                phone_number_sid=twilio_sid,
                account_sid=twilio_account_sid,
                auth_token=twilio_auth_token,
                webhook_url=inbound_webhook_url,
                status_callback_url=status_callback_url,
            )
        except Exception as exc:
            raise ValueError(f"Failed to configure Twilio webhooks for {phone_number}: {exc}")

        pn = PhoneNumber(
            phone_number=phone_number,
            label=label,
            tenant_id=tenant_id,
            status="active",
            provider="twilio",
            twilio_phone_number_sid=twilio_sid,
            twilio_account_sid=encrypt_api_key(twilio_account_sid),
            twilio_auth_token=encrypt_api_key(twilio_auth_token),
        )
        db.add(pn)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ValueError(f"Phone number {phone_number} is already assigned to another tenant")
        db.refresh(pn)
        return pn

    # ------------------------------------------------------------------
    # Sprint 2 — new provisioning methods
    # ------------------------------------------------------------------

    def purchase_phone_number(
        self,
        db: Session,
        phone_number: str,
        tenant_id: uuid.UUID,
        label: Optional[str] = None,
    ) -> PhoneNumber:
        """
        Atomically purchase a Twilio number and persist a phone_numbers row.

        Uses Secret Manager credentials (staging → test creds, no real purchase).
        Webhook is configured immediately after purchase.
        """
        from fastapi import HTTPException

        from app.core.config import settings
        from app.services.twilio_service import twilio_service

        # Global uniqueness check before touching Twilio
        existing = db.execute(
            select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Phone number {phone_number} is already registered",
            )

        inbound_webhook = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/incoming"
        status_callback = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/call-events"

        # Purchase via Twilio (Secret Manager creds injected inside get_client())
        try:
            purchase_result = twilio_service.purchase_phone_number(
                phone_number=phone_number,
                webhook_url=inbound_webhook,
                status_callback_url=status_callback,
            )
        except Exception as exc:
            logger.error("Twilio purchase failed for %s: %s", phone_number, exc)
            raise HTTPException(status_code=502, detail=f"Twilio purchase failed: {exc}")

        pn = PhoneNumber(
            phone_number=purchase_result["phone_number"],
            label=label,
            tenant_id=tenant_id,
            status="active",
            provider="twilio",
            twilio_phone_number_sid=purchase_result["sid"],
        )
        db.add(pn)
        try:
            db.flush()
            self._attach_default_configuration(db, pn)
            db.commit()
        except IntegrityError:
            db.rollback()
            # Edge case: concurrent purchase of same number — not a real purchase error,
            # just a DB uniqueness collision; the Twilio number was purchased, log and surface.
            logger.error("DB integrity error after Twilio purchase of %s", phone_number)
            raise HTTPException(
                status_code=409,
                detail=f"Phone number {phone_number} registered concurrently",
            )
        db.refresh(pn)
        return pn

    def register_external_number(
        self,
        db: Session,
        phone_number: str,
        tenant_id: uuid.UUID,
        sip_username: str,
        sip_password: str,
        label: Optional[str] = None,
    ) -> PhoneNumber:
        """Register a BYO / SIP external number (provider='external')."""
        from fastapi import HTTPException

        existing = db.execute(
            select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Phone number {phone_number} is already registered",
            )

        pn = PhoneNumber(
            phone_number=phone_number,
            label=label,
            tenant_id=tenant_id,
            status="active",
            provider="external",
            sip_username=sip_username,
            sip_password=encrypt_api_key(sip_password),
        )
        db.add(pn)
        try:
            db.flush()
            self._attach_default_configuration(db, pn)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail=f"Phone number {phone_number} is already registered",
            )
        db.refresh(pn)
        return pn

    def bind_number(
        self,
        db: Session,
        phone_number_id: uuid.UUID,
        agent_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> PhoneNumber:
        """
        Bind a phone number to an agent.

        Rules:
        - One number → one agent. Duplicate bind → 409.
        - Sets agent.status = 'ready'.
        """
        from fastapi import HTTPException

        pn = self._require_number(db, phone_number_id, tenant_id)

        if pn.assistant_id is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Phone number {pn.phone_number} is already bound to agent {pn.assistant_id}. "
                    "Unbind first."
                ),
            )

        agent = db.execute(
            select(Agent).where(Agent.id == agent_id, Agent.tenant_id == tenant_id)
        ).scalar_one_or_none()
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        pn.assistant_id = agent_id
        agent.status = "ready"
        db.commit()
        db.refresh(pn)
        return pn

    def unbind_number(
        self,
        db: Session,
        phone_number_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> PhoneNumber:
        """
        Unbind a phone number from its agent.

        Rules:
        - Must currently be bound, else 409.
        - Sets agent.status = 'pending'.
        """
        from fastapi import HTTPException

        pn = self._require_number(db, phone_number_id, tenant_id)

        if pn.assistant_id is None:
            raise HTTPException(
                status_code=409,
                detail=f"Phone number {pn.phone_number} is not bound to any agent",
            )

        agent = db.execute(
            select(Agent).where(Agent.id == pn.assistant_id)
        ).scalar_one_or_none()
        if agent is not None:
            agent.status = "pending"

        pn.assistant_id = None
        db.commit()
        db.refresh(pn)
        return pn

    def list_numbers_with_binding(self, db: Session, tenant_id: uuid.UUID) -> List[dict]:
        """Return phone numbers with binding status and agent name."""
        stmt = select(PhoneNumber).where(PhoneNumber.tenant_id == tenant_id)
        numbers = list(db.execute(stmt).scalars().all())

        result = []
        for pn in numbers:
            agent_name: Optional[str] = None
            agent_status: Optional[str] = None
            if pn.assistant_id:
                agent = db.execute(
                    select(Agent).where(Agent.id == pn.assistant_id)
                ).scalar_one_or_none()
                if agent:
                    agent_name = agent.name
                    agent_status = agent.status
            result.append(
                {
                    "id": pn.id,
                    "phone_number": pn.phone_number,
                    "provider": pn.provider,
                    "label": pn.label,
                    "status": pn.status,
                    "workspace_id": pn.tenant_id,
                    "twilio_sid": pn.twilio_phone_number_sid,
                    "binding_status": "bound" if pn.assistant_id else "unbound",
                    "agent_id": pn.assistant_id,
                    "agent_name": agent_name,
                    "agent_status": agent_status,
                    "created_at": pn.created_at,
                }
            )
        return result

    def list_bound_bindings(self, db: Session, tenant_id: uuid.UUID) -> List[dict]:
        """Phone numbers currently bound to an agent (assistant_id set)."""
        return [
            row
            for row in self.list_numbers_with_binding(db, tenant_id)
            if row.get("agent_id") is not None
        ]

    # ------------------------------------------------------------------
    # Number configuration CRUD
    # ------------------------------------------------------------------

    def upsert_number_configuration(
        self,
        db: Session,
        phone_number_id: uuid.UUID,
        tenant_id: uuid.UUID,
        recording_enabled: bool,
        max_duration_seconds: int,
        business_hours: Optional[dict],
    ) -> NumberConfiguration:
        from fastapi import HTTPException

        pn = self._require_number(db, phone_number_id, tenant_id)

        config = db.execute(
            select(NumberConfiguration).where(
                NumberConfiguration.phone_number_id == pn.id
            )
        ).scalar_one_or_none()

        if config is None:
            config = NumberConfiguration(phone_number_id=pn.id)
            db.add(config)

        config.recording_enabled = recording_enabled
        config.max_duration_seconds = max_duration_seconds
        config.business_hours = business_hours
        db.commit()
        db.refresh(config)
        return config


phone_number_service = PhoneNumberService()
