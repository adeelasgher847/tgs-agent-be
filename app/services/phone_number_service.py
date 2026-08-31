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
from decimal import Decimal
from typing import List

from sqlalchemy import func, select
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
    ) -> PhoneNumber | None:
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

    # $2 flat charge (== 2 credits, 1 credit = $1) per phone number purchased/imported
    # beyond the tenant's plan-included `free_phone_numbers` allowance.
    PHONE_NUMBER_OVERAGE_COST = Decimal("2")

    @staticmethod
    def _phone_number_overage_check(db: Session, tenant_id: uuid.UUID) -> bool:
        """
        Plan-aware phone-number allowance check — NOT a hard cap. Returns True if the
        next number purchase/import would be billed as overage (tenant already at/over
        `plan.free_phone_numbers`), after confirming the tenant can cover
        `PHONE_NUMBER_OVERAGE_COST`; raises HTTP 402 (mirrors the insufficient-credits
        pattern used for the pre-call gate) if it can't — callers must not proceed to
        Twilio in that case. Returns False (no charge) when there's no active core plan,
        `free_phone_numbers` is NULL (unlimited), or the tenant is still within its free
        allowance.
        """
        from fastapi import HTTPException

        from app.models.plan import Plan
        from app.models.tenant import Tenant
        from app.services.billing_service import BillingService

        subscription = BillingService.get_workspace_subscription(db, tenant_id)
        if subscription is None:
            return False
        plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
        if plan is None or plan.free_phone_numbers is None:
            return False

        existing_count = db.execute(
            select(func.count()).select_from(PhoneNumber).where(
                PhoneNumber.tenant_id == tenant_id,
                PhoneNumber.status == "active",
            )
        ).scalar_one()

        if existing_count < plan.free_phone_numbers:
            return False

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        current_credits = tenant.credits if tenant and tenant.credits is not None else Decimal("0")
        if tenant is None or current_credits < PhoneNumberService.PHONE_NUMBER_OVERAGE_COST:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient balance to purchase an additional phone number. "
                    f"Your plan includes {plan.free_phone_numbers} free numbers; "
                    f"additional numbers cost ${PhoneNumberService.PHONE_NUMBER_OVERAGE_COST} "
                    f"each, deducted from your credit balance."
                ),
            )
        return True

    @staticmethod
    def _charge_phone_number_overage(db: Session, tenant_id: uuid.UUID) -> None:
        """Deduct the flat overage cost from the tenant's credit balance. Caller must
        have already confirmed sufficient balance via `_phone_number_overage_check`."""
        from app.models.tenant import Tenant

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            return
        tenant.credits = (tenant.credits or Decimal("0")) - PhoneNumberService.PHONE_NUMBER_OVERAGE_COST

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

        # Unlike bind_number() (app/routers/telephony.py's dedicated /bind
        # endpoint), this path lets an agent be assigned right at number
        # creation -- must apply the same "agent.status = ready" side effect
        # bind_number does, or the agent stays stuck at whatever status it
        # already had (e.g. pending) despite genuinely being bound to a
        # working number. Real production bug: a number created with
        # assistant_id set here never flipped the agent to ready.
        if phone_number.assistant_id is not None:
            agent = db.get(Agent, phone_number.assistant_id)
            if agent is not None:
                agent.status = "ready"
                db.commit()

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
    ) -> PhoneNumber | None:
        return self._get_by_id(db, phone_number_id, tenant_id)

    def update_phone_number(
        self,
        db: Session,
        phone_number_id: uuid.UUID,
        tenant_id: uuid.UUID,
        update_data: PhoneNumberUpdate,
    ) -> PhoneNumber | None:
        pn = self._get_by_id(db, phone_number_id, tenant_id)
        if not pn:
            return None
        updates = update_data.model_dump(exclude_unset=True)
        if "agent_id" in updates:
            updates["assistant_id"] = updates.pop("agent_id")

        # Same "agent.status" side effect bind_number()/unbind_number() apply
        # (app/routers/telephony.py) -- this generic edit endpoint is a
        # second, independent way to assign/clear a number's agent, and
        # previously never touched agent.status at all, leaving a genuinely
        # bound agent stuck at whatever status it already had (e.g. still
        # "pending" despite the number correctly pointing at it). Captured
        # before applying `updates` below so we still have the OLD value to
        # compare against.
        old_assistant_id = pn.assistant_id
        reassigning_agent = "assistant_id" in updates and updates["assistant_id"] != old_assistant_id

        for field, value in updates.items():
            setattr(pn, field, value)
        pn.updated_at = datetime.utcnow()

        if reassigning_agent:
            new_assistant_id = updates["assistant_id"]
            if new_assistant_id is not None:
                new_agent = db.get(Agent, new_assistant_id)
                if new_agent is not None:
                    new_agent.status = "ready"
            if old_assistant_id is not None and old_assistant_id != new_assistant_id:
                # Only downgrade the previous agent if it isn't still bound
                # to some OTHER active number -- never wrongly mark a still-
                # working agent as not-ready just because ONE of its numbers
                # was reassigned elsewhere.
                still_bound_elsewhere = db.execute(
                    select(PhoneNumber.id).where(
                        PhoneNumber.assistant_id == old_assistant_id,
                        PhoneNumber.id != pn.id,
                        PhoneNumber.status == "active",
                    ).limit(1)
                ).first() is not None
                if not still_bound_elsewhere:
                    old_agent = db.get(Agent, old_assistant_id)
                    if old_agent is not None:
                        old_agent.status = "pending"

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
        label: str | None,
        tenant_id: uuid.UUID,
        twilio_account_sid: str,
        twilio_auth_token: str,
    ) -> PhoneNumber:
        """Import a BYO Twilio number with custom per-number credentials."""
        # Plan-aware overage check before touching Twilio: raises 402 if this import
        # would be billed as overage and the tenant can't cover it.
        needs_overage_charge = self._phone_number_overage_check(db, tenant_id)

        existing = db.execute(
            select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(
                f"Phone number {phone_number} is already assigned to another tenant"
            )

        from app.core.config import settings
        from app.services.twilio_service import twilio_service
        from twilio.base.exceptions import TwilioException

        try:
            client = twilio_service.get_client_with_credentials(twilio_account_sid, twilio_auth_token)
            owned = client.incoming_phone_numbers.list(phone_number=phone_number, limit=1)
        except TwilioException as exc:
            logger.error("Twilio API authentication failed or error occurred: %s", exc)
            raise ValueError("Invalid Twilio credentials or Account SID/Auth Token provided.")

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
            if needs_overage_charge:
                self._charge_phone_number_overage(db, tenant_id)
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
        label: str | None = None,
    ) -> PhoneNumber:
        """
        Atomically purchase a Twilio number and persist a phone_numbers row.

        Uses Secret Manager credentials (staging → test creds, no real purchase).
        Webhook is configured immediately after purchase.
        """
        from fastapi import HTTPException

        from app.core.config import settings
        from app.services.twilio_service import twilio_service

        # Plan-aware overage check before touching Twilio: raises 402 if this purchase
        # would be billed as overage and the tenant can't cover it.
        needs_overage_charge = self._phone_number_overage_check(db, tenant_id)

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
            if needs_overage_charge:
                self._charge_phone_number_overage(db, tenant_id)
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
        sip_password: str | None = None,
        label: str | None = None,
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
            sip_password=encrypt_api_key(sip_password) if sip_password else None,
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
        try:
            db.commit()
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            logger.error("bind_number commit failed for agent %s: %s", agent_id, exc)
            try:
                # Re-fetch after rollback — in-memory ``agent`` may be expired.
                agent_row = db.execute(
                    select(Agent).where(
                        Agent.id == agent_id, Agent.tenant_id == tenant_id
                    )
                ).scalar_one_or_none()
                if agent_row is not None:
                    agent_row.status = "error"
                    db.commit()
            except Exception as inner:
                db.rollback()
                logger.error(
                    "Could not persist error status for agent %s: %s", agent_id, inner
                )
            raise HTTPException(
                status_code=500,
                detail="Phone number binding failed unexpectedly. Agent marked as error.",
            )
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
            agent_name: str | None = None
            agent_status: str | None = None
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
        business_hours: dict | None,
    ) -> NumberConfiguration:

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
