"""
Credit Service Module
Vapi-style real-time billing: Per-second accurate credit deduction
"""

from sqlalchemy import or_, update as sa_update
from sqlalchemy.orm import Session
from app.models.tenant import Tenant
from app.models.agent import Agent
from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.pricing_configs import PricingConfig
from app.models.wallet_auto_recharge_config import WalletAutoRechargeConfig
from app.core.llm_models import is_openai_realtime_model
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

# Flat per-minute rate fallback when a workspace has no PricingConfig row yet.
# Matches PricingConfig.per_minute_rate's own column default.
DEFAULT_PER_MINUTE_RATE = Decimal("0.12")

# Flat per-minute surcharge for calls using an OpenAI Realtime speech-to-speech
# model (see app.core.llm_models.OPENAI_REALTIME_MODELS). Charged from
# tenant.credits on top of the normal base-rate/included-minutes handling —
# applies to the FULL call duration (both the free-allowance and
# billable/overage portions), never waived by the included-minutes pool.
OPENAI_REALTIME_SURCHARGE_PER_MINUTE = Decimal("0.50")

# Flat per-minute surcharge for calls whose agent's resolved TTS provider
# (see app.core.agent_runtime.resolve_tts_runtime) is ElevenLabs. Same
# mechanic as the realtime surcharge above: charged from tenant.credits,
# never waived by the included-minutes pool.
ELEVENLABS_TTS_SURCHARGE_PER_MINUTE = Decimal("0.05")


@dataclass(frozen=True)
class ActiveSurcharge:
    """A single named per-minute surcharge that applies to a call.

    `key` is a stable machine-readable identifier (used in logs/descriptions);
    `label` is the human-readable name surfaced in deduction descriptions and
    the pricing/usage APIs; `rate_per_minute` is charged against tenant.credits
    for the FULL call duration, independent of the included-minutes pool.
    """

    key: str
    label: str
    rate_per_minute: Decimal
    applies_when: str


# Full catalog of surcharges this service knows how to apply, independent of
# any specific call — used to advertise "what can stack on top of the base
# rate" via the pricing/usage APIs (app/api/v2/routers/workspace.py) even
# when no agent/model context is available to say whether it's *currently*
# active for a given call.
SURCHARGE_CATALOG: tuple[ActiveSurcharge, ...] = (
    ActiveSurcharge(
        key="openai_realtime",
        label="OpenAI Realtime speech-to-speech surcharge",
        rate_per_minute=OPENAI_REALTIME_SURCHARGE_PER_MINUTE,
        applies_when="Agent's model is an OpenAI Realtime speech-to-speech model (gpt-realtime / gpt-realtime-2).",
    ),
    ActiveSurcharge(
        key="elevenlabs_tts",
        label="ElevenLabs voice surcharge",
        rate_per_minute=ELEVENLABS_TTS_SURCHARGE_PER_MINUTE,
        applies_when="Agent's resolved TTS provider is ElevenLabs.",
    ),
)


class CreditService:
    """
    Service for managing tenant credits and call billing
    Vapi-style: Real-time per-second billing with accurate duration tracking

    Per-model credit costs have been retired in favor of a flat $/min rate
    (see `get_effective_rate_per_minute`); the plan-aware included-minutes
    allowance is applied before any credits are deducted.
    """

    # Monitoring interval (check every N seconds for real-time accuracy)
    MONITORING_INTERVAL = 10  # Check every 10 seconds (Vapi-style real-time)

    # Minimum time between wallet-auto-recharge trigger *attempts* for a given
    # tenant, regardless of whether the attempt succeeds or fails. Guards
    # against the 10s live-billing tick (and any other `deduct_credits`
    # caller) firing a real off-session Stripe charge on every single
    # deduction while the tenant sits below their configured threshold.
    AUTO_RECHARGE_COOLDOWN_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._active_monitors = {}  # {call_session_id: task}
        self._call_start_times = {}  # {call_session_id: start_time}
        self._last_deduction_time = {}  # {call_session_id: last_deduction_timestamp}
        self._accumulated_seconds = {}  # {call_session_id: accumulated_seconds}
        # Strongly-referenced auto-recharge charge tasks, keyed by a unique
        # per-attempt id (config_id + claimed last_triggered_at timestamp) so
        # the underlying `asyncio.to_thread(...)` task is never silently
        # garbage-collected before it completes — mirrors `_active_monitors`
        # above. Entries are removed (and any exception logged) by the
        # done-callback registered in `_maybe_trigger_wallet_auto_recharge`.
        self._active_auto_recharge_tasks: dict[str, asyncio.Task] = {}

    def get_surcharge_catalog(self) -> tuple[ActiveSurcharge, ...]:
        """All surcharges this service knows how to apply, regardless of whether
        they're currently active for any particular call — used by the pricing/usage
        APIs to advertise what can stack on top of the base per-minute rate."""
        return SURCHARGE_CATALOG

    def get_active_surcharges(
        self,
        model_name: str | None = None,
        tts_provider_slug: str | None = None,
        is_byo_elevenlabs: bool = False,
        realtime_fallen_back: bool = False,
    ) -> list[ActiveSurcharge]:
        """
        Resolve which named surcharges apply to a call given its model/TTS
        configuration. Returns a list so multiple independent surcharges can
        stack additively — extend this by appending another check, never by
        hardcoding a single combined rate.

        OpenAI Realtime speech-to-speech calls (see
        `app.voice.voice_orchestrator.VoiceOrchestrator._is_openai_realtime` /
        `_feed_openai_realtime_audio`) never invoke the separate TTS
        synthesis pipeline (`TtsPipeline` / `tts_stream_mixin`) — audio is
        generated natively by the realtime session. `agent.tts_provider_slug`
        may still hold a configured value (e.g. "elevenlabs") even though
        it's functionally unused for these calls, so the ElevenLabs surcharge
        is deliberately NOT evaluated when the model is an OpenAI Realtime
        model (and it hasn't fallen back — see `realtime_fallen_back` below),
        to avoid billing for TTS synthesis that never happened. If a future
        architecture makes both paths coexist on the same call, this early
        return should become an additive check instead.

        `is_byo_elevenlabs` (see `app.core.agent_runtime.ResolvedTtsRuntime`)
        must be True only when the tenant supplied their own ElevenLabs API
        key — the platform incurs zero ElevenLabs cost in that case, so the
        surcharge must NOT apply even though `tts_provider_slug` still
        resolves to "elevenlabs" for both platform and BYO ElevenLabs (the
        adapter_slug collapses the two).

        `realtime_fallen_back` must be True once
        `VoiceOrchestrator._fallback_to_legacy_pipeline_openai()` has kicked
        in for this call (persisted on `CallSession.call_metadata
        ["openai_realtime_fallback"]` — see that method's docstring for why
        call_metadata is the cross-process signal). Once fallen back, the
        call is no longer using OpenAI's realtime speech-to-speech API for
        the remainder of its duration, so the realtime surcharge must stop
        applying and the ElevenLabs surcharge (if applicable) must start
        being evaluated for real instead of being skipped by the early
        return below.
        """
        surcharges: list[ActiveSurcharge] = []

        is_realtime = (
            is_openai_realtime_model(model_name) if model_name else False
        ) and not realtime_fallen_back
        if is_realtime:
            surcharges.append(
                next(s for s in SURCHARGE_CATALOG if s.key == "openai_realtime")
            )
            return surcharges

        if (
            tts_provider_slug
            and tts_provider_slug.strip().lower() == "elevenlabs"
            and not is_byo_elevenlabs
        ):
            surcharges.append(
                next(s for s in SURCHARGE_CATALOG if s.key == "elevenlabs_tts")
            )

        return surcharges

    @staticmethod
    def total_surcharge_rate_per_minute(surcharges: list[ActiveSurcharge]) -> Decimal:
        """Sum of the active surcharges' per-minute rates."""
        total = Decimal("0")
        for s in surcharges:
            total += s.rate_per_minute
        return total

    def _describe_surcharges(
        self, accumulated_seconds: float, surcharges: list[ActiveSurcharge]
    ) -> tuple[float, str]:
        """
        Compute the total surcharge credits owed for `accumulated_seconds` of
        call time (surcharges apply to the FULL duration, never waived by the
        included-minutes pool) plus a human-readable breakdown string
        attributing the amount to each active surcharge, for deduction
        descriptions/logs.
        """
        if not surcharges:
            return 0.0, ""

        minutes = Decimal(str(accumulated_seconds)) / Decimal("60")
        parts: list[str] = []
        total = Decimal("0")
        for s in surcharges:
            amount = minutes * s.rate_per_minute
            total += amount
            parts.append(
                f"{s.label} on {accumulated_seconds:.1f}s ({float(amount):.4f} credits)"
            )
        return float(total), " + ".join(parts)

    def get_effective_rate_per_minute(
        self, db: Session, tenant_id: uuid.UUID
    ) -> Decimal:
        """
        Get the effective flat $/min (== credits/min, since 1 credit = $1) rate for a tenant.
        Falls back to DEFAULT_PER_MINUTE_RATE (0.12) if no PricingConfig row exists yet.
        """
        config = (
            db.query(PricingConfig)
            .filter(PricingConfig.workspace_id == tenant_id)
            .first()
        )
        if config and config.per_minute_rate is not None:
            return Decimal(str(config.per_minute_rate))
        return DEFAULT_PER_MINUTE_RATE

    def calculate_credits_for_duration(
        self, duration_seconds: float, credits_per_minute: float
    ) -> float:
        """
        Calculate credits for a specific duration (Vapi-style: per-second accurate calculation)

        Args:
            duration_seconds: Duration in seconds (can be fractional)
            credits_per_minute: Credits (== dollars) per minute

        Returns:
            Credits to deduct (as float for accurate billing)
        """
        if duration_seconds <= 0:
            return 0.0

        # Vapi-style: Calculate per-second cost (accurate, no rounding)
        credits_per_second = credits_per_minute / 60.0
        total_credits = duration_seconds * credits_per_second

        # Return float for accurate billing (no rounding until final deduction)
        return round(total_credits, 4)  # Round to 4 decimal places for precision

    def _resolve_billing_tenant_id(
        self, db: Session, tenant_id: uuid.UUID
    ) -> uuid.UUID:
        """
        Resolve which tenant's `Tenant.credits` should actually be touched
        for a credit-balance check/deduction originating from `tenant_id`.

        Wallet sharing: when `tenant_id` is a sub-account with
        `uses_master_wallet=True` AND a non-null `parent_workspace_id`, the
        PARENT tenant's credit balance is billed instead of the sub-account's
        own. Otherwise (wallet sharing off, or no parent) this is a no-op —
        the tenant bills itself, as before wallet sharing existed.

        This must NEVER be used for the included-minutes/plan-allowance
        lookup (`BillingService.get_included_minutes_status`) — that stays
        keyed on the original calling tenant's own plan/allowance regardless
        of wallet sharing. Only the actual `Tenant.credits` balance
        check/mutation is redirected here.
        """
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if (
            tenant is not None
            and tenant.uses_master_wallet
            and tenant.parent_workspace_id is not None
        ):
            return tenant.parent_workspace_id
        return tenant_id

    def get_tenant_credits(self, db: Session, tenant_id: uuid.UUID) -> float:
        """
        Get current credit balance for a tenant

        Args:
            db: Database session
            tenant_id: Tenant UUID

        Returns:
            Current credit balance (as float)
        """
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return 0.0
        return float(tenant.credits or 0)

    def has_sufficient_credits(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        model_name: str | None = None,
        tts_provider_slug: str | None = None,
        is_byo_elevenlabs: bool = False,
    ) -> tuple[bool, float, float, str]:
        """
        Check if tenant is allowed to start a call: plan-aware pre-call gate.

        If the tenant still has remaining included plan minutes this cycle, the call
        is allowed regardless of credit balance (it's covered by the plan allowance).
        Otherwise, fall back to the existing "credits > 0" check.

        Any call whose agent configuration triggers one or more surcharges (see
        `get_active_surcharges` — currently: OpenAI Realtime speech-to-speech models,
        or an ElevenLabs-resolved TTS provider) has those surcharges deducted from
        credits starting from the very first monitoring tick, regardless of remaining
        included minutes — so a tenant with a zero credit balance must not be allowed
        to start one of these calls even if they still have plenty of included minutes
        left, since the first tick's surcharge deduction would immediately force-end
        the call.

        Args:
            model_name: Agent's model name, if known at call-start time. Only
                affects the gate when it is an OpenAI Realtime model.
            tts_provider_slug: Agent's resolved TTS adapter slug (see
                `app.core.agent_runtime.resolve_tts_runtime`), if known at
                call-start time. Only affects the gate when it is "elevenlabs".
            is_byo_elevenlabs: True when the agent uses a tenant-supplied BYO
                ElevenLabs key (see `ResolvedTtsRuntime.is_byo_elevenlabs`) —
                exempts the call from the ElevenLabs surcharge gate below.

        Returns:
            Tuple of (has_sufficient, current_credits, required_credits, reason).
            `required_credits` is kept in the return shape for backward-compat callers
            but is no longer a meaningful "cost estimate" now that per-model pricing
            is retired; it's always 0.0. `reason` is a short machine/human-readable
            string identifying which rule produced the result (for diagnosable
            rejection logging/messaging at call sites) — "ok" when allowed.

        Wallet sharing: the included-minutes/plan-allowance check below is always
        against `tenant_id` (the calling tenant's own plan is unaffected by wallet
        sharing), but the credit-BALANCE check is against the resolved billing
        tenant (`_resolve_billing_tenant_id`) — the parent's balance when the
        calling tenant has wallet sharing on, otherwise the calling tenant's own.
        """
        from app.services.billing_service import BillingService

        billing_tenant_id = self._resolve_billing_tenant_id(db, tenant_id)
        current_credits = self.get_tenant_credits(db, billing_tenant_id)
        _, _, remaining_minutes = BillingService.get_included_minutes_status(
            db, tenant_id
        )

        surcharges = self.get_active_surcharges(
            model_name=model_name,
            tts_provider_slug=tts_provider_slug,
            is_byo_elevenlabs=is_byo_elevenlabs,
        )

        if remaining_minutes > 0:
            if surcharges:
                # Base rate is still waived by the included-minutes allowance, but
                # active surcharges are not — require a positive credit balance too.
                if current_credits > 0:
                    return (True, current_credits, 0.0, "ok")
                names = " + ".join(s.label for s in surcharges)
                return (
                    False,
                    current_credits,
                    0.0,
                    f"blocked: {names} requires a positive credit balance "
                    "(included minutes remain, but surcharges are never waived by the allowance)",
                )
            return (True, current_credits, 0.0, "ok")

        # No included minutes left (or pure pay-as-you-go) — require positive credit balance.
        if current_credits > 0:
            return (True, current_credits, 0.0, "ok")
        return (
            False,
            current_credits,
            0.0,
            "blocked: no included minutes remaining and no credit balance",
        )

    def deduct_credits(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        amount: float,
        call_session_id: uuid.UUID | None = None,
        description: str = None,
    ) -> tuple[bool, float]:
        """
        Deduct credits from tenant account (supports float credits)

        Args:
            db: Database session
            tenant_id: Tenant UUID
            amount: Amount of credits to deduct (can be float)
            call_session_id: Optional call session ID for tracking
            description: Optional description of the deduction

        Returns:
            Tuple of (success, remaining_credits)
        """
        billing_tenant_id = self._resolve_billing_tenant_id(db, tenant_id)
        tenant = db.query(Tenant).filter(Tenant.id == billing_tenant_id).first()
        if not tenant:
            logger.error("Tenant %s not found", billing_tenant_id)
            return (False, 0.0)

        current_credits = float(tenant.credits or 0)

        # Check if we have sufficient credits to deduct
        if current_credits <= 0:
            logger.warning("Insufficient credits for tenant %s: %s <= 0", billing_tenant_id, current_credits            )
            return (False, current_credits)

        # Deduct credits but never allow negative balance
        new_credits = max(0.0, current_credits - amount)
        tenant.credits = new_credits

        # If this deduction is associated with a specific call, accumulate per-call cost
        if call_session_id is not None:
            try:
                call_session = (
                    db.query(CallSession)
                    .filter(CallSession.id == call_session_id)
                    .first()
                )
                if call_session:
                    # Treat 'cost' as "total credits consumed for this call"
                    current_call_cost = float(call_session.cost or 0.0)
                    call_session.cost = current_call_cost + float(amount)

                    # Mirror the same value onto the associated CallLog (for dashboards)
                    call_log = (
                        db.query(CallLog)
                        .filter(CallLog.call_session_id == call_session_id)
                        .first()
                    )
                    if call_log:
                        current_log_cost = float(call_log.cost or 0.0)
                        call_log.cost = current_log_cost + float(amount)
            except Exception as e:
                # Do not block credit deduction if cost aggregation fails; just log it.
                logger.error("Failed to track per-call credit cost for session %s: %s", call_session_id, e,
                    exc_info=True,
                )

        # ✅ IMMEDIATE DATABASE UPDATE - Commit right away
        db.commit()
        db.refresh(tenant)

        # Check if credits reached 0 after deduction
        credits_exhausted = new_credits == 0 and current_credits > 0

        logger.info("✅ Deducted %.4f credits from tenant %s", amount, billing_tenant_id            + (f" (for sub-account {tenant_id})" if billing_tenant_id != tenant_id else "")
            + f". Remaining: {float(tenant.credits):.4f} (updated in DB). "
            f"Call: {call_session_id}. "
            f"Description: {description}"
        )
        logger.info("💰 CREDIT DEDUCTION: Deducted %.4f credits | Remaining: %.4f | Call: %s", amount, float(tenant.credits), call_session_id        )

        # Wallet auto-recharge: fire-and-forget, non-critical side effect. Must
        # never affect the deduction that just happened above (already
        # committed) or propagate an exception back to whatever call flow is
        # deducting credits — mirrors the fail-open pattern documented for
        # `CallSessionService.update_call_session_status()`'s other post-hooks.
        try:
            self._maybe_trigger_wallet_auto_recharge(
                db, billing_tenant_id, float(tenant.credits)
            )
        except Exception as exc:
            logger.error("Wallet auto-recharge trigger check failed for tenant %s: %s", billing_tenant_id, exc,
                exc_info=True,
            )

        # Return success=False if credits exhausted (call should end immediately)
        return (not credits_exhausted, float(tenant.credits))

    def _maybe_trigger_wallet_auto_recharge(
        self, db: Session, tenant_id: uuid.UUID, current_credits: float
    ) -> None:
        """
        After a credit deduction, check whether the tenant has wallet
        auto-recharge enabled and has just dropped below their configured
        `min_balance` — if so, atomically claim the trigger (cooldown-guarded)
        and dispatch the actual Stripe charge.

        Cooldown/idempotency: the live billing loop deducts credits every
        ~10s during an active call, so without protection a tenant sitting
        below their threshold would be charged repeatedly every tick until a
        recharge lands. `last_triggered_at` is updated to "now" via a single
        conditional `UPDATE ... WHERE last_triggered_at IS NULL OR <
        cutoff` BEFORE the Stripe charge is attempted (not after) — this is
        an atomic claim at the DB row level: two concurrent deductions (e.g.
        two active calls for the same tenant) racing this check will
        serialize on the row lock, and only the transaction that wins gets
        `rowcount == 1`, so at most one Stripe charge is dispatched per
        cooldown window even under concurrency. The claim is committed
        immediately, independent of whether the charge itself later succeeds
        or fails, so a slow/hanging Stripe call can never leave the window
        open for a second concurrent trigger.
        """
        config = (
            db.query(WalletAutoRechargeConfig)
            .filter(
                WalletAutoRechargeConfig.workspace_id == tenant_id,
                WalletAutoRechargeConfig.deleted_at.is_(None),
            )
            .first()
        )
        if not config or not config.enabled or not config.stripe_payment_method_id:
            return
        if current_credits >= float(config.min_balance):
            return

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.AUTO_RECHARGE_COOLDOWN_SECONDS)

        # The claim UPDATE also stamps last_trigger_status="pending" (and
        # clears any stale last_payment_intent_id/last_trigger_error from a
        # previous attempt) atomically with the cooldown claim itself — so
        # even if the process crashes before the Stripe call resolves, there
        # is a durable "we attempted a charge at `now`" record rather than
        # only a log line. `now` (the timestamp the claim just set) also
        # doubles as the per-attempt Stripe idempotency nonce below.
        claim_stmt = (
            sa_update(WalletAutoRechargeConfig)
            .where(
                WalletAutoRechargeConfig.id == config.id,
                WalletAutoRechargeConfig.enabled.is_(True),
                WalletAutoRechargeConfig.stripe_payment_method_id.isnot(None),
                or_(
                    WalletAutoRechargeConfig.last_triggered_at.is_(None),
                    WalletAutoRechargeConfig.last_triggered_at < cutoff,
                ),
            )
            .values(
                last_triggered_at=now,
                last_trigger_status="pending",
                last_payment_intent_id=None,
                last_trigger_error=None,
            )
        )
        result = db.execute(claim_stmt)
        db.commit()

        if result.rowcount != 1:
            # Still in cooldown (or config changed underneath us) — some other
            # deduction already claimed/attempted a trigger recently.
            return

        recharge_amount = Decimal(str(config.recharge_amount))
        stripe_payment_method_id = config.stripe_payment_method_id
        config_id = config.id

        # Per-attempt Stripe idempotency key, derived from the monotonic
        # timestamp the claim UPDATE above just wrote. Reusing this same key
        # on any retry within the same cooldown window (there won't be one,
        # by construction) or if this exact attempt is somehow re-dispatched
        # guarantees Stripe returns the original PaymentIntent instead of
        # creating a second, distinct charge.
        idempotency_key = f"auto-recharge:{config_id}:{now.isoformat()}"

        logger.warning("💳 WALLET AUTO-RECHARGE TRIGGERED: tenant %s balance %.4f < threshold %.4f; dispatching off-session charge of $%s (config %s, idempotency_key=%s).", tenant_id, current_credits, float(config.min_balance), recharge_amount, config_id, idempotency_key        )

        # Keep the trigger check itself cheap/synchronous (just the DB read +
        # claim above); only the actual blocking Stripe network call is
        # dispatched off the event loop when one is running, so a slow Stripe
        # response can never stall the 10s billing tick. `deduct_credits` is
        # called from both sync and async contexts — when there's no running
        # loop (a genuinely sync caller), fall back to a brief synchronous
        # Stripe call rather than dropping the charge.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            task_key = idempotency_key
            task = loop.create_task(
                asyncio.to_thread(
                    self._execute_auto_recharge_charge,
                    tenant_id,
                    config_id,
                    recharge_amount,
                    stripe_payment_method_id,
                    idempotency_key,
                )
            )
            self._active_auto_recharge_tasks[task_key] = task

            def _on_done(t: asyncio.Task, key: str = task_key) -> None:
                self._active_auto_recharge_tasks.pop(key, None)
                exc = None
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    logger.warning("💳 AUTO-RECHARGE task cancelled (key=%s).", key)
                    return
                if exc is not None:
                    logger.error("💳 AUTO-RECHARGE task raised an unhandled exception (key=%s): %s", key, exc,
                        exc_info=exc,
                    )

            task.add_done_callback(_on_done)
        else:
            self._execute_auto_recharge_charge(
                tenant_id,
                config_id,
                recharge_amount,
                stripe_payment_method_id,
                idempotency_key,
            )

    def _execute_auto_recharge_charge(
        self,
        tenant_id: uuid.UUID,
        config_id: uuid.UUID,
        recharge_amount: Decimal,
        stripe_payment_method_id: str,
        idempotency_key: str,
    ) -> None:
        """
        Perform the actual off-session Stripe charge for wallet auto-recharge
        and, on success, grant `recharge_amount` credits (1 credit = $1, same
        rate as everywhere else in this service). Opens its own DB session
        since this may run in a background thread (`asyncio.to_thread`)
        outside any request-scoped session — never raises, this is a
        fire-and-forget side effect, not part of any caller's control flow.
        Real money moves automatically here with no human in the loop, so
        every branch is logged as clearly as the existing credit-deduction
        logging in this file.

        `idempotency_key` is passed straight through to Stripe so a dropped
        response after Stripe already confirmed the PaymentIntent can never
        result in a second, distinct charge on retry.

        `WalletAutoRechargeConfig.last_trigger_status` /
        `last_payment_intent_id` / `last_trigger_error` are updated to their
        final outcome before returning on every branch (the "pending" row
        written by the claim UPDATE is never left stale) — on success this
        happens in the SAME commit as the credit grant.
        """
        from app.db.session import SessionLocal
        from app.services.stripe_service import (
            StripeService,
            StripeAuthenticationRequiredError,
        )

        db = SessionLocal()
        try:
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            if not tenant or not tenant.stripe_customer_id:
                error_detail = "tenant has no Stripe customer on file"
                logger.error("💳 AUTO-RECHARGE FAILED: tenant %s has no Stripe customer on file; cannot charge for auto-recharge (config %s).", tenant_id, config_id                )
                self._record_trigger_outcome(
                    db, config_id, status="failed", error=error_detail
                )
                return

            amount_cents = int(round(float(recharge_amount) * 100))
            try:
                intent = StripeService.create_off_session_payment_intent(
                    customer_id=tenant.stripe_customer_id,
                    payment_method_id=stripe_payment_method_id,
                    amount_cents=amount_cents,
                    currency="usd",
                    description=f"Wallet auto-recharge (+{recharge_amount} credits)",
                    metadata={
                        "tenant_id": str(tenant_id),
                        "purchase_type": "wallet_auto_recharge",
                        "auto_recharge_config_id": str(config_id),
                    },
                    idempotency_key=idempotency_key,
                )
            except StripeAuthenticationRequiredError as exc:
                logger.warning("⚠️ AUTO-RECHARGE AUTHENTICATION REQUIRED for tenant %s (config %s, amount $%s): Card requires 3DS/SCA authentication. Customer action required to re-authenticate card: %s", tenant_id, config_id, recharge_amount, exc                )
                self._record_trigger_outcome(
                    db,
                    config_id,
                    status="failed",
                    error=f"authentication_required: {str(exc)}"[:500],
                    payment_intent_id=exc.payment_intent_id,
                )
                return
            except Exception as exc:
                logger.error("💳 AUTO-RECHARGE CHARGE FAILED for tenant %s (config %s, amount $%s): %s", tenant_id, config_id, recharge_amount, exc,
                    exc_info=True,
                )
                self._record_trigger_outcome(
                    db, config_id, status="failed", error=str(exc)[:500]
                )
                return

            intent_status = getattr(intent, "status", None)
            if intent_status is None and isinstance(intent, dict):
                intent_status = intent.get("status")
            intent_id = getattr(intent, "id", None) or (
                intent.get("id") if isinstance(intent, dict) else None
            )

            if intent_status != "succeeded":
                logger.error("💳 AUTO-RECHARGE CHARGE NOT SUCCEEDED for tenant %s (config %s): PaymentIntent status=%r. No credits granted.", tenant_id, config_id, intent_status)
                self._record_trigger_outcome(
                    db,
                    config_id,
                    status="failed",
                    error=f"PaymentIntent status={intent_status!r}",
                    payment_intent_id=intent_id,
                )
                return

            tenant.credits = (tenant.credits or Decimal("0")) + recharge_amount
            # Capture the post-recharge balance now, before commit — avoids a
            # post-commit db.refresh() round-trip whose own failure would
            # otherwise fall into the outer except and mislabel an
            # already-committed success as "failed" (see credit_service.py
            # review history).
            new_balance = tenant.credits
            # Same transaction as the credit grant: update the trigger trail
            # to "succeeded" before the single commit below.
            self._record_trigger_outcome(
                db,
                config_id,
                status="succeeded",
                error=None,
                payment_intent_id=intent_id,
                commit=False,
            )
            db.commit()

            logger.warning("💳 AUTO-RECHARGE SUCCEEDED: tenant %s charged $%s off-session (PaymentIntent %s); credits now %.4f.", tenant_id, recharge_amount, intent_id, float(new_balance)            )
        except Exception as exc:
            logger.error("💳 AUTO-RECHARGE unexpected error for tenant %s (config %s): %s", tenant_id, config_id, exc,
                exc_info=True,
            )
            try:
                # A failure inside the try block (e.g. the commit itself)
                # leaves this session's transaction in a "pending rollback"
                # state — any further db.execute() without rolling back
                # first raises PendingRollbackError instead of persisting
                # the recovery write, silently stranding the row at
                # last_trigger_status="pending" forever.
                db.rollback()
                self._record_trigger_outcome(
                    db, config_id, status="failed", error=str(exc)[:500]
                )
            except Exception:
                logger.error("💳 AUTO-RECHARGE also failed to persist trigger outcome for config %s", config_id,
                    exc_info=True,
                )
        finally:
            db.close()

    @staticmethod
    def _record_trigger_outcome(
        db: Session,
        config_id: uuid.UUID,
        *,
        status: str,
        error: str | None,
        payment_intent_id: str | None = None,
        commit: bool = True,
    ) -> None:
        """
        Update `WalletAutoRechargeConfig.last_trigger_status` /
        `last_trigger_error` / `last_payment_intent_id` for `config_id`.
        `commit=False` lets the success path fold this update into the same
        transaction/commit as the credit grant in `_execute_auto_recharge_charge`.
        """
        values: Dict[str, Any] = {
            "last_trigger_status": status,
            "last_trigger_error": error,
        }
        if payment_intent_id is not None:
            values["last_payment_intent_id"] = payment_intent_id
        db.execute(
            sa_update(WalletAutoRechargeConfig)
            .where(WalletAutoRechargeConfig.id == config_id)
            .values(**values)
        )
        if commit:
            db.commit()

    async def start_credit_monitoring(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        agent_id: uuid.UUID,
    ):
        """
        Start monitoring and deducting credits for an active call (Vapi-style real-time)

        Args:
            db: Database session (used only for initial setup, monitoring creates its own session)
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            agent_id: Agent UUID
        """
        if str(call_session_id) in self._active_monitors:
            logger.warning("Credit monitor already active for call %s", call_session_id)
            return

        # Agent/model lookup kept for logging context, and to resolve which
        # surcharges (if any) apply for the full duration of this call.
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            logger.error("Agent %s not found", agent_id)
            return

        model_name = agent.model.model_name if agent.model else "unknown"
        credits_per_minute = float(self.get_effective_rate_per_minute(db, tenant_id))

        # Determined once per call (agents don't change mid-call) so the 10s tick
        # loop never needs to re-resolve the agent's TTS runtime just to check this.
        tts_provider_slug: str | None = None
        is_byo_elevenlabs = False
        try:
            from app.core.agent_runtime import resolve_tts_runtime

            tts_runtime = resolve_tts_runtime(agent, db=db)
            tts_provider_slug = tts_runtime.adapter_slug
            is_byo_elevenlabs = tts_runtime.is_byo_elevenlabs
        except Exception as exc:
            logger.warning("Failed to resolve TTS runtime for agent %s while starting credit monitor (ElevenLabs surcharge check skipped): %s", agent_id, exc            )
        surcharges = self.get_active_surcharges(
            model_name=model_name,
            tts_provider_slug=tts_provider_slug,
            is_byo_elevenlabs=is_byo_elevenlabs,
        )

        # Initialize tracking for this call
        call_start_time = datetime.now(timezone.utc)
        call_id_str = str(call_session_id)
        self._call_start_times[call_id_str] = call_start_time
        self._last_deduction_time[call_id_str] = call_start_time
        self._accumulated_seconds[call_id_str] = 0.0

        logger.info("Starting credit monitor for call %s. Model: %s, Rate: %s credits/min, Surcharges: %s (Vapi-style per-second billing with float precision)", call_session_id, model_name, credits_per_minute, [s.key for s in surcharges] or 'none'        )
        logger.info("💰 CREDIT MONITORING STARTED: Call %s | Model: %s | Rate: %s credits/min | Float credits enabled", call_session_id, model_name, credits_per_minute        )

        # Create monitoring task (will create its own DB session for thread-safety).
        # NOTE: tts_provider_slug/is_byo_elevenlabs (not a fixed `surcharges` list)
        # are passed through so the tick loop can re-derive which surcharges are
        # CURRENTLY active every 10s by reading CallSession.call_metadata
        # ["openai_realtime_fallback"] fresh each time — a call configured for an
        # OpenAI Realtime model can fall back mid-call to the legacy TTS pipeline
        # (see VoiceOrchestrator._fallback_to_legacy_pipeline_openai), at which
        # point the realtime surcharge must stop and the ElevenLabs surcharge (if
        # applicable) must start applying for the remainder of the call.
        task = asyncio.create_task(
            self._monitor_and_deduct_credits(
                call_session_id,
                tenant_id,
                model_name,
                credits_per_minute,
                tts_provider_slug=tts_provider_slug,
                is_byo_elevenlabs=is_byo_elevenlabs,
            )
        )
        self._active_monitors[call_id_str] = task

    def _resolve_current_surcharges(
        self,
        call_session: CallSession,
        model_name: str | None,
        tts_provider_slug: str | None,
        is_byo_elevenlabs: bool,
    ) -> list[ActiveSurcharge]:
        """
        Re-derive which surcharges are active for `call_session` RIGHT NOW —
        called on every monitoring tick (and at finalization) rather than
        once at call-start, so a mid-call OpenAI Realtime fallback (see
        `VoiceOrchestrator._fallback_to_legacy_pipeline_openai`) is picked up
        for the remainder of the call. `call_session` is re-queried by the
        caller on every tick already, so this reads whatever the orchestrator
        most recently persisted with no extra DB round-trip.
        """
        realtime_fallen_back = False
        try:
            meta = call_session.call_metadata or {}
            if isinstance(meta, dict):
                fallback_meta = meta.get("openai_realtime_fallback")
                if isinstance(fallback_meta, dict):
                    realtime_fallen_back = bool(fallback_meta.get("fell_back"))
                else:
                    realtime_fallen_back = bool(fallback_meta)
        except Exception as exc:
            logger.warning("Failed to read openai_realtime_fallback from call_metadata for call %s: %s", call_session.id, exc            )

        return self.get_active_surcharges(
            model_name=model_name,
            tts_provider_slug=tts_provider_slug,
            is_byo_elevenlabs=is_byo_elevenlabs,
            realtime_fallen_back=realtime_fallen_back,
        )

    async def _monitor_and_deduct_credits(
        self,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credits_per_minute: float,
        tts_provider_slug: str | None = None,
        is_byo_elevenlabs: bool = False,
    ):
        """
        Background task to monitor and deduct credits during call
        Vapi-style: Real-time per-second billing with accurate duration tracking, plan-aware:
        minutes within the tenant's remaining included-minutes allowance are recorded as
        usage but not charged; only minutes beyond the allowance are deducted from credits.

        Args:
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            model_name: Model name — the DB-configured model, used both for
                logging and (via `_resolve_current_surcharges`) as the
                starting point for surcharge resolution each tick.
            credits_per_minute: Flat effective $/min (== credits/min) rate for this tenant
            tts_provider_slug: Agent's resolved TTS adapter slug (see
                `app.core.agent_runtime.resolve_tts_runtime`), fixed for the
                call (agents don't change mid-call).
            is_byo_elevenlabs: True when the agent uses a tenant-supplied BYO
                ElevenLabs key — exempts the ElevenLabs surcharge.

        Surcharges are NOT resolved once and cached — `_resolve_current_surcharges`
        is called fresh on every tick (and at finalization) so a mid-call OpenAI
        Realtime fallback (`VoiceOrchestrator._fallback_to_legacy_pipeline_openai`,
        persisted on `CallSession.call_metadata["openai_realtime_fallback"]`) is
        picked up for the remainder of the call: the realtime surcharge stops
        applying and the ElevenLabs surcharge (if applicable) starts applying
        from that point onward.
        """
        call_id_str = str(call_session_id)

        # Create dedicated DB session for this async task (Vapi-style: proper session handling)
        from app.db.session import SessionLocal

        db = SessionLocal()

        # Resolved once per call (wallet-sharing config doesn't change mid-call):
        # which tenant's Tenant.credits actually gets debited. `tenant_id` itself
        # is still used everywhere else in this loop (included-minutes allowance
        # lookup, UsageRecord attribution) — only the credit-balance
        # check/deduction is redirected to the resolved billing tenant.
        billing_tenant_id = self._resolve_billing_tenant_id(db, tenant_id)

        try:
            while True:
                # Wait for monitoring interval (real-time checks)
                await asyncio.sleep(self.MONITORING_INTERVAL)

                # Check if call is still active
                call_session = (
                    db.query(CallSession)
                    .filter(CallSession.id == call_session_id)
                    .first()
                )

                if not call_session:
                    logger.info("Call session %s not found, stopping monitor", call_session_id)
                    break

                # Re-derive active surcharges every tick (not cached from call-start)
                # so a mid-call OpenAI Realtime fallback is reflected immediately.
                surcharges = self._resolve_current_surcharges(
                    call_session, model_name, tts_provider_slug, is_byo_elevenlabs
                )

                # Only deduct for active call statuses
                if call_session.status not in [
                    "active",
                    "in-progress",
                    "connected",
                    "answered",
                ]:
                    # Call ended - do final deduction for remaining time
                    await self._finalize_call_credits(
                        db,
                        call_session_id,
                        tenant_id,
                        model_name,
                        credits_per_minute,
                        call_session,
                        tts_provider_slug=tts_provider_slug,
                        is_byo_elevenlabs=is_byo_elevenlabs,
                        billing_tenant_id=billing_tenant_id,
                    )
                    break

                # Calculate duration since last deduction
                current_time = datetime.now(timezone.utc)
                last_deduction = self._last_deduction_time.get(call_id_str)

                if not last_deduction:
                    last_deduction = self._call_start_times.get(
                        call_id_str, current_time
                    )
                    self._last_deduction_time[call_id_str] = last_deduction

                # Calculate seconds since last deduction
                elapsed_seconds = (current_time - last_deduction).total_seconds()

                if elapsed_seconds > 0:
                    from app.services.billing_service import BillingService

                    # Accumulate seconds
                    accumulated = self._accumulated_seconds.get(call_id_str, 0.0)
                    accumulated += elapsed_seconds
                    self._accumulated_seconds[call_id_str] = accumulated

                    # Split accumulated time into the portion still covered by the tenant's
                    # remaining included-plan minutes ("free") and the portion beyond it
                    # ("billable") — a call must never be force-ended while fully within
                    # its free allowance, even at a zero credit balance.
                    _, _, remaining_minutes = (
                        BillingService.get_included_minutes_status(db, tenant_id)
                    )
                    remaining_seconds = float(remaining_minutes) * 60.0
                    free_seconds = min(accumulated, remaining_seconds)
                    billable_seconds = max(0.0, accumulated - free_seconds)

                    # Calculate credits owed for only the billable (overage) portion
                    credits_to_deduct_float = self.calculate_credits_for_duration(
                        billable_seconds, credits_per_minute
                    )

                    # Active surcharges (OpenAI Realtime, ElevenLabs TTS, or both,
                    # summed additively) apply to the FULL accumulated seconds
                    # (both the free-allowance and billable portions) — never
                    # waived by the included-minutes pool, unlike the base rate above.
                    surcharge_float, surcharge_desc = self._describe_surcharges(
                        accumulated, surcharges
                    )

                    total_deduction_float = credits_to_deduct_float + surcharge_float

                    deduction_success = True
                    remaining_credits = self.get_tenant_credits(db, billing_tenant_id)

                    if total_deduction_float >= 0.01:
                        description = (
                            f"Call duration: {accumulated:.1f}s "
                            f"(billable {billable_seconds:.1f}s) - Model: {model_name}"
                        )
                        if surcharge_float > 0:
                            description += f" + {surcharge_desc}"
                        # ✅ Deduct accumulated overage + active surcharge credits in one
                        # combined deduction (updates DB immediately). Targets the
                        # RESOLVED billing tenant (parent, if wallet sharing is on) —
                        # never the calling tenant_id directly.
                        deduction_success, remaining_credits = self.deduct_credits(
                            db=db,
                            tenant_id=billing_tenant_id,
                            amount=total_deduction_float,
                            call_session_id=call_session_id,
                            description=description,
                        )

                    # Record the FULL accumulated minutes as usage regardless of whether
                    # anything was charged, so the included-minutes pool actually depletes.
                    try:
                        BillingService.record_call_minutes(
                            db=db,
                            tenant_id=tenant_id,
                            call_id=call_session_id,
                            minutes=Decimal(str(accumulated)) / Decimal("60"),
                            credits_charged=(
                                Decimal(str(total_deduction_float))
                                if total_deduction_float >= 0.01
                                else Decimal("0")
                            ),
                        )
                    except Exception as exc:
                        logger.error("Failed to record call minutes for %s: %s", call_session_id, exc                        )

                    self._accumulated_seconds[call_id_str] = 0.0
                    self._last_deduction_time[call_id_str] = current_time

                    # Only force-end the call if there was a real charge (billable overage
                    # and/or realtime surcharge) AND the deduction failed or exhausted the
                    # tenant's balance.
                    if total_deduction_float >= 0.01 and (
                        not deduction_success or remaining_credits <= 0
                    ):
                        # ✅ CREDITS FINISHED - END CALL IMMEDIATELY
                        logger.warning("⛔ Credits finished for call %s. Remaining credits: %s. Ending call immediately.", call_session_id, remaining_credits                        )

                        # Update call session status immediately
                        call_session.status = "completed"
                        call_session.end_time = datetime.now(timezone.utc)
                        call_session.ended_reason = "Insufficient credits"

                        if call_session.start_time:
                            duration = (
                                call_session.end_time - call_session.start_time
                            ).total_seconds()
                            call_session.duration = int(duration)

                        # ✅ IMMEDIATE DATABASE UPDATE
                        db.commit()

                        # Try to end the Twilio call immediately
                        try:
                            await self._end_twilio_call(call_session.twilio_call_sid)
                            logger.info(
                                "✅ Twilio call ended immediately due to insufficient credits"
                            )
                        except Exception as e:
                            logger.error("Error ending Twilio call: %s", e)

                        # Broadcast call ended event
                        try:
                            from app.routers.general_websocket import (
                                broadcast_call_status_update,
                            )

                            await broadcast_call_status_update(
                                call_session_id=str(call_session_id),
                                status="completed",
                                metadata={
                                    "reason": "insufficient_credits",
                                    "message": "Call ended due to insufficient credits",
                                    "remaining_credits": remaining_credits,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        except Exception as e:
                            logger.error("Error broadcasting call end event: %s", e)

                        # Stop monitoring immediately
                        break

                    logger.info("Call %s: %.1fs elapsed (%.1fs billable, %.4f credits deducted [base %.4f + surcharge %.4f]). Remaining: %.4f", call_session_id, accumulated, billable_seconds, total_deduction_float, credits_to_deduct_float, surcharge_float, remaining_credits                    )

        except asyncio.CancelledError:
            logger.info("Credit monitor cancelled for call %s", call_session_id)
        except Exception as e:
            logger.error("Error in credit monitoring for call %s: %s", call_session_id, e)
            import traceback

            traceback.print_exc()

        finally:
            # Final deduction for any remaining time
            try:
                call_session = (
                    db.query(CallSession)
                    .filter(CallSession.id == call_session_id)
                    .first()
                )
                if call_session:
                    await self._finalize_call_credits(
                        db,
                        call_session_id,
                        tenant_id,
                        model_name,
                        credits_per_minute,
                        call_session,
                        tts_provider_slug=tts_provider_slug,
                        is_byo_elevenlabs=is_byo_elevenlabs,
                        billing_tenant_id=billing_tenant_id,
                    )
            except Exception as e:
                logger.error("Error in final credit deduction: %s", e)
            finally:
                # Close DB session (Vapi-style: proper cleanup)
                db.close()

            # Clean up monitor and tracking
            if call_id_str in self._active_monitors:
                del self._active_monitors[call_id_str]
            if call_id_str in self._call_start_times:
                del self._call_start_times[call_id_str]
            if call_id_str in self._last_deduction_time:
                del self._last_deduction_time[call_id_str]
            if call_id_str in self._accumulated_seconds:
                del self._accumulated_seconds[call_id_str]

            logger.info("Credit monitor stopped for call %s", call_session_id)

    async def _finalize_call_credits(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credits_per_minute: float,
        call_session: CallSession,
        tts_provider_slug: str | None = None,
        is_byo_elevenlabs: bool = False,
        billing_tenant_id: uuid.UUID | None = None,
    ):
        """
        Finalize credits for call end - deduct any remaining accumulated time (Vapi-style),
        plan-aware: only the portion beyond the tenant's remaining included minutes is charged
        at the base rate. Active surcharges (when applicable) apply to the full accumulated
        seconds regardless of the included-minutes split — mirrors
        `_monitor_and_deduct_credits`.

        Surcharges are re-resolved from `call_session.call_metadata` here (via
        `_resolve_current_surcharges`) rather than reused from an earlier tick,
        so a fallback that happened since the last tick is still reflected in
        the final deduction.
        """
        from app.services.billing_service import BillingService

        # `billing_tenant_id` is passed in by the caller (resolved ONCE at
        # monitoring-loop start) rather than re-derived here. Re-deriving it
        # independently at call-end would let a wallet-sharing toggle that
        # fires mid-call split a single call's cost across two tenants'
        # wallets (some ticks billed the sub-account, the final catch-up
        # deduction billed the parent, or vice versa) — the whole call must
        # be billed against exactly one resolved tenant. The calling
        # tenant_id is still used for the included-minutes allowance lookup
        # and UsageRecord attribution below regardless.
        if billing_tenant_id is None:
            billing_tenant_id = self._resolve_billing_tenant_id(db, tenant_id)

        surcharges = self._resolve_current_surcharges(
            call_session, model_name, tts_provider_slug, is_byo_elevenlabs
        )
        call_id_str = str(call_session_id)
        accumulated = self._accumulated_seconds.get(call_id_str, 0.0)

        if accumulated > 0:
            _, _, remaining_minutes = BillingService.get_included_minutes_status(
                db, tenant_id
            )
            remaining_seconds = float(remaining_minutes) * 60.0
            free_seconds = min(accumulated, remaining_seconds)
            billable_seconds = max(0.0, accumulated - free_seconds)

            # Calculate final credits for the billable (overage) portion only
            final_credits = self.calculate_credits_for_duration(
                billable_seconds, credits_per_minute
            )

            # Active surcharges apply to the FULL accumulated seconds.
            surcharge_final, surcharge_desc = self._describe_surcharges(
                accumulated, surcharges
            )

            total_final = final_credits + surcharge_final
            credits_charged = Decimal("0")

            if total_final >= 0.01:  # Only deduct if at least 0.01 credits
                description = f"Final call duration: {accumulated:.1f}s (billable {billable_seconds:.1f}s) - Model: {model_name}"
                if surcharge_final > 0:
                    description += f" + {surcharge_desc}"
                success, remaining_credits = self.deduct_credits(
                    db=db,
                    tenant_id=billing_tenant_id,
                    amount=total_final,
                    call_session_id=call_session_id,
                    description=description,
                )
                credits_charged = Decimal(str(total_final))

                logger.info("Call %s: Final deduction of %.4f credits (base %.4f + surcharge %.4f) for %.1fs. Remaining: %.4f", call_session_id, total_final, final_credits, surcharge_final, accumulated, remaining_credits                )
                logger.info("💰 FINAL CREDIT DEDUCTION: %.4f credits for %.1fs | Remaining: %.4f | Call: %s", total_final, accumulated, remaining_credits, call_session_id                )

            try:
                BillingService.record_call_minutes(
                    db=db,
                    tenant_id=tenant_id,
                    call_id=call_session_id,
                    minutes=Decimal(str(accumulated)) / Decimal("60"),
                    credits_charged=credits_charged,
                )
            except Exception as exc:
                logger.error("Failed to record final call minutes for %s: %s", call_session_id, exc                )

            self._accumulated_seconds[call_id_str] = 0.0

    async def _end_twilio_call(self, twilio_call_sid: str):
        """
        End a Twilio call

        Args:
            twilio_call_sid: Twilio call SID
        """
        if not twilio_call_sid:
            return

        try:
            from app.services.twilio_service import twilio_service

            twilio_service.end_call(twilio_call_sid)
            logger.info("Successfully ended Twilio call %s", twilio_call_sid)
        except Exception as e:
            logger.error("Error ending Twilio call %s: %s", twilio_call_sid, e)

    def stop_credit_monitoring(self, call_session_id: uuid.UUID):
        """
        Stop monitoring credits for a call

        Args:
            call_session_id: Call session UUID
        """
        call_id_str = str(call_session_id)
        if call_id_str in self._active_monitors:
            task = self._active_monitors[call_id_str]
            task.cancel()
            del self._active_monitors[call_id_str]
            logger.info("Stopped credit monitor for call %s", call_session_id)

    def get_active_monitors(self) -> Dict[str, Any]:
        """Get list of active credit monitors"""
        return {
            call_id: {"active": not task.done()}
            for call_id, task in self._active_monitors.items()
        }


# Global instance
credit_service = CreditService()
