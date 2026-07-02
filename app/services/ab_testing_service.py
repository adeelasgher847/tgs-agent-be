"""A/B prompt testing — variant assignment at dispatch time and results aggregation.

Random assignment happens once, at call-session creation, and is persisted onto
``CallSession.ab_variant`` (plus the resolved prompt text in ``call_metadata``)
before any LLM request is made — so the variant is always known even if the
call fails mid-way, and never changes for the lifetime of the call.
"""
from __future__ import annotations

import random
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.prompt_version import PromptVersion


class AbTestingService:
    def pick_variant(self, call_flow: CallFlow) -> Optional[str]:
        """Return 'a' or 'b' if the flow is eligible for A/B testing, else None."""
        if not call_flow.ab_test_enabled:
            return None
        if not call_flow.ab_prompt_a_id or not call_flow.ab_prompt_b_id:
            return None
        split_ratio = float(call_flow.ab_split_ratio)
        return "a" if random.random() < split_ratio else "b"

    def get_variant_prompt_text(
        self, db: Session, call_flow: CallFlow, variant: str
    ) -> Optional[str]:
        prompt_id = (
            call_flow.ab_prompt_a_id if variant == "a" else call_flow.ab_prompt_b_id
        )
        if prompt_id is None:
            return None
        version = db.execute(
            select(PromptVersion).where(PromptVersion.id == prompt_id)
        ).scalar_one_or_none()
        return version.prompt_text if version else None

    def assign_and_lock_variant(
        self, db: Session, call_session: CallSession, call_flow: CallFlow
    ) -> None:
        """Assign a variant (if the flow has A/B testing enabled) and persist it
        onto the call session immediately, before any LLM request."""
        variant = self.pick_variant(call_flow)
        if variant is None:
            return

        prompt_text = self.get_variant_prompt_text(db, call_flow, variant)
        if prompt_text is None:
            logger.warning(
                "A/B test enabled on flow %s but prompt version for variant '%s' "
                "could not be resolved; skipping variant assignment",
                call_flow.id,
                variant,
            )
            return

        call_session.ab_variant = variant
        call_session.call_metadata = {
            **(call_session.call_metadata or {}),
            "ab_prompt_text": prompt_text,
        }
        db.commit()
        db.refresh(call_session)
        logger.info(
            "A/B variant '%s' assigned to call session %s (flow=%s)",
            variant,
            call_session.id,
            call_flow.id,
        )


ab_testing_service = AbTestingService()
