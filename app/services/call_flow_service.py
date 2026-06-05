"""Call Flow service — ALL versioning and business logic lives here."""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.prompt_version import PromptVersion
from app.repositories.call_flow_repository import CallFlowRepository
from app.repositories.prompt_version_repository import PromptVersionRepository
from app.schemas.agent import agent_to_out
from app.schemas.call_flow import CallFlowCreate, CallFlowListResponse, CallFlowListItem, CallFlowOut, AgentRef, CallFlowUpdate
from app.schemas.prompt_version import PromptVersionOut
from app.utils.gemini_prompt_sanitizer import sanitize_prompt_for_gemini

_MAX_VERSIONS = 50


class CallFlowService:
    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_agent_or_404(
        self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Agent:
        agent = db.execute(
            select(Agent).where(
                Agent.id == agent_id,
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found in workspace",
            )
        return agent

    def _get_flow_or_404(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        load_relations: bool = False,
    ) -> CallFlow:
        repo = CallFlowRepository(db)
        flow = repo.find_by_id(
            flow_id, tenant_id=tenant_id, load_relations=load_relations
        )
        if flow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} not found",
            )
        return flow

    def _insert_prompt_version(
        self,
        db: Session,
        flow_id: uuid.UUID,
        prompt_text: str,
        notes: Optional[str],
        *,
        current_prompt_id: Optional[uuid.UUID] = None,
    ) -> PromptVersion:
        """Create a PromptVersion row, run gemini sanitizer, enforce 50-cap.

        Never prunes the version identified by *current_prompt_id* — it is the
        flow's active version and must remain reachable even after pruning.
        """
        gemini_prompt = sanitize_prompt_for_gemini(prompt_text)
        pv_repo = PromptVersionRepository(db)
        version = pv_repo.create(
            {
                "flow_id": flow_id,
                "prompt_text": prompt_text,
                "gemini_prompt": gemini_prompt,
                "notes": notes,
            }
        )

        # Enforce 50-version cap; loop in case of edge-case drift
        count = pv_repo.count_by_flow(flow_id)
        while count > _MAX_VERSIONS:
            oldest = pv_repo.find_oldest_deletable(flow_id, current_prompt_id)
            if oldest is None:
                logger.warning(
                    "Cannot prune versions for flow %s: all %d versions are protected",
                    flow_id,
                    count,
                )
                break
            logger.info(
                "Pruning oldest prompt version for flow %s: deleted version %s",
                flow_id,
                oldest.id,
            )
            pv_repo.delete(oldest)
            count -= 1

        return version

    def _prompt_changed(
        self, db: Session, flow: CallFlow, new_prompt: str
    ) -> bool:
        """Return True if new_prompt text differs from the currently active version."""
        if flow.current_prompt_id is None:
            return True
        pv_repo = PromptVersionRepository(db)
        current = pv_repo.find_by_id(flow.current_prompt_id)
        if current is None:
            return True
        return current.prompt_text != new_prompt

    # ── Serialization helpers ─────────────────────────────────────────────

    def _version_to_out(self, v: PromptVersion) -> PromptVersionOut:
        return PromptVersionOut.model_validate(v)

    def _flow_to_out(self, db: Session, flow: CallFlow) -> dict:
        pv_repo = PromptVersionRepository(db)
        versions = pv_repo.find_by_flow(flow.id, order_desc=True)

        # Full AgentOut on detail endpoints (POST 201, GET, PUT)
        agent_dict: Optional[dict] = None
        if flow.agent:
            agent_dict = agent_to_out(flow.agent).model_dump(by_alias=True, mode="json")

        out = CallFlowOut(
            id=flow.id,
            name=flow.name,
            direction=flow.direction,
            agent_id=flow.agent_id,
            agent=agent_dict,
            welcome_message_type=flow.welcome_message_type,
            custom_welcome_message=flow.custom_welcome_message,
            current_prompt_id=flow.current_prompt_id,
            prompt_versions=[self._version_to_out(v) for v in versions],
            flow_data=flow.flow_data,
            settings=flow.settings,
            created_at=flow.created_at,
            updated_at=flow.updated_at,
        )
        return out.model_dump(by_alias=True, mode="json")

    def _flow_to_list_item(self, flow: CallFlow) -> dict:
        agent_ref: Optional[AgentRef] = None
        if flow.agent:
            agent_ref = AgentRef.model_validate(flow.agent)

        item = CallFlowListItem(
            id=flow.id,
            name=flow.name,
            direction=flow.direction,
            agent_id=flow.agent_id,
            agent=agent_ref,
            welcome_message_type=flow.welcome_message_type,
            custom_welcome_message=flow.custom_welcome_message,
            current_prompt_id=flow.current_prompt_id,
            flow_data=flow.flow_data,
            settings=flow.settings,
            created_at=flow.created_at,
            updated_at=flow.updated_at,
        )
        return item.model_dump(by_alias=True, mode="json")

    # ── Public API ────────────────────────────────────────────────────────

    def create_flow(
        self, db: Session, tenant_id: uuid.UUID, body: CallFlowCreate
    ) -> dict:
        self._get_agent_or_404(db, body.agent_id, tenant_id)
        repo = CallFlowRepository(db)

        flow_data_dict = body.flow_data.model_dump() if body.flow_data else None

        flow = repo.create(
            {
                "tenant_id": tenant_id,
                "agent_id": body.agent_id,
                "name": body.name,
                "direction": body.direction.value,
                "welcome_message_type": body.welcome_message_type,
                "custom_welcome_message": body.custom_welcome_message,
                "flow_data": flow_data_dict,
                "settings": body.settings,
            }
        )

        if body.prompt and body.prompt.strip():
            # No existing current yet on create; pass None so nothing is protected
            version = self._insert_prompt_version(
                db, flow.id, body.prompt, body.notes, current_prompt_id=None
            )
            flow = repo.update(flow, {"current_prompt_id": version.id})

        db.commit()
        db.refresh(flow)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    def get_flow(
        self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> dict:
        flow = self._get_flow_or_404(db, flow_id, tenant_id, load_relations=True)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    def list_flows(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        page: int,
        limit: int,
    ) -> dict:
        repo = CallFlowRepository(db)
        rows, total = repo.find_by_workspace(tenant_id, page=page, limit=limit)
        response = CallFlowListResponse(
            data=[],
            total=total,
            page=page,
            page_size=limit,
        )
        return {
            "data": [self._flow_to_list_item(f) for f in rows],
            "total": total,
            "page": page,
            "pageSize": limit,
        }

    def update_flow(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: CallFlowUpdate,
    ) -> dict:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        repo = CallFlowRepository(db)

        # Validate new agent if being changed
        if body.agent_id is not None:
            self._get_agent_or_404(db, body.agent_id, tenant_id)

        # Gather scalar field updates
        scalar_updates: dict = {}
        if body.name is not None:
            scalar_updates["name"] = body.name
        if body.direction is not None:
            scalar_updates["direction"] = body.direction.value
        if body.agent_id is not None:
            scalar_updates["agent_id"] = body.agent_id
        if body.welcome_message_type is not None:
            scalar_updates["welcome_message_type"] = body.welcome_message_type
        if body.custom_welcome_message is not None:
            scalar_updates["custom_welcome_message"] = body.custom_welcome_message
        if body.flow_data is not None:
            scalar_updates["flow_data"] = body.flow_data.model_dump()
        if body.settings is not None:
            scalar_updates["settings"] = body.settings

        # Prompt versioning logic
        if body.prompt is not None and body.prompt.strip():
            # New prompt text → create version only if it differs from current
            if self._prompt_changed(db, flow, body.prompt):
                # Protect the existing active version from pruning
                version = self._insert_prompt_version(
                    db, flow.id, body.prompt, body.notes,
                    current_prompt_id=flow.current_prompt_id,
                )
                scalar_updates["current_prompt_id"] = version.id
        elif body.current_prompt_id is not None:
            # Explicit rollback — no prompt text provided
            pv_repo = PromptVersionRepository(db)
            target = pv_repo.find_by_id(body.current_prompt_id)
            if target is None or target.flow_id != flow.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="currentPromptId does not belong to this flow",
                )
            scalar_updates["current_prompt_id"] = body.current_prompt_id

        if scalar_updates:
            flow = repo.update(flow, scalar_updates)

        db.commit()
        db.refresh(flow)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    def delete_flow(
        self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        # 409 if any active call session is using this flow
        active = db.execute(
            select(CallSession).where(
                CallSession.call_flow_id == flow_id,
                CallSession.tenant_id == tenant_id,
                CallSession.status == "active",
            ).limit(1)
        ).scalar_one_or_none()
        if active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete a call flow that has active calls in progress",
            )

        repo = CallFlowRepository(db)
        repo.soft_delete(flow)
        db.commit()

    def get_prompt_versions(
        self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[dict]:
        self._get_flow_or_404(db, flow_id, tenant_id)
        pv_repo = PromptVersionRepository(db)
        versions = pv_repo.find_by_flow(flow_id, order_desc=True)
        return [
            self._version_to_out(v).model_dump(by_alias=True, mode="json")
            for v in versions
        ]


call_flow_service = CallFlowService()
