"""Call Flow service — ALL versioning and business logic lives here."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from scipy.stats import chi2_contingency
from sqlalchemy import case, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.inbound_rule import InboundRule, InboundRuleSet
from app.models.model import Model
from app.models.prompt_version import PromptVersion
from app.repositories.call_flow_repository import CallFlowRepository
from app.repositories.prompt_version_repository import PromptVersionRepository
from app.services.call_history_service import call_history_service
from app.schemas.agent import agent_to_out
from app.schemas.ab_testing import (
    AbResultsResponse,
    AbTestResponse,
    AbTestUpdate,
    AbTestWinnerUpdate,
    VariantMetrics,
)
from app.schemas.call_flow import (
    CallFlowCreate,
    CallFlowListItem,
    CallFlowOut,
    AgentRef,
    CallScreeningSettingsResponse,
    CallScreeningSettingsUpdate,
    CallTimingSettingsResponse,
    CallTimingSettingsUpdate,
    ComplianceDetectionSettingsResponse,
    ComplianceDetectionSettingsUpdate,
    DataRetentionSettingsResponse,
    DataRetentionSettingsUpdate,
    CallerMemorySettingsResponse,
    CallerMemorySettingsUpdate,
    CallFlowSettingsUpdate,
    CallFlowUpdate,
    FlowDataListItem,
    FlowDataResponse,
    FlowDataSaveResponse,
    FlowDataUpdate,
    FlowInboundRulesResponse,
    FlowInboundRulesUpdate,
    FlowValidationError,
    FlowValidationErrorItem,
    FlowValidationResponse,
    InboundRedirectSettingsResponse,
    InboundRedirectSettingsUpdate,
    RedirectCondition,
    IVRDTMFSettingsResponse,
    IVRDTMFSettingsUpdate,
    MetadataSettingsResponse,
    MetadataSettingsUpdate,
    PaginatedFlowDataResponse,
    PostCallActionsSettingsResponse,
    PostCallActionsSettingsUpdate,
    PostCallAnalysisSettingsResponse,
    PostCallAnalysisSettingsUpdate,
    RecordingSettingsResponse,
    RecordingSettingsUpdate,
    SystemWebhooksSettingsResponse,
    SystemWebhooksSettingsUpdate,
    SystemWebhookDeliveryOut,
    PaginatedSystemWebhookDeliveries,
    VoicemailSettingsResponse,
    VoicemailSettingsUpdate,
)
from app.models.system_webhook_log import SystemWebhookDeliveryLog
from app.core.db_encryption import encrypt_webhook_headers
from app.schemas.prompt_version import PromptVersionOut
from app.services.flow_graph_service import compile_graph, validate_graph
from app.utils.gemini_prompt_sanitizer import sanitize_prompt_for_gemini

_MAX_VERSIONS = 50
_AB_MIN_CALLS_FOR_SIGNIFICANCE = 30
_AB_SIGNIFICANCE_P_VALUE = 0.05


def _strip_flow_data_for_readonly(flow_data: dict) -> dict:
    """Return a sanitised copy of *flow_data* safe for readonly callers.

    Keeps per-node: ``id``, ``type``, ``position``, ``data.label``. Every other
    ``node.data`` field — prompt text, phone/extension numbers, webhook URLs,
    etc. — is dropped via allow-list (only ``label`` is ever copied over), so
    a new sensitive node field never needs to be enumerated here to stay safe.
    Keeps edges unchanged (structural info only, no secrets).
    Keeps top-level: ``entry_node_id``, ``version``, ``compiled_at``.
    The caller is responsible for setting ``flow_data_compiled=None`` separately.
    """
    sanitised_nodes = []
    for node in flow_data.get("nodes", []):
        raw_data: dict = node.get("data") or {}
        # Allow-list approach: start from an empty dict and keep only label
        safe_data: dict = {}
        if "label" in raw_data:
            safe_data["label"] = raw_data["label"]

        sanitised_nodes.append(
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "position": node.get("position"),
                "data": safe_data,
            }
        )

    return {
        "nodes": sanitised_nodes,
        "edges": list(flow_data.get("edges", [])),
        # Top-level metadata fields — present only when stored
        **{
            k: flow_data[k]
            for k in ("entry_node_id", "version", "compiled_at")
            if k in flow_data
        },
    }


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

    def _resolve_analysis_model(self, db: Session, model_name: str) -> Model:
        """Mirror agent_service._resolve_llm_model exactly, for the post-call
        analysis "Analysis Model" dropdown — same catalog, same validation.
        """
        name = model_name.strip()
        model = (
            db.query(Model)
            .filter(Model.model_name == name, Model.archive == False)  # noqa: E712
            .first()
        )
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{name}' is not a supported LLM model for post-call analysis.",
            )
        return model

    def _insert_prompt_version(
        self,
        db: Session,
        flow_id: uuid.UUID,
        prompt_text: str,
        notes: str | None,
        *,
        current_prompt_id: uuid.UUID | None = None,
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

    def _prompt_changed(self, db: Session, flow: CallFlow, new_prompt: str) -> bool:
        """Return True if new_prompt text differs from the currently active version."""
        if flow.current_prompt_id is None:
            return True
        pv_repo = PromptVersionRepository(db)
        current = pv_repo.find_by_id(flow.current_prompt_id)
        if current is None:
            return True
        return current.prompt_text != new_prompt

    def _update_current_version_notes(
        self, db: Session, flow: CallFlow, notes: str | None
    ) -> None:
        """Patch notes on the flow's currently active prompt version, if any."""
        if notes is None or flow.current_prompt_id is None:
            return
        pv_repo = PromptVersionRepository(db)
        current_ver = pv_repo.find_by_id(flow.current_prompt_id)
        if current_ver:
            current_ver.notes = notes
            db.add(current_ver)

    # ── Serialization helpers ─────────────────────────────────────────────

    def _version_to_out(self, v: PromptVersion) -> PromptVersionOut:
        return PromptVersionOut.model_validate(v)

    def _flow_to_out(self, db: Session, flow: CallFlow) -> dict:
        pv_repo = PromptVersionRepository(db)
        versions = pv_repo.find_by_flow(flow.id, order_desc=True)
        folder_ids = (
            CallFlowRepository(db).find_folder_ids_map([flow.id]).get(flow.id, [])
        )

        # Full AgentOut on detail endpoints (POST 201, GET, PUT)
        agent_dict: dict | None = None
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
            knowledge_base_ids=flow.knowledge_base_ids or [],
            folder_ids=folder_ids,
            public_access=flow.public_access,
            status=flow.status,
            created_at=flow.created_at,
            updated_at=flow.updated_at,
        )
        return out.model_dump(by_alias=True, mode="json")

    def flow_to_list_item_model(
        self, flow: CallFlow, folder_ids: list[uuid.UUID] | None = None
    ) -> CallFlowListItem:
        """Build the slim list-item model for a CallFlow.

        Public (no leading underscore) so other services can reuse the actual
        typed model — e.g. DashboardService, which nests this inside another
        Pydantic model rather than emitting raw JSON, so it needs the
        CallFlowListItem instance itself, not the aliased dict flow_to_list_item
        returns.
        """
        agent_ref: AgentRef | None = None
        if flow.agent:
            agent_ref = AgentRef.model_validate(flow.agent)

        return CallFlowListItem(
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
            knowledge_base_ids=flow.knowledge_base_ids or [],
            folder_ids=folder_ids or [],
            public_access=flow.public_access,
            status=flow.status,
            created_at=flow.created_at,
            updated_at=flow.updated_at,
        )

    def flow_to_list_item(
        self, flow: CallFlow, folder_ids: list[uuid.UUID] | None = None
    ) -> dict:
        """Serialize a CallFlow to the slim list-item shape (aliased dict).

        Public (no leading underscore) so other services — e.g. FolderService,
        when listing the flows attached to a folder — can reuse it instead of
        duplicating the serialization logic.
        """
        item = self.flow_to_list_item_model(flow, folder_ids)
        return item.model_dump(by_alias=True, mode="json")

    def _sync_agent_system_prompt(self, db: Session, flow: CallFlow) -> None:
        """Ensure the bound Agent's system_prompt matches the flow's current_prompt_id text."""
        if not flow.agent_id or not flow.current_prompt_id:
            return
        pv_repo = PromptVersionRepository(db)
        current_version = pv_repo.find_by_id(flow.current_prompt_id)
        if not current_version or not current_version.prompt_text:
            return
        agent = db.execute(
            select(Agent).where(Agent.id == flow.agent_id)
        ).scalar_one_or_none()
        if agent and agent.system_prompt != current_version.prompt_text:
            agent.system_prompt = current_version.prompt_text
            db.add(agent)

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
                "status": body.status.value,
            }
        )

        if body.prompt and body.prompt.strip():
            # No existing current yet on create; pass None so nothing is protected
            version = self._insert_prompt_version(
                db, flow.id, body.prompt, body.notes, current_prompt_id=None
            )
            flow = repo.update(flow, {"current_prompt_id": version.id})
            self._sync_agent_system_prompt(db, flow)

        db.commit()
        db.refresh(flow)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    def get_flow(self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID) -> dict:
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
        folder_ids_map = repo.find_folder_ids_map([f.id for f in rows])
        # Tenant-wide across every call flow/agent — NOT scoped to this page's
        # flows or affected by pagination. Callers must not assume it reflects
        # only the `data` rows returned above.
        metrics = call_history_service.get_metrics(db, tenant_id)
        return {
            "data": [
                self.flow_to_list_item(f, folder_ids_map.get(f.id, [])) for f in rows
            ],
            "total": total,
            "page": page,
            "pageSize": limit,
            "analytics": {
                "totalCalls": metrics.total_calls,
                "successRatePercent": metrics.success_rate_percent,
                "averageDurationSeconds": metrics.avg_duration_seconds,
            },
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
        if body.status is not None:
            scalar_updates["status"] = body.status.value

        # Prompt versioning logic
        if body.prompt is not None and body.prompt.strip():
            # New prompt text → create version only if it differs from current
            if self._prompt_changed(db, flow, body.prompt):
                # Protect the existing active version from pruning
                version = self._insert_prompt_version(
                    db,
                    flow.id,
                    body.prompt,
                    body.notes,
                    current_prompt_id=flow.current_prompt_id,
                )
                scalar_updates["current_prompt_id"] = version.id
            else:
                self._update_current_version_notes(db, flow, body.notes)
        elif body.current_prompt_id is not None:
            # Explicit rollback / version select — no prompt text provided
            pv_repo = PromptVersionRepository(db)
            target = pv_repo.find_by_id(body.current_prompt_id)
            if target is None or target.flow_id != flow.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="currentPromptId does not belong to this flow",
                )
            if body.notes is not None:
                target.notes = body.notes
                db.add(target)
            scalar_updates["current_prompt_id"] = body.current_prompt_id
        else:
            self._update_current_version_notes(db, flow, body.notes)

        if scalar_updates:
            flow = repo.update(flow, scalar_updates)
            if "current_prompt_id" in scalar_updates or "agent_id" in scalar_updates:
                self._sync_agent_system_prompt(db, flow)

        db.commit()
        db.refresh(flow)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    def update_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: CallFlowSettingsUpdate,
    ) -> dict:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        repo = CallFlowRepository(db)
        flow = repo.update(flow, {"public_access": body.public_access})
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
            select(CallSession)
            .where(
                CallSession.call_flow_id == flow_id,
                CallSession.tenant_id == tenant_id,
                CallSession.status == "active",
            )
            .limit(1)
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

    def delete_prompt_version(
        self,
        db: Session,
        flow_id: uuid.UUID,
        version_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> None:
        """Delete a single prompt version from a call flow.

        Guards:
        - Flow must exist and belong to tenant
        - Version must exist and belong to flow
        - Cannot delete the currently active version (flow.current_prompt_id)
        - Cannot delete a version currently assigned as an A/B test variant
        - Cannot delete the only remaining version of a flow
        """
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        pv_repo = PromptVersionRepository(db)
        version = pv_repo.find_by_id(version_id)
        if version is None or version.flow_id != flow.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt version {version_id} not found in call flow {flow_id}",
            )
        if flow.current_prompt_id == version.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the currently active prompt version. Please switch to another version first.",
            )
        if version.id in (flow.ab_prompt_a_id, flow.ab_prompt_b_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete a prompt version assigned to an active A/B test. Please update the A/B test first.",
            )
        count = pv_repo.count_by_flow(flow.id)
        if count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the only prompt version of a call flow.",
            )
        pv_repo.soft_delete(version)
        db.commit()

    # ── A/B prompt testing ──────────────────────────────────────────────────

    def update_ab_test(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: AbTestUpdate,
    ) -> AbTestResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        pv_repo = PromptVersionRepository(db)

        prompt_a = pv_repo.find_by_id(body.prompt_a_id)
        if prompt_a is None or prompt_a.flow_id != flow.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"prompt_a_id {body.prompt_a_id} does not belong to flow {flow_id}",
            )
        prompt_b = pv_repo.find_by_id(body.prompt_b_id)
        if prompt_b is None or prompt_b.flow_id != flow.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"prompt_b_id {body.prompt_b_id} does not belong to flow {flow_id}",
            )

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "ab_test_enabled": body.enabled,
                "ab_prompt_a_id": body.prompt_a_id,
                "ab_prompt_b_id": body.prompt_b_id,
                "ab_split_ratio": body.split_ratio,
            },
        )
        db.commit()
        db.refresh(flow)
        return AbTestResponse(
            ab_test_enabled=flow.ab_test_enabled,
            ab_prompt_a_id=flow.ab_prompt_a_id,
            ab_prompt_b_id=flow.ab_prompt_b_id,
            ab_split_ratio=float(flow.ab_split_ratio),
        )

    def get_ab_results(
        self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> AbResultsResponse:
        self._get_flow_or_404(db, flow_id, tenant_id)

        metrics_a = self._variant_metrics(db, flow_id, tenant_id, "a")
        metrics_b = self._variant_metrics(db, flow_id, tenant_id, "b")

        significance, recommended = self._ab_significance(
            metrics_a.calls, metrics_a.completed, metrics_b.calls, metrics_b.completed
        )

        return AbResultsResponse(
            variant_a=metrics_a,
            variant_b=metrics_b,
            statistical_significance=significance,
            recommended_variant=recommended,
        )

    def _variant_metrics(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        variant: str,
    ) -> VariantMetrics:
        row = db.execute(
            select(
                func.count().label("calls"),
                func.sum(case((CallSession.status == "completed", 1), else_=0)).label(
                    "completed"
                ),
                func.sum(case((CallSession.status == "failed", 1), else_=0)).label(
                    "failed"
                ),
                func.avg(CallSession.duration).label("avg_duration"),
                func.sum(
                    case((CallSession.transferred == True, 1), else_=0)  # noqa: E712
                ).label("transferred"),
                func.sum(
                    case((CallSession.success_evaluation == "success", 1), else_=0)
                ).label("successes"),
            ).where(
                CallSession.call_flow_id == flow_id,
                CallSession.tenant_id == tenant_id,
                CallSession.ab_variant == variant,
            )
        ).one()

        calls = row.calls or 0
        return VariantMetrics(
            calls=calls,
            completed=row.completed or 0,
            failed=row.failed or 0,
            avg_duration=float(row.avg_duration) if row.avg_duration else None,
            transfer_rate=(row.transferred / calls) if calls else 0.0,
            success_rate=(row.successes / calls) if calls else 0.0,
        )

    def _ab_significance(
        self, calls_a: int, completed_a: int, calls_b: int, completed_b: int
    ) -> tuple[bool, str]:
        """Chi-squared test on completed-vs-total contingency table.

        Guardrail: fewer than 30 calls on either variant is always inconclusive,
        regardless of p-value — too few samples for the test to be meaningful.
        """
        if (
            calls_a < _AB_MIN_CALLS_FOR_SIGNIFICANCE
            or calls_b < _AB_MIN_CALLS_FOR_SIGNIFICANCE
        ):
            return False, "inconclusive"

        contingency = [
            [completed_a, calls_a - completed_a],
            [completed_b, calls_b - completed_b],
        ]
        _, p_value, _, _ = chi2_contingency(contingency)

        if p_value >= _AB_SIGNIFICANCE_P_VALUE:
            return False, "inconclusive"

        rate_a = completed_a / calls_a
        rate_b = completed_b / calls_b
        recommended = "a" if rate_a > rate_b else "b"
        return True, recommended

    # ── Cross-session caller memory ─────────────────────────────────────────

    def update_caller_memory_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: CallerMemorySettingsUpdate,
    ) -> CallerMemorySettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "caller_memory_enabled": body.caller_memory_enabled,
                "caller_memory_window": body.caller_memory_window,
            },
        )
        db.commit()
        db.refresh(flow)
        return CallerMemorySettingsResponse(
            caller_memory_enabled=flow.caller_memory_enabled,
            caller_memory_window=flow.caller_memory_window,
        )

    def update_post_call_actions_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: PostCallActionsSettingsUpdate,
    ) -> PostCallActionsSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "email_summary_enabled": body.email_summary_enabled,
                "email_summary_recipients": [
                    str(e) for e in body.email_summary_recipients
                ],
                "summary_to_business_owner_enabled": body.summary_to_business_owner_enabled,
                "slack_summary_enabled": body.slack_summary_enabled,
                "slack_channel_id": body.slack_channel_id,
                "slack_channel_name": body.slack_channel_name,
            },
        )
        db.commit()
        db.refresh(flow)
        return PostCallActionsSettingsResponse(
            email_summary_enabled=flow.email_summary_enabled,
            email_summary_recipients=list(flow.email_summary_recipients or []),
            summary_to_business_owner_enabled=flow.summary_to_business_owner_enabled,
            slack_summary_enabled=flow.slack_summary_enabled,
            slack_channel_id=flow.slack_channel_id,
            slack_channel_name=flow.slack_channel_name,
        )

    def get_post_call_actions_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> PostCallActionsSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return PostCallActionsSettingsResponse(
            email_summary_enabled=bool(flow.email_summary_enabled),
            email_summary_recipients=list(flow.email_summary_recipients or []),
            summary_to_business_owner_enabled=bool(flow.summary_to_business_owner_enabled),
            slack_summary_enabled=bool(flow.slack_summary_enabled),
            slack_channel_id=flow.slack_channel_id,
            slack_channel_name=flow.slack_channel_name,
        )

    # ── Voicemail Detection Settings ────────────────────────────────────────

    def update_voicemail_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: VoicemailSettingsUpdate,
    ) -> VoicemailSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        action_val = (
            body.voicemail_action.value
            if hasattr(body.voicemail_action, "value")
            else str(body.voicemail_action)
        )
        flow = repo.update(
            flow,
            {
                "voicemail_detection_enabled": body.voicemail_detection_enabled,
                "voicemail_action": action_val,
                "voicemail_message": body.voicemail_message,
                "voicemail_advanced_detection_enabled": body.voicemail_advanced_detection_enabled,
                "voicemail_detection_timeout": body.voicemail_detection_timeout,
            },
        )
        db.commit()
        db.refresh(flow)
        return VoicemailSettingsResponse(
            voicemail_detection_enabled=bool(flow.voicemail_detection_enabled),
            voicemail_action=flow.voicemail_action or "hang_up",
            voicemail_message=flow.voicemail_message,
            voicemail_advanced_detection_enabled=bool(flow.voicemail_advanced_detection_enabled),
            voicemail_detection_timeout=flow.voicemail_detection_timeout or 5,
        )

    def get_voicemail_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> VoicemailSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return VoicemailSettingsResponse(
            voicemail_detection_enabled=bool(flow.voicemail_detection_enabled),
            voicemail_action=flow.voicemail_action or "hang_up",
            voicemail_message=flow.voicemail_message,
            voicemail_advanced_detection_enabled=bool(flow.voicemail_advanced_detection_enabled),
            voicemail_detection_timeout=flow.voicemail_detection_timeout or 5,
        )

    # ── Call Screening Detection Settings ───────────────────────────────────

    def update_call_screening_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: CallScreeningSettingsUpdate,
    ) -> CallScreeningSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        action_val = (
            body.call_screening_action.value
            if hasattr(body.call_screening_action, "value")
            else str(body.call_screening_action)
        )
        flow = repo.update(
            flow,
            {
                "call_screening_action": action_val,
            },
        )
        db.commit()
        db.refresh(flow)
        return CallScreeningSettingsResponse(
            call_screening_action=flow.call_screening_action or "respond",
        )

    def get_call_screening_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> CallScreeningSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return CallScreeningSettingsResponse(
            call_screening_action=flow.call_screening_action or "respond",
        )

    # ── Disable Metadata Settings ───────────────────────────────────────────

    def update_metadata_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: MetadataSettingsUpdate,
    ) -> MetadataSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "disable_metadata": bool(body.disable_metadata),
            },
        )
        db.commit()
        db.refresh(flow)
        return MetadataSettingsResponse(
            disable_metadata=bool(flow.disable_metadata),
        )

    def get_metadata_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> MetadataSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return MetadataSettingsResponse(
            disable_metadata=bool(flow.disable_metadata),
        )

    # ── IVR Phone Tree & DTMF Keypad Settings ────────────────────────────────

    def update_ivr_dtmf_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: IVRDTMFSettingsUpdate,
    ) -> IVRDTMFSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "ivr_enabled": bool(body.ivr_enabled),
                "ivr_action": body.ivr_action,
                "ivr_navigation_mode": body.ivr_navigation_mode,
                "ivr_max_attempts": body.ivr_max_attempts,
                "ivr_keypress_delay": body.ivr_keypress_delay,
                "ivr_priority_list": body.ivr_priority_list or [],
                "ivr_wait_on_hold": bool(body.ivr_wait_on_hold),
                "ivr_max_hold_time": body.ivr_max_hold_time,
                "dtmf_enabled": bool(body.dtmf_enabled),
                "dtmf_button_press_delay": body.dtmf_button_press_delay,
                "dtmf_allow_caller_interruption": bool(
                    body.dtmf_allow_caller_interruption
                ),
                "dtmf_max_digits": body.dtmf_max_digits,
                "dtmf_allowed_exceeded_attempts": body.dtmf_allowed_exceeded_attempts,
                "dtmf_exceeded_action": body.dtmf_exceeded_action,
                "dtmf_end_call_message": body.dtmf_end_call_message,
            },
        )
        db.commit()
        db.refresh(flow)
        return self._to_ivr_dtmf_response(flow)

    def get_ivr_dtmf_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> IVRDTMFSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_ivr_dtmf_response(flow)

    @staticmethod
    def _to_ivr_dtmf_response(flow: CallFlow) -> IVRDTMFSettingsResponse:
        return IVRDTMFSettingsResponse(
            ivr_enabled=bool(flow.ivr_enabled),
            ivr_action=flow.ivr_action or "dial_through",
            ivr_navigation_mode=flow.ivr_navigation_mode or "let_ai_converse",
            ivr_max_attempts=(
                flow.ivr_max_attempts if flow.ivr_max_attempts is not None else 3
            ),
            ivr_keypress_delay=(
                flow.ivr_keypress_delay if flow.ivr_keypress_delay is not None else 8
            ),
            ivr_priority_list=list(flow.ivr_priority_list or []),
            ivr_wait_on_hold=bool(flow.ivr_wait_on_hold),
            ivr_max_hold_time=(
                flow.ivr_max_hold_time if flow.ivr_max_hold_time is not None else 120
            ),
            dtmf_enabled=bool(flow.dtmf_enabled),
            dtmf_button_press_delay=(
                flow.dtmf_button_press_delay
                if flow.dtmf_button_press_delay is not None
                else 2
            ),
            dtmf_allow_caller_interruption=bool(flow.dtmf_allow_caller_interruption),
            dtmf_max_digits=(
                flow.dtmf_max_digits if flow.dtmf_max_digits is not None else 50
            ),
            dtmf_allowed_exceeded_attempts=(
                flow.dtmf_allowed_exceeded_attempts
                if flow.dtmf_allowed_exceeded_attempts is not None
                else 10
            ),
            dtmf_exceeded_action=flow.dtmf_exceeded_action or "end_call",
            dtmf_end_call_message=flow.dtmf_end_call_message,
        )

    # ── Call Timing & Silence Detection Settings ──

    def update_call_timing_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: CallTimingSettingsUpdate,
    ) -> CallTimingSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        update_dict = body.model_dump(exclude_unset=True)
        if "reminder_messages" in update_dict and update_dict["reminder_messages"] is not None:
            update_dict["reminder_messages"] = list(update_dict["reminder_messages"])

        repo = CallFlowRepository(db)
        flow = repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_call_timing_response(flow)

    def get_call_timing_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> CallTimingSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_call_timing_response(flow)

    @staticmethod
    def _to_call_timing_response(flow: CallFlow) -> CallTimingSettingsResponse:
        return CallTimingSettingsResponse(
            silence_timeout=(
                flow.silence_timeout if flow.silence_timeout is not None else 10
            ),
            end_call_after_reminder=(
                flow.end_call_after_reminder
                if flow.end_call_after_reminder is not None
                else 10
            ),
            reminder_retries=(
                flow.reminder_retries if flow.reminder_retries is not None else 1
            ),
            reminder_messages=list(flow.reminder_messages or []),
            max_call_duration=(
                flow.max_call_duration
                if flow.max_call_duration is not None
                else 1800
            ),
            max_duration_message=flow.max_duration_message,
        )

    # ── Inbound Call Redirection & Forwarding Settings ──

    def update_inbound_redirect_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: InboundRedirectSettingsUpdate,
    ) -> InboundRedirectSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        update_dict = body.model_dump(exclude_unset=True)
        if (
            "redirect_conditions" in update_dict
            and update_dict["redirect_conditions"] is not None
        ):
            update_dict["redirect_conditions"] = [
                c.model_dump() if hasattr(c, "model_dump") else dict(c)
                for c in update_dict["redirect_conditions"]
            ]

        repo = CallFlowRepository(db)
        repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_inbound_redirect_response(flow)

    def get_inbound_redirect_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> InboundRedirectSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_inbound_redirect_response(flow)

    @staticmethod
    def _to_inbound_redirect_response(
        flow: CallFlow,
    ) -> InboundRedirectSettingsResponse:
        raw_conditions = flow.redirect_conditions or []
        conditions = []
        for item in raw_conditions:
            if isinstance(item, dict):
                conditions.append(RedirectCondition(**item))
            elif isinstance(item, RedirectCondition):
                conditions.append(item)

        return InboundRedirectSettingsResponse(
            redirect_inbound_calls_enabled=bool(
                flow.redirect_inbound_calls_enabled
            ),
            redirect_forward_phone_number=flow.redirect_forward_phone_number,
            redirect_conditions=conditions,
            redirect_speak_message_enabled=bool(
                flow.redirect_speak_message_enabled
            ),
            redirect_message=flow.redirect_message,
        )

    @staticmethod
    def evaluate_redirect_conditions(
        conditions: list[dict | Any],
        context: dict[str, Any],
    ) -> bool:
        """
        Evaluate redirect conditions against context using AND logic.
        Returns True if conditions list is empty, or if ALL conditions match.
        """
        if not conditions:
            return True

        for cond in conditions:
            if hasattr(cond, "model_dump"):
                cond_dict = cond.model_dump()
            elif isinstance(cond, dict):
                cond_dict = cond
            else:
                cond_dict = {
                    "variable": getattr(cond, "variable", ""),
                    "operator": getattr(cond, "operator", ""),
                    "value": getattr(cond, "value", None),
                }

            raw_var = str(cond_dict.get("variable") or "").strip()
            var_name = raw_var
            if var_name.startswith("{{") and var_name.endswith("}}"):
                var_name = var_name[2:-2].strip()

            operator_raw = cond_dict.get("operator")
            if hasattr(operator_raw, "value"):
                operator = str(operator_raw.value).strip().lower()
            else:
                operator = str(operator_raw or "").strip().lower()

            target_val = cond_dict.get("value")
            target_val_str = (
                str(target_val).strip().lower()
                if target_val is not None
                else None
            )

            val = None
            if "." in var_name:
                parts = var_name.split(".")
                cur: Any = context
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        cur = None
                        break
                val = cur
                if val is None:
                    val = context.get(var_name)
            else:
                val = context.get(var_name)
                if (
                    val is None
                    and "_metadata" in context
                    and isinstance(context["_metadata"], dict)
                ):
                    val = context["_metadata"].get(var_name)
                if (
                    val is None
                    and "_variable" in context
                    and isinstance(context["_variable"], dict)
                ):
                    val = context["_variable"].get(var_name)

            if operator == "exists":
                if val is None:
                    return False
            elif operator == "not_empty":
                if val is None or str(val).strip() == "":
                    return False
            elif operator == "equals":
                if val is None:
                    return False
                if str(val).strip().lower() != target_val_str:
                    return False
            elif operator == "not_equals":
                curr_str = str(val).strip().lower() if val is not None else ""
                if curr_str == (target_val_str or ""):
                    return False
            else:
                return False

        return True

    @staticmethod
    def render_redirect_message_template(
        template: str,
        context: dict[str, Any],
    ) -> str:
        """
        Render {{key}} and {{_metadata.key}} template tokens in redirect announcement message.
        """
        if not template:
            return ""
        import re

        def _replace_token(match: re.Match) -> str:
            token = match.group(1).strip()
            if "." in token:
                parts = token.split(".")
                cur: Any = context
                found = True
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        found = False
                        break
                if found and cur is not None:
                    return str(cur)
            if token in context and context[token] is not None:
                return str(context[token])
            if (
                "_metadata" in context
                and isinstance(context["_metadata"], dict)
                and context["_metadata"].get(token) is not None
            ):
                return str(context["_metadata"][token])
            if (
                "_variable" in context
                and isinstance(context["_variable"], dict)
                and context["_variable"].get(token) is not None
            ):
                return str(context["_variable"][token])
            return ""

        return re.sub(
            r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}", _replace_token, template
        ).strip()

    # ── Flow Inbound Rules & Blocklist Rule Set Assignment ──

    def update_flow_inbound_rules(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: FlowInboundRulesUpdate,
    ) -> FlowInboundRulesResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        if body.inbound_rule_set_id is not None:
            rs = (
                db.execute(
                    select(InboundRuleSet).where(
                        InboundRuleSet.id == body.inbound_rule_set_id,
                        InboundRuleSet.tenant_id == tenant_id,
                        InboundRuleSet.is_deleted.is_(False),
                    )
                )
                .scalars()
                .first()
            )
            if not rs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Inbound rule set {body.inbound_rule_set_id} not found",
                )

        update_dict = {
            "inbound_rule_set_id": body.inbound_rule_set_id,
        }

        repo = CallFlowRepository(db)
        repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_flow_inbound_rules_response(db, flow)

    def get_flow_inbound_rules(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> FlowInboundRulesResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_flow_inbound_rules_response(db, flow)

    @staticmethod
    def _to_flow_inbound_rules_response(
        db: Session,
        flow: CallFlow,
    ) -> FlowInboundRulesResponse:
        if not flow.inbound_rule_set_id:
            return FlowInboundRulesResponse(
                inbound_rule_set_id=None,
                inbound_rule_set_name=None,
                active_rules_count=0,
            )

        rs = (
            db.execute(
                select(InboundRuleSet).where(
                    InboundRuleSet.id == flow.inbound_rule_set_id,
                    InboundRuleSet.tenant_id == flow.tenant_id,
                    InboundRuleSet.is_deleted.is_(False),
                )
            )
            .scalars()
            .first()
        )
        if not rs:
            return FlowInboundRulesResponse(
                inbound_rule_set_id=flow.inbound_rule_set_id,
                inbound_rule_set_name=None,
                active_rules_count=0,
            )

        count = (
            db.execute(
                select(func.count(InboundRule.id)).where(
                    InboundRule.rule_set_id == rs.id,
                    InboundRule.tenant_id == flow.tenant_id,
                    InboundRule.is_deleted.is_(False),
                )
            ).scalar()
            or 0
        )

        return FlowInboundRulesResponse(
            inbound_rule_set_id=rs.id,
            inbound_rule_set_name=rs.name,
            active_rules_count=count,
        )

    # ── Call Recording Settings ──

    def update_recording_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: RecordingSettingsUpdate,
    ) -> RecordingSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        update_dict = {
            "recording_enabled": body.recording_enabled,
            "public_recording_enabled": body.public_recording_enabled,
            "faster_inbound_pickup": body.faster_inbound_pickup,
            "stop_recording_on_transfer": body.stop_recording_on_transfer,
        }

        repo = CallFlowRepository(db)
        repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_recording_response(flow)

    def get_recording_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> RecordingSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_recording_response(flow)

    @staticmethod
    def _to_recording_response(flow: CallFlow) -> RecordingSettingsResponse:
        return RecordingSettingsResponse(
            recording_enabled=bool(
                flow.recording_enabled
                if flow.recording_enabled is not None
                else True
            ),
            public_recording_enabled=bool(
                flow.public_recording_enabled
                if flow.public_recording_enabled is not None
                else False
            ),
            faster_inbound_pickup=bool(
                flow.faster_inbound_pickup
                if flow.faster_inbound_pickup is not None
                else False
            ),
            stop_recording_on_transfer=bool(
                flow.stop_recording_on_transfer
                if flow.stop_recording_on_transfer is not None
                else False
            ),
        )

    # ── Compliance & Detection Settings ──

    def update_compliance_detection_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: ComplianceDetectionSettingsUpdate,
    ) -> ComplianceDetectionSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        update_dict = {
            "compliance_monitoring_enabled": body.compliance_monitoring_enabled,
            "anti_bot_detection_enabled": body.anti_bot_detection_enabled,
            "terminate_on_fake_voice": body.terminate_on_fake_voice,
        }

        repo = CallFlowRepository(db)
        repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_compliance_detection_response(flow)

    def get_compliance_detection_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ComplianceDetectionSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_compliance_detection_response(flow)

    @staticmethod
    def _to_compliance_detection_response(
        flow: CallFlow,
    ) -> ComplianceDetectionSettingsResponse:
        return ComplianceDetectionSettingsResponse(
            compliance_monitoring_enabled=bool(
                flow.compliance_monitoring_enabled
                if flow.compliance_monitoring_enabled is not None
                else False
            ),
            anti_bot_detection_enabled=bool(
                flow.anti_bot_detection_enabled
                if flow.anti_bot_detection_enabled is not None
                else False
            ),
            terminate_on_fake_voice=bool(
                flow.terminate_on_fake_voice
                if flow.terminate_on_fake_voice is not None
                else False
            ),
        )

    # ── Data Retention Policy Settings ──

    def update_data_retention_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: DataRetentionSettingsUpdate,
    ) -> DataRetentionSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        update_dict = {
            "retention_policy_enabled": body.retention_policy_enabled,
            "retention_transcript_enabled": body.retention_transcript_enabled,
            "retention_transcript_days": body.retention_transcript_days,
            "retention_summary_enabled": body.retention_summary_enabled,
            "retention_summary_days": body.retention_summary_days,
            "retention_recording_enabled": body.retention_recording_enabled,
            "retention_recording_days": body.retention_recording_days,
        }

        repo = CallFlowRepository(db)
        repo.update(flow, update_dict)
        db.commit()
        db.refresh(flow)
        return self._to_data_retention_response(flow)

    def get_data_retention_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> DataRetentionSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        return self._to_data_retention_response(flow)

    @staticmethod
    def _to_data_retention_response(
        flow: CallFlow,
    ) -> DataRetentionSettingsResponse:
        return DataRetentionSettingsResponse(
            retention_policy_enabled=bool(
                flow.retention_policy_enabled
                if flow.retention_policy_enabled is not None
                else False
            ),
            retention_transcript_enabled=bool(
                flow.retention_transcript_enabled
                if flow.retention_transcript_enabled is not None
                else False
            ),
            retention_transcript_days=int(
                flow.retention_transcript_days
                if flow.retention_transcript_days is not None
                else 30
            ),
            retention_summary_enabled=bool(
                flow.retention_summary_enabled
                if flow.retention_summary_enabled is not None
                else False
            ),
            retention_summary_days=int(
                flow.retention_summary_days
                if flow.retention_summary_days is not None
                else 30
            ),
            retention_recording_enabled=bool(
                flow.retention_recording_enabled
                if flow.retention_recording_enabled is not None
                else False
            ),
            retention_recording_days=int(
                flow.retention_recording_days
                if flow.retention_recording_days is not None
                else 30
            ),
        )

    # ── System Webhooks (pre-inbound / dynamic routing / post-call / status) ──

    def update_system_webhooks_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: SystemWebhooksSettingsUpdate,
    ) -> SystemWebhooksSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        fields = {
            "pre_inbound_webhook_url": body.pre_inbound_webhook_url,
            "pre_inbound_webhook_query_params": list(
                body.pre_inbound_webhook_query_params or []
            ),
            "pre_inbound_webhook_static_metadata": dict(
                body.pre_inbound_webhook_static_metadata or {}
            ),
            "dynamic_inbound_routing_enabled": body.dynamic_inbound_routing_enabled,
            "post_call_webhook_url": body.post_call_webhook_url,
            "post_call_webhook_query_params": list(
                body.post_call_webhook_query_params or []
            ),
            "post_call_webhook_custom_payload_enabled": (
                body.post_call_webhook_custom_payload_enabled
            ),
            "post_call_webhook_custom_payload_template": (
                body.post_call_webhook_custom_payload_template
            ),
            "status_webhook_enabled": body.status_webhook_enabled,
            "status_webhook_url": body.status_webhook_url,
            "status_webhook_query_params": list(body.status_webhook_query_params or []),
        }

        # Headers are the one deliberate asymmetry from strict full-replace:
        # the API never echoes back decrypted header values (see
        # SystemWebhooksSettingsResponse), so the caller has nothing valid to
        # resubmit on an otherwise-full-replace PUT. `None` in the request
        # means "leave the stored ciphertext unchanged"; an explicit `{}`
        # clears it back to NULL. Every other field above is a true
        # full-replace, matching update_post_call_actions_settings's shape.
        if body.pre_inbound_webhook_headers is not None:
            fields["pre_inbound_webhook_headers_encrypted"] = (
                encrypt_webhook_headers(body.pre_inbound_webhook_headers, db)
                if body.pre_inbound_webhook_headers
                else None
            )
        if body.post_call_webhook_headers is not None:
            fields["post_call_webhook_headers_encrypted"] = (
                encrypt_webhook_headers(body.post_call_webhook_headers, db)
                if body.post_call_webhook_headers
                else None
            )
        if body.status_webhook_headers is not None:
            fields["status_webhook_headers_encrypted"] = (
                encrypt_webhook_headers(body.status_webhook_headers, db)
                if body.status_webhook_headers
                else None
            )

        repo = CallFlowRepository(db)
        flow = repo.update(flow, fields)
        db.commit()
        db.refresh(flow)

        return SystemWebhooksSettingsResponse(
            pre_inbound_webhook_url=flow.pre_inbound_webhook_url,
            pre_inbound_webhook_headers_configured=bool(
                flow.pre_inbound_webhook_headers_encrypted
            ),
            pre_inbound_webhook_query_params=list(
                flow.pre_inbound_webhook_query_params or []
            ),
            pre_inbound_webhook_static_metadata=dict(
                flow.pre_inbound_webhook_static_metadata or {}
            ),
            dynamic_inbound_routing_enabled=flow.dynamic_inbound_routing_enabled,
            post_call_webhook_url=flow.post_call_webhook_url,
            post_call_webhook_headers_configured=bool(
                flow.post_call_webhook_headers_encrypted
            ),
            post_call_webhook_query_params=list(
                flow.post_call_webhook_query_params or []
            ),
            post_call_webhook_custom_payload_enabled=(
                flow.post_call_webhook_custom_payload_enabled
            ),
            post_call_webhook_custom_payload_template=(
                flow.post_call_webhook_custom_payload_template
            ),
            status_webhook_enabled=flow.status_webhook_enabled,
            status_webhook_url=flow.status_webhook_url,
            status_webhook_headers_configured=bool(
                flow.status_webhook_headers_encrypted
            ),
            status_webhook_query_params=list(flow.status_webhook_query_params or []),
        )

    def get_system_webhooks_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> SystemWebhooksSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return SystemWebhooksSettingsResponse(
            pre_inbound_webhook_url=flow.pre_inbound_webhook_url,
            pre_inbound_webhook_headers_configured=bool(
                flow.pre_inbound_webhook_headers_encrypted
            ),
            pre_inbound_webhook_query_params=list(
                flow.pre_inbound_webhook_query_params or []
            ),
            pre_inbound_webhook_static_metadata=dict(
                flow.pre_inbound_webhook_static_metadata or {}
            ),
            dynamic_inbound_routing_enabled=flow.dynamic_inbound_routing_enabled,
            post_call_webhook_url=flow.post_call_webhook_url,
            post_call_webhook_headers_configured=bool(
                flow.post_call_webhook_headers_encrypted
            ),
            post_call_webhook_query_params=list(
                flow.post_call_webhook_query_params or []
            ),
            post_call_webhook_custom_payload_enabled=(
                flow.post_call_webhook_custom_payload_enabled
            ),
            post_call_webhook_custom_payload_template=(
                flow.post_call_webhook_custom_payload_template
            ),
            status_webhook_enabled=flow.status_webhook_enabled,
            status_webhook_url=flow.status_webhook_url,
            status_webhook_headers_configured=bool(
                flow.status_webhook_headers_encrypted
            ),
            status_webhook_query_params=list(flow.status_webhook_query_params or []),
        )

    def list_system_webhook_deliveries(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        webhook_kind: str | None,
        page: int,
        page_size: int,
    ) -> PaginatedSystemWebhookDeliveries:
        # Confirms the flow exists in this tenant before querying deliveries —
        # same 404 scoping as every other system-webhooks endpoint.
        self._get_flow_or_404(db, flow_id, tenant_id)

        query = db.query(SystemWebhookDeliveryLog).filter(
            SystemWebhookDeliveryLog.tenant_id == tenant_id,
            SystemWebhookDeliveryLog.call_flow_id == flow_id,
        )
        if webhook_kind is not None:
            query = query.filter(SystemWebhookDeliveryLog.webhook_kind == webhook_kind)

        total = query.count()
        items = (
            query.order_by(SystemWebhookDeliveryLog.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return PaginatedSystemWebhookDeliveries(
            items=[SystemWebhookDeliveryOut.model_validate(d) for d in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_post_call_analysis_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: PostCallAnalysisSettingsUpdate,
    ) -> PostCallAnalysisSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        analysis_model_name: str | None = None
        if body.analysis_model is not None:
            model = self._resolve_analysis_model(db, body.analysis_model)
            analysis_model_name = model.model_name

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "post_call_analysis_variables": [
                    v.model_dump() for v in body.variables_to_extract
                ],
                "post_call_analysis_model": analysis_model_name,
            },
        )
        db.commit()
        db.refresh(flow)
        return PostCallAnalysisSettingsResponse(
            variables_to_extract=list(flow.post_call_analysis_variables or []),
            analysis_model=flow.post_call_analysis_model,
        )

    def get_post_call_analysis_settings(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> PostCallAnalysisSettingsResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)

        return PostCallAnalysisSettingsResponse(
            variables_to_extract=list(flow.post_call_analysis_variables or []),
            analysis_model=flow.post_call_analysis_model,
        )

    def promote_ab_winner(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: AbTestWinnerUpdate,
    ) -> dict:
        flow = self._get_flow_or_404(db, flow_id, tenant_id, load_relations=True)

        winning_prompt_id = (
            flow.ab_prompt_a_id if body.variant == "a" else flow.ab_prompt_b_id
        )
        if winning_prompt_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Flow {flow_id} has no prompt version assigned to variant '{body.variant}'",
            )

        repo = CallFlowRepository(db)
        flow = repo.update(
            flow,
            {
                "current_prompt_id": winning_prompt_id,
                "ab_test_enabled": False,
            },
        )
        self._sync_agent_system_prompt(db, flow)
        db.commit()
        db.refresh(flow)
        if flow.agent is None:
            flow.agent = db.execute(
                select(Agent).where(Agent.id == flow.agent_id)
            ).scalar_one_or_none()
        return self._flow_to_out(db, flow)

    # ── Visual Flow Editor ────────────────────────────────────────────────

    def list_flow_data(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        page: int,
        limit: int,
    ) -> PaginatedFlowDataResponse:
        repo = CallFlowRepository(db)
        try:
            rows, total = repo.find_flow_data_by_workspace(
                tenant_id, page=page, limit=limit
            )
        except SQLAlchemyError:
            logger.exception(
                "Failed to list flow-data for tenant %s (page=%s, limit=%s)",
                tenant_id,
                page,
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list flow data",
            )

        return PaginatedFlowDataResponse(
            data=[
                FlowDataListItem(
                    flow_id=f.id,
                    name=f.name,
                    flow_data=f.flow_data,
                    flow_data_compiled=f.compiled_plan,
                    updated_at=f.updated_at,
                )
                for f in rows
            ],
            total=total,
            page=page,
            page_size=limit,
        )

    def get_flow_data(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        readonly: bool = False,
    ) -> FlowDataResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        if readonly:
            raw = (
                _strip_flow_data_for_readonly(flow.flow_data)
                if flow.flow_data
                else None
            )
            # Flow graph was already validated and pre-compiled at save time;
            # skip redundant O(V+E) graph traversal on readonly GET requests.
            return FlowDataResponse(
                flow_data=raw,
                flow_data_compiled=None,  # never expose compiled plan in readonly mode
                validation_errors=[],
            )
        validation_errors = validate_graph(flow.flow_data) if flow.flow_data else []
        return FlowDataResponse(
            flow_data=flow.flow_data,
            flow_data_compiled=flow.compiled_plan,
            validation_errors=[FlowValidationError(**e) for e in validation_errors],
        )

    def validate_flow_data(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: FlowDataUpdate | None = None,
    ) -> FlowValidationResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        flow_data = body.flow_data.model_dump() if body else (flow.flow_data or {})
        validation_errors = validate_graph(flow_data)
        return FlowValidationResponse(
            valid=not validation_errors,
            errors=[
                FlowValidationErrorItem(node_id=e.get("node_id"), message=e["message"])
                for e in validation_errors
            ],
        )

    def update_flow_data(
        self,
        db: Session,
        flow_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: FlowDataUpdate,
    ) -> FlowDataSaveResponse:
        flow = self._get_flow_or_404(db, flow_id, tenant_id)
        flow_data = body.flow_data.model_dump()

        validation_errors = validate_graph(flow_data)
        if validation_errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "flow_validation_failed",
                    "message": "Flow graph validation failed",
                    "validationErrors": [
                        {
                            "code": e["code"],
                            "message": e["message"],
                            "nodeId": e.get("node_id"),
                        }
                        for e in validation_errors
                    ],
                },
            )

        # version increments on every save; compiled_at is set to the moment
        # the pre-compiled executor plan below is built — both are top-level
        # fields of the flow_data JSONB document itself, per the ticket schema.
        previous_version = (
            (flow.flow_data or {}).get("version", 0) if flow.flow_data else 0
        )
        flow_data["version"] = (previous_version or 0) + 1
        flow_data["compiled_at"] = datetime.now(timezone.utc).isoformat()

        compiled = compile_graph(flow_data)

        repo = CallFlowRepository(db)
        flow = repo.update(flow, {"flow_data": flow_data, "compiled_plan": compiled})
        db.commit()
        db.refresh(flow)
        return FlowDataSaveResponse(
            version=flow_data["version"],
            validated=True,
            flow_data=flow_data,
            flow_data_compiled=compiled,
        )


call_flow_service = CallFlowService()
