"""Wires FlowExecutor (pre-compiled visual call-flow graphs) into the live
bidirectional voice pipeline.

Mixed into ``BidirectionalStreamHandler``. Every entry point degrades to a
no-op when the agent's call flow has no ``compiled_plan`` — the caller
falls through to the existing default LLM pipeline unchanged.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Dict

from app.core.logger import logger
from app.voice.flow_executor import (
    FlowExecutor,
    FlowExecutorError,
    NodeExecutionResult,
    PipelineState,
)


# ---------------------------------------------------------------------------
# Pure, module-level helper — importable and independently testable.
# ---------------------------------------------------------------------------


def _interpolate_variables(text: str, variables: Dict[str, Any]) -> str:
    """Replace ``{variable_name}`` placeholders in *text* from *variables*.

    Missing variables are replaced with an empty string and a warning is
    emitted.  The function is pure (no side-effects beyond logging) so it is
    safe to call from tests without a running voice pipeline.
    """

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            logger.warning(
                "FlowExecutor: variable '%s' not found in state, using empty string",
                name,
            )
            return ""
        return str(variables[name])

    return re.sub(r"\{([^}]+)\}", _replace, text)


class FlowPipelineMixin:
    """Requires ``self.call_flow``, ``self._tts_pipeline`` on the host class."""

    # Class-level defaults: some tests build the handler via ``object.__new__``
    # and never run ``__init__``/``_flow_init`` — these ensure every flow hook
    # can safely check ``self._flow_executor`` without an AttributeError.
    _flow_executor: FlowExecutor | None = None
    _flow_state: PipelineState | None = None

    def _flow_init(self) -> None:
        """Build the FlowExecutor + PipelineState from the loaded call_flow.

        Called once from ``_load_session_data()``. Leaves ``self._flow_executor``
        as ``None`` when there is no compiled flow — every other flow hook
        checks that attribute first and is a no-op if it is unset.
        """
        self._flow_executor: FlowExecutor | None = None
        self._flow_state: PipelineState | None = None

        compiled = (
            getattr(self.call_flow, "compiled_plan", None)
            if self.call_flow
            else None
        )
        if not compiled:
            return

        flow_data = getattr(self.call_flow, "flow_data", None) or {}
        entry_node_id = flow_data.get("entry_node_id")

        try:
            executor = FlowExecutor(compiled, entry_node_id)
            start_node_id = executor.start_node_id()
        except FlowExecutorError as exc:
            logger.warning(
                "Call flow %s has no usable compiled graph: %s",
                getattr(self.call_flow, "id", None),
                exc,
            )
            return

        self._flow_executor = executor
        self._flow_state = PipelineState(current_node_id=start_node_id)

    async def _flow_start(self) -> bool:
        """Run the flow from its start node. Returns True if it handled the greeting."""
        if not self._flow_executor or not self._flow_state:
            return False
        await self._flow_run_chain(transcript=None)
        return True

    async def _flow_on_transcript(self, transcript: str, confidence: float) -> bool:
        """Advance the flow using the caller's final transcript.

        Returns True if the flow consumed this turn (caller must skip the
        default LLM path); False if there's no active flow.
        """
        if not self._flow_executor or not self._flow_state:
            return False
        await self._flow_run_chain(transcript=transcript)
        return True

    async def _flow_run_chain(self, transcript: str | None) -> None:
        """Advance nodes until hitting one that needs caller input or ends the call.

        ``FlowExecutor.next_node_id`` routes purely on node type/config (the
        ``branch`` node's own condition, or every other node type's single
        ``default`` edge) — it never routes on transcript content. ``transcript``
        is only consumed here, once, to store the caller's utterance into the
        variable declared by the ``collect_input`` node that requested it.
        """
        executor = self._flow_executor
        state = self._flow_state

        while True:
            next_node_id = executor.next_node_id(
                state.current_node_id, None, state.variables
            )
            if next_node_id is None:
                logger.warning(
                    "FlowExecutor: no matching edge from node %s — stalling turn",
                    state.current_node_id,
                )
                return

            try:
                result = executor.execute_node(next_node_id, state)
            except FlowExecutorError as exc:
                logger.error("FlowExecutor: %s", exc)
                return

            if result.action == "speak":
                await self._flow_speak(result)
                continue

            if result.action == "branch":
                continue

            if result.action == "wait_for_input":
                # Store the caller's utterance into the variable declared by
                # this collect_input node.  ``transcript`` is the value passed
                # into _flow_run_chain at the top of this call; ``remaining_transcript``
                # has already been cleared to None for subsequent loop links.
                variable_name = result.config.get("variable_name")
                if variable_name and transcript is not None:
                    state.variables[variable_name] = transcript
                    logger.debug(
                        "FlowExecutor: stored transcript into variable '%s'",
                        variable_name,
                    )
                # Speak the collect_input prompt (with interpolation) if present.
                prompt = result.config.get("prompt")
                if prompt:
                    prompt_result = NodeExecutionResult(
                        node_id=result.node_id,
                        node_type=result.node_type,
                        action="speak",
                        config=result.config,
                        speech_text=prompt,
                    )
                    await self._flow_speak(prompt_result)
                return

            if result.action == "transfer":
                await self._transfer_after_agent_request()
                return

            if result.action == "end_call":
                await self._end_call_after_agent_request()
                return

            if result.action == "kb_lookup":
                await self._flow_kb_lookup(result, state)
                continue

    async def _flow_speak(self, result: NodeExecutionResult) -> None:
        text = (result.speech_text or "").strip()
        if not text or not self._tts_pipeline:
            return
        # Interpolate {variable_name} placeholders before sending to TTS.
        text = _interpolate_variables(text, self._flow_state.variables)
        if not text:
            return
        await self._add_to_transcript("agent", text, "flow_node")
        await self._tts_pipeline.queue_tts(
            {
                "text": text,
                "chunk_id": f"flow_{result.node_id}",
                "use_ssml": self._use_ssml,
                "is_final": True,
            }
        )
        self._twilio_buffer_primed = False

    async def _flow_kb_lookup(
        self, result: NodeExecutionResult, state: PipelineState
    ) -> None:
        """Perform a pgvector KB search without blocking the async event loop.

        Reads ``kb_id``, ``query_variable``, ``result_variable``,
        ``min_confidence`` (default 0.7), and ``fallback_message`` from
        ``result.config``.  When ``kb_id`` is a valid UUID, retrieval is
        scoped to that single knowledge base (tenant scoping via
        ``tenant_id`` is still enforced unconditionally in
        ``rag_service.retrieve()``); an empty/invalid ``kb_id`` falls back
        to searching across all KBs bound to the agent's tenant, preserving
        prior behavior. The underlying ``rag_service.retrieve()`` is
        synchronous; it is dispatched to the default thread-pool executor so
        the async event loop is never blocked.
        """
        config = result.config
        kb_id_raw = config.get("kb_id")
        kb_id: uuid.UUID | None = None
        if kb_id_raw:
            try:
                kb_id = uuid.UUID(str(kb_id_raw))
            except (ValueError, TypeError):
                logger.warning(
                    "FlowExecutor: kb_lookup node has invalid kb_id %r; ignoring KB scoping",
                    kb_id_raw,
                )
        query_variable = config.get("query_variable", "")
        result_variable = config.get("result_variable", "")
        min_confidence: float = float(config.get("min_confidence", 0.7))
        fallback_message: str = config.get("fallback_message", "")

        query = state.variables.get(query_variable, "")
        if not query:
            logger.warning(
                "FlowExecutor: kb_lookup query_variable '%s' is empty; skipping lookup",
                query_variable,
            )
            if fallback_message:
                fallback_result = NodeExecutionResult(
                    node_id=result.node_id,
                    node_type=result.node_type,
                    action="speak",
                    config=config,
                    speech_text=fallback_message,
                )
                await self._flow_speak(fallback_result)
            return

        # Tenant/agent IDs come from the host class (BidirectionalStreamHandler).
        tenant_id = getattr(self, "tenant_id", None)
        agent_id = getattr(self, "agent_id", None)

        logger.debug(
            "FlowExecutor: kb_lookup node=%s kb_id=%s query_variable=%s query=%r",
            result.node_id,
            kb_id,
            query_variable,
            query[:80],
        )

        from app.services.embedding_service import embed_text_for_rag
        from app.services.rag_service import rag_service

        def _sync_retrieve():
            return rag_service.retrieve(
                user_text=query,
                tenant_id=tenant_id,
                agent_id=agent_id,
                embedding_func=embed_text_for_rag,
                kb_id=kb_id,
            )

        try:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, _sync_retrieve)
        except Exception as exc:
            logger.error(
                "FlowExecutor: kb_lookup retrieval failed: %s", exc, exc_info=True
            )
            chunks = []

        # Apply min_confidence threshold (mirrors rag_context.py behaviour).
        passing = [c for c in chunks if (c.score or 0.0) >= min_confidence]

        if passing:
            # Store the highest-scoring chunk's text as the result value.
            best = max(passing, key=lambda c: c.score or 0.0)
            if result_variable:
                state.variables[result_variable] = best.text
                logger.debug(
                    "FlowExecutor: kb_lookup stored result in variable '%s' (score=%.3f)",
                    result_variable,
                    best.score or 0.0,
                )
        else:
            logger.info(
                "FlowExecutor: kb_lookup no chunks above min_confidence=%.2f for node=%s; "
                "using fallback",
                min_confidence,
                result.node_id,
            )
            if fallback_message:
                fallback_result = NodeExecutionResult(
                    node_id=result.node_id,
                    node_type=result.node_type,
                    action="speak",
                    config=config,
                    speech_text=fallback_message,
                )
                await self._flow_speak(fallback_result)

