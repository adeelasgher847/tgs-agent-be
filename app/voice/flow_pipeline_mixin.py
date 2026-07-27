"""Wires FlowExecutor (pre-compiled visual call-flow graphs) into the live
bidirectional voice pipeline.

Mixed into ``BidirectionalStreamHandler``. Every entry point degrades to a
no-op when the agent's call flow has no ``flow_data_compiled`` — the caller
falls through to the existing default LLM pipeline unchanged.
"""

from __future__ import annotations

from typing import Optional

from app.core.logger import logger
from app.voice.flow_executor import (
    FlowExecutor,
    FlowExecutorError,
    NodeExecutionResult,
    PipelineState,
)


class FlowPipelineMixin:
    """Requires ``self.call_flow``, ``self._tts_pipeline`` on the host class."""

    # Class-level defaults: some tests build the handler via ``object.__new__``
    # and never run ``__init__``/``_flow_init`` — these ensure every flow hook
    # can safely check ``self._flow_executor`` without an AttributeError.
    _flow_executor: Optional[FlowExecutor] = None
    _flow_state: Optional[PipelineState] = None

    def _flow_init(self) -> None:
        """Build the FlowExecutor + PipelineState from the loaded call_flow.

        Called once from ``_load_session_data()``. Leaves ``self._flow_executor``
        as ``None`` when there is no compiled flow — every other flow hook
        checks that attribute first and is a no-op if it is unset.
        """
        self._flow_executor: Optional[FlowExecutor] = None
        self._flow_state: Optional[PipelineState] = None

        compiled = (
            getattr(self.call_flow, "flow_data_compiled", None)
            if self.call_flow
            else None
        )
        if not compiled:
            return

        try:
            executor = FlowExecutor(compiled)
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

    async def _flow_run_chain(self, transcript: Optional[str]) -> None:
        """Advance nodes until hitting one that needs caller input or ends the call.

        Only the first transition in the chain is evaluated against
        ``transcript`` — every subsequent link (e.g. a ``branch`` node's
        onward edge, or a ``greeting``'s auto-advance to the next node) is
        evaluated with ``transcript=None`` so only ``always``/``fallback``
        edges fire, matching the intent that a single utterance only
        answers the question posed by the node that requested it.
        """
        executor = self._flow_executor
        state = self._flow_state
        remaining_transcript = transcript

        while True:
            next_node_id = executor.next_node_id(
                state.current_node_id, remaining_transcript, state.variables
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

            remaining_transcript = (
                None  # only the originating edge consumes the transcript
            )

            if result.action == "speak":
                await self._flow_speak(result)
                continue
            if result.action == "branch":
                continue
            if result.action == "wait_for_input":
                return
            if result.action == "transfer":
                await self._transfer_after_agent_request()
                return
            if result.action == "end_call":
                await self._end_call_after_agent_request()
                return

    async def _flow_speak(self, result: NodeExecutionResult) -> None:
        text = (result.speech_text or "").strip()
        if not text or not self._tts_pipeline:
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
