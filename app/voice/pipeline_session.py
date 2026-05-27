"""
PipelineSession — per-call container satisfying the ticket's PipelineSession contract.

Holds the three live objects that span the full call lifetime:
  - stt_pipeline   : active Deepgram STT session (or None before pickup)
  - llm_cancel     : asyncio.Event — set to abort any in-flight Vertex stream
  - tts_pipeline   : active TtsPipeline (or None before pickup)
  - history        : in-memory conversation history shared with the handler

Incremental migration: BidirectionalStreamHandler creates one instance in
__init__ and aliases its own fields (self._llm_cancel_event, self._conversation_history_cache,
etc.) to the session's attributes so existing call-sites keep working without a
big-bang refactor.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from app.voice.stt_pipeline import SttPipeline
    from app.voice.tts_pipeline import TtsPipeline


@dataclass
class PipelineSession:
    """
    Satisfies ticket requirement: per-call object holding STT emitter, LLM
    cancel handle, TTS pipeline reference, and in-memory conversation history.
    """

    # STT session — set after Deepgram connect, None before pickup
    stt_pipeline: Optional["SttPipeline"] = None

    # LLM cancel handle — set in _cancel_inflight_llm_response; cleared at each turn start
    llm_cancel: asyncio.Event = field(default_factory=asyncio.Event)

    # TTS pipeline — set after call_start, None otherwise
    tts_pipeline: Optional["TtsPipeline"] = None

    # In-memory conversation history: (role, content) pairs, max 40 (20 turns)
    history: List[Tuple[str, str]] = field(default_factory=list)

    def reset_llm_cancel(self) -> None:
        """Clear the cancel event at the start of each new generation turn."""
        self.llm_cancel.clear()

    def cancel_llm(self) -> None:
        """Signal any in-flight LLM stream to stop (barge-in / new STT final)."""
        self.llm_cancel.set()

    def bind_stt(self, pipeline: Optional["SttPipeline"]) -> None:
        """Register the active Deepgram STT pipeline for this call."""
        self.stt_pipeline = pipeline

    def bind_tts(self, pipeline: Optional["TtsPipeline"]) -> None:
        """Register the active TTS pipeline for this call."""
        self.tts_pipeline = pipeline

    def clear_pipelines(self) -> None:
        """Clear STT/TTS references on shutdown (LLM cancel + history remain until GC)."""
        self.stt_pipeline = None
        self.tts_pipeline = None
