"""
PipelineSession — per-call state container for the voice pipeline.

Holds references to the STT emitter, LLM cancel handle, TTS pipeline, and
in-memory conversation history.  BidirectionalStreamHandler stores one of these
as self._pipeline_session.

The existing handler attributes (_tts_cancel, _conversation_history_cache, etc.)
remain in place for backward compatibility — PipelineSession holds the same
objects by reference so both access paths stay in sync.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.voice.llm_prompt_builder import prune_history_to_turns


@dataclass
class PipelineSession:
    """Aggregate of per-call voice-pipeline state."""

    # STT emitter (SttPipeline or similar); may be None before pickup.
    stt_emitter: Any = field(default=None)

    # asyncio.Event set to signal the Vertex stream to stop mid-generation.
    # Must be the same object as handler._tts_cancel so barge-in wires correctly.
    llm_cancel: asyncio.Event = field(default_factory=asyncio.Event)

    # TtsPipeline instance; same object as handler._tts_pipeline.
    tts_pipeline: Any = field(default=None)

    # In-memory conversation history: (role, content) tuples.
    # Same list object as handler._conversation_history_cache.
    history: list[tuple[str, str]] = field(default_factory=list)

    def cancel_llm(self) -> None:
        """Signal in-flight LLM stream to stop (barge-in / new utterance)."""
        self.llm_cancel.set()

    def reset_llm_cancel(self) -> None:
        """Clear the cancel signal for the next turn."""
        self.llm_cancel.clear()

    def append_turn(self, role: str, content: str) -> None:
        """Append a single message to the in-memory history."""
        if content:
            self.history.append((role, content))

    def get_pruned_history(self, max_turns: int) -> list[tuple[str, str]]:
        """Return a pruned snapshot of history (does not mutate self.history)."""
        return prune_history_to_turns(self.history, max_turns)
