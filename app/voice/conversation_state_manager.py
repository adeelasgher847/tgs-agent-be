"""
ConversationStateManager: State machine for voice conversation.

Tracks:
- Conversation state (waiting, user speaking, processing, agent speaking, interrupted)
- Message history (last 12 turns)
- Mood/sentiment
- Audio timeline

Single source of truth for orchestration.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """Conversation state machine"""
    WAITING_FOR_INPUT = "waiting_for_input"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"  # LLM generating
    AGENT_SPEAKING = "agent_speaking"  # TTS streaming
    INTERRUPTED = "interrupted"  # Barge-in occurred
    CALL_ENDED = "call_ended"


class Mood(str, Enum):
    """User mood detection"""
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"


@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: str  # "user" or "agent"
    text: str
    timestamp_ms: int
    confidence: Optional[float] = None  # STT confidence
    duration_ms: Optional[int] = None  # Speaking duration
    response_time_ms: Optional[int] = None  # Agent response latency


class ConversationStateManager:
    """
    Manages call state and conversation context.
    
    IMPORTANT: This is the single source of truth for orchestration.
    All state changes must go through this manager.
    """

    def __init__(self, call_id: str, agent_config: Dict):
        self.call_id = call_id
        self.agent_config = agent_config

        # State tracking
        self.state = ConversationState.WAITING_FOR_INPUT
        self.state_updated_at_ms = time.time() * 1000

        # Conversation history
        self.messages: List[ConversationMessage] = []
        self.turn_number = 0

        # User state
        self.interim_text = ""
        self.final_text = ""
        self.user_speaking = False
        self.user_speaking_start_ms = 0

        # Agent state
        self.agent_is_speaking = False
        self.agent_speaking_start_ms = 0

        # Mood & context
        self.current_mood = Mood.NEUTRAL
        self.respond_briefly_flag = False

        # Audio timeline
        self.call_start_ms = time.time() * 1000

    def _elapsed_ms(self) -> int:
        """Milliseconds since call start"""
        return int((time.time() * 1000) - self.call_start_ms)

    async def transition_state(self, new_state: ConversationState, context: Optional[str] = None) -> None:
        """
        Transition to new state with validation.
        
        Validates state transitions are legal:
        WaitingForInput → UserSpeaking → Processing → AgentSpeaking → WaitingForInput
        """
        # Validate transition
        valid_transitions = {
            ConversationState.WAITING_FOR_INPUT: [
                ConversationState.USER_SPEAKING,
                ConversationState.CALL_ENDED,
            ],
            ConversationState.USER_SPEAKING: [
                ConversationState.PROCESSING,
                ConversationState.WAITING_FOR_INPUT,
                ConversationState.CALL_ENDED,
            ],
            ConversationState.PROCESSING: [
                ConversationState.AGENT_SPEAKING,
                ConversationState.INTERRUPTED,
                ConversationState.CALL_ENDED,
            ],
            ConversationState.AGENT_SPEAKING: [
                ConversationState.WAITING_FOR_INPUT,
                ConversationState.INTERRUPTED,
                ConversationState.CALL_ENDED,
            ],
            ConversationState.INTERRUPTED: [
                ConversationState.WAITING_FOR_INPUT,
                ConversationState.CALL_ENDED,
            ],
            ConversationState.CALL_ENDED: [],
        }

        if new_state not in valid_transitions.get(self.state, []):
            logger.warning(
                f"[{self.call_id}] Invalid state transition: {self.state} → {new_state}"
            )
            return

        old_state = self.state
        self.state = new_state
        self.state_updated_at_ms = self._elapsed_ms()

        logger.debug(
            f"[{self.call_id}] State transition: {old_state} → {new_state} "
            f"(@ {self.state_updated_at_ms}ms) {context or ''}"
        )

    def set_interim_text(self, text: str) -> None:
        """Update interim transcript from STT"""
        self.interim_text = text
        if not self.user_speaking:
            self.user_speaking = True
            self.user_speaking_start_ms = self._elapsed_ms()

    def set_final_text(self, text: str) -> None:
        """Update final transcript from STT"""
        self.final_text = text

    def agent_speaking_start(self) -> None:
        """Agent started speaking (TTS started)"""
        self.agent_is_speaking = True
        self.agent_speaking_start_ms = self._elapsed_ms()

    def agent_speaking_end(self) -> None:
        """Agent stopped speaking (TTS complete or barge-in)"""
        self.agent_is_speaking = False

    def cancel_agent_speaking(self) -> None:
        """Barge-in: cancel agent speech immediately"""
        self.agent_is_speaking = False

    def cancel_user_speaking(self) -> None:
        """Call ended: reset user speaking state"""
        self.user_speaking = False
        self.interim_text = ""
        self.final_text = ""

    def update_mood(self, mood: Mood) -> None:
        """Update detected mood from turn signals"""
        if mood != self.current_mood:
            logger.debug(f"[{self.call_id}] Mood updated: {self.current_mood} → {mood}")
            self.current_mood = mood

    async def add_to_history(
        self,
        role: str,
        text: str,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None,
        response_time_ms: Optional[int] = None,
    ) -> None:
        """
        Add message to conversation history (max 12 messages).
        
        Called after STT final or LLM completion.
        """
        message = ConversationMessage(
            role=role,
            text=text,
            timestamp_ms=self._elapsed_ms(),
            confidence=confidence,
            duration_ms=duration_ms,
            response_time_ms=response_time_ms,
        )
        self.messages.append(message)

        # Keep only last 12 messages
        self.messages = self.messages[-12:]
        self.turn_number = len(self.messages) // 2  # Approximate turn number

        logger.debug(
            f"[{self.call_id}] Added {role} message to history (len: {len(self.messages)})"
        )

    def get_messages_for_llm(self) -> List[Dict]:
        """
        Get conversation history formatted for LLM prompt.
        
        Returns:
            List of {"role": "user"/"assistant", "content": "text"} dicts
        """
        return [
            {
                "role": "user" if msg.role == "user" else "assistant",
                "content": msg.text,
            }
            for msg in self.messages
        ]

    def get_call_summary(self) -> Dict:
        """Get summary for call logging/DB storage"""
        return {
            'call_id': self.call_id,
            'state': self.state.value,
            'turn_number': self.turn_number,
            'final_mood': self.current_mood.value,
            'message_count': len(self.messages),
            'duration_ms': self._elapsed_ms(),
            'messages': [
                {
                    'role': msg.role,
                    'text': msg.text,
                    'confidence': msg.confidence,
                }
                for msg in self.messages
            ],
        }

    def get_telemetry(self) -> Dict:
        """Get telemetry for monitoring"""
        return {
            'call_id': self.call_id,
            'state': self.state.value,
            'user_speaking': self.user_speaking,
            'agent_speaking': self.agent_is_speaking,
            'mood': self.current_mood.value,
            'turns': self.turn_number,
            'elapsed_ms': self._elapsed_ms(),
        }
