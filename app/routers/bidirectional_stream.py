"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Target latency: 400–500ms (Vapi-style).

STT → LLM → TTS FLOW:
- Twilio sends audio every 20ms (MULAW 8kHz). We push each chunk to Deepgram STT.
- By default, LLM+TTS runs on **final** STT only (`VOICE_ENABLE_INTERIM_LLM=False`), matching
  stable one-reply-per-utterance behavior. Deepgram emits many more partials than classic
  Google STT; early interim LLM is opt-in and gated by min word count + confidence.
- Optional: first qualifying interim can start one LLM+TTS run (see `VOICE_ENABLE_INTERIM_LLM`).
- When final arrives after an interim run, we regenerate only if `_should_regenerate_on_final` says so
  (same-utterance extensions skip a second LLM).

PARALLEL TTS PIPELINE (Vapi-style):
- User Speech → STT Interim → LLM Chunk 1 → TTS Chunk 1 Playing
                             ↓ LLM Chunk 2 → TTS Chunk 2 Generating (parallel)
                             ↓ LLM Chunk 3 → TTS Chunk 3 Queued
- TTS generation and playback happen in parallel; new TTS patches embed with current
  playback via crossfade (build_crossfade_bridge) — no distortion, no sudden buffer noise.

NATURAL CONVERSATION FEATURES (Vapi-Style):
1. SSML Support:
   - <prosody> tags with varied rate (95%, 98%, 100%, 102%, 105%)
   - <prosody> tags with varied pitch (-1st, 0st, +1st, +2st)
   - <break> tags for natural pauses (150-200ms)
   - <audio> tags for breath sounds (DISABLED by default - can cause distortion)
   - Example: <prosody rate="95%" pitch="+1st">Alright.</prosody><break time="180ms"/>

2. Micro-Pauses & Hesitation Fillers:
   - Hesitation patterns: "Hmm <break time='120ms'/> I think..."
   - "Uh", "Well", "Let me see" WITH breaks (15% chance)
   - SPOKEN boundary fillers at EVERY 10-word mid-chunk break (100%):
     * "<break time='80-100ms'/><prosody>uhh/umm/uh/hmm</prosody>" pattern
     * ALWAYS added between chunks to eliminate tak-tak distortion
     * No silent gaps - natural spoken connectors keep audio flowing
   - Example: <speak>Hmm <break time="120ms"/> I think I can help with that.</speak>

3. Turn-Taking & Barge-In:
   - ENABLED - Agent stops when user speech is detected over playing TTS
   - Gate: `_is_tts_playing` + min word count + STT confidence floor + filler reject
   - Checked FIRST before interim gating (highest priority!)
   - TTS queue cleared (prevents old audio from resuming)
   - Barge-in still uses interim; normal replies prefer final when interim LLM is off

4. Persona & Variability:
   - Subtle prosody variations (95%-105% rate, ±1 semitone pitch)
   - Randomized breath/pause durations
   - Consistent voice persona from agent configuration

SMART CHUNKING WITH OVERLAP:
- 10 words per chunk for balanced quality + performance
- Major punctuation (. ! ? ;) triggers immediate chunk
- Minor punctuation (, : —) with 5+ words triggers chunk
- OVERLAP TECHNIQUE: Last 2 words of chunk 1 + filler + first words of chunk 2
  Example: Chunk 1: "Hello how are you doing today I'm doing" + SAVE("great thank")
           Chunk 2: "great thank" + "uhh" + "you for asking how"
  Result: Seamless transition with spoken fillers, no tak-tak distortion!

CACHING & LOW-LATENCY STRATEGIES:
1. Auto-Greeting on Connect (inbound only):
   - After pickup, optional one-time TTS from greeting_message (or legacy first_message if set)
   - greeting_message is not used on outbound calls
   - Bypasses LLM for that opening only; LLM is not instructed to repeat it verbatim on later "hi"

2. Pre-cached Common Phrases (disabled — implementation commented in code):
   - 36+ common phrases were pre-generated at connect to warm the MULAW TTS cache
   - Greetings, acknowledgements, confirmations; <50ms when cache hit
   - Re-enable by uncommenting asyncio.create_task(self._precache_common_phrases()) and the method

3. Quick Acknowledgement Pattern (5-Word Rule + Probability):
   - Eligible when user says 5+ words; then only ~38% chance we send "Got it" (more natural).
   - Never used for emotional/serious content (help, emergency, problem, etc.).
   - Short ack plays first; full response streams in parallel.
   - Example: "Got it" → "checking that now..." (full reply)

4. Adaptive Max Tokens:
   - Yes/No queries: 15 tokens (ultra-fast)
   - Short queries (1-3 words): 25 tokens (fast)
   - Medium queries (4-7 words): 35 tokens (balanced)
   - Complex queries: Full configured tokens
   - 30-60% faster LLM generation for simple queries

5. TTS Client Pre-warming:
   - Google TTS client initialized at startup
   - Avoids first-call penalty (~500ms saved)

TTS GAPLESS / NO DISTORTION (Vapi-style):
- Micro fade-in (25ms) on first frame to avoid clicks/pops.
- Crossfade at chunk boundaries (build_crossfade_bridge) so new TTS embeds with
  currently playing audio — no tak-tak, no network delay spikes.
- Jitter buffer primed with 3×20ms silence (60ms) once per utterance; no sudden
  buffer underrun noise.
- Google TTS streaming: https://cloud.google.com/text-to-speech/docs/create-audio-text-streaming
  (Chirp 3: HD, MULAW 8kHz, StreamingSynthesizeConfig.)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Any, Optional, Dict, Iterable, List
import time
from datetime import datetime, timezone, date
import uuid
import re
from app.core.logger import logger

# Deepgram STT is used via SttPipeline (app/voice/stt_pipeline.py).

from app.services.call_session_service import call_session_service
from app.services.voice_screening_qualification_service import (
    apply_resume_candidate_status_after_voice_screening,
    is_jd_recruitment_voice_context,
    persist_voice_screening_status_signal,
)
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.models.appointment import Appointment
from app.services.transcript_service import transcript_service
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.groq_service import groq_service
from app.services.rag_service import rag_service
from app.services.credit_service import credit_service
from app.services.twilio_service import twilio_service
from app.utils.voice_twilio_utils import (
    get_twilio_credentials_for_call,
    twilio_caller_id_for_transfer_dial,
)
from app.services.google_tts_service import google_tts_service
from app.utils.tts_adapter import get_tts_adapter
from app.voice.tone_adapter import tone_adapter
from app.voice.turn_signals import TurnContext, build_turn_context, build_user_signals_block
from app.utils.tts_preprocessing import detect_emotion
from app.core.config import settings
from app.routers.general_websocket import broadcast_call_status_update
from app.utils.tts_preprocessing import preprocess_for_tts, quick_clean
from app.voice.stt_pipeline import SttPipeline
from app.voice.tts_pipeline import TtsPipeline
from app.voice.background_audio import BackgroundAudioManager
from app.voice.conversation_orchestrator import (
    VOICE_TUNABLES,
    ConversationOrchestrator,
    should_send_quick_ack,
)
from app.voice.voice_orchestrator import VoiceOrchestrator
from app.voice.rag_context import build_rag_context_block, build_rag_context_block_with_trace
from app.voice.tts_only_session import TtsOnlySession
from app.voice.metrics import VoiceTurnMetrics

# Import utilities and services
from app.utils.audio_utils import (
    ulaw_to_linear_sample,
    stream_mulaw_bytes_over_twilio,
    crossfade_mulaw_segments,
    build_crossfade_bridge,
    MULAW_FRAME_BYTES,
)
from app.utils.ssml_utils import (
    strip_ssml_tags,
    add_natural_ssml,
    clean_text_for_tts,
    smart_chunk_text
)
from app.services.bidirectional_stream_service import (
    generate_mulaw_tts,
    build_streaming_twiml,
    build_tts_only_twiml,
)
from app.utils.eleven_tts_text import (
    build_elevenlabs_audio_tag_prompt_block,
    get_elevenlabs_voice_prompt_rule_lines,
    prepare_tts_text_for_provider,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
    supports_elevenlabs_audio_tags,
)


from app.voice.booking_mixin import BookingMixin
from app.voice.tts_stream_mixin import TtsStreamMixin
from app.voice.call_control_mixin import CallControlMixin

router = APIRouter()

# Voice screening / hangup tokens (case-insensitive; models vary casing/spacing)
_RE_VOICE_END_CALL = re.compile(r"\[\s*END_CALL\s*\]", re.IGNORECASE)
_RE_VOICE_SCREENING_QUALIFIED = re.compile(r"\[\s*SCREENING_QUALIFIED\s*\]", re.IGNORECASE)

# User utterance indicates opt-out / wrong call — force hangup (recruitment outbound safety net)
_SCREENING_USER_DECLINE_RE = re.compile(
    r"(not\s+interested|no\s+thanks|no\s+thank\s+you|don'?t\s+call|stop\s+calling|wrong\s+(number|person|call)|"
    r"not\s+available|i\s+am\s+not\s+available|i'?m\s+not\s+available|can'?t\s+talk|"
    r"take\s+me\s+off|remove\s+me|not\s+looking|already\s+placed|not\s+for\s+me|"
    r"wrong\s+call|decline\s+this|reject\s+this)",
    re.IGNORECASE,
)

# Vapi-style customEndpointingRules analogue: when the agent asks for email, we either
# defer a longer Deepgram endpointing for the first STT session or reconnect once with
# DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED so spelling pauses do not split finals prematurely.
# ── Speculative TTS opener rules ─────────────────────────────────────────────
# Compiled once at module load.  Ordered most-specific first so the first match
# wins.  Each entry: (pattern, predicted_first_chunk_text).
# The prediction only needs to be accurate enough to be useful — even a 30-40%
# hit rate eliminates TTS synthesis latency for those turns entirely.
_SPECULATIVE_OPENER_RULES: list[tuple[re.Pattern, str]] = [
    # Booking / scheduling (specific before generic "what")
    (re.compile(r"\b(book|schedule|appointment|reserve|reserv|slot|meeting|available|availability|reschedule)\b", re.I), "Sure, let me check that."),
    # Pricing / cost
    (re.compile(r"\b(price|cost|how much|fee|rate|charge|pricing|quote)\b", re.I), "Sure,"),
    # Simple yes / affirmation (full utterance match)
    (re.compile(r"^(yes|yeah|yep|yup|sure|absolutely|of course|correct|right|exactly|definitely|perfect|great)[\s.,!?]*$", re.I), "Great!"),
    # Simple no / negative (full utterance match)
    (re.compile(r"^(no|nope|not really|nah|never mind|nevermind|cancel that)[\s.,!?]*$", re.I), "I understand."),
    # Help / problem
    (re.compile(r"\b(help|support|issue|problem|trouble|not working|broken|error|wrong)\b", re.I), "I understand."),
    # Greeting (full utterance)
    (re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|what'?s up)[\s!.,?]*$", re.I), "Hey!"),
    # Acknowledgement / filler (full utterance)
    (re.compile(r"^(okay|ok|alright|got it|sounds good|fine|cool|makes sense)[\s.,!?]*$", re.I), "Great."),
    # General information request
    (re.compile(r"\b(what|how|when|where|why|who|which|tell me|can you|could you|do you|does|is there)\b", re.I), "Sure,"),
]

_EMAIL_AGENT_PROMPT_FOR_EXTENDED_STT_RE = re.compile(
    r"(?i)(?:"
    r"(?:provide|share|send|give)\s+(?:us\s+)?(?:your\s+)?(?:e-?mail\s+address|e-?mail|email)|"
    r"(?:what(?:'s|\s+is)|may\s+i\s+have|can\s+i\s+(?:have|get))\s+(?:your\s+)?(?:e-?mail|email)(?:\s+address)?|"
    r"(?:your\s+)?(?:e-?mail|email)\s+address(?:,?\s*please)?|"
    r"\bspell\b.*\b(?:e-?mail|email)|\b(?:e-?mail|email)\b.*\bspell\b"
    r")",
)


class BidirectionalStreamHandler(BookingMixin, TtsStreamMixin, CallControlMixin):
    """Handles real-time bidirectional voice streaming (400–500ms target, Vapi-style gapless TTS)."""

    # Expose tunables on the class so they remain easy to discover in-context,
    # while the actual values live in VOICE_TUNABLES above.
    STT_INTERIM_INTERVAL_MS = VOICE_TUNABLES.stt_interim_interval_ms

    # Quick acknowledgement: 5-word rule + probability (Vapi+ naturalness)
    QUICK_ACK_MIN_WORDS = VOICE_TUNABLES.quick_ack.min_words
    QUICK_ACK_PROBABILITY = VOICE_TUNABLES.quick_ack.probability
    QUICK_ACK_SKIP_PHRASES = VOICE_TUNABLES.quick_ack.skip_phrases

    # Conversation context: keep the prompt small for latency (voice calls)
    HISTORY_MAX_MESSAGES = VOICE_TUNABLES.history_max_messages

    # Incremental TTS: flush when we have a complete thought/sentence
    TTS_FLUSH_MIN_WORDS = VOICE_TUNABLES.tts_flush_min_words
    TTS_FLUSH_MAX_WORDS = VOICE_TUNABLES.tts_flush_max_words  # keep chunks short for fast TTS start
    
    def __init__(
        self,
        websocket: WebSocket,
        call_session_id: str,
        agent_id: str,
        db: Session
    ):
        self.websocket = websocket
        self.call_session_id = call_session_id
        self.agent_id = agent_id
        self.db = db
        
        # STT (Input) state - Deepgram streaming with endpointing (speech_final)
        self.stream_sid = None
        self.call_sid = None
        self.current_speech = ""
        self._stt_pipeline: Optional[SttPipeline] = None
        # Interim → early LLM (optional; default off in settings for Deepgram stability)
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._enable_interim_llm: bool = bool(
            getattr(settings, "VOICE_ENABLE_INTERIM_LLM", False)
        )
        self._min_interim_words: int = max(
            1, int(getattr(settings, "VOICE_MIN_INTERIM_WORDS", 4))
        )
        self._min_interim_confidence: float = float(
            getattr(settings, "VOICE_MIN_INTERIM_CONFIDENCE", 0.52)
        )
        self._rag_prefetch_min_words: int = max(
            1, int(getattr(settings, "VOICE_RAG_PREFETCH_MIN_WORDS", 1) or 1)
        )
        self._rag_prefetch_min_confidence: float = float(
            getattr(settings, "VOICE_RAG_PREFETCH_MIN_CONFIDENCE", 0.05) or 0.05
        )
        self._min_interim_interval_sec = self.STT_INTERIM_INTERVAL_MS / 1000.0
        
        # TTS (Output) state - Parallel Pipeline
        self.is_speaking = False
        # _is_tts_playing: True ONLY when audio frames are actively streaming to
        # Twilio.  Used as the barge-in gate instead of is_speaking (which flips
        # True when synthesis tasks are created, before any audio reaches Twilio).
        # This prevents false-positive barge-in cancellation during the LLM→TTS
        # synthesis phase — the root cause of "2-3 words then silence".
        self._is_tts_playing: bool = False
        self._tts_cancel = asyncio.Event()   # barge-in cancel signal
        self._tts_lock = asyncio.Lock()      # serialize TTS streams
        self._tts_worker_task = None         # Backwards-compatible handle to pipeline worker
        self._tts_generation_tasks = []      # Track parallel TTS generation (reserved)
        self._prev_tts_tail = b""            # Last streamed audio tail for crossfade bridge
        self._tts_overlap_bytes = 400        # 50ms overlap at 8kHz (Vapi's approach for smooth transitions)
        self._twilio_buffer_primed = False   # Track if jitter buffer has been primed
        self._tts_pipeline: Optional[TtsPipeline] = None
        self._elevenlabs_prev_tts_text = ""
        self._use_ssml = True                # Enable SSML by default
        # Quick-ack dedup guard: prevent repeated acknowledgements for the same
        # user turn when interim/final regeneration paths overlap.
        self._last_quick_ack_user_norm: str = ""
        self._last_quick_ack_mono: float = 0.0
        # Latency instrumentation timestamps (time.perf_counter() — single clock for all deltas)
        self._metric_stt_final_ts: float = 0.0        # STT final received
        self._metric_gen_start_ts: float = 0.0        # LLM generation started
        self._metric_first_token_ts: float = 0.0      # First LLM token received
        self._metric_first_audio_ts: float = 0.0      # First audio frame sent to Twilio
        self._metric_barge_in_ts: float = 0.0         # Barge-in event fired
        self._metric_audio_cut_ts: float = 0.0        # Audio stream actually stopped
        
        # Session data
        self.call_session = None
        self.agent = None
        self._last_offered_calendar_slots: List[datetime] = []
        self._last_requested_calendar_date: Optional[date] = None
        self._last_selected_calendar_slot: Optional[datetime] = None
        self._load_session_data()
        
        # User pickup detection (VAPI-style: actual user audio = user picked up)
        self._user_picked_up = False
        self._first_media_received = False
        self._in_progress_sent = False  # Track if in-progress status has been sent
        self._skip_audio_until = None  # Timestamp until which to skip audio (system messages)
        self._audio_level_samples = []  # Track audio levels to detect actual user audio
        _r = int(getattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 70) or 70)
        self._min_audio_level_threshold = max(20, min(250, _r))  # linear RMS; clamp to avoid misconfig
        self._stt_min_final_confidence: float = float(
            getattr(settings, "VOICE_STT_MIN_FINAL_CONFIDENCE", 0.26) or 0.26
        )
        self._stt_min_final_confidence = max(0.15, min(0.45, self._stt_min_final_confidence))
        self._enable_soft_final_fallback: bool = bool(
            getattr(settings, "VOICE_STT_ENABLE_SOFT_FINAL_FALLBACK", True)
        )
        self._stt_soft_min_final_confidence: float = float(
            getattr(settings, "VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE", 0.16) or 0.16
        )
        self._stt_soft_min_final_confidence = max(0.10, min(0.35, self._stt_soft_min_final_confidence))
        self._stt_soft_min_words: int = int(
            getattr(settings, "VOICE_STT_SOFT_MIN_WORDS", 2) or 2
        )
        self._stt_soft_min_words = max(1, min(6, self._stt_soft_min_words))
        self._barge_in_min_conf: float = float(
            getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE", 0.26) or 0.26
        )
        self._barge_in_min_conf = max(0.15, min(0.5, self._barge_in_min_conf))
        self._barge_in_min_conf_1w: float = float(
            getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE_1W", 0.52) or 0.52
        )
        self._barge_in_min_conf_1w = max(0.4, min(0.75, self._barge_in_min_conf_1w))
        self._barge_in_min_words: int = int(
            getattr(settings, "VOICE_BARGE_IN_MIN_WORDS", 2) or 2
        )
        self._barge_in_min_words = max(1, min(4, self._barge_in_min_words))
        self._barge_in_rejected_while_playing: int = 0
        self._audio_samples_needed = max(
            4, int(getattr(settings, "VOICE_PICKUP_SAMPLE_WINDOW", 6) or 6)
        )
        self._audio_non_silent_needed = max(
            1,
            min(
                self._audio_samples_needed,
                int(
                    getattr(
                        settings,
                        "VOICE_PICKUP_MIN_NON_SILENT_FRAMES",
                        self._audio_samples_needed,
                    )
                    or self._audio_samples_needed
                ),
            ),
        )
        self._stream_sid_ready = asyncio.Event()
        
        # Goodbye detection state
        self._call_ended = False  # Track if call has been ended due to goodbye detection
        # True when the agent's final streamed reply included successful-screening tokens before hangup
        self._pending_resume_screening_qualify = False

        # STT feed gate: set to False on stop/disconnect so media frames are not pushed
        # to Deepgram after the call ends (prevents silence-frame leak post call-end).
        self._stt_active = True

        # Signals the main receive loop to break cleanly when the call ends internally
        # (goodbye phrase, [END_CALL] token, voicemail, etc.) so the WebSocket closes
        # without waiting for Twilio to send a `stop` event.
        self._stop_event = asyncio.Event()
        # Post-call appointment finalization (exactly one task per handler lifetime)
        self._post_call_orchestration_scheduled = False
        # Longer Deepgram endpointing after agent asks for email (one-time upgrade per call).
        self._email_stt_endpointing_upgraded = False
        self._stt_deferred_endpointing_ms: Optional[int] = None

        # One response per turn: first interim that passes gates starts the LLM (dev-style: commit agent at stream end)
        self._turn_response_started = False  # True after first interim triggers LLM for this turn
        self._turn_response_seed_text = ""
        # In-flight LLM+TTS; wrapped in a task so barge-in can cancel while we await (like dev, but cancelable)
        self._llm_response_task: Optional[asyncio.Task] = None
        self._auto_greeting_sent = False
        self._recording_started = False
        self._screening_decline_handled = False

        # Deepgram can emit the same is_final twice within a short window — avoid triple LLM/transcript
        self._stt_last_final_raw: str = ""
        self._stt_last_final_monotonic: float = 0.0
        # Widened from 2.5s to 6.0s: Twilio sometimes re-endpoints the same short utterance
        # ("Hello?", "Yes", "Okay") up to ~5s apart, producing two finals with identical text
        # that both trigger the LLM → duplicate agent replies. One STT final per user turn is the goal.
        self._STT_DEDUP_FINAL_WINDOW_SEC: float = 6.0

        # Turn coordinator: remember the last few (user_norm, agent_norm, ts) pairs so we can
        #   (a) short-circuit a generate when the same user turn repeats within 15s, and
        #   (b) suppress duplicate agent transcript writes within 25s (belt-and-suspenders so
        #       the visible transcript never shows the same agent line twice).
        # Bounded to the last 5 entries — O(1) cost, no added latency.
        self._recent_agent_pairs: list[tuple[str, str, float]] = []
        self._DUP_USER_TURN_WINDOW_SEC: float = 15.0
        self._AGENT_LINE_DEDUP_WINDOW_SEC: float = 25.0
        self._RECENT_AGENT_PAIRS_MAX: int = 5

        # RAG prefetch: fired on the first qualifying interim so vector-DB retrieval
        # overlaps Deepgram endpointing time instead of blocking LLM start.
        self._rag_prefetch_task: Optional[asyncio.Task] = None
        self._rag_prefetch_user_text: str = ""

        # Speculative TTS prefetch: fired on STT final (and on qualifying interim)
        # to synthesise the predicted first-chunk text during LLM TTFT.  When the
        # prediction is correct the first TtsPipeline chunk is a cache hit → zero
        # synthesis latency for chunk 0.  Wrong predictions cost one wasted TTS API
        # call and are silently discarded.
        self._speculative_prefetch_task: Optional[asyncio.Task] = None

        # Duplicate-transcript guard for _complete_llm_turn_after_stt_final.
        # Set INSIDE _llm_turn_serial_lock so tasks 2/3 that queued up behind
        # task 1 drop out immediately once task 1 has answered the transcript.
        self._llm_last_answered_transcript: str = ""
        self._llm_last_answered_ts: float = 0.0

        # KB / business-knowledge blocks: agent-level data that never changes during
        # a call.  Fetched once in the background at call start so every subsequent
        # turn pays zero DB latency for these blocks (saves 50-200ms per slow turn).
        self._cached_inbound_kb_block: str = ""
        self._cached_business_knowledge_block: str = ""
        self._kb_cache_ready: bool = False  # True once the background fetch completes

        # In-memory conversation history: avoids re-parsing the growing
        # call_transcript JSON on every turn (saves 5-30ms, grows with call length).
        # Entries are (role, content) tuples matching the transcript filter rules.
        self._conversation_history_cache: list[tuple[str, str]] = []

        # Final transcript bookkeeping (dedupe, DB writes): hold briefly — never across LLM+TTS.
        self._voice_transcript_lock = asyncio.Lock()
        # One completion at a time (interim/final regen policy, task awaits). Matches product
        # platforms (Vapi-style): transcript ingestion stays concurrent; generation serializes.
        self._llm_turn_serial_lock = asyncio.Lock()
        self._voice_metrics = VoiceTurnMetrics()

        # Background audio manager (dev-branch style embedded ambience loop).
        self._background_audio = BackgroundAudioManager(
            websocket=self.websocket,
            get_stream_sid=lambda: self.stream_sid,
            is_speaking_flag=lambda: self.is_speaking,
        )
        asyncio.create_task(self._background_audio.load_from_base64_async())

        # ── OLD direct pipeline wiring (replaced by VoiceOrchestrator below) ──
        # self._tts_pipeline = TtsPipeline(self)
        # self._tts_worker_task = self._tts_pipeline._worker_task
        # self._conversation = ConversationOrchestrator(self)
        # ─────────────────────────────────────────────────────────────────────

        # Pre-cache quick-ack phrases so they hit the LRU audio cache instead of
        # paying TTS API latency on the first "Got it" / "Sure" of the call.
        asyncio.create_task(self._precache_common_phrases())

        # Fetch KB/business-knowledge blocks in the background at call start.
        # These are agent-level and don't change per-turn; caching them here
        # eliminates the 50-200ms parallel executor cost on every slow-path turn.
        asyncio.create_task(self._prefetch_kb_blocks_at_call_start())

        # ── Voice Orchestration Layer ─────────────────────────────────────────
        # VoiceOrchestrator owns SttPipeline + TtsPipeline lifecycle and
        # coordinates the full turn flow (pickup → STT → barge-in → LLM → TTS).
        # It writes _tts_pipeline and _tts_worker_task back onto this handler
        # so all existing methods that reference self._tts_pipeline keep working.
        self._voice_orchestrator = VoiceOrchestrator(self)
    
    def _load_session_data(self):
        """Load call session and agent data"""
        try:
            session_uuid = uuid.UUID(self.call_session_id)
            self.call_session = call_session_service.get_call_session_by_id(self.db, session_uuid)
            
            if self.call_session and self.agent_id:
                agent_uuid = uuid.UUID(self.agent_id)
                self.agent = agent_service.get_agent_by_id(
                    self.db,
                    agent_uuid,
                    self.call_session.tenant_id
                )
                # Lazy safety net: ensure prompt KB exists for older agents
                # created before auto-ingest rollout.
                agent_service.ensure_agent_prompt_ingested(self.db, self.agent)
        except Exception as e:
            logger.error(f"Error loading session data: {e}", exc_info=True)

    def _jd_recruitment_screening_active(self) -> bool:
        """JD-backed recruitment screening only — skips booking/general outbound calls."""
        if not self.call_session or not isinstance(self.call_session.call_metadata, dict):
            return False
        return is_jd_recruitment_voice_context(self.call_session.call_metadata.get("jd_context"))

    async def _precache_common_phrases(self):
        """
        Pre-generate and cache common phrases for instant playback.
        Runs in background during initialization.
        """
        try:
            # Let call setup/pickup settle first so warmup does not contend with
            # first-turn STT/LLM/TTS work.
            await asyncio.sleep(1.5)
            # Common phrases for instant responses (greetings, confirmations, acknowledgements)
            common_phrases = [
                # Greetings
                "Hello",
                "Hi there",
                "Hi",
                "Good morning",
                "Good afternoon",
                "Good evening",
                
                # Acknowledgements (Quick feedback)
                "Got it",
                "I see",
                "Okay",
                "Sure",
                "Alright",
                "Perfect",
                "Great",
                "Understood",
                
                # Confirmations
                "Yes",
                "No",
                "Absolutely",
                "Of course",
                
                # Thinking/Processing
                "Let me check that",
                "One moment please",
                "Just a second",
                "Let me see",
                
                # Transitions
                "Thank you",
                "Thanks",
                "You're welcome",
                
                # Closings
                "Goodbye",
                "Have a great day",
                "Thank you for calling",
                "Talk to you later",
            ]
            
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            
            sem = asyncio.Semaphore(4)

            async def _warm_phrase(phrase: str) -> None:
                async with sem:
                    try:
                        await generate_mulaw_tts(
                            text=phrase,
                            lang=lang,
                            voice=voice,
                            use_chirp3_hd=True,
                            speaking_rate=1.0,
                            use_ssml=False,
                        )
                    except Exception:
                        return

            await asyncio.gather(*(_warm_phrase(phrase) for phrase in common_phrases))
        except Exception as e:
            logger.error(f"Error in precache_common_phrases: {e}")

    async def _prefetch_kb_blocks_at_call_start(self) -> None:
        """
        Fetch inbound-KB and business-knowledge context blocks once at call start.
        These are agent/tenant-level and don't change during a call, so caching here
        saves the 50-200ms parallel-executor cost that was previously paid on every
        slow-path turn inside generate_and_stream_response.
        """
        try:
            tenant_uuid = self.call_session.tenant_id if self.call_session else None
            agent_uuid = self.agent.id if self.agent else None
            if not (tenant_uuid or agent_uuid):
                self._kb_cache_ready = True
                return

            loop = asyncio.get_running_loop()

            async def _fetch_inbound_kb() -> str:
                if not (self.agent and self.agent.is_inbound_agent and tenant_uuid and agent_uuid):
                    return ""
                try:
                    return await loop.run_in_executor(
                        None,
                        lambda: agent_service.build_inbound_kb_documents_context_block(
                            db=self.db, inbound_agent_id=agent_uuid, tenant_id=tenant_uuid,
                        ),
                    )
                except Exception as exc:
                    logger.warning("[KB prefetch] inbound KB fetch failed: %s", exc)
                    return ""

            async def _fetch_business_knowledge() -> str:
                if not tenant_uuid:
                    return ""
                try:
                    return await loop.run_in_executor(
                        None,
                        lambda: agent_service.build_business_knowledge_context_block(
                            db=self.db, tenant_id=tenant_uuid, agent_id=agent_uuid,
                        ),
                    )
                except Exception as exc:
                    logger.warning("[KB prefetch] business knowledge fetch failed: %s", exc)
                    return ""

            kb, bk = await asyncio.gather(_fetch_inbound_kb(), _fetch_business_knowledge())
            self._cached_inbound_kb_block = kb
            self._cached_business_knowledge_block = bk
            self._kb_cache_ready = True
            logger.debug("[KB prefetch] cache ready (inbound_kb=%d chars, bk=%d chars)", len(kb), len(bk))
        except Exception as exc:
            logger.warning("[KB prefetch] call-start fetch failed: %s", exc)
            self._kb_cache_ready = True  # allow the call to proceed without KB context

    # ── Speculative TTS prefetch ──────────────────────────────────────────────

    @staticmethod
    def _predict_opener_phrase(user_text: str) -> Optional[str]:
        """
        Fast rule-based prediction of the most likely first TTS chunk text.

        Runs in ~0.1ms (no LLM call, no DB).  Called at Deepgram endpointing time
        so synthesis can start during LLM TTFT.  Returns None when no confident
        prediction can be made (no synthesis waste on those turns).
        """
        text = (user_text or "").strip()
        if not text:
            return None
        for pattern, phrase in _SPECULATIVE_OPENER_RULES:
            if pattern.search(text):
                return phrase
        return None

    async def _run_speculative_tts_prefetch(self, user_text: str) -> None:
        """
        Synthesise the predicted opener phrase and inject it into TtsPipeline's
        LRU audio cache so the real chunk-0 is a cache hit (zero synthesis latency).

        Fires as a background task at STT-final time (and optionally on interim)
        so synthesis overlaps LLM TTFT.  If the prediction is wrong the cached
        audio is simply never read and evicted normally.
        """
        try:
            phrase = self._predict_opener_phrase(user_text)
            if not phrase:
                return
            pipeline = self._tts_pipeline
            if pipeline is None:
                return
            if self._tts_cancel.is_set():
                return

            from app.voice.tts_pipeline import TtsPipeline as _TtsPipeline
            cache_key = _TtsPipeline._cache_key(phrase)

            # Skip if already warm (e.g. common phrase pre-cached at call start)
            if pipeline._get_cached(cache_key) is not None:
                logger.debug("[SpecTTS] '%s' already cached — skipping synthesis", phrase)
                return

            task = {"text": phrase, "use_ssml": False, "is_final": False}
            audio_bytes = await self._prefetch_tts_audio(task)
            if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                if self._tts_pipeline and not self._tts_cancel.is_set():
                    self._tts_pipeline._put_cached(cache_key, audio_bytes)
                    logger.debug(
                        "[SpecTTS] pre-synthesized '%s' (%d bytes) — chunk-0 will be cache hit",
                        phrase, len(audio_bytes),
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("[SpecTTS] prefetch failed: %s", exc)

    async def handle_media_message(self, message: dict):
        """
        Handle incoming MULAW audio from Twilio.

        The heavy lifting (pickup detection, grace period, STT pipeline lifecycle,
        barge-in, and STT callbacks) now lives in VoiceOrchestrator.on_audio_chunk().
        This method is intentionally thin — it just decodes the base64 payload and
        hands the raw bytes to the orchestrator.

        ── OLD direct pipeline code (now owned by VoiceOrchestrator) ────────────
        # if not self._user_picked_up:  ... RMS pickup detection ...
        # if self._skip_audio_until ... grace period ...
        # if not self._stt_active: return
        # if self._stt_pipeline is None: ... SttPipeline(...) ...
        # await self._stt_pipeline.feed_audio_chunk(audio_data)
        # ────────────────────────────────────────────────────────────────────────
        """
        try:
            payload = message.get("media", {}).get("payload")
            if not payload:
                return

            audio_data = base64.b64decode(payload)
            await self._voice_orchestrator.on_audio_chunk(audio_data)

        except Exception as e:
            logger.error(f"Error handling media message: {e}", exc_info=True)

    def _schedule_recreate_stt_for_email_collection(self, agent_text: str) -> None:
        """
        Defer STT session reconnect to the next event-loop tick.
        Delegates to VoiceOrchestrator which owns the SttPipeline lifecycle.

        ── OLD direct implementation (now in VoiceOrchestrator) ─────────────────
        # async def _deferred(): await self._maybe_recreate_stt_for_email_collection(text)
        # asyncio.create_task(_deferred())
        # ────────────────────────────────────────────────────────────────────────
        """
        self._voice_orchestrator.schedule_stt_recreate_for_email(agent_text)

    # _maybe_recreate_stt_for_email_collection is now handled inside
    # VoiceOrchestrator._maybe_upgrade_stt_for_email — kept as a no-op stub
    # for any call-site that might reference it directly.
    async def _maybe_recreate_stt_for_email_collection(self, agent_text: str) -> None:
        await self._voice_orchestrator._maybe_upgrade_stt_for_email(agent_text)
    
    # Removed chunk-based STT processing; relying on Deepgram streaming endpointing

    async def _cancel_inflight_llm_response(self) -> None:
        """Stop background LLM+TTS for this turn (barge-in or final regen)."""
        t = self._llm_response_task
        self._llm_response_task = None
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        if self._tts_pipeline:
            await self._tts_pipeline.cancel_current_and_clear_queue()
        # Discard stale RAG prefetch so the next turn gets a fresh retrieval.
        prefetch = getattr(self, "_rag_prefetch_task", None)
        if prefetch and not prefetch.done():
            prefetch.cancel()
        self._rag_prefetch_task = None
        self._rag_prefetch_user_text = ""

        # Cancel any in-flight speculative TTS synthesis (barge-in resets the turn).
        spec = getattr(self, "_speculative_prefetch_task", None)
        if spec and not spec.done():
            spec.cancel()
        self._speculative_prefetch_task = None

    async def _respond_screening_decline_and_end(self) -> None:
        """User declined screening — short goodbye TTS then hang up (no LLM)."""
        try:
            await self._cancel_inflight_llm_response()
            self._pending_resume_screening_qualify = False
            msg = "Thanks for letting me know. Take care."
            self._tts_cancel.clear()
            self._prev_tts_tail = b""
            await self._add_to_transcript("agent", msg, "agent_response")
            if self._tts_pipeline:
                await self._tts_pipeline.queue_tts({
                    "text": msg,
                    "use_ssml": self._use_ssml,
                    "is_final": True,
                    "end_call_after": True,
                })
        except Exception as e:
            logger.error("respond_screening_decline_and_end: %s", e, exc_info=True)

    async def _complete_llm_turn_after_stt_final(self, transcript: str, confidence: float) -> None:
        """
        Run after the user's final message is in the DB.

        KEY DESIGN: _llm_turn_serial_lock is held only for the fast state-check
        (~1ms).  generate_and_stream_response runs OUTSIDE the lock so a new STT
        final from the user can acquire the lock immediately (no 2.6s wait) and
        cancel this generation if a new utterance arrives.  This is what brings
        stt_final→gen_start from 2-19s down to ~0ms.
        """
        tstrip = (transcript or "").strip()

        # Recruitment: user opted out / wrong call — end immediately (backend safety net)
        if (
            tstrip
            and not self._screening_decline_handled
            and self.call_session
            and self._jd_recruitment_screening_active()
            and _SCREENING_USER_DECLINE_RE.search(tstrip)
        ):
            self._screening_decline_handled = True
            asyncio.create_task(self._respond_screening_decline_and_end())
            return

        # ── Phase 1: state check (inside lock, fast) ──────────────────────────
        _should_generate = False
        _need_cancel = False

        async with self._llm_turn_serial_lock:
            _now_m = time.monotonic()

            # Duplicate-transcript gate: near-simultaneous STT finals for the
            # same utterance (Deepgram re-endpoints) all queue here.  Once the
            # first sets _llm_last_answered_transcript, the rest bail instantly.
            if (
                tstrip
                and tstrip == self._llm_last_answered_transcript
                and (_now_m - self._llm_last_answered_ts) < 10.0
            ):
                logger.info(
                    "[LLMlock] dropping duplicate transcript '%s' (answered %.1fs ago)",
                    tstrip[:40], _now_m - self._llm_last_answered_ts,
                )
                return

            if self._turn_response_started:
                should_regenerate = self._should_regenerate_on_final(transcript)
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                if should_regenerate:
                    _should_generate = True
                    _need_cancel = True
                    self._llm_last_answered_transcript = tstrip
                    self._llm_last_answered_ts = _now_m
                else:
                    # Interim LLM already running with the correct transcript —
                    # let it finish without a second generation.
                    pass
                return
            else:
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                _should_generate = True
                _need_cancel = True  # cancel any stale task from a previous turn
                self._llm_last_answered_transcript = tstrip
                self._llm_last_answered_ts = _now_m
        # ── Lock released ──────────────────────────────────────────────────────

        if not _should_generate:
            return

        # ── Phase 2: cancel previous + generate (outside lock) ────────────────
        # Cancelling here (outside lock) means a new STT final can acquire the
        # lock while we're waiting for TTS cancel (~250ms) — it will see the
        # updated _llm_last_answered_* and bail, preventing a race.
        if _need_cancel:
            await self._cancel_inflight_llm_response()

        self._tts_cancel.clear()
        try:
            await asyncio.wait_for(
                self.generate_and_stream_response(transcript, confidence, is_greeting=False),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            logger.error("[LLM] generate_and_stream_response timed out (12s) — aborting turn")

    def _should_accept_final_transcript(self, transcript: str, confidence: float) -> bool:
        """
        Primary gate for Deepgram final transcripts:
        - Keep strong confidence threshold for normal flow.
        - Optional soft fallback accepts likely-human speech at lower confidence so
          soft callers are not dropped mid-call.
        """
        text = (transcript or "").strip()
        if not text:
            return False
        if confidence >= self._stt_min_final_confidence:
            return True
        if not self._enable_soft_final_fallback:
            return False
        if confidence < self._stt_soft_min_final_confidence:
            return False
        words = text.split()
        if len(words) < self._stt_soft_min_words:
            return False
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if alpha_chars < 3:
            return False
        # Ignore pure filler-ish very short low-confidence utterances.
        low = re.sub(r"[^a-z ]+", "", text.lower()).strip()
        filler = {"uh", "um", "hmm", "mm", "ah", "er", "huh", "hmm hmm", "uh huh"}
        if low in filler:
            return False
        return True

    _BARGE_IN_FILLER_WORDS = frozenset(
        {
            "uh",
            "um",
            "hmm",
            "mm",
            "ah",
            "er",
            "huh",
            "mhm",
            "mmm",
            "eh",
            "ugh",
        }
    )

    def _is_stt_filler_for_barge_in(self, transcript: str) -> bool:
        """Reject phantom Deepgram hits (uh/mm, uh huh) from cutting active TTS."""
        low = re.sub(r"[^a-z ]+", "", (transcript or "").lower()).strip()
        if not low:
            return True
        if low in self._BARGE_IN_FILLER_WORDS:
            return True
        tokens = low.split()
        return bool(tokens) and all(t in self._BARGE_IN_FILLER_WORDS for t in tokens)

    def _should_barge_in_on_stt(self, transcript: str, confidence: float) -> bool:
        """
        True when STT looks like real user speech over the agent (not noise/filler).

        Gates (all required): non-empty text, not filler-only, word count ≥
        VOICE_BARGE_IN_MIN_WORDS, and confidence at or above
        VOICE_BARGE_IN_MIN_CONFIDENCE (multi-word) or VOICE_BARGE_IN_MIN_CONFIDENCE_1W
        when min words is 1.
        """
        text = (transcript or "").strip()
        if not text or self._is_stt_filler_for_barge_in(text):
            return False
        word_count = len(text.split())
        if word_count < self._barge_in_min_words:
            return False
        if word_count >= 2:
            return confidence >= self._barge_in_min_conf
        return confidence >= self._barge_in_min_conf_1w

    def _log_barge_in_suppressed(self, transcript: str, confidence: float, reason: str) -> None:
        """Staging/debug: count STT events that looked like speech but failed barge-in gates."""
        self._barge_in_rejected_while_playing += 1
        logger.debug(
            "[Barge-in] suppressed (%s): words=%d conf=%.2f rejected_total=%d text=%r",
            reason,
            len((transcript or "").split()),
            confidence,
            self._barge_in_rejected_while_playing,
            (transcript or "")[:40],
        )

    async def _process_transcript(self, transcript: str, confidence: float):
        """Process a transcript (final result)"""
        try:
            # Barge-in on FINAL events: cut playing TTS before DB work when STT passes gates.
            if self._is_tts_playing and self._should_barge_in_on_stt(transcript, confidence):
                self._metric_barge_in_ts = time.perf_counter()
                logger.info(
                    "[Barge-in/final] TTS cut by final STT: %r",
                    (transcript or "")[:40],
                )
                await self._cancel_inflight_llm_response()
                self._metric_audio_cut_ts = time.perf_counter()
                _cut_ms = (self._metric_audio_cut_ts - self._metric_barge_in_ts) * 1000
                logger.info(
                    "[Metrics] interruption_detection_to_audio_cut=%.0f ms (final)",
                    _cut_ms,
                )
                self._is_tts_playing = False
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                # Fall through — still process transcript and generate new response.

            async with self._voice_transcript_lock:
                if not self._should_accept_final_transcript(transcript, confidence):
                    return

                # Skip duplicate finals (e.g. same "Hello?" endpointed multiple times).
                tstrip = (transcript or "").strip()
                _now = time.monotonic()
                if (
                    tstrip
                    and tstrip == self._stt_last_final_raw
                    and (_now - self._stt_last_final_monotonic) < self._STT_DEDUP_FINAL_WINDOW_SEC
                ):
                    logger.debug("STT: skipping duplicate final within %ss", self._STT_DEDUP_FINAL_WINDOW_SEC)
                    return
                self._stt_last_final_raw = tstrip
                self._stt_last_final_monotonic = _now

                # STT self-echo guard: discard transcripts that closely match
                # recently spoken agent text. Occurs when the STT mic picks up
                # the agent's TTS output on the phone line (sidetone / open mic).
                if self._is_agent_self_echo(tstrip):
                    logger.info("STT: suppressing agent self-echo: %r", tstrip)
                    return

                self._voice_metrics.begin_turn_at_stt_final()
                self._metric_stt_final_ts = time.perf_counter()

                # 🎯 Check for goodbye words FIRST - end call if detected
                if await self._check_and_end_call_if_goodbye(transcript):
                    return  # Stop processing - call is ending

                # 🎯 Check for voicemail detection - end call if detected
                if await self._check_and_end_call_if_voicemail(transcript):
                    return  # Stop processing - call is ending

                # 🎯 Send "in-progress" status when confident word is detected (like "hello")
                if not self._in_progress_sent and confidence >= 0.1 and len(transcript.split()) > 0:
                    await self._send_in_progress_status(transcript, confidence)
                    self._in_progress_sent = True

                # Add to transcript (always)
                await self._add_to_transcript(
                    "client",
                    transcript,
                    "speech",
                    confidence,
                    defer_post_write=True,
                )
                self._update_booking_memory_from_user_turn(transcript)

                user_turn_norm = self._normalize_turn_text(transcript)
                if self._has_recent_duplicate_reply_for(user_turn_norm):
                    logger.info(
                        "TurnCoordinator: suppressing duplicate generate for user turn=%r (within %ss)",
                        transcript,
                        self._DUP_USER_TURN_WINDOW_SEC,
                    )
                    self._turn_response_started = False
                    self._turn_response_seed_text = ""
                    self._last_interim_text = ""
                    return

            # Speculative TTS prefetch: synthesise the predicted opener phrase during
            # LLM TTFT so chunk-0 is a cache hit when the real first flush arrives.
            # Fire-and-forget — cancelled on barge-in alongside the LLM task.
            if self._speculative_prefetch_task and not self._speculative_prefetch_task.done():
                self._speculative_prefetch_task.cancel()
            self._speculative_prefetch_task = asyncio.create_task(
                self._run_speculative_tts_prefetch(transcript)
            )

            # Never hold _voice_transcript_lock across LLM+TTS — queued Deepgram finals would
            # stall here for one full generation (~10–60s), inflating stt_final→gen_start.
            await self._complete_llm_turn_after_stt_final(transcript, confidence)

        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        At most one early LLM+TTS run per user utterance when `VOICE_ENABLE_INTERIM_LLM` is True.
        Default is False: LLM runs on **final** STT only, avoiding double replies from Deepgram
        partials (e.g. "I'm" then "I'm feeling sad"). Barge-in still uses interim. When enabled,
        gates use `VOICE_MIN_INTERIM_WORDS` + `VOICE_MIN_INTERIM_CONFIDENCE`.
        """
        try:
            if not transcript:
                return

            word_count = len(transcript.split())

            # ── Barge-in gate ────────────────────────────────────────────────────
            # Cut when audio is actively playing AND STT looks like real speech.
            # `_is_tts_playing` only gates on streaming audio; word count, confidence,
            # and filler rejection block phantom Deepgram hits on silence.
            is_barge_in = False
            if self._is_tts_playing:
                if self._should_barge_in_on_stt(transcript, confidence):
                    is_barge_in = True
                elif (transcript or "").strip():
                    if self._is_stt_filler_for_barge_in(transcript):
                        self._log_barge_in_suppressed(transcript, confidence, "filler")
                    elif word_count < self._barge_in_min_words:
                        self._log_barge_in_suppressed(transcript, confidence, "min_words")
                    else:
                        self._log_barge_in_suppressed(transcript, confidence, "confidence")
            if is_barge_in:
                self._metric_barge_in_ts = time.perf_counter()
                logger.info(
                    "[Barge-in] triggered: words=%d conf=%.2f transcript=%r",
                    word_count, confidence, (transcript or "")[:40],
                )
                await self._cancel_inflight_llm_response()
                self._metric_audio_cut_ts = time.perf_counter()
                _cut_ms = (self._metric_audio_cut_ts - self._metric_barge_in_ts) * 1000
                logger.info(
                    "[Metrics] interruption_detection_to_audio_cut=%.0f ms", _cut_ms
                )
                self._is_tts_playing = False
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                return

            # RAG prefetch: fire vector-DB retrieval as soon as we have a confident
            # partial so it overlaps Deepgram endpointing time. This saves ~2s because
            # the result is ready (or nearly so) by the time generate_and_stream_response
            # would otherwise block waiting for Pinecone. Fires once per user turn
            # regardless of VOICE_ENABLE_INTERIM_LLM so even final-only mode benefits.
            _prefetch = getattr(self, "_rag_prefetch_task", None)
            if (
                not _prefetch
                and not self._turn_response_started
                and word_count >= self._rag_prefetch_min_words
                and confidence >= self._rag_prefetch_min_confidence
            ):
                self._rag_prefetch_user_text = transcript
                self._rag_prefetch_task = asyncio.create_task(
                    self._prefetch_rag_context(transcript)
                )

            # Speculative TTS prefetch: also fire on interim so synthesis starts during
            # the Deepgram endpointing window (200ms), maximising overlap with LLM TTFT.
            # Only fire once per turn (guard: task not already running).
            if (
                not self._turn_response_started
                and word_count >= self._min_interim_words
                and confidence >= self._min_interim_confidence
                and (
                    self._speculative_prefetch_task is None
                    or self._speculative_prefetch_task.done()
                )
            ):
                self._speculative_prefetch_task = asyncio.create_task(
                    self._run_speculative_tts_prefetch(transcript)
                )

            # Final-only mode: do not start LLM from partials (barge-in already handled).
            if not self._enable_interim_llm:
                return

            # At most one interim LLM start per user turn; further partials are ignored
            if self._turn_response_started:
                return

            if self._should_defer_interim_response(transcript):
                return

            if confidence < self._min_interim_confidence or word_count < self._min_interim_words:
                return

            now = asyncio.get_event_loop().time()
            if (now - self._last_interim_sent_ts) < self._min_interim_interval_sec:
                return

            if self._last_interim_text and transcript.startswith(self._last_interim_text):
                advanced = transcript[len(self._last_interim_text) :].strip()
                if not advanced:
                    return

            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
            self._turn_response_started = True
            self._turn_response_seed_text = transcript

            async def _run_interim() -> None:
                await self.generate_and_stream_response(
                    transcript,
                    confidence,
                    is_greeting=False,
                )

            if self._llm_response_task and not self._llm_response_task.done():
                self._llm_response_task.cancel()
                try:
                    await self._llm_response_task
                except asyncio.CancelledError:
                    pass
            # Fire LLM task without awaiting here — returning immediately keeps the
            # STT reader loop unblocked so barge-in interims can still be processed
            # during the full 8-10s LLM+TTS cycle.
            self._llm_response_task = asyncio.create_task(_run_interim())
        except Exception as e:
            logger.error(f"Error processing interim: {e}")
    
    async def _send_quick_acknowledgement(self, user_text: str):
        """
        Send instant acknowledgement for longer queries while generating full response.
        Probability-based (QUICK_ACK_PROBABILITY) so we don't say "Got it" every time — more natural.
        Skips emotional/serious content so we never ack with "Got it" to e.g. "I have an emergency".
        """
        import random
        
        text = (user_text or "").strip()
        if not text:
            return
        now_mono = time.monotonic()
        user_norm = self._normalize_turn_text(text)
        if (
            user_norm
            and user_norm == self._last_quick_ack_user_norm
            and (now_mono - self._last_quick_ack_mono) < 12.0
        ):
            return
        # First check if this text is even eligible for a quick acknowledgement
        if not should_send_quick_ack(text, VOICE_TUNABLES.quick_ack):
            return

        # Apply probability filter so we don't say "Got it" every single time
        if random.random() >= VOICE_TUNABLES.quick_ack.probability:
            return
        acks = [
            "Got it",
            "I see",
            "Okay",
            "Alright",
            "Sure",
            "Mm-hmm",
            "Oh, okay",
            "One moment",
            "Hang on a sec",
            "Let me check that",
        ]
        ack = random.choice(acks)
        if not self._tts_pipeline:
            return
        await self._tts_pipeline.queue_tts({
            "text": ack,
            "chunk_id": "quick_ack",
            "use_ssml": False,
            "is_acknowledgement": True,
            "is_final": False
        })
        self._last_quick_ack_user_norm = user_norm
        self._last_quick_ack_mono = now_mono

    async def _prefetch_rag_context(self, user_text: str) -> tuple:
        """
        Run RAG vector-DB retrieval in the background so the result is available
        (or nearly so) by the time generate_and_stream_response needs it.

        Called as asyncio.create_task() from _maybe_process_interim on the first
        qualifying partial — this overlaps Deepgram's endpointing silence window
        (~200ms) and eliminates what would otherwise be a sequential ~2s RAG block
        before every LLM start.
        """
        try:
            tenant_uuid = self.call_session.tenant_id if self.call_session else None
            agent_uuid = self.agent.id if self.agent else None
            rag_agent_scope = None if (self.agent and self.agent.is_inbound_agent) else agent_uuid
            loop = asyncio.get_running_loop()

            def _build():
                return build_rag_context_block_with_trace(
                    user_text=user_text,
                    tenant_id=tenant_uuid,
                    agent_id=rag_agent_scope,
                )

            return await asyncio.wait_for(
                loop.run_in_executor(None, _build),
                timeout=settings.RAG_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.debug("[RAG prefetch] timed out for '%s…'", user_text[:20])
        except Exception as exc:
            logger.debug("[RAG prefetch] failed: %s", exc)
        # Fallback: empty context (LLM still runs, just without RAG)
        tenant_uuid = self.call_session.tenant_id if self.call_session else None
        agent_uuid = self.agent.id if self.agent else None
        rag_agent_scope = None if (self.agent and self.agent.is_inbound_agent) else agent_uuid
        return build_rag_context_block_with_trace(
            user_text="",
            tenant_id=tenant_uuid,
            agent_id=rag_agent_scope,
        )

    async def generate_and_stream_response(
        self,
        user_text: str,
        confidence: float,
        is_greeting: bool = False,
    ):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        Agent reply is always committed to the transcript at end of stream (dev-style for interim and final).

        Args:
            user_text: User's input text (empty for greeting)
            confidence: STT confidence score
            is_greeting: If True, plays inbound one-time opening (greeting_message / first_message) without LLM
        """
        try:
            from datetime import datetime, timezone
            import json
            
            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting (inbound pickup only)
            if is_greeting:
                greeting_text = None
                inbound = (
                    self.call_session is not None
                    and (self.call_session.call_type or "").lower() == "inbound"
                )
                outbound_screening = (
                    self.call_session is not None
                    and (self.call_session.call_type or "").lower() == "outbound"
                    and self._jd_recruitment_screening_active()
                )
                if self.agent and inbound:
                    if getattr(self.agent, "greeting_message", None):
                        greeting_text = self.agent.greeting_message.strip()
                    elif getattr(self.agent, "first_message", None):
                        greeting_text = self.agent.first_message.strip()

                if greeting_text is None and outbound_screening and self.agent:
                    vc: dict = {}
                    if isinstance(self.call_session.call_metadata, dict):
                        raw_vc = self.call_session.call_metadata.get("voice_dynamic_context")
                        if isinstance(raw_vc, dict):
                            vc = raw_vc
                    job_title = (vc.get("job_title") or "").strip() or "the role we're hiring for"
                    cand = (vc.get("candidate_name") or "").strip()
                    aname = (self.agent.name or "our team").strip()
                    if cand:
                        greeting_text = (
                            f"Hi {cand}, I'm {aname}, calling about a quick screening for the {job_title} role. "
                            f"Is now a good time for a short call?"
                        )
                    else:
                        greeting_text = (
                            f"Hi, I'm {aname}, calling about a quick screening for the {job_title} role. "
                            f"Is now a good time for a short call?"
                        )

                if not greeting_text:
                    # No greeting configured — wait for user to speak first
                    return

                # Add greeting to transcript
                await self._add_to_transcript("agent", greeting_text, "greeting")

                # Queue greeting TTS directly (skip LLM!)
                if not self._tts_pipeline:
                    return
                await self._tts_pipeline.queue_tts({
                    "text": greeting_text,
                    "chunk_id": "greeting",
                    "use_ssml": self._use_ssml,
                    "is_final": True
                })

                # Mark as not primed for the greeting
                self._twilio_buffer_primed = False

                return  # Done! No LLM needed for greeting
            
            # Reset TTS state for new response generation
            self._tts_cancel.clear()
            self._prev_tts_tail = b""           # Reset crossfade state so new response starts clean
            self._twilio_buffer_primed = False  # Ensure micro-fade and buffer priming for new utterance
            self._is_tts_playing = False        # Audio not yet streaming for this turn

            self._voice_metrics.start_generation()
            self._metric_gen_start_ts = time.perf_counter()
            _stt_to_gen_ms = (self._metric_gen_start_ts - self._metric_stt_final_ts) * 1000
            if self._metric_stt_final_ts > 0:
                logger.info("[Metrics] stt_final_to_gen_start=%.0f ms", _stt_to_gen_ms)
            slowpath_started_at = time.perf_counter()

            def _remaining_slowpath_budget(default_timeout: float) -> float:
                budget = float(getattr(settings, "VOICE_SLOWPATH_BUDGET_SEC", 0.55) or 0.55)
                budget = max(0.12, min(3.0, budget))
                elapsed = time.perf_counter() - slowpath_started_at
                remaining = max(0.05, budget - elapsed)
                return min(max(0.05, default_timeout), remaining)

            booking_intent_turn = self._is_booking_intent_turn(user_text)
            follow_up_active = bool(
                self.call_session
                and self.call_session.call_metadata
                and self.call_session.call_metadata.get("appointment_id")
            )
            low_latency_fastpath = self._should_use_latency_fastpath(
                user_text, booking_intent_turn
            )

            # Fire quick-ack immediately (fire-and-forget) so the user hears audio
            # while the LLM+RAG+KB pipeline runs.  Only on the slow path (fastpath
            # queries have sub-500ms LLM TTFT so a quick-ack would finish before the
            # LLM chunk is ready, creating a silence gap rather than masking latency).
            # In V2 TtsPipeline, synthesis is parallel: the LLM's first chunk starts
            # synthesising in the background while quick-ack is still playing, so by
            # the time quick-ack ends the real audio is typically already ready.
            if user_text and not low_latency_fastpath:
                asyncio.create_task(self._send_quick_acknowledgement(user_text))

            turn_context = build_turn_context(
                user_text,
                confidence,
                booking_context_active=self._is_booking_context_active(user_text),
                is_final=True,
            )

            # ------- RAG: build knowledge base context in voice layer -------
            tenant_uuid = self.call_session.tenant_id if self.call_session else None
            agent_uuid = self.agent.id if self.agent else None
            rag_agent_scope = None if (self.agent and self.agent.is_inbound_agent) else agent_uuid

            # RAG retrieval: use prefetched result when available (fired on interim
            # so it overlaps Deepgram endpointing), otherwise run synchronously.
            rag_context_block = ""
            rag_trace: dict = {}
            _prefetch = self._rag_prefetch_task
            self._rag_prefetch_task = None  # Consume once per turn
            if low_latency_fastpath:
                rag_trace = {"status": "skipped_fastpath"}
                if _prefetch and not _prefetch.done():
                    _prefetch.cancel()
            else:
                try:
                    if _prefetch is not None:
                        if _prefetch.done():
                            # Already finished — zero-cost read
                            rag_context_block, rag_trace = _prefetch.result()
                            logger.debug("[RAG] used prefetch result (done)")
                        else:
                            # Still running — wait with reduced timeout (most work already done)
                            prefetch_wait_cap = float(
                                getattr(settings, "VOICE_RAG_PREFETCH_AWAIT_SEC", 0.18) or 0.18
                            )
                            rag_context_block, rag_trace = await asyncio.wait_for(
                                asyncio.shield(_prefetch),
                                timeout=_remaining_slowpath_budget(
                                    min(max(0.05, prefetch_wait_cap), settings.RAG_RETRIEVAL_TIMEOUT_SEC)
                                ),
                            )
                            logger.debug("[RAG] awaited in-flight prefetch")
                    else:
                        # No prefetch fired (e.g., very short utterance, final-only path)
                        loop = asyncio.get_running_loop()

                        def _build_rag():
                            return build_rag_context_block_with_trace(
                                user_text=user_text,
                                tenant_id=tenant_uuid,
                                agent_id=rag_agent_scope,
                            )

                        rag_context_block, rag_trace = await asyncio.wait_for(
                            loop.run_in_executor(None, _build_rag),
                            timeout=_remaining_slowpath_budget(settings.RAG_RETRIEVAL_TIMEOUT_SEC),
                        )
                except asyncio.TimeoutError:
                    rag_context_block, rag_trace = build_rag_context_block_with_trace(
                        user_text="",
                        tenant_id=tenant_uuid,
                        agent_id=rag_agent_scope,
                    )
                    rag_trace["status"] = "timeout"
                    rag_trace["timeout"] = True
                except Exception as e:
                    logger.error("RAG context build failed unexpectedly: %s", e, exc_info=True)
                    rag_context_block, rag_trace = build_rag_context_block_with_trace(
                        user_text="",
                        tenant_id=tenant_uuid,
                        agent_id=rag_agent_scope,
                    )
                    rag_trace["status"] = "failure"
                    rag_trace["error"] = str(e)

            # Use call-start-cached KB blocks (fetched once in _prefetch_kb_blocks_at_call_start).
            # These are agent/tenant-level and don't change during the call, so serving
            # them from cache costs 0ms versus the previous 50-200ms parallel executor fetch.
            # Fall back to live fetch only if the cache background task hasn't finished yet
            # (race on the very first turn of an extremely fast caller — rare in practice).
            inbound_kb_docs_context_block = ""
            business_knowledge_block = ""
            if not low_latency_fastpath:
                if self._kb_cache_ready:
                    inbound_kb_docs_context_block = self._cached_inbound_kb_block
                    business_knowledge_block = self._cached_business_knowledge_block
                else:
                    skip_live_kb = bool(
                        getattr(settings, "VOICE_SKIP_LIVE_KB_FETCH_ON_COLD_START", True)
                    )
                    if skip_live_kb:
                        logger.debug("[KB] cache cold; skipping live fetch for latency budget")
                    else:
                        # Cache not ready yet — live fetch as fallback.
                        _loop = asyncio.get_running_loop()

                        async def _fetch_inbound_kb_live() -> str:
                            if not (self.agent and self.agent.is_inbound_agent and tenant_uuid and agent_uuid):
                                return ""
                            try:
                                return await _loop.run_in_executor(
                                    None,
                                    lambda: agent_service.build_inbound_kb_documents_context_block(
                                        db=self.db, inbound_agent_id=agent_uuid, tenant_id=tenant_uuid,
                                    ),
                                )
                            except Exception as exc:
                                logger.warning("KB live-fetch (inbound) failed: %s", exc)
                                return ""

                        async def _fetch_bk_live() -> str:
                            if not tenant_uuid:
                                return ""
                            try:
                                return await _loop.run_in_executor(
                                    None,
                                    lambda: agent_service.build_business_knowledge_context_block(
                                        db=self.db, tenant_id=tenant_uuid, agent_id=agent_uuid,
                                    ),
                                )
                            except Exception as exc:
                                logger.warning("KB live-fetch (business) failed: %s", exc)
                                return ""

                        try:
                            inbound_kb_docs_context_block, business_knowledge_block = await asyncio.wait_for(
                                asyncio.gather(_fetch_inbound_kb_live(), _fetch_bk_live()),
                                timeout=_remaining_slowpath_budget(0.25),
                            )
                        except asyncio.TimeoutError:
                            logger.debug("[KB] live fetch timeout; continuing without live KB blocks")
                        except Exception as exc:
                            logger.debug("[KB] live fetch failed; continuing without live KB blocks: %s", exc)

            # One-line RAG summary (safe: no chunk text, no secrets)
            try:
                logger.info(
                    "RAG trace summary: status=%s timeout=%s initial=%s filtered=%s retrieve_error=%s",
                    rag_trace.get("status"),
                    rag_trace.get("timeout"),
                    rag_trace.get("initial_retrieved_count"),
                    rag_trace.get("filtered_count"),
                    rag_trace.get("retrieve_error"),
                )
            except Exception:
                # Logging must never break voice calls.
                pass
            
            # Build history text from in-memory cache (avoids re-parsing the growing
            # call_transcript JSON on every turn — O(n) JSON parse that gets slower
            # as the call continues).  _conversation_history_cache is appended to
            # directly by _add_to_transcript so it's always up-to-date.
            # Fallback: if cache is empty but DB has a transcript, seed from DB once.
            if not self._conversation_history_cache and self.call_session and self.call_session.call_transcript:
                try:
                    raw = self.call_session.call_transcript
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    for msg in (parsed or []):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content") or msg.get("message", "")
                            mtype = msg.get("message_type", "")
                            if content and role in ("client", "agent") and mtype not in ("greeting", "system", "status"):
                                self._conversation_history_cache.append((role, content))
                except Exception:
                    pass

            history_text = ""
            if self._conversation_history_cache:
                try:
                    filtered = self._conversation_history_cache

                    # Booking flows use a slightly wider window to keep service/date/slot
                    # in context, but capped at 20 to avoid prompt bloat that inflates LLM TTFT.
                    # (Previous default of 39 was adding ~1000 tokens to the prompt on long calls.)
                    max_msgs = getattr(self, "HISTORY_MAX_MESSAGES", 50)
                    if low_latency_fastpath:
                        # Keep at least 12 on fast path — intake flows span 6+ phases and
                        # cutting to 6 drops early context (name, issue, location already given),
                        # causing the agent to re-ask questions the caller already answered.
                        max_msgs = min(max_msgs, 60)
                    if self._is_booking_context_active(user_text):
                        max_msgs = min(max(max_msgs, 60), 60)
                    if len(filtered) > max_msgs:
                        filtered = filtered[-max_msgs:]

                    history_text = "\n".join(
                        f"{role.capitalize()}: {content}" for role, content in filtered
                    )
                except Exception:
                    history_text = ""
            
            booking_memory_block = self._build_booking_memory_block()
            follow_up_appt_block = self._build_follow_up_appointment_block()

            # Build system prompt with agent personality + history
            agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
            agent_language = self.agent.language if self.agent and self.agent.language else "en"
            v_add = ""
            if self.call_session and self.call_session.call_metadata:
                voice_ctx = self.call_session.call_metadata.get("voice_dynamic_context")
                if isinstance(voice_ctx, dict):
                    v_add = (voice_ctx.get("system_prompt_addendum") or "").strip()
            v_block = (
                f"\n\n# THIS CALL — CANDIDATE & ROLE\n{v_add}\n"
                if v_add
                else ""
            )
            tts_provider = getattr(self.agent, "tts_provider", None) if self.agent else None
            tts_provider_slug = (getattr(tts_provider, "slug", None) or "").lower()
            elevenlabs_audio_tags_enabled = supports_elevenlabs_audio_tags(tts_provider_slug)
            if elevenlabs_audio_tags_enabled:
                output_plain_text_rule, no_ssml_rule_base, no_ssml_rule = (
                    get_elevenlabs_voice_prompt_rule_lines()
                )
            else:
                output_plain_text_rule = (
                    "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. "
                    "Prosody is handled by the system."
                )
                no_ssml_rule_base = (
                    "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
                )
                no_ssml_rule = "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."
            elevenlabs_audio_tag_block = build_elevenlabs_audio_tag_prompt_block(tts_provider_slug)

            # Inbound: opening may play once via TTS at pickup — do not instruct LLM to repeat verbatim on every "hi".
            _inbound_call = (
                self.call_session is not None
                and (self.call_session.call_type or "").lower() == "inbound"
            )
            _has_opening_cfg = bool(self.agent) and (
                (getattr(self.agent, "greeting_message", None) or "").strip()
                or (getattr(self.agent, "first_message", None) or "").strip()
            )
            greeting_instruction_block = (
                "\n- GREETING: A short opening may play once automatically at call start (inbound). "
                "If the caller says hi, hello, or similar later, acknowledge in one brief phrase only — "
                "do not repeat the full opening greeting verbatim.\n"
                if (_inbound_call and _has_opening_cfg)
                else ""
            )

            _recruitment_screening_block = ""
            if self._jd_recruitment_screening_active():
                _recruitment_screening_block = """
# RECRUITMENT OUTBOUND SCREENING (THIS CALL — OVERRIDES CONFLICTING GENERIC RULES)
- If the user says not interested, wrong number, wrong person, wrong call, not available, stop calling, or cannot do this now (clear no) — reply with ONE short polite sentence and end with [END_CALL] only. No follow-up questions. No persuasion.
- Never ask a question that is already answered in "Previous conversation" above; move to the next step in YOUR ROLE block.
- On successful completion of the full screening per YOUR ROLE block, your last reply must include [SCREENING_QUALIFIED] immediately before [END_CALL] as instructed there.
- If an intro already played at call start, do not repeat the same full intro; continue the flow.
"""

            # When no business facts are loaded, inject an explicit "do not invent" guard
            # so the LLM never fills the empty section with hallucinated details.
            _bk_block = business_knowledge_block or (
                "# AUTHORITATIVE BUSINESS FACTS\n"
                "No verified business facts are loaded for this call.\n"
                "CRITICAL: Do NOT invent or assume ANY business details (name, address, phone, "
                "email, services, prices, hours, or any other specifics).\n"
                "If the caller asks about the business, say that specific information is not "
                "available to you right now and offer to help in another way."
            )

            # Base prompt for phone conversations (voice-first, plain text only, no SSML)
            base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.
{v_block}
# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{output_plain_text_rule}
- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [1], [2], or similar annotations.
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{follow_up_appt_block}
{rag_context_block}
{inbound_kb_docs_context_block}
{_recruitment_screening_block}
# CRITICAL RULES
1. CONVERSATION CONTINUITY: Read "Previous conversation" above before every reply. Any information already given by the caller (name, location, issue, timing) is still valid — do not ask for it again. Do not restart from the beginning of your flow mid-call. If the caller corrects or updates a previously given answer (e.g., corrects their name), acknowledge it and continue from the current step, not step one.
2. NO REPETITION: Never ask a question that was already asked and answered in the transcript. Move to the next unanswered item only.
3. HANDLING SILENCE: If the user says something vague, ask a single clarifying question.
4. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
5. BUSINESS FACTS: For any question about the business name, address, phone, email, website, services, or pricing — answer using AUTHORITATIVE BUSINESS FACTS below. Never say you don't know if the answer is there. Never invent details that are not written there.
6. SERVICE SCOPE: Strictly follow "BUSINESS SCOPE & POLICY — STRICT RULES" inside AUTHORITATIVE BUSINESS FACTS. Only offer the services listed there. If the caller asks for something we don't offer, politely decline and pivot to what we actually do.
7. SERVICE AREA: If Service Areas are listed and restricted and the caller is outside them, apologize, briefly name the areas we cover, say a short goodbye, and end your response with exactly [END_CALL]. If Service Areas describe global/remote/worldwide coverage, never refuse based on location.
{no_ssml_rule_base}

{elevenlabs_audio_tag_block}

{_bk_block}
# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD] (ISO date or the date the caller asked about).
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,location=<caller city and state>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, location, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).
- NEVER emit [BOOK_APPOINTMENT:...] until the caller has clearly stated their city and state AND (if service areas are restricted) that location is confirmed to be within the covered areas.

# GOAL
Continue the conversation based on the history above. Be {agent_name}."""

            # Use agent's custom system prompt if available, otherwise use base prompt
            if self.agent and self.agent.system_prompt:
                # Agent has custom system prompt - grounding rules placed BEFORE custom
                # instructions so factual constraints cannot be overridden by agent config.
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# GROUNDING RULES (NON-NEGOTIABLE — APPLY BEFORE READING CUSTOM INSTRUCTIONS)
These rules override any conflicting custom instructions below. Never deviate from them.
1. BUSINESS FACTS: Answer questions about business name, address, phone, email, website, services, or pricing ONLY using the AUTHORITATIVE BUSINESS FACTS section below. Never invent or assume any detail not explicitly written there. If a fact is absent, say it is not available.
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.

{_bk_block}

# CUSTOM INSTRUCTIONS
{self.agent.system_prompt}
{v_block}
# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
{output_plain_text_rule}
- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [breathes], [excited], [1], [2], or any similar annotation. These will not be rendered — they will be read aloud literally.
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{follow_up_appt_block}
{rag_context_block}
{inbound_kb_docs_context_block}
{_recruitment_screening_block}
# CRITICAL RULES
1. CONVERSATION CONTINUITY: Read "Previous conversation" above before every reply. Any information the caller already gave (name, location, issue, timing) is still valid — do not ask for it again, and do not restart your intake flow from the beginning mid-call. If the caller corrects a previously given answer (e.g., corrects their name), acknowledge it and continue from the current step, not step one.
2. NO REPETITION: Never ask a question that was already asked and answered in the transcript above. Move to the next unanswered item only.
3. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD].
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,location=<caller city and state>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, location, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).
- NEVER emit [BOOK_APPOINTMENT:...] until the caller has clearly stated their city and state AND (if service areas are restricted) that location is confirmed to be within the covered areas.

# GOAL
Follow your custom instructions. Continue from the history above. Be {agent_name}."""
            elif self.agent and self.agent.model and self.agent.model.system_prompt:
                # Model has system prompt - grounding rules placed BEFORE model instructions.
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# GROUNDING RULES (NON-NEGOTIABLE — APPLY BEFORE READING MODEL INSTRUCTIONS)
These rules override any conflicting model instructions below. Never deviate from them.
1. BUSINESS FACTS: Answer questions about business name, address, phone, email, website, services, or pricing ONLY using the AUTHORITATIVE BUSINESS FACTS section below. Never invent or assume any detail not explicitly written there. If a fact is absent, say it is not available.
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.

{_bk_block}

# MODEL INSTRUCTIONS
{self.agent.model.system_prompt}
{v_block}
# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use fillers like "uhm," "well," "I see" occasionally.
{output_plain_text_rule}
- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [breathes], [excited], [1], [2], or any similar annotation. These will not be rendered — they will be read aloud literally.{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{follow_up_appt_block}
{rag_context_block}
{inbound_kb_docs_context_block}
# CRITICAL RULES
1. CONVERSATION CONTINUITY: Read "Previous conversation" above before every reply. Any information the caller already gave (name, location, issue, timing) is still valid — do not ask for it again, and do not restart your intake flow from the beginning mid-call. If the caller corrects a previously given answer (e.g., corrects their name), acknowledge it and continue from the current step, not step one.
2. NO REPETITION: Never ask a question that was already asked and answered in the transcript above. Move to the next unanswered item only.
3. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD].
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,location=<caller city and state>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, location, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).
- NEVER emit [BOOK_APPOINTMENT:...] until the caller has clearly stated their city and state AND (if service areas are restricted) that location is confirmed to be within the covered areas.

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
            else:
                # Use base prompt
                system_prompt = base_prompt

            # Use _bk_block (never empty) so the service area gate is derived from
            # the actual loaded knowledge — not the raw block which is "" on cold cache.
            call_policy_block = agent_service.build_call_policy_block(
                business_knowledge_block=_bk_block,
                transfer_route=getattr(self.agent, "transfer_route", None) if self.agent else None,
            )
            if call_policy_block:
                system_prompt = call_policy_block + "\n" + system_prompt

            # Prepend current date/time so the agent knows what "today", "tomorrow",
            # and "past slots" mean. Also injected into appointment booking flow.
            _now_local = datetime.now(timezone.utc)
            _now_str = _now_local.strftime("%A, %B %d, %Y at %I:%M %p UTC")
            _user_signals = build_user_signals_block(turn_context)
            system_prompt = (
                f"# CURRENT DATE & TIME\nNow: {_now_str}\n\n"
                f"{_user_signals}\n\n"
                + system_prompt
            )

            # Get agent's configured model and provider
            llm_service = None
            model_name = "gemini-1.5-flash"  # Default fallback
            api_key = None
            temperature = 0.15
            max_tokens = 100
            if self.agent and self.agent.model:
                model_name = self.agent.model.model_name
                
                # Decrypt API key if available
                if self.agent.model.api_key:
                    try:
                        from app.core.security import decrypt_api_key
                        api_key = decrypt_api_key(self.agent.model.api_key)
                    except Exception as e:
                        logger.error(f"Failed to decrypt agent API key: {e}")
                        api_key = None  # Will fallback to settings.OPENAI_API_KEY or settings.GOOGLE_API_KEY
                else:
                    api_key = None  # Will use global key from .env
                
                # Use agent-specific config if available
                if self.agent.agent_temperature is not None:
                    temperature = self.agent.agent_temperature / 100.0  # Convert 0-100 to 0-1
                elif self.agent.model.temperature is not None:
                    temperature = self.agent.model.temperature / 100.0
                
                if self.agent.agent_max_tokens:
                    max_tokens = self.agent.agent_max_tokens
                elif self.agent.model.max_tokens:
                    max_tokens = self.agent.model.max_tokens

                # Booking / follow-up turns need enough completion budget for action tokens.
                if booking_intent_turn or follow_up_active:
                    max_tokens = max(max_tokens, 180)
                elif low_latency_fastpath:
                    max_tokens = min(max_tokens, 80)
                
                # Select service based on provider
                if self.agent.provider:
                    provider_name = self.agent.provider.name.lower()
                    if "openai" in provider_name:
                        llm_service = openai_service
                    elif "gemini" in provider_name or "google" in provider_name:
                        llm_service = gemini_service
                    elif "groq" in provider_name:
                        llm_service = groq_service
                    else:
                        # Default to Gemini
                        llm_service = gemini_service
                else:
                    llm_service = gemini_service
            else:
                # Fallback to Gemini
                llm_service = gemini_service
            
            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            _tts_time_flush_s = max(
                0.08,
                float(getattr(settings, "VOICE_TTS_TIME_FLUSH_SEC", 0.10) or 0.10),
            )
            _time_flush_target_words = 3 if low_latency_fastpath else 4
            logger.info(f"🧠 Calling LLM ({llm_service.__class__.__name__ if hasattr(llm_service, '__class__') else 'Service'}) for response to: '{user_text[:20]}...'")
            
            async def try_stream(service, model: str, api_key_override: str = None) -> str:
                nonlocal chunk_counter
                import re
                import time

                response_accum = ""
                tts_buffer = ""
                end_call_after = False
                transfer_after = False
                _transfer_re = re.compile(r"\[\s*TRANSFER_CALL\s*\]", re.IGNORECASE)
                first_tts_chunk = True
                last_flush_ts = time.perf_counter()
                deferred_memory_scheduled = False
                self._pending_resume_screening_qualify = False

                def _schedule_deferred_memory_once() -> None:
                    nonlocal deferred_memory_scheduled
                    if deferred_memory_scheduled:
                        return
                    deferred_memory_scheduled = True
                    asyncio.create_task(
                        self._deferred_conversation_memory_update(turn_context, user_text)
                    )

                def _find_flush_index(buf: str):
                    """
                    Return an index (end-exclusive) where we can safely flush.
                    Prefer sentence boundaries. Fallback to comma/semicolon if buffer is getting long.
                    """
                    if not buf:
                        return None

                    nl = buf.find("\n")
                    if nl != -1:
                        prefix = buf[:nl].strip()
                        if len(prefix.split()) >= self.TTS_FLUSH_MIN_WORDS:
                            return nl

                    # Prefer sentence boundaries: ., !, ? followed by whitespace/newline/end
                    last_boundary = None
                    for m in re.finditer(r"([.!?])(\s+|$)", buf):
                        last_boundary = m.end(1)

                    if last_boundary is not None:
                        prefix = buf[:last_boundary].strip()
                        if len(prefix.split()) >= self.TTS_FLUSH_MIN_WORDS:
                            return last_boundary

                    # If the buffer is getting long, allow a softer boundary split
                    words = buf.split()
                    if len(words) >= self.TTS_FLUSH_MAX_WORDS:
                        last_soft = None
                        for m in re.finditer(r"([,;:])(\s+|$)", buf):
                            last_soft = m.end(1)
                        if last_soft is not None:
                            prefix = buf[:last_soft].strip()
                            if len(prefix.split()) >= self.TTS_FLUSH_MIN_WORDS:
                                return last_soft

                    return None

                def _find_time_flush_index(buf: str):
                    """
                    Time-based flush (Vapi-style): if punctuation is delayed, flush on a safe word boundary
                    so we can start speaking fast. Returns an index (end-exclusive) or None.
                    """
                    if not buf:
                        return None
                    words = buf.split()
                    # Lower floor to 2 words (was hardcoded 5): allows the 150ms time
                    # gate to flush short confident fragments like "Sure," or "Got it."
                    # without waiting for a full sentence to accumulate first.
                    if len(words) < max(self.TTS_FLUSH_MIN_WORDS, 2):
                        return None

                    # Flush around ~4-6 words to start speaking quickly.
                    target_words = min(_time_flush_target_words, len(words))
                    m = re.match(rf"^(?:\\S+\\s+){{{target_words - 1}}}\\S+", buf)
                    if not m:
                        return None
                    return m.end()

                _first_token_marked = False
                _first_token_recorded = False
                async for chunk in service.stream_text(
                    prompt=user_text,
                    system_prompt=system_prompt,
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key_override
                ):
                    if not chunk:
                        continue
                    # If barge-in requested, stop generating
                    if self._tts_cancel.is_set():
                        break

                    response_accum += chunk
                    tts_buffer += chunk

                    if not _first_token_marked:
                        _first_token_marked = True
                        _vm = getattr(self, "_voice_metrics", None)
                        if _vm:
                            _vm.mark_llm_first_token()
                    if not _first_token_recorded:
                        _first_token_recorded = True
                        self._metric_first_token_ts = time.perf_counter()

                    # Detect END_CALL early (may appear late, but handle if it appears mid-stream)
                    if _RE_VOICE_END_CALL.search(response_accum):
                        end_call_after = True
                        # Remove it from TTS buffer immediately so it never gets spoken
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    if _transfer_re.search(response_accum):
                        transfer_after = True
                        end_call_after = False
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    # Remove OUTCOME tokens from any in-flight buffer (never spoken)
                    if "[OUTCOME:" in tts_buffer:
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    # Backend-only screening outcome token (never spoken)
                    if _RE_VOICE_SCREENING_QUALIFIED.search(response_accum):
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    # Avoid spoken "final confirmation" before backend booking succeeds.
                    if "[BOOK_APPOINTMENT:" in response_accum:
                        tts_buffer = self._prepare_tts_text(tts_buffer)

                    # Flush complete thoughts early for faster perceived latency
                    flush_idx = _find_flush_index(tts_buffer)
                    # If punctuation-based flush isn't available, do a time-based flush (tunable)
                    if flush_idx is None:
                        now_ts = time.perf_counter()
                        if (now_ts - last_flush_ts) >= _tts_time_flush_s:
                            flush_idx = _find_time_flush_index(tts_buffer)
                    if flush_idx is not None and not self._tts_cancel.is_set():
                        flush_text = tts_buffer[:flush_idx].strip()
                        tts_buffer = tts_buffer[flush_idx:].lstrip()

                        flush_text = self._prepare_tts_text(flush_text)
                        if flush_text:
                            if self._use_ssml:
                                clean_text = strip_ssml_tags(flush_text)
                                enhanced_text = preprocess_for_tts(
                                    clean_text,
                                    start_break_ms=0,
                                    between_sentence_break_ms=100,
                                )
                            else:
                                enhanced_text = quick_clean(flush_text)

                            enhanced_text = tone_adapter(enhanced_text, turn_context, self._use_ssml)

                            chunk_counter += 1
                            if self._tts_pipeline:
                                await self._tts_pipeline.queue_tts({
                                    "text": enhanced_text,
                                    "chunk_id": chunk_counter,
                                    "use_ssml": self._use_ssml,
                                    "is_final": False,
                                })
                                _vm = getattr(self, "_voice_metrics", None)
                                if _vm:
                                    _vm.mark_first_tts_queued()
                                _schedule_deferred_memory_once()
                            first_tts_chunk = False
                            last_flush_ts = time.perf_counter()

                # Flush any remaining text as the FINAL chunk
                full_response = response_accum.strip()
                end_call_after = end_call_after or bool(_RE_VOICE_END_CALL.search(full_response))
                if (
                    end_call_after
                    and not _transfer_re.search(full_response)
                ):
                    # DB apply_resume_candidate_status_after_voice_screening enforces jd_context + recruitment checks
                    self._pending_resume_screening_qualify = False
                    try:
                        persisted_status = persist_voice_screening_status_signal(
                            self.db,
                            self.call_session,
                            full_response,
                        )
                        self._pending_resume_screening_qualify = persisted_status is not None
                    except Exception:
                        pass

                if _transfer_re.search(full_response):
                    transfer_after = True
                    end_call_after = False
                    self._pending_resume_screening_qualify = False

                # Never speak control tokens
                tts_buffer = self._prepare_tts_text(tts_buffer)

                if tts_buffer and not self._tts_cancel.is_set():
                    if self._use_ssml:
                        clean_text = strip_ssml_tags(tts_buffer)
                        enhanced_text = preprocess_for_tts(
                            clean_text,
                            start_break_ms=0,
                            between_sentence_break_ms=150,
                        )
                    else:
                        enhanced_text = quick_clean(tts_buffer)

                    enhanced_text = tone_adapter(enhanced_text, turn_context, self._use_ssml)

                    chunk_counter += 1
                    if self._tts_pipeline:
                        await self._tts_pipeline.queue_tts({
                            "text": enhanced_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._use_ssml,
                            "is_final": True,
                            "end_call_after": end_call_after and not transfer_after,
                            "transfer_after": transfer_after,
                        })
                        _vm = getattr(self, "_voice_metrics", None)
                        if _vm:
                            _vm.mark_first_tts_queued()
                        _schedule_deferred_memory_once()
                elif transfer_after and not self._tts_cancel.is_set() and self._tts_pipeline:
                    chunk_counter += 1
                    await self._tts_pipeline.queue_tts({
                        "text": "One moment.",
                        "chunk_id": chunk_counter,
                        "use_ssml": self._use_ssml,
                        "is_final": True,
                        "end_call_after": False,
                        "transfer_after": True,
                    })
                    _vm = getattr(self, "_voice_metrics", None)
                    if _vm:
                        _vm.mark_first_tts_queued()
                    _schedule_deferred_memory_once()

                return response_accum.strip()

            final_text = None
            try:
                # Use agent's configured model and service
                final_text = await try_stream(llm_service, model_name, api_key)
            except Exception as e:
                logger.warning(f"⚠️ Primary LLM failed ({model_name}): {e}. Attempting fallback...")
                # Fallback: try alternate service
                try:
                    if llm_service == openai_service:
                        # Fallback to Gemini
                        final_text = await try_stream(gemini_service, "gemini-1.5-flash", None)
                    else:
                        # Fallback to OpenAI
                        final_text = await try_stream(openai_service, "gpt-3.5-turbo", None)
                except Exception as e:
                    logger.warning(f"⚠️ Secondary LLM fallback failed: {e}. Attempting VoiceLoggingService fallback...")
                    # Last fallback: non-streaming fast response via VoiceLoggingService
                    try:
                        final_text = await VoiceLoggingService.generate_agent_response(
                            speech_text=user_text,
                            confidence=confidence,
                            agent=self.agent,
                            db=self.db,
                            call_session_id=self.call_session.id if self.call_session else None
                        )
                        # Queue fallback response
                        if final_text and not self._tts_cancel.is_set():
                            safe_tts_text = self._prepare_tts_text(final_text)
                            out_text = tone_adapter(safe_tts_text.strip(), turn_context, self._use_ssml)
                            if not out_text:
                                out_text = "One moment please."
                            chunk_counter += 1
                            if self._tts_pipeline and out_text:
                                await self._tts_pipeline.queue_tts({
                                    "text": out_text,
                                    "use_ssml": self._use_ssml,
                                    "is_final": True,
                                })
                                asyncio.create_task(
                                    self._deferred_conversation_memory_update(turn_context, user_text)
                                )
                    except Exception as e:
                        logger.warning(f"⚠️ VoiceLoggingService fallback failed: {e}. Using ultimate fallback.")
                        # Ultimate fallback response
                        final_text = "I apologize, I'm having trouble responding right now. Could you please repeat that?"
                        utt = tone_adapter(final_text, turn_context, self._use_ssml)
                        chunk_counter += 1
                        if self._tts_pipeline:
                            await self._tts_pipeline.queue_tts({
                                "text": utt,
                                "chunk_id": chunk_counter,
                                "use_ssml": self._use_ssml,
                                "is_final": True
                            })
                            asyncio.create_task(
                                self._deferred_conversation_memory_update(turn_context, user_text)
                            )

            if final_text:
                # Strip control tokens from transcript (never saved to history)
                transcript_text = _RE_VOICE_END_CALL.sub(
                    "", self._strip_control_tokens_for_tts(final_text)
                ).strip()
                if "[BOOK_APPOINTMENT:" in final_text:
                    transcript_text = self._strip_premature_booking_confirmation(transcript_text)
                if transcript_text:
                    await self._add_to_transcript(
                        "agent",
                        transcript_text,
                        "agent_response",
                        message_metadata={
                            "user_text": user_text,
                            "rag_trace": rag_trace,
                        },
                    )
                    self._schedule_recreate_stt_for_email_collection(transcript_text)

                # Handle calendar tokens (fire-and-forget — TTS already queued above).
                # Two-step reliability: if booking intent exists but the LLM omitted the
                # token, run _extract_calendar_action_token as a background task so it
                # never blocks the voice pipeline.  The extracted token fires its own
                # calendar handler asynchronously after the spoken response is done.
                if re.search(r"\[\s*CHECK_SLOTS\s*:", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_check_slots_token(final_text))
                elif re.search(r"\[\s*BOOK_APPOINTMENT\s*:", final_text, flags=re.IGNORECASE):
                    pass  # handled below
                elif booking_intent_turn:
                    # No token emitted — run extraction in the background (non-blocking)
                    async def _deferred_extraction() -> None:
                        try:
                            token = await asyncio.wait_for(
                                self._extract_calendar_action_token(
                                    llm_service=llm_service,
                                    model_name=model_name,
                                    api_key=api_key,
                                    user_text=user_text,
                                    assistant_text=final_text,
                                    history_text=history_text,
                                    temperature=temperature,
                                ),
                                timeout=5.0,
                            )
                            if token:
                                logger.info("[CalAction] deferred extraction: %s", token[:120])
                                combined = f"{final_text}\n{token}"
                                if re.search(r"\[\s*CHECK_SLOTS\s*:", token, flags=re.IGNORECASE):
                                    await self._handle_check_slots_token(combined)
                                elif re.search(r"\[\s*BOOK_APPOINTMENT\s*:", token, flags=re.IGNORECASE):
                                    await self._handle_book_appointment_token(combined)
                        except asyncio.TimeoutError:
                            logger.debug("[CalAction] deferred extraction timed out")
                        except Exception as exc:
                            logger.debug("[CalAction] deferred extraction error: %s", exc)
                    asyncio.create_task(_deferred_extraction())

                if re.search(r"\[\s*BOOK_APPOINTMENT\s*:", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_book_appointment_token(final_text))
                if re.search(r"\[\s*FOLLOWUP_CONFIRM\s*\]", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_followup_confirm_token(final_text))
                if re.search(r"\[\s*FOLLOWUP_CANCEL\s*\]", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_followup_cancel_token(final_text))
                if re.search(r"\[\s*FOLLOWUP_RESCHEDULE\s*:", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_followup_reschedule_token(final_text))

                try:
                    self._voice_metrics.log_turn_summary(
                        logger,
                        user_preview=(user_text or "")[:56],
                        session_hint=str(self.call_session_id or ""),
                    )
                    if bool(getattr(settings, "VOICE_SLO_ENABLED", True)):
                        breaches = self._voice_metrics.get_slo_breaches(
                            stt_final_to_gen_start_s=float(
                                getattr(settings, "VOICE_SLO_STT_FINAL_TO_GEN_START_SEC", 0.35) or 0.35
                            ),
                            gen_start_to_llm_first_token_s=float(
                                getattr(settings, "VOICE_SLO_GEN_START_TO_LLM_FIRST_TOKEN_SEC", 0.90)
                                or 0.90
                            ),
                            gen_start_to_first_tts_q_s=float(
                                getattr(settings, "VOICE_SLO_GEN_START_TO_FIRST_TTS_QUEUE_SEC", 1.40)
                                or 1.40
                            ),
                            gen_start_to_now_warn_s=float(
                                getattr(settings, "VOICE_SLO_GEN_START_TO_NOW_WARN_SEC", 2.00)
                                or 2.00
                            ),
                        )
                        if breaches:
                            logger.warning(
                                "[VoiceSLO] breach session=%s user=%r %s",
                                str(self.call_session_id or ""),
                                (user_text or "")[:56],
                                " | ".join(breaches),
                            )
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in generate_and_stream_response: {e}", exc_info=True)

    async def _deferred_conversation_memory_update(
        self, turn_context: TurnContext, user_text: str
    ) -> None:
        """
        Non-blocking hook after first TTS is queued. Extend with embeddings or summaries
        without adding latency to STT → LLM → TTS.
        """
        try:
            logger.debug(
                "deferred turn context: mood=%s phase=%s is_final=%s (user chars=%d)",
                turn_context.mood_label(),
                turn_context.conversation_phase,
                turn_context.is_final,
                len((user_text or "")),
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("deferred conversation memory update failed: %s", e, exc_info=True)
    
    # ── Booking and calendar token methods live in app/voice/booking_mixin.py ──

    # ── TTS streaming methods live in app/voice/tts_stream_mixin.py ──

    # ── Call control methods live in app/voice/call_control_mixin.py ──

    async def handle_start_message(self, message: dict):
        """Handle stream start - Just WebSocket connection (NOT user pickup!)"""
        try:
            self.stream_sid = message.get("streamSid")
            if self.stream_sid:
                self._stream_sid_ready.set()
            # Notify orchestrator so it can perform any stream-SID-dependent setup.
            self._voice_orchestrator.set_stream_sid(self.stream_sid)
            start = message.get("start", {})
            self.call_sid = start.get("callSid")

            try:
                configured_edge = str(getattr(settings, "TWILIO_EDGE", "") or "").strip().lower()
                expected_edge = "umatilla"  # Oregon
                strict_align = bool(getattr(settings, "VOICE_REGION_ALIGNMENT_STRICT", True))
                server_region = str(getattr(settings, "SERVER_REGION", "us-west-2") or "us-west-2")
                if configured_edge:
                    logger.info(
                        "[RegionAlign] server_region=%s twilio_edge=%s expected_edge=%s",
                        server_region,
                        configured_edge,
                        expected_edge,
                    )
                if strict_align and configured_edge and configured_edge != expected_edge:
                    logger.warning(
                        "[RegionAlign] Twilio edge mismatch for Oregon deployment: configured=%s expected=%s",
                        configured_edge,
                        expected_edge,
                    )
            except Exception:
                pass

            # Inbound calls use Connect/Stream, so start recording explicitly here.
            # This enables recording-status webhook -> call_session.recording_url persistence.
            if (
                self.call_sid
                and self.call_session
                and self.call_session.call_type == "inbound"
                and not self._recording_started
            ):
                try:
                    account_sid, auth_token = get_twilio_credentials_for_call(
                        self.db, self.call_session
                    )
                    started = twilio_service.start_recording_with_credentials(
                        self.call_sid, account_sid, auth_token
                    )
                    if started:
                        self._recording_started = True
                except Exception as rec_err:
                    logger.warning(
                        "Could not start inbound recording (call_sid=%s, session=%s): %s",
                        self.call_sid,
                        self.call_session.id if self.call_session else None,
                        rec_err,
                    )

            # DO NOT start credit monitoring or greeting here!
            # Wait for first media packet (user actually picks up - VAPI-style)

        except Exception as e:
            logger.error(f"Error in handle_start_message: {e}", exc_info=True)
    
    async def _handle_user_pickup(self):
        """Handle user pickup - called when actual user audio detected (not Twilio system messages)"""
        try:
            if self._user_picked_up:
                return  # Already handled
            
            self._user_picked_up = True
            self._voice_metrics.mark_call_pickup()

            # ❌ Credit monitoring moved to _send_in_progress_status() 
            # Credit deduction will start when connected status is sent (first media packet + connected status)
            
            # Don't send in-progress status here - wait for confident word detection
            # Status will be sent in _process_transcript() when confident transcript is detected

            # Start ambient background loop after brief stabilization delay.
            asyncio.create_task(self._start_background_audio_with_delay())
            
            # 👋 One-time opening TTS after pickup (inbound only, only if configured).
            if (
                self.call_session
                and self.call_session.call_type == "inbound"
                and not self._auto_greeting_sent
                and self.agent
                and (
                    (getattr(self.agent, "greeting_message", None) or "").strip()
                    or (getattr(self.agent, "first_message", None) or "").strip()
                )
            ):
                self._auto_greeting_sent = True
                asyncio.create_task(self._schedule_inbound_greeting_after_delay())
            elif (
                self.call_session
                and (self.call_session.call_type or "").lower() == "outbound"
                and not self._auto_greeting_sent
                and self._jd_recruitment_screening_active()
            ):
                # Outbound screening: play scripted intro (LLM often skipped intro)
                self._auto_greeting_sent = True
                asyncio.create_task(self._schedule_outbound_screening_intro_after_delay())
        
        except Exception as e:
            logger.error(f"Error in _handle_user_pickup: {e}", exc_info=True)

    async def _schedule_outbound_screening_intro_after_delay(self) -> None:
        """Brief delay after pickup, then scripted recruitment intro via same path as inbound greeting."""
        try:
            delay_sec = float(getattr(settings, "VOICE_OUTBOUND_SCREENING_INTRO_DELAY_SEC", 0.55) or 0.55)
        except Exception:
            delay_sec = 0.55
        delay_sec = max(0.0, min(delay_sec, 5.0))
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)
        if self._stop_event.is_set():
            return
        if not self.call_session or (self.call_session.call_type or "").lower() != "outbound":
            return
        if self._tts_cancel.is_set():
            return
        await self.generate_and_stream_response("", 1.0, is_greeting=True)

    async def _schedule_inbound_greeting_after_delay(self) -> None:
        """
        Delay inbound auto-greeting slightly after pickup to let telephony audio settle.
        Uses a configurable delay and exits safely if call state has already stopped.
        """
        try:
            delay_sec = float(getattr(settings, "VOICE_INBOUND_GREETING_DELAY_SEC", 2.0) or 0.0)
        except Exception:
            delay_sec = 2.0
        delay_sec = max(0.0, min(delay_sec, 10.0))

        if delay_sec > 0:
            await asyncio.sleep(delay_sec)

        # Skip if call has already ended or session state is no longer valid.
        if self._stop_event.is_set():
            return
        if not self.call_session or self.call_session.call_type != "inbound":
            return
        if not self.agent:
            return
        if self._tts_cancel.is_set():
            return
        if not (
            (getattr(self.agent, "greeting_message", None) or "").strip()
            or (getattr(self.agent, "first_message", None) or "").strip()
        ):
            return

        await self.generate_and_stream_response(
            user_text="",
            confidence=1.0,
            is_greeting=True,
        )
    
    async def _full_shutdown(self) -> None:
        """
        Unified, idempotent shutdown for all pipelines (STT, LLM, TTS).

        Called from every call-end path:
          - Twilio `stop` event  (handle_stop_message)
          - User goodbye phrase  (_check_and_end_call_if_goodbye)
          - Agent [END_CALL]     (_end_call_after_agent_request)
          - Agent [TRANSFER_CALL] (_transfer_after_agent_request)
          - Voicemail detected   (_check_and_end_call_if_voicemail)
          - WebSocket finally    (route-level cleanup)

        Sets _stop_event so the main receive loop breaks out immediately
        instead of hanging at `await websocket.receive_text()`.
        """
        # Idempotent guard — first caller wins, rest are no-ops
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        self._stt_active = False

        # Stop background audio loop safely.
        try:
            await self._background_audio.stop_loop()
        except Exception:
            pass

        # ── VoiceOrchestrator handles LLM cancel + TTS shutdown + STT close ────
        # OLD direct code (now delegated to orchestrator):
        # t = self._llm_response_task; t.cancel(); self._llm_response_task = None
        # self._tts_cancel.set()
        # await self._tts_pipeline.shutdown()
        # await self._stt_pipeline.aclose()
        # ────────────────────────────────────────────────────────────────────────
        try:
            await self._voice_orchestrator.shutdown()
        except Exception:
            pass

        # Finalize voice appointment booking from transcript (exactly once per call handler)
        if not self._post_call_orchestration_scheduled:
            self._post_call_orchestration_scheduled = True
            asyncio.create_task(self._post_call_appointment_workflow())

    def _post_call_appointment_sync(self) -> None:
        from app.db.session import SessionLocal
        from app.services.post_call_appointment_service import post_call_appointment_service

        db = SessionLocal()
        try:
            post_call_appointment_service.process_call_session(
                db, uuid.UUID(self.call_session_id)
            )
        except Exception as e:
            logger.error("Post-call appointment processing failed: %s", e, exc_info=True)
        finally:
            db.close()

    async def _post_call_appointment_workflow(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._post_call_appointment_sync)
        except Exception as e:
            logger.error("Post-call appointment workflow error: %s", e, exc_info=True)

    async def handle_stop_message(self, message: dict):
        """Handle Twilio stream `stop` event — delegates to unified shutdown."""
        try:
            await self._full_shutdown()
        except Exception as e:
            logger.error(f"Error in handle_stop_message: {e}", exc_info=True)


async def _receive_or_stop(
    ws: WebSocket, stop_event: asyncio.Event
) -> Optional[str]:
    """
    Race websocket.receive_text() against an internal stop_event.

    Returns:
        str  — the raw text received from Twilio.
        None — stop_event fired first (call ended internally).

    Cancels the losing task cleanly so there are no dangling coroutines.
    """
    recv_task = asyncio.create_task(ws.receive_text())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, pending = await asyncio.wait(
            [recv_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel and await the loser to suppress "task was destroyed" warnings
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if stop_task in done:
            # Internal shutdown — tell the caller to break the loop
            recv_task.cancel()
            return None
        return recv_task.result()
    except Exception:
        return None


@router.websocket("/ws/bidirectional/{callSessionId}/{agentId}")
async def bidirectional_stream_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    Bidirectional WebSocket for real-time voice AI
    
    Handles:
    - Incoming audio (STT) from Twilio
    - Outgoing audio (TTS) to Twilio
    - Real-time streaming for ultra-low latency
    
    Target: <3 seconds response time
    """
    # Accept connection
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"Failed to accept bidirectional WebSocket: {e}")
        return
    
    # Get database session
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Create handler
    handler = BidirectionalStreamHandler(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db
    )
    
    media_count = 0
    
    try:
        while True:
            # Race: receive next Twilio message OR stop_event (internal call end)
            data = await _receive_or_stop(websocket, handler._stop_event)

            if data is None:
                # Internal end-call path triggered _full_shutdown + set _stop_event
                logger.info(f"🛑 Internal stop event fired for session {callSessionId} — closing WebSocket")
                break

            message = json.loads(data)
            event = message.get("event")
            
            if event == "connected":
                pass
            
            elif event == "start":
                await handler.handle_start_message(message)
            
            elif event == "media":
                await handler.handle_media_message(message)
            
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Bidirectional WebSocket disconnected for session {callSessionId}")
    
    except Exception as e:
        logger.error(f"Unexpected error in bidirectional WebSocket: {e}", exc_info=True)
    
    finally:
        # Ensure all pipelines are fully shut down (idempotent — safe if already done)
        if handler is not None:
            try:
                await handler._full_shutdown()
            except Exception as e:
                logger.debug(f"Pipeline cleanup in finally: {e}")

        # Explicitly close the WebSocket so Twilio gets an immediate close frame
        # instead of waiting for the TCP connection to time out.
        try:
            await websocket.close()
        except Exception:
            pass

        db.close()


@router.websocket("/ws/tts-only/{callSessionId}/{agentId}")
async def tts_only_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    TTS-ONLY WebSocket for streaming audio playback.
    Thin composition layer that delegates to TtsOnlySession.
    """
    try:
        await websocket.accept()
    except Exception:
        return

    from app.db.session import SessionLocal

    db = SessionLocal()
    session = TtsOnlySession(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db,
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info(f"🔌 TTS-ONLY WebSocket disconnected for session {callSessionId}")
    except Exception as e:
        logger.error(f"Unexpected error in TTS-ONLY WebSocket: {e}", exc_info=True)
    finally:
        db.close()