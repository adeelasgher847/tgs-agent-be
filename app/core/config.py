from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql+psycopg2://postgres:admin@localhost:5432/voiceagent"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Environment — controls which Twilio credentials are used and Secret Manager behaviour.
    # Values: "development" | "staging" | "production"
    ENVIRONMENT: str = "development"

    # Twilio Configuration
    # In production/staging these should come from Secret Manager (see app/core/secret_manager.py).
    # They are kept here as fallbacks for local development only.
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = "+13466602410"
    ALLOW_UNAUTHENTICATED_WEBHOOKS: bool = False

    # Twilio test credentials — used automatically when ENVIRONMENT="staging".
    # Set via Secret Manager or .env.staging; never commit real values.
    TWILIO_TEST_ACCOUNT_SID: str = ""
    TWILIO_TEST_AUTH_TOKEN: str = ""

    # GCP Secret Manager project ID (required in staging/production).
    GCP_PROJECT_ID: str = ""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    APP_VERSION: str = "1.0.0"

    # CORS — comma-separated list of allowed origins.
    # Example: "https://app.example.com,https://admin.example.com"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # Webhook Configuration
    WEBHOOK_BASE_URL: str = "https://tgs-agent-be.onrender.com"
    N8N_WEBHOOK_URL: str = ""  # n8n webhook URL for scheduled calls
    N8N_WEBHOOK_SECRET: str = ""  # Secret for verifying n8n webhook requests
    # Email settings (SendGrid)
    SENDGRID_API_KEY: str = ""
    SENDGRID_SENDER_EMAIL: str = ""
    # Legacy SMTP settings (no longer used)
    
    # Password reset settings
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30
    FRONTEND_URL: str = "http://localhost:3000"
    
    GEMINI_API_KEY: str = ""
    # Default LLM when an agent has no ticket llm_model or legacy model relation.
    DEFAULT_LLM_MODEL: str = "gemini-1.5-flash"
    DEFAULT_LLM_PROVIDER: str = "gemini"
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    
    # Rime Labs TTS Configuration
    RIME_API_KEY: str = ""

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = ""
    # Symmetric encryption key for agent.encrypted_elevenlabs_api_key (pgp_sym_encrypt).
    # In production/staging load from Secret Manager; never commit a real value.
    ELEVENLABS_ENCRYPTION_KEY: str = ""
    # When True, voice LLM prompts may suggest bracketed audio tags for ElevenLabs TTS only
    # ([breathes], [pause], [excited], [sad], …). Set False if your TTS model reads brackets out loud.
    ENABLE_ELEVENLABS_AUDIO_TAGS: bool = True
    
    # Google Cloud Speech-to-Text Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # Path to service account JSON file for Vertex AI + STT
    GOOGLE_CLOUD_PROJECT_ID: str = ""
    # Vertex AI — used by VertexGeminiService for voice LLM calls (ADC, no api_key needed)
    VERTEX_AI_LOCATION: str = "us-central1"
    # History pruning for Vertex Gemini voice path (1 turn = 1 user + 1 model message)
    VOICE_LLM_HISTORY_MAX_TURNS: int = 20
    # Default temperature for Vertex Gemini voice path (0–1 scale)
    VOICE_LLM_DEFAULT_TEMPERATURE: float = 0.3
    # Canned fallback spoken when the Vertex LLM errors (quota, timeout, filter)
    VOICE_LLM_FALLBACK_MESSAGE: str = "I am sorry, I did not catch that"
    GOOGLE_STT_LANGUAGE_CODE: str = "en-US"  # Default language
    # Deprecated fallback; prefer STT_SAMPLE_RATE for provider-neutral STT settings.
    GOOGLE_STT_SAMPLE_RATE: int = 8000
    GOOGLE_STT_ENCODING: str = "MULAW"  # Twilio's audio encoding

    # Deepgram Speech-to-Text (replaces Google STT for streaming + batch)
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_STT_LANGUAGE: str = "en"  # Deepgram listen param; override in .env if needed
    # Silence (ms) before Deepgram marks speech_final. 300ms splits spelling/email pauses;
    # ~900ms matches typical telephony spelling tolerance (Vapi-style longer listen window).
    DEEPGRAM_STT_ENDPOINTING_MS: int = 350
    # After the agent asks for email, bidirectional stream may reopen STT once with this value.
    DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED: int = 500
    # Telecom-oriented silence window for spelling/email (when mode is extended or email-recreate runs).
    # Ignored unless VOICE_STT_ENDPOINTING_MODE == "extended" or email flow bumps endpointing.
    # One-time Deepgram reconnect with extended endpointing when agent transcript matches email ask.
    VOICE_STT_ENDPOINTING_EMAIL_PROMPT_RECREATES_STT: bool = True
    # Initial Deepgram endpointing profile for the first STT session:
    #   normal     → DEEPGRAM_STT_ENDPOINTING_MS
    #   extended   → max(base, DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED)
    #   aggressive → faster finals (lower ms, clamped) for snappier turns
    VOICE_STT_ENDPOINTING_MODE: str = "aggressive"
    # Secondary dedup in STT pipeline: normalized text, same window idea as handler (seconds).
    VOICE_STT_FINAL_NORMALIZED_DEDUP_SEC: float = 6.0
    STT_SAMPLE_RATE: int = 8000  # provider-neutral STT sample rate (Twilio MULAW default)

    # Google Cloud Text-to-Speech (TTS) endpoint/voice overrides
    # Docs: https://cloud.google.com/text-to-speech/docs/endpoints
    CLOUD_TTS_ENDPOINT: str = ""  # e.g. https://us-texttospeech.googleapis.com
    GOOGLE_TTS_VOICE_NAME: str = ""  # Optional exact voice name override (e.g. en-US-Chirp3-HD-Achernar)
    
    # Voice Conversation Settings - VAPI-STYLE REAL-TIME STREAMING
    USE_GATHER_APPROACH: bool = False  # Using real-time bidirectional streaming
    USE_BIDIRECTIONAL_STREAMING: bool = True  # ✅ ENABLED - Real-time STT + TTS with Adaptive VAD
    USE_WEBSOCKET_TTS: bool = True  # ✅ ENABLED - 20ms chunk streaming (MULAW 8kHz)

    # Voice streaming tunables (phase 6 centralization)
    VOICE_STT_INTERIM_INTERVAL_MS: int = 30
    # Deepgram fires many more partials than classic Google STT. Running LLM on every
    # interim → double replies + TTS "breaks." Default: final STT only (one reply per
    # utterance). Set True for lower first-token latency at the cost of stability.
    VOICE_ENABLE_INTERIM_LLM: bool = True
    # When interim LLM is enabled, these gates reduce junk triggers ("I'm", "Do you", …)
    VOICE_MIN_INTERIM_WORDS: int = 2
    VOICE_MIN_INTERIM_CONFIDENCE: float = 0.14
    # Inbound MULAW → linear RMS: frames above this count as "speech" for user-pickup detection.
    # Lower = softer voices register sooner (e.g. 60–70); higher = stricter, needs louder speech
    # (legacy default was 100). Too low picks up line noise.
    VOICE_MIN_AUDIO_RMS_FOR_PICKUP: int = 20
    # Drop Deepgram final transcripts below this (0.0–1.0). Default slightly below 0.30 so
    # quiet/soft speech is not rejected as often; too low adds garbage.
    VOICE_STT_MIN_FINAL_CONFIDENCE: float = 0.15
    # Optional adaptive fallback: accept lower-confidence finals when they still look like
    # real speech (multi-word, alpha content). Helps callers with soft volume mid-call.
    VOICE_STT_ENABLE_SOFT_FINAL_FALLBACK: bool = True
    VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE: float = 0.12
    VOICE_STT_SOFT_MIN_WORDS: int = 2
    # Barge-in (user talks over agent): require at least this many STT words while TTS plays.
    # Default 2 filters phantom 1-word Deepgram hits ("uh", noise artefacts) on silence.
    VOICE_BARGE_IN_MIN_WORDS: int = 2
    # Min STT confidence when word count >= VOICE_BARGE_IN_MIN_WORDS.
    VOICE_BARGE_IN_MIN_CONFIDENCE: float = 0.26
    # Only used when VOICE_BARGE_IN_MIN_WORDS == 1 (one-word interrupts like "stop").
    VOICE_BARGE_IN_MIN_CONFIDENCE_1W: float = 0.52
    VOICE_HISTORY_MAX_MESSAGES: int = 50
    VOICE_TTS_FLUSH_MIN_WORDS: int = 4
    # Smaller max keeps per-chunk synthesis short (~300ms for ElevenLabs) so the
    # playback gate chain never backs up — eliminates "arr arr" / mid-chunk silence.
    VOICE_TTS_FLUSH_MAX_WORDS: int = 6
    # If no sentence boundary yet, flush after this many seconds (once min words met).
    VOICE_TTS_TIME_FLUSH_SEC: float = 0.10
    # Keep a short (but non-zero) guard after pickup so ringback artifacts are skipped
    # without delaying real user speech by multiple seconds.
    VOICE_POST_PICKUP_STT_GRACE_SEC: float = 0.35
    # Inbound auto-greeting delay after user pickup (seconds).
    # Keep small but non-zero so call audio stabilizes before greeting starts.
    VOICE_INBOUND_GREETING_DELAY_SEC: float = 0.5
    # Pickup detector window and threshold (RMS frames over threshold) before STT starts.
    VOICE_PICKUP_SAMPLE_WINDOW: int = 6
    VOICE_PICKUP_MIN_NON_SILENT_FRAMES: int = 4
    # Allow RAG prefetch to start earlier than interim-LLM gates.
    VOICE_RAG_PREFETCH_MIN_WORDS: int = 1
    VOICE_RAG_PREFETCH_MIN_CONFIDENCE: float = 0.05
    # TTS speed/volume bounds — shared by API schema (TtsSettingsJsonSchema) and
    # runtime clamping (resolve_tts_runtime). Tune per deploy without code changes.
    TTS_SPEED_MIN: float = 0.25
    TTS_SPEED_MAX: float = 2.0
    TTS_VOLUME_MIN: float = 0.0
    TTS_VOLUME_MAX: float = 2.0
    # Start TTS streaming sooner for short first chunks.
    VOICE_TTS_STREAM_MIN_WORDS: int = 2
    # Twilio jitter buffer priming frames (20ms each) for low-latency voice output.
    VOICE_TTS_PRIME_FRAMES: int = 1
    VOICE_QUICK_ACK_MIN_WORDS: int = 5
    # Quick-ack: fires on slow-path queries only (fastpath is excluded at call site).
    # In V2 TtsPipeline, LLM chunk synthesis runs in parallel with quick-ack playback
    # so the "shutter then silence" gap only occurs when LLM TTFT > quick-ack duration.
    # 0.35 = fires roughly every third slow-path turn; set 0.0 to disable entirely.
    VOICE_QUICK_ACK_PROBABILITY: float = 0.35
    # Fast-path for very short/simple turns to reduce first-token latency:
    # skip heavy RAG/KB context for obvious non-booking smalltalk.
    VOICE_ENABLE_LATENCY_FASTPATH: bool = True
    VOICE_FASTPATH_MAX_WORDS: int = 7

    # Vapi-style intelligent contact recovery (additive — never downgrades intake confidence).
    # 1) Deterministic email STT-artifact cleanup: strip commas/spaces inside an email span
    #    ("ali.sa,ee,b@gmail.com" -> "ali.saeeb@gmail.com") before strict validation.
    EMAIL_STT_CLEANUP_ENABLED: bool = True
    # 2) Natural confirmation path: when the agent repeats a name ("just to confirm, your
    #    name is Alex Carter") and the caller affirms ("yes" / "correct"), mark name_confident
    #    without requiring letter-by-letter spelling.
    VOICE_NATURAL_NAME_CONFIRMATION: bool = True
    # 3) Post-call LLM recovery for contact (name/email) when strict intake gate would fail.
    #    Only invoked AFTER the call ends, behind a flag, and only if OPENAI_API_KEY is set.
    POST_CALL_LLM_CONTACT_RECOVERY: bool = True
    POST_CALL_LLM_CONTACT_RECOVERY_MODEL: str = "gpt-4o-mini"
    # If name is already confident but email is missing, still run post-call contact LLM once.
    POST_CALL_LLM_EMAIL_RECOVERY_WHEN_NAME_OK: bool = True
    # If the model returns a valid normalized email but email_confident=false, trust it when
    # the same address is clearly present in the transcript (reduces false negatives).
    POST_CALL_LLM_EMAIL_ANCHOR_TRUST: bool = True

    # Stripe settings
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_ID_FREE: str = ""
    STRIPE_PRICE_ID_PRO: str = ""
    
    # Billing settings
    FREE_PLAN_AGENT_LIMIT: int = 2
    FREE_PLAN_MONTHLY_CALLS: int = 100
    PRO_PLAN_AGENT_LIMIT: int = 50
    PRO_PLAN_MONTHLY_CALLS: int = 10000
    
    # Rate limiting settings
    REDIS_URL: str = "redis://localhost:6379"
    RATE_LIMIT_ENABLED: bool = True
    
    # Login rate limiting — per-IP, stricter than global API limit (enforce_login_rate_limit)
    LOGIN_RATE_LIMIT: int = 10
    LOGIN_RATE_WINDOW: int = 60  # seconds
    
    # Webhook rate limiting (requests per minute)
    WEBHOOK_RATE_LIMIT: int = 100
    WEBHOOK_RATE_WINDOW: int = 60  # seconds
    
    # General API rate limiting — global sliding-window middleware
    API_RATE_LIMIT: int = 60   # requests per window per identity
    API_RATE_WINDOW: int = 60  # seconds

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # Vector / RAG configuration
    # Optional generic vector DB URL (used as a fallback host for Pinecone).
    VECTOR_DB_URL: Optional[str] = None
    # Default embedding dimension for text-embedding models
    VECTOR_DIMENSION: int = 1536  # e.g. OpenAI text-embedding-3-small

    # Pinecone (preferred vector store for RAG)
    PINECONE_API_KEY: str = ""
    # Optional: direct index host, if you copy it from Pinecone console.
    # Example: "your-index-host.svc.us-east-1-aws.pinecone.io"
    PINECONE_INDEX_HOST: Optional[str] = None
    # Optional: index name; if host is not provided, we can resolve host from this.
    PINECONE_INDEX_NAME: Optional[str] = None
    
    # Twilio Edge hint (for logging/observability; set actual edge in Twilio Console)
    TWILIO_EDGE: Optional[str] = "umatilla"  # e.g., "ashburn", "singapore", "dublin"
    # App deployment region hint for latency diagnostics.
    SERVER_REGION: str = "us-west-2"  # Oregon
    # If enabled, log warnings when Twilio edge does not match expected regional edge.
    VOICE_REGION_ALIGNMENT_STRICT: bool = True

    # RAG behavior tuning (voice-first defaults)
    # Master switch for latency A/B tests and emergency fail-open behavior.
    # Set RAG_ENABLED=false in .env to skip retrieval entirely.
    RAG_ENABLED: bool = True
    # These defaults are intentionally conservative to avoid prompt bloat/latency.
    # Primary embedding model (OpenAI by default).
    # Make embedding model configurable because some OpenAI projects do not
    # have access to all embedding models.
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Fallback embedding model/provider used when the primary embedding call fails.
    RAG_FALLBACK_EMBEDDING_PROVIDER: str = "gemini"
    RAG_FALLBACK_EMBEDDING_MODEL: str = "gemini-embedding-001"
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.4
    # Hard cap for the size of the rendered context block injected into prompts.
    # This is character-based (approx). For token-accurate sizing, you would need a tokenizer.
    RAG_MAX_CONTEXT_CHARS: int = 6000

    # Voice latency guardrails
    # If Pinecone or embedding generation is slow, we must fail fast and
    # return an empty knowledge context to avoid breaking the voice UX.
    RAG_RETRIEVAL_TIMEOUT_SEC: float = 0.45
    # Slow-path budget: cap cumulative waits for RAG/KB on a turn.
    VOICE_SLOWPATH_BUDGET_SEC: float = 0.55
    # How long we wait for an in-flight RAG prefetch before failing open.
    VOICE_RAG_PREFETCH_AWAIT_SEC: float = 0.18
    # When KB cache isn't ready on early turns, skip live DB fetch to protect latency.
    VOICE_SKIP_LIVE_KB_FETCH_ON_COLD_START: bool = True
    # Prevent extremely large STT transcripts from being embedded.
    RAG_MAX_QUERY_CHARS: int = 3000

    # Optional retrieval reranking (lexical overlap).
    # Keep disabled by default to avoid changing retrieval semantics unexpectedly.
    RAG_ENABLE_RERANK: bool = False
    # Weight for Pinecone vector similarity vs lexical overlap.
    # Higher means more trust in vector similarity.
    RAG_RERANK_VECTOR_WEIGHT: float = 0.8

    # Voice latency SLO thresholds (seconds) for observability.
    VOICE_SLO_ENABLED: bool = True
    VOICE_SLO_STT_FINAL_TO_GEN_START_SEC: float = 0.35
    VOICE_SLO_GEN_START_TO_LLM_FIRST_TOKEN_SEC: float = 0.90
    VOICE_SLO_GEN_START_TO_FIRST_TTS_QUEUE_SEC: float = 1.40
    VOICE_SLO_GEN_START_TO_NOW_WARN_SEC: float = 2.00
    
    # Trello — platform-managed inbound call boards (optional)
    TRELLO_PLATFORM_API_KEY: str = ""
    TRELLO_PLATFORM_API_TOKEN: str = ""

    # Monday.com Configuration
    MONDAY_API_KEY: str = ""  # Monday.com Personal API Token
    MONDAY_BOARD_ID: str = ""  # Monday.com Board ID for scheduled calls
    MONDAY_WORKSPACE_ID: Optional[str] = None  # Optional workspace to create tenant boards in

    # Resume ↔ job matching (recruiting): LLM + rules
    # hybrid = blend (recommended); rules = heuristics only; ai = LLM scores (rules if LLM fails)
    RECRUIT_MATCH_MODE: str = "hybrid"
    # Weight of LLM vs rules when match_mode=hybrid (0–1). Higher = trust AI more.
    RECRUIT_MATCH_AI_WEIGHT: float = 0.68
    RECRUIT_MATCH_LLM_PROVIDER: str = "auto"  # auto | openai | gemini
    RECRUIT_MATCH_OPENAI_MODEL: str = "gpt-4o-mini"
    RECRUIT_MATCH_GEMINI_MODEL: str = "gemini-1.5-flash"
    RECRUIT_MATCH_LLM_TEMPERATURE: float = 0.12
    RECRUIT_MATCH_LLM_MAX_TOKENS: int = 600
    RECRUIT_MATCH_MAX_PROMPT_CHARS: int = 14000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()