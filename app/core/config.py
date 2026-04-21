from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql+psycopg2://postgres:admin@localhost:5432/voiceagent"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = "+13466602410"  # TODO: Replace with your actual Twilio phone number from Twilio Console
    ALLOW_UNAUTHENTICATED_WEBHOOKS: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
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
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = ""
    
    # Google Cloud Speech-to-Text Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # Path to service account JSON file
    GOOGLE_CLOUD_PROJECT_ID: str = ""
    GOOGLE_STT_LANGUAGE_CODE: str = "en-US"  # Default language
    # Deprecated fallback; prefer STT_SAMPLE_RATE for provider-neutral STT settings.
    GOOGLE_STT_SAMPLE_RATE: int = 8000
    GOOGLE_STT_ENCODING: str = "MULAW"  # Twilio's audio encoding

    # Deepgram Speech-to-Text (replaces Google STT for streaming + batch)
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_STT_LANGUAGE: str = "en"  # Deepgram listen param; override in .env if needed
    DEEPGRAM_STT_ENDPOINTING_MS: int = 300  # silence (ms) before utterance end / speech_final
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
    VOICE_HISTORY_MAX_MESSAGES: int = 12
    VOICE_TTS_FLUSH_MIN_WORDS: int = 2
    VOICE_TTS_FLUSH_MAX_WORDS: int = 12
    VOICE_QUICK_ACK_MIN_WORDS: int = 5
    VOICE_QUICK_ACK_PROBABILITY: float = 0.38
    
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
    
    # Login rate limiting (requests per minute)
    LOGIN_RATE_LIMIT: int = 5
    LOGIN_RATE_WINDOW: int = 60  # seconds
    
    # Webhook rate limiting (requests per minute)
    WEBHOOK_RATE_LIMIT: int = 100
    WEBHOOK_RATE_WINDOW: int = 60  # seconds
    
    # General API rate limiting (requests per minute)
    API_RATE_LIMIT: int = 1000
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

    # RAG behavior tuning (voice-first defaults)
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
    RAG_RETRIEVAL_TIMEOUT_SEC: float = 2.0
    # Prevent extremely large STT transcripts from being embedded.
    RAG_MAX_QUERY_CHARS: int = 3000

    # Optional retrieval reranking (lexical overlap).
    # Keep disabled by default to avoid changing retrieval semantics unexpectedly.
    RAG_ENABLE_RERANK: bool = False
    # Weight for Pinecone vector similarity vs lexical overlap.
    # Higher means more trust in vector similarity.
    RAG_RERANK_VECTOR_WEIGHT: float = 0.8
    
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