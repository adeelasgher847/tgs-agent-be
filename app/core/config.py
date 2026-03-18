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
    GOOGLE_STT_SAMPLE_RATE: int = 8000  # Twilio uses 8kHz for MULAW
    GOOGLE_STT_ENCODING: str = "MULAW"  # Twilio's audio encoding

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
    
    FRONTEND_URL: str = "http://localhost:3000"  
    
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
    
    # Twilio settings
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    
    # Webhook settings
    ALLOW_UNAUTHENTICATED_WEBHOOKS: bool = False
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
    
    # Monday.com Configuration
    MONDAY_API_KEY: str = ""  # Monday.com Personal API Token
    MONDAY_BOARD_ID: str = ""  # Monday.com Board ID for scheduled calls
    MONDAY_WORKSPACE_ID: Optional[str] = None  # Optional workspace to create tenant boards in

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()