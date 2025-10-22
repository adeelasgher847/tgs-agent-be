from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # Email settings for password reset
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "mubeenhussain8@gmail.com"
    SMTP_PASSWORD: str = "luse tpvz rsqb ahij"
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    
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
    
    # Voice Conversation Settings
    USE_GATHER_APPROACH: bool = True  # Use <Gather> for STT (proven reliable!)
    USE_BIDIRECTIONAL_STREAMING: bool = False  # Disabled - using Gather for STT
    USE_WEBSOCKET_TTS: bool = True  # ✅ NEW - Stream TTS via WebSocket for instant playback (no HTTP fetch delay!)
    
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()