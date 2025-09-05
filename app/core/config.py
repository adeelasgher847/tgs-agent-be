from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql://neondb_owner:npg_O0gvul4bTMPH@ep-raspy-lab-afr28nzh-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = "AC4071b00aa2ea5985a36201ad45a98ed1"
    TWILIO_AUTH_TOKEN: str = "2940ee281d4991f6fe3169afd470a620"
    TWILIO_PHONE_NUMBER: str = "+1234567890"  # Replace with your actual Twilio phone number
    ALLOW_UNAUTHENTICATED_WEBHOOKS: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
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
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = ""
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = "app.log"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings() 