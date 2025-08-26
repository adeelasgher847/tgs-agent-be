from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql+psycopg2://postgres:123456@localhost:5432/voiceagent"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    # Email settings for password reset
    SMTP_HOST: str = "mubeenhussain8@gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "voice_agent"
    SMTP_PASSWORD: str = "luse tpvz rsqb ahij"
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    
    # Password reset settings
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30
    FRONTEND_URL: str = "http://localhost:3000"  

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings() 