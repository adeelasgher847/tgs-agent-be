from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql+psycopg2://postgres:1234@localhost:5432/voiceagent"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()