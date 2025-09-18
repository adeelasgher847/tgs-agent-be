from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    ADMIN_ROLE: str = "admin"
    
    DATABASE_URL: str = "postgresql://neondb_owner:npg_O0gvul4bTMPH@ep-raspy-lab-afr28nzh-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
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

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()