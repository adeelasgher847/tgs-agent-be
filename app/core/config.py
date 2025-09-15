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

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()