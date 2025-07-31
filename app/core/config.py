from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    # Role IDs
    ADMIN_ROLE_ID: int = 1
    
    DATABASE_URL: str = "postgresql+psycopg2://postgres:1234@localhost:5432/voiceagent"
    SECRET_KEY: str = "supersecretkey"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings() 