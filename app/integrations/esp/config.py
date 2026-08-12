from pydantic_settings import BaseSettings, SettingsConfigDict

class EspSettings(BaseSettings):
    api_base_url: str = "https://expressserviceprotection.inlineadmin.com"
    query_id: str = "38"
    session_cookie: str = ""

    model_config = SettingsConfigDict(
        env_prefix="ESP_",
        env_file=".env",
        extra="ignore"
    )

esp_settings = EspSettings()
