"""
app/core/config.py — Application settings.

Flat env-var fields remain on `Settings` for full backwards compatibility
(60+ call sites use `settings.FLAT_NAME`).  Eight domain sub-models are
assembled by a model_validator and exposed as grouped views:

    settings.db.url          # same value as settings.DATABASE_URL
    settings.auth.secret_key # same value as settings.SECRET_KEY
    settings.twilio.account_sid
    settings.llm.openai_api_key
    settings.tts.provider
    settings.crm.hubspot_client_id
    settings.server.environment
    settings.redis.url
"""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Domain sub-models
# Each field carries a validation_alias matching the canonical env-var name so
# the sub-model can also be instantiated directly from a flat env dict via
# SubModel.model_validate(os.environ).  populate_by_name=True allows
# construction by the Python field name as well (used in the Settings validator
# below and in tests).
# ---------------------------------------------------------------------------


class DbSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(
        default="postgresql+psycopg2://postgres:admin@localhost:5432/voiceagent",
        validation_alias="DATABASE_URL",
    )
    pool_size: int = Field(default=10, validation_alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, validation_alias="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, validation_alias="DATABASE_POOL_TIMEOUT")
    statement_timeout: int = Field(
        default=30000, validation_alias="DATABASE_STATEMENT_TIMEOUT"
    )


class AuthSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    secret_key: str = Field(default="", validation_alias="SECRET_KEY")
    algorithm: str = Field(default="", validation_alias="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=15, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=7, validation_alias="REFRESH_TOKEN_EXPIRE_DAYS"
    )
    password_reset_token_expire_minutes: int = Field(
        default=30, validation_alias="PASSWORD_RESET_TOKEN_EXPIRE_MINUTES"
    )
    webhook_secret_encryption_key: str = Field(
        default="", validation_alias="WEBHOOK_SECRET_ENCRYPTION_KEY"
    )
    sso_encryption_key: str = Field(default="", validation_alias="SSO_ENCRYPTION_KEY")
    google_client_id: str = Field(default="", validation_alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(
        default="", validation_alias="GOOGLE_CLIENT_SECRET"
    )


class TwilioSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    account_sid: str = Field(default="", validation_alias="TWILIO_ACCOUNT_SID")
    auth_token: str = Field(default="", validation_alias="TWILIO_AUTH_TOKEN")
    phone_number: str = Field(default="", validation_alias="TWILIO_PHONE_NUMBER")
    allow_unauthenticated_webhooks: bool = Field(
        default=False, validation_alias="ALLOW_UNAUTHENTICATED_WEBHOOKS"
    )
    test_account_sid: str = Field(
        default="", validation_alias="TWILIO_TEST_ACCOUNT_SID"
    )
    test_auth_token: str = Field(default="", validation_alias="TWILIO_TEST_AUTH_TOKEN")
    edge: str | None = Field(default="umatilla", validation_alias="TWILIO_EDGE")


class LlmSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # OpenAI
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", validation_alias="OPENAI_BASE_URL")
    openai_api_version: str = Field(default="", validation_alias="OPENAI_API_VERSION")
    openai_realtime_transcription_model: str = Field(
        default="whisper-1", validation_alias="OPENAI_REALTIME_TRANSCRIPTION_MODEL"
    )
    # Gemini / Vertex
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    google_application_credentials: str = Field(
        default="", validation_alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    google_cloud_project_id: str = Field(
        default="", validation_alias="GOOGLE_CLOUD_PROJECT_ID"
    )
    vertex_ai_location: str = Field(
        default="us-central1", validation_alias="VERTEX_AI_LOCATION"
    )
    # Provider selection
    provider: str = Field(default="google", validation_alias="LLM_PROVIDER")
    default_model: str = Field(
        default="gemini-1.5-flash", validation_alias="DEFAULT_LLM_MODEL"
    )
    default_provider: str = Field(
        default="gemini", validation_alias="DEFAULT_LLM_PROVIDER"
    )
    # Voice LLM tuning
    history_max_turns: int = Field(
        default=20, validation_alias="VOICE_LLM_HISTORY_MAX_TURNS"
    )
    default_temperature: float = Field(
        default=0.3, validation_alias="VOICE_LLM_DEFAULT_TEMPERATURE"
    )
    fallback_message: str = Field(
        default="I am sorry, I did not catch that",
        validation_alias="VOICE_LLM_FALLBACK_MESSAGE",
    )
    # Deepgram STT
    deepgram_api_key: str = Field(default="", validation_alias="DEEPGRAM_API_KEY")
    deepgram_stt_model: str = Field(
        default="nova-3", validation_alias="DEEPGRAM_STT_MODEL"
    )
    deepgram_stt_language: str = Field(
        default="en", validation_alias="DEEPGRAM_STT_LANGUAGE"
    )
    deepgram_stt_endpointing_ms: int = Field(
        default=350, validation_alias="DEEPGRAM_STT_ENDPOINTING_MS"
    )
    deepgram_stt_endpointing_ms_extended: int = Field(
        default=500, validation_alias="DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED"
    )
    deepgram_stt_utterance_end_ms: int = Field(
        default=1000, validation_alias="DEEPGRAM_STT_UTTERANCE_END_MS"
    )
    # Google STT
    google_stt_language_code: str = Field(
        default="en-US", validation_alias="GOOGLE_STT_LANGUAGE_CODE"
    )
    google_stt_sample_rate: int = Field(
        default=8000, validation_alias="GOOGLE_STT_SAMPLE_RATE"
    )
    google_stt_encoding: str = Field(
        default="MULAW", validation_alias="GOOGLE_STT_ENCODING"
    )


class TtsSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str = Field(default="elevenlabs", validation_alias="TTS_PROVIDER")
    api_key: str = Field(default="", validation_alias="TTS_API_KEY")
    rime_api_key: str = Field(default="", validation_alias="RIME_API_KEY")
    hume_api_key: str = Field(default="", validation_alias="HUME_API_KEY")
    hume_sample_rate_hz: int = Field(
        default=48000, validation_alias="HUME_TTS_SAMPLE_RATE_HZ"
    )
    elevenlabs_api_key: str = Field(default="", validation_alias="ELEVENLABS_API_KEY")
    elevenlabs_encryption_key: str = Field(
        default="", validation_alias="ELEVENLABS_ENCRYPTION_KEY"
    )
    enable_audio_tags: bool = Field(
        default=True, validation_alias="ENABLE_ELEVENLABS_AUDIO_TAGS"
    )
    cloud_endpoint: str = Field(default="", validation_alias="CLOUD_TTS_ENDPOINT")
    google_voice_name: str = Field(default="", validation_alias="GOOGLE_TTS_VOICE_NAME")
    speed_min: float = Field(default=0.25, validation_alias="TTS_SPEED_MIN")
    speed_max: float = Field(default=2.0, validation_alias="TTS_SPEED_MAX")
    volume_min: float = Field(default=0.0, validation_alias="TTS_VOLUME_MIN")
    volume_max: float = Field(default=2.0, validation_alias="TTS_VOLUME_MAX")


class CrmSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # HubSpot
    hubspot_client_id: str = Field(default="", validation_alias="HUBSPOT_CLIENT_ID")
    hubspot_client_secret: str = Field(
        default="", validation_alias="HUBSPOT_CLIENT_SECRET"
    )
    hubspot_redirect_uri: str = Field(
        default="", validation_alias="HUBSPOT_REDIRECT_URI"
    )
    hubspot_token_encryption_key: str = Field(
        default="", validation_alias="HUBSPOT_TOKEN_ENCRYPTION_KEY"
    )
    # Salesforce
    salesforce_client_id: str = Field(
        default="", validation_alias="SALESFORCE_CLIENT_ID"
    )
    salesforce_client_secret: str = Field(
        default="", validation_alias="SALESFORCE_CLIENT_SECRET"
    )
    salesforce_redirect_uri: str = Field(
        default="", validation_alias="SALESFORCE_REDIRECT_URI"
    )
    salesforce_token_encryption_key: str = Field(
        default="", validation_alias="SALESFORCE_TOKEN_ENCRYPTION_KEY"
    )
    # GoHighLevel (GHL)
    ghl_client_id: str = Field(default="", validation_alias="GHL_CLIENT_ID")
    ghl_client_secret: str = Field(default="", validation_alias="GHL_CLIENT_SECRET")
    ghl_redirect_uri: str = Field(default="", validation_alias="GHL_REDIRECT_URI")
    ghl_token_encryption_key: str = Field(
        default="", validation_alias="GHL_TOKEN_ENCRYPTION_KEY"
    )
    # Slack
    slack_client_id: str = Field(default="", validation_alias="SLACK_CLIENT_ID")
    slack_client_secret: str = Field(default="", validation_alias="SLACK_CLIENT_SECRET")
    slack_redirect_uri: str = Field(default="", validation_alias="SLACK_REDIRECT_URI")
    slack_token_encryption_key: str = Field(
        default="", validation_alias="SLACK_TOKEN_ENCRYPTION_KEY"
    )
    # Calendly
    calendly_client_id: str = Field(default="", validation_alias="CALENDLY_CLIENT_ID")
    calendly_client_secret: str = Field(
        default="", validation_alias="CALENDLY_CLIENT_SECRET"
    )
    calendly_redirect_uri: str = Field(
        default="", validation_alias="CALENDLY_REDIRECT_URI"
    )
    calendly_token_encryption_key: str = Field(
        default="", validation_alias="CALENDLY_TOKEN_ENCRYPTION_KEY"
    )
    # Monday.com
    monday_api_key: str = Field(default="", validation_alias="MONDAY_API_KEY")
    monday_board_id: str = Field(default="", validation_alias="MONDAY_BOARD_ID")
    monday_workspace_id: str | None = Field(
        default=None, validation_alias="MONDAY_WORKSPACE_ID"
    )
    # Trello
    trello_api_key: str = Field(default="", validation_alias="TRELLO_PLATFORM_API_KEY")
    trello_api_token: str = Field(
        default="", validation_alias="TRELLO_PLATFORM_API_TOKEN"
    )
    # AWS SES
    aws_ses_sender_email: str = Field(
        default="", validation_alias="AWS_SES_SENDER_EMAIL"
    )
    # Stripe
    stripe_publishable_key: str = Field(
        default="", validation_alias="STRIPE_PUBLISHABLE_KEY"
    )
    stripe_secret_key: str = Field(default="", validation_alias="STRIPE_SECRET_KEY")
    stripe_webhook_secret: str = Field(
        default="", validation_alias="STRIPE_WEBHOOK_SECRET"
    )
    stripe_incall_webhook_secret: str = Field(
        default="", validation_alias="STRIPE_INCALL_WEBHOOK_SECRET"
    )
    stripe_price_id_free: str = Field(
        default="", validation_alias="STRIPE_PRICE_ID_FREE"
    )
    stripe_price_id_pro: str = Field(default="", validation_alias="STRIPE_PRICE_ID_PRO")
    payment_page_base_url: str = Field(
        default="https://pay.yourdomain.com", validation_alias="PAYMENT_PAGE_BASE_URL"
    )
    # Billing plan limits
    free_plan_agent_limit: int = Field(
        default=2, validation_alias="FREE_PLAN_AGENT_LIMIT"
    )
    free_plan_monthly_calls: int = Field(
        default=100, validation_alias="FREE_PLAN_MONTHLY_CALLS"
    )
    pro_plan_agent_limit: int = Field(
        default=50, validation_alias="PRO_PLAN_AGENT_LIMIT"
    )
    pro_plan_monthly_calls: int = Field(
        default=10000, validation_alias="PRO_PLAN_MONTHLY_CALLS"
    )


class ServerSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    app_version: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    admin_role: str = Field(default="admin", validation_alias="ADMIN_ROLE")
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        validation_alias="ALLOWED_ORIGINS",
    )
    # API docs (HTTP Basic)
    api_docs_enabled: bool = Field(default=True, validation_alias="API_DOCS_ENABLED")
    api_docs_username: str = Field(default="", validation_alias="API_DOCS_USERNAME")
    api_docs_password: str = Field(default="", validation_alias="API_DOCS_PASSWORD")
    # Webhooks & URLs
    webhook_base_url: str = Field(
        default="https://tgs-agent-be.onrender.com", validation_alias="WEBHOOK_BASE_URL"
    )
    n8n_webhook_url: str = Field(default="", validation_alias="N8N_WEBHOOK_URL")
    n8n_webhook_secret: str = Field(default="", validation_alias="N8N_WEBHOOK_SECRET")
    frontend_url: str = Field(
        default="http://localhost:3000", validation_alias="FRONTEND_URL"
    )
    # GCP infra
    gcp_project_id: str = Field(default="", validation_alias="GCP_PROJECT_ID")
    server_region: str = Field(default="us-west-2", validation_alias="SERVER_REGION")
    # LiveKit
    livekit_url: str = Field(default="", validation_alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", validation_alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", validation_alias="LIVEKIT_API_SECRET")
    livekit_token_ttl: int = Field(default=3600, validation_alias="LIVEKIT_TOKEN_TTL")
    livekit_room_empty_timeout: int = Field(
        default=30, validation_alias="LIVEKIT_ROOM_EMPTY_TIMEOUT"
    )
    livekit_max_participants: int = Field(
        default=2, validation_alias="LIVEKIT_MAX_PARTICIPANTS"
    )
    livekit_enabled: bool = Field(default=True, validation_alias="LIVEKIT_ENABLED")
    # GCS recordings
    gcs_recordings_bucket: str = Field(
        default="", validation_alias="GCS_RECORDINGS_BUCKET"
    )
    gcs_recordings_signed_url_expiry_seconds: int = Field(
        default=3600, validation_alias="GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS"
    )
    gcs_recordings_prefix: str = Field(
        default="recordings", validation_alias="GCS_RECORDINGS_PREFIX"
    )
    # GCS knowledge base
    gcs_kb_bucket: str = Field(default="", validation_alias="GCS_KB_BUCKET")
    gcs_kb_prefix: str = Field(default="kb-files", validation_alias="GCS_KB_PREFIX")
    # AWS S3 storage
    aws_access_key_id: str = Field(default="", validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(
        default="", validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_region_name: str = Field(
        default="us-east-1", validation_alias="AWS_REGION_NAME"
    )
    s3_recordings_bucket: str = Field(
        default="", validation_alias="S3_RECORDINGS_BUCKET"
    )
    s3_kb_bucket: str = Field(default="", validation_alias="S3_KB_BUCKET")
    # Concurrency
    outbound_max_concurrent_per_workspace: int = Field(
        default=10, validation_alias="OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE"
    )
    max_batch_concurrency: int = Field(
        default=5, validation_alias="MAX_BATCH_CONCURRENCY"
    )


class RedisSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    rate_limit_enabled: bool = Field(
        default=True, validation_alias="RATE_LIMIT_ENABLED"
    )
    login_rate_limit: int = Field(default=10, validation_alias="LOGIN_RATE_LIMIT")
    login_rate_window: int = Field(default=60, validation_alias="LOGIN_RATE_WINDOW")
    webhook_rate_limit: int = Field(default=100, validation_alias="WEBHOOK_RATE_LIMIT")
    webhook_rate_window: int = Field(default=60, validation_alias="WEBHOOK_RATE_WINDOW")
    api_rate_limit: int = Field(default=60, validation_alias="API_RATE_LIMIT")
    api_rate_window: int = Field(default=60, validation_alias="API_RATE_WINDOW")
    public_token_rate_limit: int = Field(
        default=20, validation_alias="PUBLIC_TOKEN_RATE_LIMIT"
    )
    public_token_rate_window: int = Field(
        default=60, validation_alias="PUBLIC_TOKEN_RATE_WINDOW"
    )


# ---------------------------------------------------------------------------
# Main Settings
# All flat env-var fields are kept so existing callers (settings.DATABASE_URL,
# settings.OPENAI_API_KEY, …) continue to work without modification.
# The eight sub-model fields below are populated by _assemble_sub_models after
# all flat fields have been read from the environment.
# ---------------------------------------------------------------------------


class Settings(BaseSettings):

    ADMIN_ROLE: str = "admin"

    DATABASE_URL: str = "postgresql+psycopg2://postgres:admin@localhost:5432/voiceagent"

    # Connection pool tuning
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30  # seconds to wait for a connection from the pool
    DATABASE_STATEMENT_TIMEOUT: int = (
        30000  # milliseconds; auto-terminates slow queries in PG
    )

    SECRET_KEY: str = ""
    ALGORITHM: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Environment — controls which Twilio credentials are used and Secret Manager behaviour.
    # Values: "development" | "staging" | "production"
    ENVIRONMENT: str = "development"

    # Twilio Configuration
    # In production/staging these should come from Secret Manager (see app/core/secret_manager.py).
    # They are kept here as fallbacks for local development only.
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    ALLOW_UNAUTHENTICATED_WEBHOOKS: bool = False

    # Twilio test credentials — used automatically when ENVIRONMENT="staging".
    # Set via Secret Manager or .env.staging; never commit real values.
    TWILIO_TEST_ACCOUNT_SID: str = ""
    TWILIO_TEST_AUTH_TOKEN: str = ""

    # GCP Secret Manager project ID (required in staging/production).
    GCP_PROJECT_ID: str = ""

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    APP_VERSION: str = "1.0.0"

    # Betterstack status page badge — surfaced in GET /api/v1/health when set.
    # Betterstack monitors https://api.yourdomain.com/api/v1/health (HTTP 200
    # expected, 30s interval) and serves the public status page at
    # status.yourdomain.com; this URL points at that page's embeddable badge.
    BETTERSTACK_BADGE_URL: str | None = None

    # Swagger / committed OpenAPI at GET /api/docs (HTTP Basic — not dashboard JWT).
    API_DOCS_ENABLED: bool = True
    API_DOCS_USERNAME: str = ""
    API_DOCS_PASSWORD: str = ""

    # CORS — comma-separated list of allowed origins.
    # Example: "https://app.example.com,https://admin.example.com"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Webhook Configuration
    WEBHOOK_BASE_URL: str = "https://tgs-agent-be.onrender.com"
    N8N_WEBHOOK_URL: str = ""  # n8n webhook URL for scheduled calls
    N8N_WEBHOOK_SECRET: str = ""  # Secret for verifying n8n webhook requests
    # Email settings (AWS SES) — sender identity; delivery uses AWS_ACCESS_KEY_ID /
    # AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME (shared with S3, declared below).
    AWS_SES_SENDER_EMAIL: str = ""

    # Password reset settings
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30
    FRONTEND_URL: str = "http://localhost:3000"

    GEMINI_API_KEY: str = ""
    # Default LLM when an agent has no ticket llm_model or legacy model relation.
    DEFAULT_LLM_MODEL: str = "gemini-1.5-flash"
    DEFAULT_LLM_PROVIDER: str = "gemini"
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    # Input-audio transcription model for OpenAI Realtime sessions (separate
    # from the main gpt-realtime*/gpt-realtime-2 model itself). Defaults to
    # whisper-1 — OpenAI's oldest, most universally-enabled transcription
    # model — since newer options like gpt-4o-mini-transcribe require
    # per-project access that isn't guaranteed to be enabled (confirmed via
    # a real call: model_not_found for gpt-4o-mini-transcribe on a project
    # that otherwise has full Realtime API access). Override per-environment
    # if a project has access to a better model.
    OPENAI_REALTIME_TRANSCRIPTION_MODEL: str = "whisper-1"

    # Rime Labs TTS Configuration
    RIME_API_KEY: str = ""

    # Hume AI TTS Configuration
    HUME_API_KEY: str = ""
    # ASSUMED default — Hume's PCM streaming output sample rate is not
    # documented publicly as of this integration (2026-08). 48000 Hz is a
    # reasonable industry-standard guess for a modern neural TTS PCM stream,
    # but MUST be verified against a real Hume account/API key before this
    # goes to production traffic; override via env var if Hume support
    # confirms a different value. Drives PCMStreamDownsampler's src_rate_hz
    # when converting Hume's PCM16LE chunks down to mulaw 8kHz for Twilio.
    HUME_TTS_SAMPLE_RATE_HZ: int = 48000

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str = ""

    # Outbound number reputation monitoring (First Orion / Hiya)
    REPUTATION_API_KEY: str = ""
    # Symmetric encryption key for agent.encrypted_elevenlabs_api_key (pgp_sym_encrypt).
    # In production/staging load from Secret Manager; never commit a real value.
    ELEVENLABS_ENCRYPTION_KEY: str = ""
    # Symmetric encryption key for webhookendpoint.secret (pgp_sym_encrypt).
    # Reads transparently handle legacy JWT-encrypted secrets for backwards compat.
    # In production/staging load from Secret Manager; never commit a real value.
    WEBHOOK_SECRET_ENCRYPTION_KEY: str = ""
    # Symmetric encryption key for OIDC client secrets (Fernet).
    SSO_ENCRYPTION_KEY: str = ""
    # When True, voice LLM prompts may suggest bracketed audio tags for ElevenLabs TTS only
    # ([breathes], [pause], [excited], [sad], …). Set False if your TTS model reads brackets out loud.
    ENABLE_ELEVENLABS_AUDIO_TAGS: bool = True

    # HubSpot CRM OAuth (app/services/hubspot_service.py).
    # client_id/client_secret kept here as local-dev fallbacks only — in
    # staging/production they are read from Secret Manager (see
    # app/core/secret_manager.py::get_hubspot_oauth_credentials).
    HUBSPOT_CLIENT_ID: str = ""
    HUBSPOT_CLIENT_SECRET: str = ""
    HUBSPOT_REDIRECT_URI: str = (
        ""  # defaults to {WEBHOOK_BASE_URL}/api/v1/integrations/hubspot/callback
    )
    # Symmetric encryption key for workspaceintegration.access_token / refresh_token
    # (pgp_sym_encrypt) — same scheme as ELEVENLABS_ENCRYPTION_KEY above.
    HUBSPOT_TOKEN_ENCRYPTION_KEY: str = ""

    # Salesforce CRM OAuth (app/services/salesforce_service.py).
    # client_id/client_secret kept here as local-dev fallbacks only — in
    # staging/production they are read from Secret Manager (see
    # app/core/secret_manager.py::get_salesforce_oauth_credentials).
    SALESFORCE_CLIENT_ID: str = ""
    SALESFORCE_CLIENT_SECRET: str = ""
    SALESFORCE_REDIRECT_URI: str = (
        ""  # defaults to {WEBHOOK_BASE_URL}/api/v1/integrations/salesforce/callback
    )
    # Symmetric encryption key for workspaceintegration.access_token / refresh_token
    # (AES-256-GCM) — same scheme as HUBSPOT_TOKEN_ENCRYPTION_KEY above.
    SALESFORCE_TOKEN_ENCRYPTION_KEY: str = ""
    # Fixed OAuth login host — https://login.salesforce.com for production orgs,
    # https://test.salesforce.com for sandboxes. NOT per-tenant: the data-plane
    # instance_url returned in the token response is what varies per org.
    SALESFORCE_LOGIN_URL: str = "https://login.salesforce.com"
    SALESFORCE_API_VERSION: str = "v59.0"
    # Salesforce token responses (unlike HubSpot's) do not include expires_in —
    # session length is an org-configurable setting with no fixed default we can
    # read via the API. We assume a conservative TTL and rely on AC #4's
    # refresh-60s-early rule to force a refresh well before typical session
    # timeouts (Salesforce's own default session timeout is 2 hours).
    SALESFORCE_ACCESS_TOKEN_TTL_SECONDS: int = 1800

    # GoHighLevel (GHL) CRM OAuth (app/services/ghl_service.py).
    # client_id/client_secret kept here as local-dev fallbacks only — in
    # staging/production they are read from Secret Manager (see
    # app/core/secret_manager.py::get_ghl_oauth_credentials).
    GHL_CLIENT_ID: str = ""
    GHL_CLIENT_SECRET: str = ""
    GHL_REDIRECT_URI: str = (
        ""  # defaults to {WEBHOOK_BASE_URL}/api/v1/integrations/leadconnector/callback
    )
    # Symmetric encryption key for workspaceintegration.access_token / refresh_token
    # (AES-256-GCM) — same scheme as HUBSPOT_TOKEN_ENCRYPTION_KEY above.
    GHL_TOKEN_ENCRYPTION_KEY: str = ""
    GHL_API_BASE_URL: str = "https://services.leadconnectorhq.com"
    GHL_API_VERSION: str = "2021-07-28"

    # Slack post-call-summary OAuth (app/services/slack_service.py).
    # client_id/client_secret kept here as local-dev fallbacks only — in
    # staging/production they are read from Secret Manager (see
    # app/core/secret_manager.py::get_slack_oauth_credentials).
    SLACK_CLIENT_ID: str = ""
    SLACK_CLIENT_SECRET: str = ""
    SLACK_REDIRECT_URI: str = (
        ""  # defaults to {WEBHOOK_BASE_URL}/api/v1/integrations/slack/callback
    )
    # Symmetric encryption key for workspaceintegration.access_token
    # (AES-256-GCM) — same scheme as HUBSPOT_TOKEN_ENCRYPTION_KEY above.
    SLACK_TOKEN_ENCRYPTION_KEY: str = ""

    # Calendly calendar OAuth (app/services/calendly_service.py).
    CALENDLY_CLIENT_ID: str = ""
    CALENDLY_CLIENT_SECRET: str = ""
    CALENDLY_REDIRECT_URI: str = (
        ""  # e.g. {WEBHOOK_BASE_URL}/api/v2/integrations/calendly/callback
    )
    # Symmetric encryption key for calendlyintegration.access_token / refresh_token
    # (AES-256-GCM) — same scheme as HUBSPOT_TOKEN_ENCRYPTION_KEY above.
    CALENDLY_TOKEN_ENCRYPTION_KEY: str = ""

    # Google Cloud Speech-to-Text Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str = (
        ""  # Path to service account JSON file for Vertex AI + STT
    )
    GOOGLE_CLOUD_PROJECT_ID: str = ""
    # Vertex AI — used by VertexGeminiService for voice LLM calls (ADC, no api_key needed)
    VERTEX_AI_LOCATION: str = "us-central1"
    # History pruning for Vertex Gemini voice path (1 turn = 1 user + 1 model message)
    VOICE_LLM_HISTORY_MAX_TURNS: int = 20
    # Default temperature for Vertex Gemini voice path (0–1 scale)
    VOICE_LLM_DEFAULT_TEMPERATURE: float = 0.3
    # Canned fallback spoken when the Vertex LLM errors (quota, timeout, filter)
    VOICE_LLM_FALLBACK_MESSAGE: str = "I am sorry, I did not catch that"
    GOOGLE_STT_LANGUAGE_CODE: str = "en-US"  # Default language
    # Deprecated fallback; prefer STT_SAMPLE_RATE for provider-neutral STT settings.
    GOOGLE_STT_SAMPLE_RATE: int = 8000
    GOOGLE_STT_ENCODING: str = "MULAW"  # Twilio's audio encoding

    # Speechmatics Speech-to-Text (optional provider — native VAD/end-of-utterance)
    SPEECHMATICS_API_KEY: str = ""
    # Leave blank to use the speechmatics-rt SDK default (wss://eu2.rt.speechmatics.com/v2)
    SPEECHMATICS_RT_URL: str = ""
    SPEECHMATICS_STT_MODEL: str = "enhanced"  # "enhanced" | "standard"
    SPEECHMATICS_STT_LANGUAGE: str = "en"
    # Silence (seconds) before Speechmatics' server-side EndOfUtterance fires.
    # Range 0-2s per Speechmatics docs; must stay below SPEECHMATICS_MAX_DELAY.
    SPEECHMATICS_EOT_SILENCE_TRIGGER_SEC: float = 0.5
    # Max seconds of latency Speechmatics may take to finalize a transcript segment.
    SPEECHMATICS_MAX_DELAY_SEC: float = 2.0

    # ElevenLabs Scribe v2 Realtime STT — reuses ELEVENLABS_API_KEY (TTS, below)
    # rather than a separate key; same ElevenLabs account for both.
    ELEVENLABS_SCRIBE_MODEL: str = "scribe_v2_realtime"
    ELEVENLABS_SCRIBE_LANGUAGE: str = "en"
    # Native server-side VAD/commit-strategy tuning (commit_strategy="vad").
    # Ranges per ElevenLabs docs: silence_threshold 0.3-3.0s, vad_threshold 0.1-0.9,
    # min_speech/min_silence_duration_ms 50-2000ms.
    ELEVENLABS_SCRIBE_VAD_SILENCE_THRESHOLD_SECS: float = 1.5
    ELEVENLABS_SCRIBE_VAD_THRESHOLD: float = 0.4
    ELEVENLABS_SCRIBE_MIN_SPEECH_DURATION_MS: int = 100
    ELEVENLABS_SCRIBE_MIN_SILENCE_DURATION_MS: int = 100

    # xAI Grok Speech-to-Text (wss://api.x.ai/v1/stt) — dedicated key, no
    # existing xAI credential to reuse.
    XAI_API_KEY: str = ""
    XAI_STT_LANGUAGE: str = "en"
    # Silence (ms) before a turn-boundary check fires. Low by design (xAI
    # default is 10ms) -- Smart Turn's ML judgment, not raw silence duration,
    # is what actually governs speech_final; see below.
    XAI_STT_ENDPOINTING_MS: int = 10
    # Native server-side turn detection: Smart Turn ML model demotes false-
    # positive silence boundaries (mid-sentence pauses, dictating numbers) to
    # chunk-final instead of firing speech_final. Threshold range 0.0-1.0;
    # 0.5 = "balanced" per xAI's own docs.
    XAI_STT_SMART_TURN_THRESHOLD: float = 0.5
    # Safety net: force speech_final after this many ms of silence regardless
    # of Smart Turn confidence, so a session can't hang indefinitely. Range 1-5000ms.
    XAI_STT_SMART_TURN_TIMEOUT_MS: int = 3000

    # AssemblyAI Universal-Streaming STT (wss://streaming.assemblyai.com/v3/ws)
    # — dedicated key, no existing AssemblyAI credential to reuse.
    ASSEMBLYAI_API_KEY: str = ""
    # No `language` query param exists on this API -- language is baked into
    # the speech_model choice (english vs multilingual variants). Kept only
    # for interface parity/logging, same rationale as xAI's unused `model`.
    ASSEMBLYAI_STT_LANGUAGE: str = "en"
    # Only universal-streaming-multilingual is seeded in the sttmodel catalog
    # (product decision -- AssemblyAI also offers universal-3-5-pro and
    # universal-streaming-english, deliberately not offered here).
    ASSEMBLYAI_STT_SPEECH_MODEL: str = "universal-streaming-multilingual"
    # Silence (ms) before AssemblyAI's native end-of-turn detection considers
    # a turn boundary. Range 50-10000ms per docs; no vendor-documented default
    # -- 400ms chosen empirically as a conservative responsiveness/false-positive
    # tradeoff, not a measured value. Revisit if end-of-turn detection feels
    # too eager or too laggy in production.
    ASSEMBLYAI_STT_MIN_TURN_SILENCE_MS: int = 400
    # Vendor default per docs; same 50-10000ms range as min_turn_silence.
    ASSEMBLYAI_STT_MAX_TURN_SILENCE_MS: int = 1536
    # Vendor default per docs. Range 0.0-1.0. Applies to the Universal
    # English/Multilingual model family (this deployment's only model).
    ASSEMBLYAI_STT_END_OF_TURN_CONFIDENCE_THRESHOLD: float = 0.4
    # Vendor default for the Universal-family models per docs. Range 0.0-1.0.
    ASSEMBLYAI_STT_VAD_THRESHOLD: float = 0.4
    # Whether finalized Turn transcripts are punctuation/cased-formatted.
    ASSEMBLYAI_STT_FORMAT_TURNS: bool = True

    # Deepgram Speech-to-Text (replaces Google STT for streaming + batch)
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_STT_LANGUAGE: str = (
        "en"  # Deepgram listen param; override in .env if needed
    )
    # Silence (ms) before Deepgram marks speech_final. 300ms splits spelling/email pauses;
    # ~900ms matches typical telephony spelling tolerance (Vapi-style longer listen window).
    DEEPGRAM_STT_ENDPOINTING_MS: int = 350
    # After the agent asks for email, bidirectional stream may reopen STT once with this value.
    DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED: int = 500
    # Word-timing-based fallback for end-of-turn detection (Deepgram's UtteranceEnd event),
    # robust to phone-line noise that can prevent silence-based endpointing/speech_final
    # from firing. Deepgram requires >=1000ms; only takes effect if speech_final never
    # arrives for the current utterance -- never fires early, never double-finalizes.
    DEEPGRAM_STT_UTTERANCE_END_MS: int = 1000
    # Telecom-oriented silence window for spelling/email (when mode is extended or email-recreate runs).
    # Ignored unless VOICE_STT_ENDPOINTING_MODE == "extended" or email flow bumps endpointing.
    # One-time Deepgram reconnect with extended endpointing when agent transcript matches email ask.
    VOICE_STT_ENDPOINTING_EMAIL_PROMPT_RECREATES_STT: bool = True
    # Initial Deepgram endpointing profile for the first STT session:
    #   normal     → DEEPGRAM_STT_ENDPOINTING_MS
    #   extended   → max(base, DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED)
    #   aggressive → faster finals (lower ms, clamped) for snappier turns
    VOICE_STT_ENDPOINTING_MODE: str = "aggressive"
    # Secondary dedup in STT pipeline: normalized text, same window idea as handler (seconds).
    VOICE_STT_FINAL_NORMALIZED_DEDUP_SEC: float = 6.0
    STT_SAMPLE_RATE: int = (
        8000  # provider-neutral STT sample rate (Twilio MULAW default)
    )
    # Multi-provider STT silence threshold: ms of no audio before treating utterance as done.
    # Applies to Google STT path; Deepgram uses its own endpointing above.
    SILENCE_THRESHOLD_MS: int = 1500

    # Google Cloud Text-to-Speech (TTS) endpoint/voice overrides
    # Docs: https://cloud.google.com/text-to-speech/docs/endpoints
    CLOUD_TTS_ENDPOINT: str = ""  # e.g. https://us-texttospeech.googleapis.com
    GOOGLE_TTS_VOICE_NAME: str = (
        ""  # Optional exact voice name override (e.g. en-US-Chirp3-HD-Achernar)
    )

    # Voice Conversation Settings - VAPI-STYLE REAL-TIME STREAMING
    USE_GATHER_APPROACH: bool = False  # Using real-time bidirectional streaming
    USE_BIDIRECTIONAL_STREAMING: bool = (
        True  # ✅ ENABLED - Real-time STT + TTS with Adaptive VAD
    )
    USE_WEBSOCKET_TTS: bool = True  # ✅ ENABLED - 20ms chunk streaming (MULAW 8kHz)

    # Voice streaming tunables (phase 6 centralization)
    VOICE_STT_INTERIM_INTERVAL_MS: int = 30
    # Deepgram fires many more partials than classic Google STT. Running LLM on every
    # interim → double replies + TTS "breaks." Default: final STT only (one reply per
    # utterance). Set True for lower first-token latency at the cost of stability.
    VOICE_ENABLE_INTERIM_LLM: bool = True
    # When interim LLM is enabled, these gates reduce junk triggers ("I'm", "Do you", …)
    VOICE_MIN_INTERIM_WORDS: int = 2
    VOICE_MIN_INTERIM_CONFIDENCE: float = 0.14
    # Inbound MULAW → linear RMS: frames above this count as "speech" for user-pickup detection.
    # Lower = softer voices register sooner (e.g. 60–70); higher = stricter, needs louder speech
    # (legacy default was 100). Too low picks up line noise.
    VOICE_MIN_AUDIO_RMS_FOR_PICKUP: int = 20
    # Drop Deepgram final transcripts below this (0.0–1.0). Default slightly below 0.30 so
    # quiet/soft speech is not rejected as often; too low adds garbage.
    VOICE_STT_MIN_FINAL_CONFIDENCE: float = 0.15
    # Optional adaptive fallback: accept lower-confidence finals when they still look like
    # real speech (multi-word, alpha content). Helps callers with soft volume mid-call.
    VOICE_STT_ENABLE_SOFT_FINAL_FALLBACK: bool = True
    VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE: float = 0.12
    VOICE_STT_SOFT_MIN_WORDS: int = 2
    # Barge-in (user talks over agent): require at least this many STT words while TTS plays.
    # Default 2 filters phantom 1-word Deepgram hits ("uh", noise artefacts) on silence.
    VOICE_BARGE_IN_MIN_WORDS: int = 2
    # Min STT confidence when word count >= VOICE_BARGE_IN_MIN_WORDS.
    VOICE_BARGE_IN_MIN_CONFIDENCE: float = 0.26
    # Only used when VOICE_BARGE_IN_MIN_WORDS == 1 (one-word interrupts like "stop").
    VOICE_BARGE_IN_MIN_CONFIDENCE_1W: float = 0.52
    VOICE_HISTORY_MAX_MESSAGES: int = 50
    VOICE_TTS_FLUSH_MIN_WORDS: int = 4
    # Smaller max keeps per-chunk synthesis short (~300ms for ElevenLabs) so the
    # playback gate chain never backs up — eliminates "arr arr" / mid-chunk silence.
    VOICE_TTS_FLUSH_MAX_WORDS: int = 8
    # If no sentence boundary yet, flush after this many seconds (once min words met).
    VOICE_TTS_TIME_FLUSH_SEC: float = 0.25
    # Browser "Share Demo Link" calls (app.voice.livekit_browser_call_handler) have no
    # Twilio phone number, so recording_config_service.get_recording_enabled_for_call's
    # NumberConfiguration lookup can never resolve for them (assistant_phone_number is
    # the literal string "web_agent"). There is currently no per-tenant/agent/call-flow
    # recording toggle independent of a phone number — this is a process-wide interim
    # default until that's added via a proper schema change (flag for db-migration).
    VOICE_BROWSER_DEMO_RECORDING_ENABLED: bool = True
    # Keep a short (but non-zero) guard after pickup so ringback artifacts are skipped
    # without delaying real user speech by multiple seconds.
    VOICE_POST_PICKUP_STT_GRACE_SEC: float = 0.35
    # Inbound auto-greeting delay after user pickup (seconds).
    # Keep small but non-zero so call audio stabilizes before greeting starts.
    VOICE_INBOUND_GREETING_DELAY_SEC: float = 0.5
    # Pickup detector window and threshold (RMS frames over threshold) before STT starts.
    VOICE_PICKUP_SAMPLE_WINDOW: int = 6
    VOICE_PICKUP_MIN_NON_SILENT_FRAMES: int = 4
    # Allow RAG prefetch to start earlier than interim-LLM gates.
    VOICE_RAG_PREFETCH_MIN_WORDS: int = 1
    VOICE_RAG_PREFETCH_MIN_CONFIDENCE: float = 0.05
    # TTS speed/volume bounds — shared by API schema (TtsSettingsJsonSchema) and
    # runtime clamping (resolve_tts_runtime). Tune per deploy without code changes.
    TTS_SPEED_MIN: float = 0.25
    TTS_SPEED_MAX: float = 2.0
    TTS_VOLUME_MIN: float = 0.0
    TTS_VOLUME_MAX: float = 2.0
    # Start TTS streaming sooner for short first chunks.
    VOICE_TTS_STREAM_MIN_WORDS: int = 2
    # Twilio jitter buffer priming frames (20ms each) for low-latency voice output.
    VOICE_TTS_PRIME_FRAMES: int = 1
    # Phase 4C-2/4C-3: extra 0xFF silence frames (20ms each) appended after a
    # non-final TTS chunk that ends at a real sentence boundary, so
    # multi-sentence responses get a small human-like breath between
    # sentences. Shared by both transports (app.voice.tts_stream_mixin,
    # app.voice.livekit_browser_call_handler) — see HumanizationDecision.pacing.
    # Set to 0 to fully disable (identical to pre-4C-2 behavior).
    # Phase 4C-3: enabled at 3 frames (60ms) — code-level correctness only;
    # this value has NOT been validated against real call audio (no live-call
    # testing available in this environment). Revisit after real listening.
    VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES: int = 3
    VOICE_QUICK_ACK_MIN_WORDS: int = 5
    # Quick-ack: fires on slow-path queries only (fastpath is excluded at call site).
    # In V2 TtsPipeline, LLM chunk synthesis runs in parallel with quick-ack playback
    # so the "shutter then silence" gap only occurs when LLM TTFT > quick-ack duration.
    # 0.35 = fires roughly every third slow-path turn; set 0.0 to disable entirely.
    VOICE_QUICK_ACK_PROBABILITY: float = 0.35
    # Fast-path for very short/simple turns to reduce first-token latency:
    # skip heavy RAG/KB context for obvious non-booking smalltalk.
    VOICE_ENABLE_LATENCY_FASTPATH: bool = True
    VOICE_FASTPATH_MAX_WORDS: int = 7
    # Centralized, provider-agnostic humanization decision layer
    # (app.voice.humanization_engine). Phase 3A: decision-only, not yet wired
    # into any call flow. When False, analyze_response() always returns a
    # neutral "no humanization" decision.
    VOICE_ENABLE_HUMANIZATION_ENGINE: bool = True

    # Phase 6-3: route ElevenLabs TTS chunks for a turn through a persistent
    # WebSocket `stream-input` session (app.services.elevenlabs_ws_session)
    # instead of one independent HTTP request per flushed chunk. Enabled by
    # default (True) for continuous, turn-level ElevenLabs synthesis —
    # improves cross-sentence prosody/voice continuity over the old
    # one-HTTP-request-per-chunk path. Gated behind this flag + the
    # elevenlabs-only supports_streaming_session capability
    # (app.voice.tts_provider_capabilities) so Google/Rime and the existing
    # ElevenLabs HTTP path are completely unaffected regardless of this
    # setting. See app.voice.tts_pipeline.TtsPipeline._try_elevenlabs_ws_route.
    VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED: bool = True

    # Vapi-style intelligent contact recovery (additive — never downgrades intake confidence).
    # 1) Deterministic email STT-artifact cleanup: strip commas/spaces inside an email span
    #    ("ali.sa,ee,b@gmail.com" -> "ali.saeeb@gmail.com") before strict validation.
    EMAIL_STT_CLEANUP_ENABLED: bool = True
    # 2) Natural confirmation path: when the agent repeats a name ("just to confirm, your
    #    name is Alex Carter") and the caller affirms ("yes" / "correct"), mark name_confident
    #    without requiring letter-by-letter spelling.
    VOICE_NATURAL_NAME_CONFIRMATION: bool = True
    # 3) Post-call LLM recovery for contact (name/email) when strict intake gate would fail.
    #    Only invoked AFTER the call ends, behind a flag, and only if OPENAI_API_KEY is set.
    POST_CALL_LLM_CONTACT_RECOVERY: bool = True
    POST_CALL_LLM_CONTACT_RECOVERY_MODEL: str = "gpt-4o-mini"
    # If name is already confident but email is missing, still run post-call contact LLM once.
    POST_CALL_LLM_EMAIL_RECOVERY_WHEN_NAME_OK: bool = True
    # If the model returns a valid normalized email but email_confident=false, trust it when
    # the same address is clearly present in the transcript (reduces false negatives).
    POST_CALL_LLM_EMAIL_ANCHOR_TRUST: bool = True

    # Stripe settings
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""  # Billing/subscription webhook secret
    STRIPE_INCALL_WEBHOOK_SECRET: str = (
        ""  # In-call payment webhook secret (separate endpoint)
    )
    STRIPE_PRICE_ID_FREE: str = ""
    STRIPE_PRICE_ID_PRO: str = ""

    # In-call payment page URL — returned to the agent as the caller-facing payment link.
    # Format: "{PAYMENT_PAGE_BASE_URL}/pay/{payment_intent_id}?client_secret={client_secret}"
    PAYMENT_PAGE_BASE_URL: str = "https://pay.yourdomain.com"

    # Billing settings
    FREE_PLAN_AGENT_LIMIT: int = 2
    FREE_PLAN_MONTHLY_CALLS: int = 100
    PRO_PLAN_AGENT_LIMIT: int = 50
    PRO_PLAN_MONTHLY_CALLS: int = 10000

    # Rate limiting settings
    REDIS_URL: str = "redis://localhost:6379"
    RATE_LIMIT_ENABLED: bool = True

    # Login rate limiting — per-IP, stricter than global API limit (enforce_login_rate_limit)
    LOGIN_RATE_LIMIT: int = 10
    LOGIN_RATE_WINDOW: int = 60  # seconds

    # Webhook rate limiting (requests per minute)
    WEBHOOK_RATE_LIMIT: int = 100
    WEBHOOK_RATE_WINDOW: int = 60  # seconds

    # General API rate limiting — global sliding-window middleware
    API_RATE_LIMIT: int = 60  # requests per window per identity
    API_RATE_WINDOW: int = 60  # seconds

    # Public Web SDK token endpoint — per-IP, no auth required (BE1 ticket)
    PUBLIC_TOKEN_RATE_LIMIT: int = 20
    PUBLIC_TOKEN_RATE_WINDOW: int = 60  # seconds

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # Vector / RAG configuration
    # Optional generic vector DB URL (used as a fallback host for Pinecone).
    VECTOR_DB_URL: str | None = None
    # Default embedding dimension for text-embedding models
    VECTOR_DIMENSION: int = 1536  # e.g. OpenAI text-embedding-3-small

    # Pinecone (preferred vector store for RAG)
    PINECONE_API_KEY: str = ""
    # Optional: direct index host, if you copy it from Pinecone console.
    # Example: "your-index-host.svc.us-east-1-aws.pinecone.io"
    PINECONE_INDEX_HOST: str | None = None
    # Optional: index name; if host is not provided, we can resolve host from this.
    PINECONE_INDEX_NAME: str | None = None

    # Twilio Edge hint (for logging/observability; set actual edge in Twilio Console)
    TWILIO_EDGE: str | None = "umatilla"  # e.g., "ashburn", "singapore", "dublin"
    # App deployment region hint for latency diagnostics.
    SERVER_REGION: str = "us-west-2"  # Oregon
    # If enabled, log warnings when Twilio edge does not match expected regional edge.
    VOICE_REGION_ALIGNMENT_STRICT: bool = True

    # RAG behavior tuning (voice-first defaults)
    # Master switch for latency A/B tests and emergency fail-open behavior.
    # Set RAG_ENABLED=false in .env to skip retrieval entirely.
    RAG_ENABLED: bool = True
    # These defaults are intentionally conservative to avoid prompt bloat/latency.
    # Primary embedding model (OpenAI by default).
    # Make embedding model configurable because some OpenAI projects do not
    # have access to all embedding models.
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Fallback embedding model/provider used when the primary embedding call fails.
    RAG_FALLBACK_EMBEDDING_PROVIDER: str = "gemini"
    RAG_FALLBACK_EMBEDDING_MODEL: str = "gemini-embedding-002"
    RAG_TOP_K: int = 5
    # Similarity floor (0-1, higher = more similar) shared by both the Twilio path
    # (rag_context.py) and the LiveKit path (kb_retrieval_service._query_single_kb,
    # which returns 1 - cosine_distance as `score` — same direction/semantics).
    # Lowered from 0.4 -> 0.2 based on production telemetry showing relevant KB
    # matches were being filtered out at 0.4. 0.2 is a deliberately low floor --
    # verify with recall/precision metrics before lowering further. Irrelevant
    # results at this threshold risk degrading LLM answer quality (hallucination
    # risk) more than returning no KB context would.
    RAG_SCORE_THRESHOLD: float = 0.2
    # Hard cap for the size of the rendered context block injected into prompts.
    # This is character-based (approx). For token-accurate sizing, you would need a tokenizer.
    RAG_MAX_CONTEXT_CHARS: int = 6000

    # Voice latency guardrails
    # If Pinecone or embedding generation is slow, we must fail fast and
    # return an empty knowledge context to avoid breaking the voice UX.
    RAG_RETRIEVAL_TIMEOUT_SEC: float = 0.45
    # LiveKit/browser-call path (kb_retrieval_service via ConversationOrchestrator)
    # budget — distinct from RAG_RETRIEVAL_TIMEOUT_SEC (Twilio's bidirectional_stream
    # path) so tuning one never silently changes the other's latency contract.
    # Originally set to 0.7s from local/CI samples of 271-450ms. Production
    # diagnostics on the live EC2 deployment (2026-08-16) showed this environment's
    # actual baseline round-trip + inference time to OpenAI's /v1/embeddings
    # endpoint is itself 270-510ms with real variance (confirmed via raw httpx
    # calls bypassing app code entirely), leaving ~0ms margin at 0.7s and causing
    # frequent false-positive timeouts — KB context dropped even when a result was
    # about to arrive. Raised to 1.2s to give this real-world baseline enough
    # headroom while staying under the ~1.5-2s range where voice-call silence
    # becomes clearly perceptible to callers. Revisit if p95 KB latency data
    # (once available) suggests a tighter or looser value.
    RAG_KB_RETRIEVAL_TIMEOUT_SEC: float = 1.2
    # Slow-path budget: cap cumulative waits for RAG/KB on a turn.
    VOICE_SLOWPATH_BUDGET_SEC: float = 0.55
    # How long we wait for an in-flight RAG prefetch before failing open.
    VOICE_RAG_PREFETCH_AWAIT_SEC: float = 0.18
    # When KB cache isn't ready on early turns, skip live DB fetch to protect latency.
    VOICE_SKIP_LIVE_KB_FETCH_ON_COLD_START: bool = True
    # Prevent extremely large STT transcripts from being embedded.
    RAG_MAX_QUERY_CHARS: int = 3000

    # Optional retrieval reranking (lexical overlap).
    # Keep disabled by default to avoid changing retrieval semantics unexpectedly.
    RAG_ENABLE_RERANK: bool = False
    # Weight for Pinecone vector similarity vs lexical overlap.
    # Higher means more trust in vector similarity.
    RAG_RERANK_VECTOR_WEIGHT: float = 0.8

    # Voice latency SLO thresholds (seconds) for observability.
    VOICE_SLO_ENABLED: bool = True
    VOICE_SLO_STT_FINAL_TO_GEN_START_SEC: float = 0.35
    VOICE_SLO_GEN_START_TO_LLM_FIRST_TOKEN_SEC: float = 0.90
    VOICE_SLO_GEN_START_TO_FIRST_TTS_QUEUE_SEC: float = 1.40
    VOICE_SLO_GEN_START_TO_NOW_WARN_SEC: float = 2.00

    # Trello — platform-managed inbound call boards (optional)
    TRELLO_PLATFORM_API_KEY: str = ""
    TRELLO_PLATFORM_API_TOKEN: str = ""

    # Monday.com Configuration
    MONDAY_API_KEY: str = ""  # Monday.com Personal API Token
    MONDAY_BOARD_ID: str = ""  # Monday.com Board ID for scheduled calls
    MONDAY_WORKSPACE_ID: str | None = (
        None  # Optional workspace to create tenant boards in
    )

    # LiveKit — self-hosted real-time audio transport (GKE internal, port 7880)
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_TOKEN_TTL: int = 3600  # seconds; 1 hour
    LIVEKIT_ROOM_EMPTY_TIMEOUT: int = 30  # seconds before auto-close when empty
    LIVEKIT_MAX_PARTICIPANTS: int = 2  # enforced at SDK CreateRoomRequest level
    LIVEKIT_ENABLED: bool = True

    # GCS call recordings — Sprint 4
    # Bucket must have a lifecycle rule: delete recordings/ prefix objects after 90 days.
    # Infra: GCS lifecycle rule deletes recordings/ prefix objects after 90 days (set in bucket policy, not here).
    GCS_RECORDINGS_BUCKET: str = ""
    GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS: int = 3600
    GCS_RECORDINGS_PREFIX: str = "recordings"

    # GCS knowledge-base file storage — Sprint 5
    GCS_KB_BUCKET: str = ""
    GCS_KB_PREFIX: str = "kb-files"

    # AWS S3 storage (GCS → S3 migration)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION_NAME: str = "us-east-1"
    S3_RECORDINGS_BUCKET: str = ""
    S3_KB_BUCKET: str = ""

    # HIPAA — Google Cloud DLP + CMEK
    # GCP_PROJECT_ID is declared above (line ~245); no second declaration here.

    # Outbound call concurrency — max simultaneous outbound calls per workspace.
    # Counts outbound sessions with status IN (initiated, ringing, connected, in-progress).
    # Increase at the tenant level by changing this value (no per-tenant override yet).
    OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE: int = 10

    # Batch calls — max records picked per ARQ job tick (SKIP LOCKED window size).
    MAX_BATCH_CONCURRENCY: int = 5

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

    # ------------------------------------------------------------------
    # On-premise / BYO (Bring Your Own) provider configuration.
    # Lets a self-hosted Docker Compose deployment pick its LLM/TTS/telephony
    # vendor via env vars instead of per-tenant DB rows. Values are deployment-
    # wide defaults; existing per-tenant DB config (if any) still wins.
    # See docs/on-premise/README.md for the full reference.
    # ------------------------------------------------------------------

    # BYO LLM — 'google' (Gemini/Vertex) | 'openai' | 'azure_openai'
    LLM_PROVIDER: str = "google"
    # Override the OpenAI SDK base URL to point at Azure OpenAI or any
    # Ollama/vLLM/LiteLLM OpenAI-compatible endpoint. Leave blank for api.openai.com.
    OPENAI_BASE_URL: str = ""
    # Required when LLM_PROVIDER=azure_openai (Azure OpenAI's REST API version, e.g. "2024-10-21").
    OPENAI_API_VERSION: str = ""

    # BYO TTS — 'rime' | 'elevenlabs'
    TTS_PROVIDER: str = "elevenlabs"
    # Generic TTS key — auto-applied to RIME_API_KEY/ELEVENLABS_API_KEY below
    # (whichever TTS_PROVIDER selects) when that provider-specific var is unset.
    TTS_API_KEY: str = ""

    # BYO telephony — 'twilio' | 'sip'
    TELEPHONY_PROVIDER: str = "twilio"
    # Generic SIP trunk credentials, used when TELEPHONY_PROVIDER=sip.
    SIP_TRUNK_URI: str = ""
    SIP_USERNAME: str = ""
    SIP_PASSWORD: str = ""

    # OpenTelemetry distributed tracing
    OTEL_TRACING_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SERVICE_NAME: str = "tgs-agent-be"

    # ------------------------------------------------------------------
    # Domain sub-model views (populated by _assemble_sub_models below).
    # Use settings.db.url, settings.auth.secret_key, etc. in new code;
    # flat names above remain available for all existing callers.
    # ------------------------------------------------------------------
    db: DbSettings | None = None
    auth: AuthSettings | None = None
    twilio: TwilioSettings | None = None
    llm: LlmSettings | None = None
    tts: TtsSettings | None = None
    crm: CrmSettings | None = None
    server: ServerSettings | None = None
    redis: RedisSettings | None = None

    @model_validator(mode="after")
    def _apply_byo_tts_api_key(self) -> "Settings":
        """Mirror the generic TTS_API_KEY into the provider-specific key that
        TTS_PROVIDER selects, when that specific key was not already set."""
        if self.TTS_API_KEY:
            if self.TTS_PROVIDER == "rime" and not self.RIME_API_KEY:
                self.RIME_API_KEY = self.TTS_API_KEY
            elif self.TTS_PROVIDER == "elevenlabs" and not self.ELEVENLABS_API_KEY:
                self.ELEVENLABS_API_KEY = self.TTS_API_KEY
        return self

    @model_validator(mode="after")
    def _assemble_sub_models(self) -> "Settings":
        """Build grouped sub-model views from the flat env-var fields above.

        Runs after _apply_byo_tts_api_key so TTS key mirroring is already done
        before TtsSettings captures elevenlabs_api_key / rime_api_key.
        """
        self.db = DbSettings(
            url=self.DATABASE_URL,
            pool_size=self.DATABASE_POOL_SIZE,
            max_overflow=self.DATABASE_MAX_OVERFLOW,
            pool_timeout=self.DATABASE_POOL_TIMEOUT,
            statement_timeout=self.DATABASE_STATEMENT_TIMEOUT,
        )
        self.auth = AuthSettings(
            secret_key=self.SECRET_KEY,
            algorithm=self.ALGORITHM,
            access_token_expire_minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES,
            refresh_token_expire_days=self.REFRESH_TOKEN_EXPIRE_DAYS,
            password_reset_token_expire_minutes=self.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES,
            webhook_secret_encryption_key=self.WEBHOOK_SECRET_ENCRYPTION_KEY,
            sso_encryption_key=self.SSO_ENCRYPTION_KEY,
            google_client_id=self.GOOGLE_CLIENT_ID,
            google_client_secret=self.GOOGLE_CLIENT_SECRET,
        )
        self.twilio = TwilioSettings(
            account_sid=self.TWILIO_ACCOUNT_SID,
            auth_token=self.TWILIO_AUTH_TOKEN,
            phone_number=self.TWILIO_PHONE_NUMBER,
            allow_unauthenticated_webhooks=self.ALLOW_UNAUTHENTICATED_WEBHOOKS,
            test_account_sid=self.TWILIO_TEST_ACCOUNT_SID,
            test_auth_token=self.TWILIO_TEST_AUTH_TOKEN,
            edge=self.TWILIO_EDGE,
        )
        self.llm = LlmSettings(
            openai_api_key=self.OPENAI_API_KEY,
            openai_base_url=self.OPENAI_BASE_URL,
            openai_api_version=self.OPENAI_API_VERSION,
            openai_realtime_transcription_model=self.OPENAI_REALTIME_TRANSCRIPTION_MODEL,
            gemini_api_key=self.GEMINI_API_KEY,
            google_application_credentials=self.GOOGLE_APPLICATION_CREDENTIALS,
            google_cloud_project_id=self.GOOGLE_CLOUD_PROJECT_ID,
            vertex_ai_location=self.VERTEX_AI_LOCATION,
            provider=self.LLM_PROVIDER,
            default_model=self.DEFAULT_LLM_MODEL,
            default_provider=self.DEFAULT_LLM_PROVIDER,
            history_max_turns=self.VOICE_LLM_HISTORY_MAX_TURNS,
            default_temperature=self.VOICE_LLM_DEFAULT_TEMPERATURE,
            fallback_message=self.VOICE_LLM_FALLBACK_MESSAGE,
            deepgram_api_key=self.DEEPGRAM_API_KEY,
            deepgram_stt_model=self.DEEPGRAM_STT_MODEL,
            deepgram_stt_language=self.DEEPGRAM_STT_LANGUAGE,
            deepgram_stt_endpointing_ms=self.DEEPGRAM_STT_ENDPOINTING_MS,
            deepgram_stt_endpointing_ms_extended=self.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED,
            deepgram_stt_utterance_end_ms=self.DEEPGRAM_STT_UTTERANCE_END_MS,
            google_stt_language_code=self.GOOGLE_STT_LANGUAGE_CODE,
            google_stt_sample_rate=self.GOOGLE_STT_SAMPLE_RATE,
            google_stt_encoding=self.GOOGLE_STT_ENCODING,
        )
        self.tts = TtsSettings(
            provider=self.TTS_PROVIDER,
            api_key=self.TTS_API_KEY,
            rime_api_key=self.RIME_API_KEY,
            hume_api_key=self.HUME_API_KEY,
            hume_sample_rate_hz=self.HUME_TTS_SAMPLE_RATE_HZ,
            elevenlabs_api_key=self.ELEVENLABS_API_KEY,
            elevenlabs_encryption_key=self.ELEVENLABS_ENCRYPTION_KEY,
            enable_audio_tags=self.ENABLE_ELEVENLABS_AUDIO_TAGS,
            cloud_endpoint=self.CLOUD_TTS_ENDPOINT,
            google_voice_name=self.GOOGLE_TTS_VOICE_NAME,
            speed_min=self.TTS_SPEED_MIN,
            speed_max=self.TTS_SPEED_MAX,
            volume_min=self.TTS_VOLUME_MIN,
            volume_max=self.TTS_VOLUME_MAX,
        )
        self.crm = CrmSettings(
            hubspot_client_id=self.HUBSPOT_CLIENT_ID,
            hubspot_client_secret=self.HUBSPOT_CLIENT_SECRET,
            hubspot_redirect_uri=self.HUBSPOT_REDIRECT_URI,
            hubspot_token_encryption_key=self.HUBSPOT_TOKEN_ENCRYPTION_KEY,
            salesforce_client_id=self.SALESFORCE_CLIENT_ID,
            salesforce_client_secret=self.SALESFORCE_CLIENT_SECRET,
            salesforce_redirect_uri=self.SALESFORCE_REDIRECT_URI,
            salesforce_token_encryption_key=self.SALESFORCE_TOKEN_ENCRYPTION_KEY,
            ghl_client_id=self.GHL_CLIENT_ID,
            ghl_client_secret=self.GHL_CLIENT_SECRET,
            ghl_redirect_uri=self.GHL_REDIRECT_URI,
            ghl_token_encryption_key=self.GHL_TOKEN_ENCRYPTION_KEY,
            slack_client_id=self.SLACK_CLIENT_ID,
            slack_client_secret=self.SLACK_CLIENT_SECRET,
            slack_redirect_uri=self.SLACK_REDIRECT_URI,
            slack_token_encryption_key=self.SLACK_TOKEN_ENCRYPTION_KEY,
            calendly_client_id=self.CALENDLY_CLIENT_ID,
            calendly_client_secret=self.CALENDLY_CLIENT_SECRET,
            calendly_redirect_uri=self.CALENDLY_REDIRECT_URI,
            calendly_token_encryption_key=self.CALENDLY_TOKEN_ENCRYPTION_KEY,
            monday_api_key=self.MONDAY_API_KEY,
            monday_board_id=self.MONDAY_BOARD_ID,
            monday_workspace_id=self.MONDAY_WORKSPACE_ID,
            trello_api_key=self.TRELLO_PLATFORM_API_KEY,
            trello_api_token=self.TRELLO_PLATFORM_API_TOKEN,
            aws_ses_sender_email=self.AWS_SES_SENDER_EMAIL,
            stripe_publishable_key=self.STRIPE_PUBLISHABLE_KEY,
            stripe_secret_key=self.STRIPE_SECRET_KEY,
            stripe_webhook_secret=self.STRIPE_WEBHOOK_SECRET,
            stripe_incall_webhook_secret=self.STRIPE_INCALL_WEBHOOK_SECRET,
            stripe_price_id_free=self.STRIPE_PRICE_ID_FREE,
            stripe_price_id_pro=self.STRIPE_PRICE_ID_PRO,
            payment_page_base_url=self.PAYMENT_PAGE_BASE_URL,
            free_plan_agent_limit=self.FREE_PLAN_AGENT_LIMIT,
            free_plan_monthly_calls=self.FREE_PLAN_MONTHLY_CALLS,
            pro_plan_agent_limit=self.PRO_PLAN_AGENT_LIMIT,
            pro_plan_monthly_calls=self.PRO_PLAN_MONTHLY_CALLS,
        )
        self.server = ServerSettings(
            host=self.HOST,
            port=self.PORT,
            debug=self.DEBUG,
            app_version=self.APP_VERSION,
            environment=self.ENVIRONMENT,
            admin_role=self.ADMIN_ROLE,
            allowed_origins=self.ALLOWED_ORIGINS,
            api_docs_enabled=self.API_DOCS_ENABLED,
            api_docs_username=self.API_DOCS_USERNAME,
            api_docs_password=self.API_DOCS_PASSWORD,
            webhook_base_url=self.WEBHOOK_BASE_URL,
            n8n_webhook_url=self.N8N_WEBHOOK_URL,
            n8n_webhook_secret=self.N8N_WEBHOOK_SECRET,
            frontend_url=self.FRONTEND_URL,
            gcp_project_id=self.GCP_PROJECT_ID,
            server_region=self.SERVER_REGION,
            livekit_url=self.LIVEKIT_URL,
            livekit_api_key=self.LIVEKIT_API_KEY,
            livekit_api_secret=self.LIVEKIT_API_SECRET,
            livekit_token_ttl=self.LIVEKIT_TOKEN_TTL,
            livekit_room_empty_timeout=self.LIVEKIT_ROOM_EMPTY_TIMEOUT,
            livekit_max_participants=self.LIVEKIT_MAX_PARTICIPANTS,
            livekit_enabled=self.LIVEKIT_ENABLED,
            gcs_recordings_bucket=self.GCS_RECORDINGS_BUCKET,
            gcs_recordings_signed_url_expiry_seconds=self.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
            gcs_recordings_prefix=self.GCS_RECORDINGS_PREFIX,
            gcs_kb_bucket=self.GCS_KB_BUCKET,
            gcs_kb_prefix=self.GCS_KB_PREFIX,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            aws_region_name=self.AWS_REGION_NAME,
            s3_recordings_bucket=self.S3_RECORDINGS_BUCKET,
            s3_kb_bucket=self.S3_KB_BUCKET,
            outbound_max_concurrent_per_workspace=self.OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE,
            max_batch_concurrency=self.MAX_BATCH_CONCURRENCY,
        )
        self.redis = RedisSettings(
            url=self.REDIS_URL,
            rate_limit_enabled=self.RATE_LIMIT_ENABLED,
            login_rate_limit=self.LOGIN_RATE_LIMIT,
            login_rate_window=self.LOGIN_RATE_WINDOW,
            webhook_rate_limit=self.WEBHOOK_RATE_LIMIT,
            webhook_rate_window=self.WEBHOOK_RATE_WINDOW,
            api_rate_limit=self.API_RATE_LIMIT,
            api_rate_window=self.API_RATE_WINDOW,
            public_token_rate_limit=self.PUBLIC_TOKEN_RATE_LIMIT,
            public_token_rate_window=self.PUBLIC_TOKEN_RATE_WINDOW,
        )
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
