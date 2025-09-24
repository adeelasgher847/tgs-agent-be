from sqlalchemy.orm import Session
from app.models.model import Model as ModelORM
from app.models.provider import Provider as ProviderORM
from app.schemas.agent import gemini_client

class ProviderIntegrationService:
    def create_remote_agent(self, db: Session, model_id, agent_name: str) -> str | None:
        if not model_id:
            return None
        model = db.query(ModelORM).filter(ModelORM.id == model_id, ModelORM.is_active == True).first()
        if not model:
            return None
        provider = db.query(ProviderORM).filter(ProviderORM.id == model.provider_id, ProviderORM.is_active == True).first()
        if not provider:
            return None

        provider_name = (provider.name or "").strip().lower()
        if provider_name in ("gemini", "google", "google-ai", "google ai", "gemini-1.5-flash"):
            return gemini_client.create_agent(agent_name)
        # Future: openai, elevenlabs, etc.
        return None

provider_integration_service = ProviderIntegrationService()