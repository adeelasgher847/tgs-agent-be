from app.db.base_class import Base
from app.models.user import User
from app.models.tenant import Tenant
from app.models.role import Role
from app.models.agent import Agent
from app.models.password_reset import PasswordResetToken
from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.transcript_message import TranscriptMessage
from app.models.phone_number import PhoneNumber
from app.models.refresh_token import RefreshToken
from app.models.invite import Invite
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.usage_record import UsageRecord
from app.models.provider import Provider
from app.models.product import Product
from app.models.model import Model
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.models.scheduled_call import ScheduledCall
from app.models.tenant_crm_config import CRMConfig

# Knowledge base / RAG
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.knowledge_base_chunk import KnowledgeBaseChunk

# Calendar
from app.models.business_hours import BusinessHours
from app.models.blocked_slot import BlockedSlot
from app.models.appointment import Appointment
from app.models.slot_reservation import SlotReservation
from app.models.tenant_inbound_crm_config import TenantInboundCRMConfig
from app.models.call_log_crm_sync import CallLogCRMSync
from app.models.job_description import JobDescription

# Business knowledge base
from app.models.business_knowledge import BusinessKnowledge

# Recruiting / resumes
from app.models.resume import Resume
from app.models.resume_interview import ResumeInterview
