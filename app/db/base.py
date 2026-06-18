from app.db.base_class import Base
from app.models.user import User
from app.models.tenant import Tenant
from app.models.role import Role
from app.models.agent import Agent
from app.models.transfer_route import TransferRoute
from app.models.password_reset import PasswordResetToken
from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.transcript_message import TranscriptMessage
from app.models.phone_number import PhoneNumber, NumberConfiguration
from app.models.refresh_token import RefreshToken
from app.models.invite import Invite
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.usage_record import UsageRecord  # noqa: F401  (replaces legacy usage_records.py)
from app.models.provider import Provider
from app.models.product import Product
from app.models.model import Model
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.models.scheduled_call import ScheduledCall
from app.models.tenant_crm_config import CRMConfig

# Knowledge base / RAG (pgvector-backed)
from app.models.knowledge_base_document import KnowledgeBase, KnowledgeBaseDocument  # noqa: F401
from app.models.kb_file import KbFile  # noqa: F401
from app.models.knowledge_base_chunk import KbChunk, KnowledgeBaseChunk  # noqa: F401

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

# Call flows and versioning
from app.models.call_flow import CallFlow
from app.models.prompt_version import PromptVersion
from app.models.folder import Folder
from app.models.folder_flow import FolderFlow

# Recruiting / resumes
from app.models.resume import Resume
from app.models.resume_interview import ResumeInterview
from app.models.api_key import Apikey
from app.models.stripe_checkout_fulfillment import StripeCheckoutFulfillment

# STT catalog
from app.models.stt_provider import STTProvider
from app.models.stt_model import STTModel

# Batch outbound calls
from app.models.batch_job import BatchJob
from app.models.batch_call_record import BatchCallRecord

# Custom webhooks
from app.models.webhook import WebhookEndpoint, WebhookDelivery

# Branding, pricing, RBAC
from app.models.branding_configs import BrandingConfig  # noqa: F401
from app.models.pricing_configs import PricingConfig  # noqa: F401
from app.models.rbac_roles import RbacRole  # noqa: F401

# Smart Callback Scheduler
from app.models.callback_schedule import CallbackSchedule  # noqa: F401

# HIPAA audit trail
from app.models.audit_log import AuditLog  # noqa: F401

# GDPR data export
from app.models.data_export_job import DataExportJob  # noqa: F401

# Web SDK — public call-token domain whitelist
from app.models.allowed_domain import AllowedDomain  # noqa: F401
