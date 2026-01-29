from datetime import datetime, timezone
import csv
import io
import httpx
import asyncio
import uuid
from typing import Optional, Tuple

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.agent import Agent
from app.models.scheduled_call import ScheduledCall
from app.models.tenant import Tenant
from app.schemas.scheduled_call import CSVUploadResponse
from app.services.monday_service import MondayService
from app.core.logger import logger


class ScheduledCallService:
    @staticmethod
    def _get_tenant(db: Session, tenant_id: uuid.UUID) -> Tenant:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return tenant

    @staticmethod
    def get_or_create_board_for_user(
        db: Session, 
        user_id: uuid.UUID, 
        tenant_id: uuid.UUID,
        crm_config_id: uuid.UUID
    ) -> Tuple[ScheduledCall, dict]:
        """
        Get or create CRM container (board/list/project) for a user.
        All tenants of this user will use the same container.
        Items are identified by tenant_id field/column.
        
        Args:
            db: Database session
            user_id: User ID
            tenant_id: Tenant ID (for validation)
            crm_config_id: CRM configuration ID to use
            
        Returns:
            Tuple of (ScheduledCall record, field_map dictionary)
        """
        from app.models.user import User
        from app.services.crm_config_service import CRMConfigService
        from app.services.crm_service_factory import CRMServiceFactory
        
        from app.services.billing_service import BillingService
        
        # Check if container already exists for this user
        board_record = db.query(ScheduledCall).filter(ScheduledCall.user_id == user_id).first()

        # Get CRM config (needed for both new and existing records)
        crm_config_service = CRMConfigService()
        crm_config = crm_config_service.get_crm_config_by_id(db, crm_config_id)
        if not crm_config:
            raise HTTPException(status_code=404, detail="CRM configuration not found")
            
        # Get CRM service (needed for both new and existing records)
        crm_service = CRMServiceFactory.get_service(crm_config)
        
        if not board_record:
            # Get user email for container name
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Create new container for this user (each user gets their own container)
            # Multiple tenants of the same user will share this container
            container_name = f"Scheduled Calls - {user.email}"
            additional_config = {}
            if crm_config.additional_config:
                import json
                additional_config = json.loads(crm_config.additional_config)
                
                # Handle double nesting if present
                if "additional_config" in additional_config and isinstance(additional_config.get("additional_config"), dict):
                    additional_config = additional_config["additional_config"]
            
            # Filter additional_config based on CRM type
            # Jira: Each user gets their own auto-generated project (don't use project_key from config)
            # ClickUp: Pass space_id and folder_id
            # Trello: No additional params needed
            # Monday.com: Pass workspace_id if provided
            if crm_config.crm_type == "jira":
                container_kwargs = {}
                # Don't use project_key from additional_config for per-user projects
                # Each user should get their own auto-generated project
                # project_key in additional_config is ignored - always auto-create per-user projects
            elif crm_config.crm_type == "clickup":
                container_kwargs = {}
                if "space_id" in additional_config:
                    container_kwargs["space_id"] = additional_config["space_id"]
                if "folder_id" in additional_config:
                    container_kwargs["folder_id"] = additional_config["folder_id"]
            elif crm_config.crm_type == "monday":
                container_kwargs = {}
                if "workspace_id" in additional_config:
                    container_kwargs["workspace_id"] = additional_config["workspace_id"]
            else:
                # Trello and others - pass all additional_config
                container_kwargs = additional_config
            
            container = crm_service.create_container(container_name, **container_kwargs)
            container_id = container["id"]
            container_url = container["url"]
            
            # Don't update global CRM config - each user gets their own container
            # Container ID is stored in ScheduledCall record (user-specific)
            
            # Create ScheduledCall record
            board_record = ScheduledCall(
                user_id=user_id,
                tenant_crm_config_id=crm_config_id,
                crm_container_id=container_id,
                crm_container_url=container_url,
                crm_type=crm_config.crm_type,
                # Legacy fields for backward compatibility
                monday_board_id=container_id if crm_config.crm_type == "monday" else None,
                monday_board_url=container_url if crm_config.crm_type == "monday" else None,
            )
            db.add(board_record)
            db.commit()
            db.refresh(board_record)
        else:
            # Verify existing container uses the same CRM config
            if board_record.tenant_crm_config_id and board_record.tenant_crm_config_id != crm_config_id:
                raise HTTPException(
                    status_code=400, 
                    detail=f"User already has a container configured with a different CRM. Please use the existing CRM config."
                )
            
            # Update existing record if it's missing new fields (for backward compatibility)
            # Or if it contains invalid data like "="
            if board_record.crm_container_id == "=":
                logger.warning(f"⚠️ Invalid container ID '=' detected for user {user_id}. Clearing it to trigger auto-fix.")
                board_record.crm_container_id = None
                
            if not board_record.crm_container_id or not board_record.crm_type:
                # Try to use legacy fields or CRM config first
                if not board_record.crm_container_id:
                    board_record.crm_container_id = board_record.monday_board_id or crm_config.container_id
                if not board_record.crm_container_url:
                    board_record.crm_container_url = board_record.monday_board_url or crm_config.container_url
                    if not board_record.crm_container_url and board_record.crm_container_id:
                        board_record.crm_container_url = crm_service.build_container_url(board_record.crm_container_id)
                if not board_record.crm_type:
                    board_record.crm_type = crm_config.crm_type
                if not board_record.tenant_crm_config_id:
                    board_record.tenant_crm_config_id = crm_config_id
                
                # If still no container_id, create a new container automatically (auto-fix)
                if not board_record.crm_container_id:
                    logger.warning(f"⚠️ Container ID missing for user {user_id}. Auto-creating new container...")
                    user = db.query(User).filter(User.id == user_id).first()
                    if not user:
                        raise HTTPException(status_code=404, detail="User not found")
                    
                    container_name = f"Scheduled Calls - {user.email}"
                    
                    additional_config = {}
                    if crm_config.additional_config:
                        import json
                        additional_config = json.loads(crm_config.additional_config)
                        
                        # Handle double nesting if present
                        if "additional_config" in additional_config and isinstance(additional_config.get("additional_config"), dict):
                            additional_config = additional_config["additional_config"]
                    
                    # Filter additional_config based on CRM type
                    if crm_config.crm_type == "jira":
                        container_kwargs = {}
                        # Don't use project_key from additional_config for per-user projects
                        # Each user should get their own auto-generated project
                        # project_key in additional_config is ignored - always auto-create per-user projects
                    elif crm_config.crm_type == "clickup":
                        container_kwargs = {}
                        if "space_id" in additional_config:
                            container_kwargs["space_id"] = additional_config["space_id"]
                        if "folder_id" in additional_config:
                            container_kwargs["folder_id"] = additional_config["folder_id"]
                    elif crm_config.crm_type == "monday":
                        container_kwargs = {}
                        if "workspace_id" in additional_config:
                            container_kwargs["workspace_id"] = additional_config["workspace_id"]
                    else:
                        # Trello and others - pass all additional_config
                        container_kwargs = additional_config
                    
                    try:
                        container = crm_service.create_container(container_name, **container_kwargs)
                        
                        # Validate container response
                        if not container or not container.get("id"):
                            raise ValueError(f"Container creation returned invalid response: {container}")
                        
                        board_record.crm_container_id = container["id"]
                        board_record.crm_container_url = container.get("url", "")
                        if not board_record.crm_container_url and board_record.crm_container_id:
                            board_record.crm_container_url = crm_service.build_container_url(board_record.crm_container_id)
                        board_record.crm_type = crm_config.crm_type
                        board_record.tenant_crm_config_id = crm_config_id
                        logger.info(f"✅ Auto-created {crm_config.crm_type} container: {board_record.crm_container_id}")
                    except HTTPException:
                        # Re-raise HTTPExceptions as-is
                        raise
                    except Exception as e:
                        error_msg = str(e)
                        # If it's a ValueError from Monday.com service, use its message directly
                        if isinstance(e, ValueError) and ("Monday.com API" in error_msg or "authentication failed" in error_msg):
                            error_msg = error_msg
                        else:
                            error_msg = f"Failed to auto-create {crm_config.crm_type} container: {error_msg}"
                        logger.error(f"❌ {error_msg}")
                        # Re-raise with more context
                        raise HTTPException(
                            status_code=500,
                            detail=error_msg
                        )
                
                db.commit()
                db.refresh(board_record)

        # Final check - if still no container_id, raise error
        if not board_record.crm_container_id:
            raise HTTPException(
                status_code=500, 
                detail="Container ID is missing. Failed to create container automatically. Please try again."
            )
        
        if not board_record.crm_type:
            raise HTTPException(
                status_code=500, 
                detail="CRM type is missing. Please recreate the container."
            )
        
        try:
            field_map = crm_service.ensure_required_fields(board_record.crm_container_id)
        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"❌ Error ensuring required fields for {board_record.crm_type}: {error_msg}", exc_info=True)
            
            # If it's a CRM authentication issue, return 400 (Bad Config) instead of 500
            if "authentication failed" in error_msg.lower() or "401" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"The {board_record.crm_type.title()} configuration is invalid or the API key has expired. Please update it in CRM Settings."
                )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Failed to prepare {board_record.crm_type} container: {error_msg}"
            )

        return board_record, field_map

    @staticmethod
    def get_board_for_user(db: Session, user_id: uuid.UUID) -> Optional[ScheduledCall]:
        """Get Monday.com board for a user."""
        return db.query(ScheduledCall).filter(ScheduledCall.user_id == user_id).first()

    @staticmethod
    def clear_board_items(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID) -> Tuple[ScheduledCall, int]:
        """
        Delete only items belonging to this tenant from user's container.
        Works with all CRMs (Monday.com, ClickUp, Jira, Trello).
        Items are filtered by tenant_id field/column.
        """
        from app.services.crm_config_service import CRMConfigService
        from app.services.crm_service_factory import CRMServiceFactory
        
        board_record = ScheduledCallService.get_board_for_user(db, user_id)
        if not board_record:
            raise HTTPException(status_code=404, detail="Container not found for user")

        # Get CRM config and service
        if not board_record.tenant_crm_config_id:
            # Fallback to Monday.com for backward compatibility
            try:
                field_map = MondayService.ensure_required_columns(board_record.monday_board_id)
                deleted = MondayService.delete_items_by_tenant_static(
                    board_id=board_record.monday_board_id,
                    tenant_id=str(tenant_id),
                    column_map=field_map
                )
                return board_record, deleted
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to clear tenant items: {exc}")

        crm_config_service = CRMConfigService()
        crm_config = crm_config_service.get_crm_config_by_id(db, board_record.tenant_crm_config_id)
        if not crm_config:
            raise HTTPException(status_code=404, detail="CRM configuration not found")

        try:
            # Get CRM service
            crm_service = CRMServiceFactory.get_service(crm_config)
            
            # Get field map
            field_map = crm_service.ensure_required_fields(board_record.crm_container_id)
            
            # Delete items by tenant
            deleted = crm_service.delete_items_by_tenant(
                container_id=board_record.crm_container_id,
                tenant_id=str(tenant_id),
                field_map=field_map
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear tenant items from {board_record.crm_type}: {exc}")

        return board_record, deleted

    @staticmethod
    async def send_n8n_webhook(
        schedule_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        phone_number: str,
        agent_id: uuid.UUID,
        call_time_utc: datetime
    ):
        """Send webhook to n8n with scheduled call data"""
        if not settings.N8N_WEBHOOK_URL:
            logger.warning("⚠️ N8N_WEBHOOK_URL not configured, skipping webhook")
            return
        
        payload = {
            "schedule_id": str(schedule_id),
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "phone_number": phone_number,
            "agent_id": str(agent_id),
            "call_time_utc": call_time_utc.isoformat(),
            "webhook_secret": settings.N8N_WEBHOOK_SECRET  # Include secret for n8n to use
        }
        
        # Prepare headers with secret (preferred method)
        headers = {}
        if settings.N8N_WEBHOOK_SECRET:
            headers["X-N8N-Webhook-Secret"] = settings.N8N_WEBHOOK_SECRET
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(settings.N8N_WEBHOOK_URL, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(f"✅ n8n webhook sent for schedule_id: {schedule_id}")
        except Exception as e:
            logger.error(f"⚠️ Failed to send n8n webhook for schedule_id {schedule_id}: {e}", exc_info=True)
            # Don't fail the entire operation if webhook fails

    @staticmethod
    async def parse_csv_and_send_to_crm(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        csv_content: str,
        crm_config_id: uuid.UUID,
        default_agent_id: uuid.UUID,  # Required parameter - agent selected before upload
        default_phone_number_id: Optional[str] = None  # ✅ Optional phone number ID for all calls in CSV
    ) -> CSVUploadResponse:
        """
        Parse CSV file and create items in CRM container (Monday.com, ClickUp, Jira, Trello).
        n8n will automatically pick them up via cron trigger.
        
        Expected CSV format (2 columns only):
        phone_number,call_time_utc
        
        - phone_number: Phone number to call (required)
        - call_time_utc: Scheduled time in UTC (required) - ISO format or YYYY-MM-DD HH:MM:SS
        - agent_id: Taken from default_agent_id parameter (all calls use same agent)
        - phone_number_id: Taken from default_phone_number_id parameter (all calls use same phone number)
        
        tenant_id and user_id are automatically taken from logged-in user.
        
        Example CSV:
        phone_number,call_time_utc
        +1234567890,2024-12-02T14:30:00Z
        +0987654321,2024-12-02T14:31:00Z
        """
        from app.services.crm_config_service import CRMConfigService
        from app.services.crm_service_factory import CRMServiceFactory
        
        board_record, field_map = ScheduledCallService.get_or_create_board_for_user(
            db, user_id, tenant_id, crm_config_id
        )
        
        # Get CRM service
        crm_config_service = CRMConfigService()
        crm_config = crm_config_service.get_crm_config_by_id(db, crm_config_id)
        crm_service = CRMServiceFactory.get_service(crm_config)
        
        # Generate unique batch_id for this CSV upload
        batch_id = str(uuid.uuid4())
        logger.info(f"📦 Generated batch_id: {batch_id} for CSV upload")
        
        # Verify agent once before processing all rows
        agent = db.query(Agent).filter(
            and_(
                Agent.id == default_agent_id,
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False
            )
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found or doesn't belong to tenant")
        
        reader = csv.DictReader(io.StringIO(csv_content))
        successful_rows = 0
        failed_rows = 0
        errors = []
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
            try:
                # Validate required fields
                if not row.get('phone_number'):
                    errors.append(f"Row {row_num}: Missing phone_number")
                    failed_rows += 1
                    continue
                
                if not row.get('call_time_utc'):
                    errors.append(f"Row {row_num}: Missing call_time_utc")
                    failed_rows += 1
                    continue
                
                # Use default_agent_id for all rows (no need to check CSV for agent_id)
                agent_uuid = default_agent_id
                
                # Parse call_time_utc
                call_time_str = row['call_time_utc'].strip()
                try:
                    if 'T' in call_time_str or '+' in call_time_str or call_time_str.endswith('Z'):
                        if call_time_str.endswith('Z'):
                            call_time_str = call_time_str.replace('Z', '+00:00')
                        scheduled_time_utc = datetime.fromisoformat(call_time_str)
                    else:
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
                            try:
                                scheduled_time_utc = datetime.strptime(call_time_str, fmt)
                                scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                        else:
                            raise ValueError(f"Unable to parse date format: {call_time_str}")
                    
                    if scheduled_time_utc.tzinfo is None:
                        scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
                    else:
                        scheduled_time_utc = scheduled_time_utc.astimezone(timezone.utc)
                        
                except Exception as e:
                    errors.append(f"Row {row_num}: Invalid call_time_utc format: {str(e)}")
                    failed_rows += 1
                    continue
                
                # Create CRM item (synchronous, but fast)
                try:
                    result = crm_service.create_scheduled_call_item(
                        container_id=board_record.crm_container_id,
                        field_map=field_map,
                        phone_number=row['phone_number'],
                        agent_id=str(agent_uuid),
                        call_time_utc=scheduled_time_utc.isoformat(),
                        tenant_id=str(tenant_id),
                        user_id=str(user_id),
                        batch_id=batch_id,  # Same batch_id for all items in this CSV
                        phone_number_id=default_phone_number_id  # ✅ Pass phone_number_id for all CSV calls
                    )
                    
                    if result:
                        successful_rows += 1
                        logger.info(f"✅ Row {row_num}: Added to {board_record.crm_type} - {row['phone_number']}")
                    else:
                        errors.append(f"Row {row_num}: Failed to create {board_record.crm_type} item")
                        failed_rows += 1
                        
                except Exception as e:
                    errors.append(f"Row {row_num}: {board_record.crm_type} error: {str(e)}")
                    failed_rows += 1
                    continue
                
            except Exception as e:
                errors.append(f"Row {row_num}: Unexpected error: {str(e)}")
                failed_rows += 1
                continue
        
        return CSVUploadResponse(
            total_rows=successful_rows + failed_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            errors=errors,
            board_id=board_record.crm_container_id,
            board_url=board_record.crm_container_url,
            batch_id=batch_id  # Return batch_id so user knows which batch was created
        )

    @staticmethod
    async def create_single_scheduled_call(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        phone_number: str,
        agent_id: uuid.UUID,
        call_time_utc: str,
        crm_config_id: uuid.UUID,
        phone_number_id: Optional[str] = None  # ✅ Add phone_number_id parameter
    ) -> dict:
        """
        Create a single scheduled call item in CRM container (Monday.com, ClickUp, Jira, Trello).
        Generates a unique batch_id for this single call.
        
        Args:
            db: Database session
            tenant_id: Tenant ID
            user_id: User ID
            phone_number: Phone number to call
            agent_id: Agent UUID
            call_time_utc: Scheduled time in UTC (ISO format string)
            crm_config_id: CRM configuration ID to use
            phone_number_id: Optional phone number ID from DB to use for call
        
        Returns:
            Dictionary with item_id, container_id, container_url, batch_id, etc.
        """
        from app.services.crm_config_service import CRMConfigService
        from app.services.crm_service_factory import CRMServiceFactory
        
        # Get or create container for user
        board_record, field_map = ScheduledCallService.get_or_create_board_for_user(
            db, user_id, tenant_id, crm_config_id
        )
        
        # Get CRM service
        crm_config_service = CRMConfigService()
        crm_config = crm_config_service.get_crm_config_by_id(db, crm_config_id)
        crm_service = CRMServiceFactory.get_service(crm_config)
        
        # Verify agent exists and belongs to tenant
        agent = db.query(Agent).filter(
            and_(
                Agent.id == agent_id,
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False
            )
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found or doesn't belong to tenant")
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            raise HTTPException(status_code=400, detail="Phone number must start with +")
        
        # Parse call_time_utc
        try:
            call_time_str = call_time_utc.strip()
            if 'T' in call_time_str or '+' in call_time_str or call_time_str.endswith('Z'):
                if call_time_str.endswith('Z'):
                    call_time_str = call_time_str.replace('Z', '+00:00')
                scheduled_time_utc = datetime.fromisoformat(call_time_str)
            else:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
                    try:
                        scheduled_time_utc = datetime.strptime(call_time_str, fmt)
                        scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Unable to parse date format: {call_time_str}")
            
            if scheduled_time_utc.tzinfo is None:
                scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
            else:
                scheduled_time_utc = scheduled_time_utc.astimezone(timezone.utc)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid call_time_utc format: {str(e)}")
        
        # Generate unique batch_id for this single call
        batch_id = str(uuid.uuid4())
        
        # Create CRM item with batch_id
        try:
            result = crm_service.create_scheduled_call_item(
                container_id=board_record.crm_container_id,
                field_map=field_map,
                phone_number=phone_number,
                agent_id=str(agent_id),
                call_time_utc=scheduled_time_utc.isoformat(),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                batch_id=batch_id,
                phone_number_id=phone_number_id  # ✅ Pass phone_number_id
            )
            
            if not result:
                raise HTTPException(status_code=500, detail=f"Failed to create {board_record.crm_type} item")
            
            item_id = result.get("id") or result.get("key") or result.get("shortLink", "")
            
            return {
                "item_id": item_id,
                "board_id": board_record.crm_container_id,
                "board_url": board_record.crm_container_url,
                "phone_number": phone_number,
                "agent_id": str(agent_id),
                "call_time_utc": scheduled_time_utc.isoformat(),
                "batch_id": batch_id,
                "crm_type": board_record.crm_type,
                "message": f"Scheduled call created in {board_record.crm_type} container. Batch ID: {batch_id}"
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create {board_record.crm_type} item: {str(e)}")

