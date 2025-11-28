from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import csv
import io
import httpx
import asyncio
from app.models.scheduled_call import ScheduledCall
from app.models.agent import Agent
from app.schemas.scheduled_call import ScheduledCallCreate, CSVUploadResponse
from app.core.config import settings

class ScheduledCallService:
    @staticmethod
    def create_scheduled_call(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        call_data: ScheduledCallCreate
    ) -> ScheduledCall:
        """Create a new scheduled call"""
        scheduled_call = ScheduledCall(
            tenant_id=tenant_id,
            user_id=user_id,
            phone_number=call_data.phone_number,
            agent_id=call_data.agent_id,
            scheduled_time_utc=call_data.scheduled_time_utc,
            status=call_data.status
        )
        db.add(scheduled_call)
        db.commit()
        db.refresh(scheduled_call)
        return scheduled_call

    @staticmethod
    async def send_n8n_webhook(scheduled_call: ScheduledCall):
        """Send webhook to n8n with scheduled call data"""
        if not settings.N8N_WEBHOOK_URL:
            print("⚠️ N8N_WEBHOOK_URL not configured, skipping webhook")
            return
        
        payload = {
            "schedule_id": str(scheduled_call.id),
            "tenant_id": str(scheduled_call.tenant_id),
            "user_id": str(scheduled_call.user_id),
            "phone_number": scheduled_call.phone_number,
            "agent_id": str(scheduled_call.agent_id),
            "call_time_utc": scheduled_call.scheduled_time_utc.isoformat(),
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
                print(f"✅ n8n webhook sent for schedule_id: {scheduled_call.id}")
        except Exception as e:
            print(f"⚠️ Failed to send n8n webhook for schedule_id {scheduled_call.id}: {e}")
            # Don't fail the entire operation if webhook fails

    @staticmethod
    async def parse_csv_and_create_calls(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        csv_content: str
    ) -> CSVUploadResponse:
        """
        Parse CSV file and create scheduled calls.
        
        Expected CSV format:
        phone_number,agent_id,call_time_utc,status
        
        - phone_number: Phone number to call (required)
        - agent_id: UUID of the agent (required)
        - call_time_utc: Scheduled time in UTC (required) - ISO format (YYYY-MM-DDTHH:MM:SSZ) or YYYY-MM-DD HH:MM:SS
        - status: Status (optional, defaults to "pending")
        
        Example CSV:
        phone_number,agent_id,call_time_utc,status
        +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-01-15T14:30:00Z,pending
        +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-01-15 16:00:00,pending
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        successful_rows = 0
        failed_rows = 0
        errors = []
        webhook_tasks = []
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
            try:
                # Validate required fields
                if not row.get('phone_number'):
                    errors.append(f"Row {row_num}: Missing phone_number")
                    failed_rows += 1
                    continue
                
                if not row.get('agent_id'):
                    errors.append(f"Row {row_num}: Missing agent_id")
                    failed_rows += 1
                    continue
                
                if not row.get('call_time_utc'):
                    errors.append(f"Row {row_num}: Missing call_time_utc")
                    failed_rows += 1
                    continue
                
                # Parse call_time_utc (already in UTC)
                call_time_str = row['call_time_utc'].strip()
                try:
                    # Try parsing ISO format first (with Z or +00:00)
                    if 'T' in call_time_str or '+' in call_time_str or call_time_str.endswith('Z'):
                        # Handle ISO format
                        if call_time_str.endswith('Z'):
                            call_time_str = call_time_str.replace('Z', '+00:00')
                        scheduled_time_utc = datetime.fromisoformat(call_time_str)
                    else:
                        # Try parsing common formats (assume UTC)
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
                            try:
                                scheduled_time_utc = datetime.strptime(call_time_str, fmt)
                                # Make it timezone-aware (UTC)
                                scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                        else:
                            raise ValueError(f"Unable to parse date format: {call_time_str}")
                    
                    # Ensure it's timezone-aware and in UTC
                    if scheduled_time_utc.tzinfo is None:
                        scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)
                    else:
                        # Convert to UTC if it has timezone info
                        scheduled_time_utc = scheduled_time_utc.astimezone(timezone.utc)
                        
                except Exception as e:
                    errors.append(f"Row {row_num}: Invalid call_time_utc format: {str(e)}")
                    failed_rows += 1
                    continue
                
                # Parse agent_id
                try:
                    agent_uuid = uuid.UUID(row['agent_id'])
                except ValueError:
                    errors.append(f"Row {row_num}: Invalid agent_id format: {row['agent_id']}")
                    failed_rows += 1
                    continue
                
                # Verify agent exists and belongs to tenant
                agent = db.query(Agent).filter(
                    and_(
                        Agent.id == agent_uuid,
                        Agent.tenant_id == tenant_id,
                        Agent.is_deleted == False
                    )
                ).first()
                
                if not agent:
                    errors.append(f"Row {row_num}: Agent not found or doesn't belong to tenant")
                    failed_rows += 1
                    continue
                
                # Get status (optional, default to "pending")
                status = row.get('status', 'pending').strip().lower()
                if status not in ["pending", "scheduled", "failed", "completed"]:
                    status = "pending"
                
                # Create scheduled call
                call_data = ScheduledCallCreate(
                    phone_number=row['phone_number'],
                    agent_id=agent_uuid,
                    scheduled_time_utc=scheduled_time_utc,
                    status=status
                )
                
                scheduled_call = ScheduledCallService.create_scheduled_call(
                    db=db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    call_data=call_data
                )
                
                # Queue webhook task
                webhook_tasks.append(ScheduledCallService.send_n8n_webhook(scheduled_call))
                
                successful_rows += 1
                
            except Exception as e:
                errors.append(f"Row {row_num}: Unexpected error: {str(e)}")
                failed_rows += 1
                continue
        
        # Send all webhooks in parallel (fire and forget)
        if webhook_tasks:
            try:
                await asyncio.gather(*webhook_tasks, return_exceptions=True)
            except Exception as e:
                print(f"⚠️ Error sending webhooks: {e}")
        
        return CSVUploadResponse(
            total_rows=successful_rows + failed_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            errors=errors
        )

    @staticmethod
    def get_pending_calls(
        db: Session,
        tenant_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        current_time: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[ScheduledCall], int]:
        """
        Get all pending calls based on current UTC time with pagination.
        Optionally filter by tenant_id or user_id.
        
        Returns:
            tuple: (list of calls, total count)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Base query - only pending status
        query = db.query(ScheduledCall).filter(
            and_(
                ScheduledCall.status == "pending",
                ScheduledCall.scheduled_time_utc >= current_time
            )
        )
        
        if tenant_id:
            query = query.filter(ScheduledCall.tenant_id == tenant_id)
        
        if user_id:
            query = query.filter(ScheduledCall.user_id == user_id)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        calls = query.order_by(ScheduledCall.scheduled_time_utc.asc()).offset(skip).limit(limit).all()
        
        return calls, total

    @staticmethod
    def update_scheduled_call_status(
        db: Session,
        call_id: uuid.UUID,
        new_status: str,
        tenant_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None
    ) -> ScheduledCall:
        """
        Update the status of a scheduled call.
        
        Validates that:
        - The call exists
        - The call belongs to the specified tenant
        - Optionally, the call belongs to the specified user
        - The new status is valid
        
        Returns:
            Updated ScheduledCall object
        """
        # Validate status
        valid_statuses = ["pending", "scheduled", "failed", "completed"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        
        # Find the scheduled call
        query = db.query(ScheduledCall).filter(
            and_(
                ScheduledCall.id == call_id,
                ScheduledCall.tenant_id == tenant_id
            )
        )
        
        # Optionally filter by user_id
        if user_id:
            query = query.filter(ScheduledCall.user_id == user_id)
        
        scheduled_call = query.first()
        
        if not scheduled_call:
            raise ValueError("Scheduled call not found or access denied")
        
        # Update status
        scheduled_call.status = new_status
        db.commit()
        db.refresh(scheduled_call)
        
        return scheduled_call

