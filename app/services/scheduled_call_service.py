from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import Optional
from datetime import datetime, timezone
import uuid
import csv
import io
import httpx
import asyncio
from app.models.agent import Agent
from app.schemas.scheduled_call import CSVUploadResponse
from app.core.config import settings


class ScheduledCallService:
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
            print("⚠️ N8N_WEBHOOK_URL not configured, skipping webhook")
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
                print(f"✅ n8n webhook sent for schedule_id: {schedule_id}")
        except Exception as e:
            print(f"⚠️ Failed to send n8n webhook for schedule_id {schedule_id}: {e}")
            # Don't fail the entire operation if webhook fails

    @staticmethod
    async def parse_csv_and_send_webhooks(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        csv_content: str
    ) -> CSVUploadResponse:
        """
        Parse CSV file and send webhooks to n8n for automation.
        No DB storage - just parse, validate, and send webhooks.
        
        Expected CSV format:
        phone_number,agent_id,call_time_utc
        
        - phone_number: Phone number to call (required)
        - agent_id: UUID of the agent (required)
        - call_time_utc: Scheduled time in UTC (required) - ISO format (YYYY-MM-DDTHH:MM:SSZ) or YYYY-MM-DD HH:MM:SS
        
        Example CSV:
        phone_number,agent_id,call_time_utc
        +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-01-15T14:30:00Z
        +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-01-15 16:00:00
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
                
                # Generate unique schedule_id for tracking
                schedule_id = uuid.uuid4()
                
                # Queue webhook task
                webhook_tasks.append(
                    ScheduledCallService.send_n8n_webhook(
                        schedule_id=schedule_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        phone_number=row['phone_number'],
                        agent_id=agent_uuid,
                        call_time_utc=scheduled_time_utc
                    )
                )
                
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
