from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import csv
import io
from app.models.scheduled_call import ScheduledCall
from app.models.agent import Agent
from app.schemas.scheduled_call import ScheduledCallCreate, CSVUploadResponse
import pytz

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
    def parse_csv_and_create_calls(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        csv_content: str,
        user_timezone: str = "UTC"
    ) -> CSVUploadResponse:
        """
        Parse CSV file and create scheduled calls.
        
        Expected CSV format:
        phone_number,agent_id,scheduled_time,timezone
        
        scheduled_time should be in format: YYYY-MM-DD HH:MM:SS or ISO format
        timezone should be a valid timezone string (e.g., 'America/New_York', 'UTC', 'Europe/London')
        """
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
                
                if not row.get('agent_id'):
                    errors.append(f"Row {row_num}: Missing agent_id")
                    failed_rows += 1
                    continue
                
                if not row.get('scheduled_time'):
                    errors.append(f"Row {row_num}: Missing scheduled_time")
                    failed_rows += 1
                    continue
                
                # Get timezone (default to user_timezone parameter or UTC)
                tz_str = row.get('timezone', user_timezone)
                if not tz_str:
                    tz_str = "UTC"
                
                # Parse scheduled_time
                scheduled_time_str = row['scheduled_time'].strip()
                try:
                    # Try parsing ISO format first
                    if 'T' in scheduled_time_str or '+' in scheduled_time_str or scheduled_time_str.endswith('Z'):
                        scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
                    else:
                        # Try parsing common formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M']:
                            try:
                                scheduled_time = datetime.strptime(scheduled_time_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            raise ValueError(f"Unable to parse date format: {scheduled_time_str}")
                except Exception as e:
                    errors.append(f"Row {row_num}: Invalid scheduled_time format: {str(e)}")
                    failed_rows += 1
                    continue
                
                # Convert to user timezone if not timezone-aware
                if scheduled_time.tzinfo is None:
                    try:
                        user_tz = pytz.timezone(tz_str)
                        scheduled_time = user_tz.localize(scheduled_time)
                    except pytz.exceptions.UnknownTimeZoneError:
                        errors.append(f"Row {row_num}: Unknown timezone: {tz_str}")
                        failed_rows += 1
                        continue
                else:
                    # If already timezone-aware, convert to the specified timezone first
                    try:
                        user_tz = pytz.timezone(tz_str)
                        scheduled_time = scheduled_time.astimezone(user_tz)
                    except pytz.exceptions.UnknownTimeZoneError:
                        errors.append(f"Row {row_num}: Unknown timezone: {tz_str}")
                        failed_rows += 1
                        continue
                
                # Convert to UTC
                scheduled_time_utc = scheduled_time.astimezone(timezone.utc)
                
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
                
                # Create scheduled call
                call_data = ScheduledCallCreate(
                    phone_number=row['phone_number'],
                    agent_id=agent_uuid,
                    scheduled_time_utc=scheduled_time_utc,
                    status="pending"
                )
                
                ScheduledCallService.create_scheduled_call(
                    db=db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    call_data=call_data
                )
                
                successful_rows += 1
                
            except Exception as e:
                errors.append(f"Row {row_num}: Unexpected error: {str(e)}")
                failed_rows += 1
                continue
        
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

