"""
Call Log Service Module
Handles call log management including creation, updates, retrieval, and statistics
"""

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, desc, asc
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.models.agent import Agent
from app.schemas.call_log import (
    CallLogCreate, CallLogUpdate, CallLogFilters, CallLogStats, CallLogDashboardResponse
)
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime, timedelta

class CallLogService:
    """Service class for handling call log operations"""
    
    def create_call_log(self, db: Session, call_log_data: CallLogCreate) -> CallLog:
        """
        Create a new call log entry
        
        Args:
            db: Database session
            call_log_data: Call log creation data
            
        Returns:
            CallLog object
        """
        call_log = CallLog(**call_log_data.dict())
        db.add(call_log)
        db.commit()
        db.refresh(call_log)
        return call_log
    
    def get_call_log_by_id(self, db: Session, log_id: uuid.UUID) -> Optional[CallLog]:
        """Get call log by ID"""
        return db.query(CallLog).filter(CallLog.id == log_id).first()
    
    def get_call_log_by_call_id(self, db: Session, call_id: str, tenant_id: uuid.UUID) -> Optional[CallLog]:
        """Get call log by call ID and tenant"""
        return db.query(CallLog).filter(
            CallLog.call_id == call_id,
            CallLog.tenant_id == tenant_id
        ).first()
    
    def update_call_log(self, db: Session, log_id: uuid.UUID, update_data: CallLogUpdate) -> Optional[CallLog]:
        """Update call log"""
        call_log = self.get_call_log_by_id(db, log_id)
        if call_log:
            for field, value in update_data.dict(exclude_unset=True).items():
                setattr(call_log, field, value)
            call_log.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(call_log)
        return call_log
    
    def get_call_logs_with_filters(
        self, 
        db: Session, 
        tenant_id: uuid.UUID,
        filters: CallLogFilters,
        page: int = 1,
        per_page: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[CallLogDashboardResponse], int, CallLogStats]:
        """
        Get call logs with filtering, pagination, and statistics
        
        Args:
            db: Database session
            tenant_id: Tenant ID to filter by
            filters: Filter criteria
            page: Page number (1-based)
            per_page: Items per page
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Tuple of (logs, total_count, stats)
        """
        # Build base query with joins
        query = db.query(CallLog).join(CallSession).join(Agent).filter(
            CallLog.tenant_id == tenant_id
        )
        
        # Apply filters
        if filters.call_type:
            query = query.filter(CallLog.call_type == filters.call_type)
        
        if filters.success_evaluation:
            query = query.filter(CallLog.success_evaluation == filters.success_evaluation)
        
        if filters.agent_id:
            query = query.filter(CallSession.agent_id == filters.agent_id)
        
        if filters.date_from:
            query = query.filter(CallLog.created_at >= filters.date_from)
        
        if filters.date_to:
            query = query.filter(CallLog.created_at <= filters.date_to)
        
        if filters.transferred is not None:
            query = query.filter(CallLog.transferred == filters.transferred)
        
        if filters.ended_reason:
            query = query.filter(CallLog.ended_reason.ilike(f"%{filters.ended_reason}%"))
        
        if filters.assistant_phone_number:
            query = query.filter(CallLog.assistant_phone_number == filters.assistant_phone_number)
        
        if filters.customer_phone_number:
            query = query.filter(CallLog.customer_phone_number == filters.customer_phone_number)
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        sort_column = getattr(CallLog, sort_by, CallLog.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        offset = (page - 1) * per_page
        call_logs = query.offset(offset).limit(per_page).all()
        
        # Convert to dashboard response format
        dashboard_logs = []
        for log in call_logs:
            # Get agent name from the joined session
            agent_name = "Unknown Agent"
            if log.call_session and log.call_session.agent:
                agent_name = log.call_session.agent.name
            
            dashboard_logs.append(CallLogDashboardResponse(
                id=log.id,
                call_id=log.call_id,
                assistant_name=agent_name,
                assistant_phone_number=log.assistant_phone_number,
                customer_phone_number=log.customer_phone_number,
                call_type=log.call_type,
                ended_reason=log.ended_reason,
                success_evaluation=log.success_evaluation,
                start_time=log.start_time,
                duration=log.duration,
                cost=log.cost,
                transferred=log.transferred,
                created_at=log.created_at
            ))
        
        # Calculate statistics
        stats = self._calculate_call_log_stats(db, tenant_id, filters)
        
        return dashboard_logs, total, stats
    
    def _calculate_call_log_stats(self, db: Session, tenant_id: uuid.UUID, filters: CallLogFilters) -> CallLogStats:
        """Calculate statistics for call logs"""
        # Base query for stats
        query = db.query(CallLog).filter(CallLog.tenant_id == tenant_id)
        
        # Apply same filters as main query
        if filters.date_from:
            query = query.filter(CallLog.created_at >= filters.date_from)
        if filters.date_to:
            query = query.filter(CallLog.created_at <= filters.date_to)
        
        # Total calls
        total_calls = query.count()
        
        # Successful and failed calls
        successful_calls = query.filter(CallLog.success_evaluation == "success").count()
        failed_calls = query.filter(CallLog.success_evaluation == "fail").count()
        
        # Transferred calls
        transferred_calls = query.filter(CallLog.transferred == True).count()
        
        # Total cost
        total_cost_result = query.with_entities(func.sum(CallLog.cost)).scalar()
        total_cost = float(total_cost_result) if total_cost_result else 0.0
        
        # Average duration
        avg_duration_result = query.with_entities(func.avg(CallLog.duration)).scalar()
        average_duration = float(avg_duration_result) if avg_duration_result else None
        
        # Calls by type
        calls_by_type = {}
        type_counts = query.with_entities(
            CallLog.call_type, func.count(CallLog.id)
        ).group_by(CallLog.call_type).all()
        for call_type, count in type_counts:
            calls_by_type[call_type] = count
        
        # Calls by agent
        calls_by_agent = {}
        agent_counts = query.join(CallSession).join(Agent).with_entities(
            Agent.name, func.count(CallLog.id)
        ).group_by(Agent.name).all()
        for agent_name, count in agent_counts:
            calls_by_agent[agent_name] = count
        
        # Calls by ended reason
        calls_by_ended_reason = {}
        reason_counts = query.with_entities(
            CallLog.ended_reason, func.count(CallLog.id)
        ).group_by(CallLog.ended_reason).all()
        for reason, count in reason_counts:
            if reason:  # Only include non-null reasons
                calls_by_ended_reason[reason] = count
        
        return CallLogStats(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            transferred_calls=transferred_calls,
            total_cost=total_cost,
            average_duration=average_duration,
            calls_by_type=calls_by_type,
            calls_by_agent=calls_by_agent,
            calls_by_ended_reason=calls_by_ended_reason
        )
    
    def get_recent_call_logs(self, db: Session, tenant_id: uuid.UUID, limit: int = 10) -> List[CallLogDashboardResponse]:
        """Get recent call logs for dashboard"""
        call_logs = db.query(CallLog).join(CallSession).join(Agent).filter(
            CallLog.tenant_id == tenant_id
        ).order_by(desc(CallLog.created_at)).limit(limit).all()
        
        dashboard_logs = []
        for log in call_logs:
            agent_name = "Unknown Agent"
            if log.call_session and log.call_session.agent:
                agent_name = log.call_session.agent.name
            
            dashboard_logs.append(CallLogDashboardResponse(
                id=log.id,
                call_id=log.call_id,
                assistant_name=agent_name,
                assistant_phone_number=log.assistant_phone_number,
                customer_phone_number=log.customer_phone_number,
                call_type=log.call_type,
                ended_reason=log.ended_reason,
                success_evaluation=log.success_evaluation,
                start_time=log.start_time,
                duration=log.duration,
                cost=log.cost,
                transferred=log.transferred,
                created_at=log.created_at
            ))
        
        return dashboard_logs
    
    def export_call_logs(self, db: Session, tenant_id: uuid.UUID, filters: CallLogFilters) -> List[Dict[str, Any]]:
        """Export call logs to CSV format"""
        # Build query with same filters as get_call_logs_with_filters
        query = db.query(CallLog).join(CallSession).join(Agent).filter(
            CallLog.tenant_id == tenant_id
        )
        
        # Apply filters (same logic as above)
        if filters.call_type:
            query = query.filter(CallLog.call_type == filters.call_type)
        if filters.success_evaluation:
            query = query.filter(CallLog.success_evaluation == filters.success_evaluation)
        if filters.agent_id:
            query = query.filter(CallSession.agent_id == filters.agent_id)
        if filters.date_from:
            query = query.filter(CallLog.created_at >= filters.date_from)
        if filters.date_to:
            query = query.filter(CallLog.created_at <= filters.date_to)
        if filters.transferred is not None:
            query = query.filter(CallLog.transferred == filters.transferred)
        if filters.ended_reason:
            query = query.filter(CallLog.ended_reason.ilike(f"%{filters.ended_reason}%"))
        
        call_logs = query.order_by(desc(CallLog.created_at)).all()
        
        export_data = []
        for log in call_logs:
            agent_name = "Unknown Agent"
            if log.call_session and log.call_session.agent:
                agent_name = log.call_session.agent.name
            
            export_data.append({
                "call_id": log.call_id,
                "assistant_name": agent_name,
                "assistant_phone_number": log.assistant_phone_number,
                "customer_phone_number": log.customer_phone_number,
                "call_type": log.call_type,
                "ended_reason": log.ended_reason,
                "success_evaluation": log.success_evaluation,
                "start_time": log.start_time.isoformat() if log.start_time else None,
                "duration": log.duration,
                "cost": log.cost,
                "transferred": log.transferred,
                "created_at": log.created_at.isoformat()
            })
        
        return export_data

# Create service instance
call_log_service = CallLogService()
