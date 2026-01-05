from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import json
from app.core.logger import logger

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.agent import Agent
from app.schemas.call_log import (
    CallLogResponse, 
    CallLogFilters, 
    CallLogStats, 
    CallLogList
)
from app.services.call_log_service import CallLogService
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/call-logs", response_model=CallLogList)
async def get_call_logs(
    # Pagination
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    
    # Filters
    call_type: Optional[str] = Query(None, description="Filter by call type (inbound, outbound, web)"),
    success_evaluation: Optional[str] = Query(None, description="Filter by success (success, fail, null)"),
    agent_id: Optional[uuid.UUID] = Query(None, description="Filter by agent ID"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    transferred: Optional[bool] = Query(None, description="Filter by transferred calls"),
    ended_reason: Optional[str] = Query(None, description="Filter by ended reason"),
    
    # User and database
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get call logs with filtering and pagination
    Comprehensive call logging system for monitoring all call activities
    """
    try:
        logger.info(f"📊 GETTING CALL LOGS")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        logger.debug(f"📄 Page: {page}, Per Page: {per_page}")
        logger.debug(f"🔍 Filters: type={call_type}, success={success_evaluation}, agent={agent_id}")
        
        # Create filters object
        filters = CallLogFilters(
            call_type=call_type,
            success_evaluation=success_evaluation,
            agent_id=agent_id,
            date_from=date_from,
            date_to=date_to,
            transferred=transferred,
            ended_reason=ended_reason
        )
        
        # Get call logs using service
        call_logs_result = CallLogService.get_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters,
            page=page,
            per_page=per_page
        )
        
        logger.info(f"✅ Found {call_logs_result['total']} call logs")
        logger.debug(f"📊 Stats: {call_logs_result['stats']}")
        
        return create_success_response(
            call_logs_result,
            f"Retrieved {len(call_logs_result['logs'])} call logs successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting call logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get call logs: {str(e)}")


@router.get("/call-logs/{call_log_id}", response_model=CallLogResponse)
async def get_call_log_detail(
    call_log_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific call log
    """
    try:
        logger.info(f"📋 GETTING CALL LOG DETAIL")
        logger.debug(f"🆔 Call Log ID: {call_log_id}")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        
        # Get call log detail
        call_log = CallLogService.get_call_log_by_id(
            db=db,
            call_log_id=call_log_id,
            tenant_id=user.current_tenant_id
        )
        
        if not call_log:
            raise HTTPException(status_code=404, detail="Call log not found")
        
        logger.info(f"✅ Found call log: {call_log.call_id}")
        logger.debug(f"📞 Phone: {call_log.customer_phone_number}")
        logger.debug(f"⏱️ Duration: {call_log.duration} seconds")
        logger.debug(f"📊 Status: {call_log.success_evaluation}")
        
        return create_success_response(
            call_log,
            "Call log detail retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting call log detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get call log detail: {str(e)}")


@router.get("/call-logs/stats", response_model=CallLogStats)
async def get_call_logs_stats(
    # Date range
    date_from: Optional[datetime] = Query(None, description="Stats from date"),
    date_to: Optional[datetime] = Query(None, description="Stats to date"),
    
    # User and database
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get call logs statistics and analytics
    """
    try:
        logger.info(f"📈 GETTING CALL LOGS STATS")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        logger.debug(f"📅 Date Range: {date_from} to {date_to}")
        
        # Get call logs statistics
        stats = CallLogService.get_call_logs_stats(
            db=db,
            tenant_id=user.current_tenant_id,
            date_from=date_from,
            date_to=date_to
        )
        
        logger.info(f"📊 Total Calls: {stats.total_calls}")
        logger.debug(f"✅ Successful: {stats.successful_calls}")
        logger.debug(f"❌ Failed: {stats.failed_calls}")
        logger.debug(f"🔄 Transferred: {stats.transferred_calls}")
        logger.debug(f"💰 Total Cost: ${stats.total_cost}")
        logger.debug(f"⏱️ Avg Duration: {stats.average_duration} seconds")
        
        return create_success_response(
            stats,
            "Call logs statistics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting call logs stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get call logs stats: {str(e)}")


@router.get("/call-logs/agent/{agent_id}")
async def get_agent_call_logs(
    agent_id: uuid.UUID,
    # Pagination
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    
    # Date range
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    
    # User and database
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get call logs for a specific agent
    """
    try:
        logger.info(f"🤖 GETTING AGENT CALL LOGS")
        logger.debug(f"🆔 Agent ID: {agent_id}")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        
        # Verify agent belongs to tenant
        agent = db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.tenant_id == user.current_tenant_id
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create filters for agent
        filters = CallLogFilters(
            agent_id=agent_id,
            date_from=date_from,
            date_to=date_to
        )
        
        # Get agent call logs
        call_logs_result = CallLogService.get_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters,
            page=page,
            per_page=per_page
        )
        
        logger.info(f"✅ Found {call_logs_result['total']} calls for agent: {agent.name}")
        
        return create_success_response(
            {
                "agent": {
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description
                },
                "call_logs": call_logs_result
            },
            f"Retrieved {len(call_logs_result['logs'])} call logs for agent {agent.name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting agent call logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get agent call logs: {str(e)}")


@router.get("/call-logs/recent")
async def get_recent_call_logs(
    limit: int = Query(10, ge=1, le=50, description="Number of recent calls"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get recent call logs for quick monitoring
    """
    try:
        logger.info(f"🕐 GETTING RECENT CALL LOGS")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        logger.debug(f"📊 Limit: {limit}")
        
        # Get recent call logs
        recent_logs = CallLogService.get_recent_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            limit=limit
        )
        
        logger.info(f"✅ Found {len(recent_logs)} recent call logs")
        
        return create_success_response(
            {
                "recent_logs": recent_logs,
                "count": len(recent_logs)
            },
            f"Retrieved {len(recent_logs)} recent call logs"
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting recent call logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get recent call logs: {str(e)}")


@router.get("/call-logs/export")
async def export_call_logs(
    # Date range
    date_from: Optional[datetime] = Query(None, description="Export from date"),
    date_to: Optional[datetime] = Query(None, description="Export to date"),
    
    # Format
    format: str = Query("json", description="Export format (json, csv)"),
    
    # User and database
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Export call logs in various formats
    """
    try:
        logger.info(f"📤 EXPORTING CALL LOGS")
        logger.debug(f"👤 User: {user.email}")
        logger.debug(f"🏢 Tenant: {user.current_tenant_id}")
        logger.debug(f"📅 Date Range: {date_from} to {date_to}")
        logger.debug(f"📄 Format: {format}")
        
        # Get all call logs for export
        filters = CallLogFilters(
            date_from=date_from,
            date_to=date_to
        )
        
        # Get all call logs (no pagination for export)
        call_logs_result = CallLogService.get_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters,
            page=1,
            per_page=10000  # Large number to get all
        )
        
        if format.lower() == "csv":
            # Convert to CSV format
            csv_data = CallLogService.export_to_csv(call_logs_result['logs'])
            return create_success_response(
                {"csv_data": csv_data, "count": len(call_logs_result['logs'])},
                f"Exported {len(call_logs_result['logs'])} call logs to CSV"
            )
        else:
            # Return JSON format
            return create_success_response(
                {
                    "call_logs": call_logs_result['logs'],
                    "stats": call_logs_result['stats'],
                    "count": len(call_logs_result['logs'])
                },
                f"Exported {len(call_logs_result['logs'])} call logs to JSON"
            )
        
    except Exception as e:
        logger.error(f"❌ Error exporting call logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export call logs: {str(e)}")