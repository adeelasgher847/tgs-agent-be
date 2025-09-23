"""
VICIdial-style API Router
Provides VICIdial-compatible API endpoints for call management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.models.agent import Agent
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.services.call_session_service import call_session_service
from app.services.call_log_service import call_log_service
from app.schemas.call_log import CallLogCreate
from app.utils.response import create_success_response
from sqlalchemy import func

router = APIRouter()

@router.get("/api.php")
async def vicidial_api(
    request: Request,
    source: str = Query(..., description="Source application identifier"),
    user: str = Query(..., description="API username"),
    pass_: str = Query(..., alias="pass", description="API password"),
    function: str = Query(..., description="API function to execute"),
    phone_number: Optional[str] = Query(None, description="Phone number for the call"),
    campaign_id: Optional[str] = Query(None, description="Campaign/Agent ID"),
    call_type: Optional[str] = Query("outbound", description="Type of call (inbound/outbound/web)"),
    notes: Optional[str] = Query(None, description="Additional notes for the call"),
    db: Session = Depends(get_db)
):
    """
    VICIdial-style API endpoint for call management
    
    Example usage:
    GET /api.php?source=myapp&user=apiuser&pass=apipass&function=add_call&phone_number=123456789&campaign_id=OUTBOUND1
    
    Supported functions:
    - add_call: Add a new call to the system
    - get_calls: Retrieve call logs
    - get_stats: Get call statistics
    """
    
    # Authenticate API user (you can implement proper API key authentication here)
    if not _authenticate_api_user(user, pass_):
        raise HTTPException(status_code=401, detail="Invalid API credentials")
    
    try:
        if function == "add_call":
            return await _handle_add_call(
                source=source,
                phone_number=phone_number,
                campaign_id=campaign_id,
                call_type=call_type,
                notes=notes,
                db=db
            )
        elif function == "get_calls":
            return await _handle_get_calls(
                source=source,
                campaign_id=campaign_id,
                db=db
            )
        elif function == "get_stats":
            return await _handle_get_stats(
                source=source,
                campaign_id=campaign_id,
                db=db
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown function: {function}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _handle_add_call(
    source: str,
    phone_number: str,
    campaign_id: str,
    call_type: str,
    notes: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Handle add_call function - similar to VICIdial's add_call"""
    
    if not phone_number:
        raise HTTPException(status_code=400, detail="phone_number is required for add_call")
    
    if not campaign_id:
        raise HTTPException(status_code=400, detail="campaign_id is required for add_call")
    
    # Find agent by campaign_id (you can map this to your agent system)
    agent = db.query(Agent).filter(Agent.name.ilike(f"%{campaign_id}%")).first()
    if not agent:
        # Create a default agent if not found
        agent = Agent(
            name=campaign_id,
            tenant_id=uuid.uuid4(),  # You might want to get this from API user context
            system_prompt="Default agent for API calls",
            created_by=uuid.uuid4(),  # You might want to get this from API user context
            updated_by=uuid.uuid4()
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
    
    # Create call session
    call_session = call_session_service.create_call_session(
        db=db,
        user_id=uuid.uuid4(),  # Anonymous user for API calls
        agent_id=agent.id,
        tenant_id=agent.tenant_id,
        twilio_call_sid=f"api_call_{uuid.uuid4().hex[:8]}",
        from_number="api_source",
        to_number=phone_number,
        call_type=call_type,
        assistant_phone_number="api_agent",
        customer_phone_number=phone_number
    )
    
    # Update call log with additional information
    call_log = db.query(CallLog).filter(CallLog.call_session_id == call_session.id).first()
    if call_log and notes:
        call_log.notes = notes
        call_log.call_metadata = {
            "source": source,
            "campaign_id": campaign_id,
            "api_call": True
        }
        db.commit()
    
    return {
        "status": "SUCCESS",
        "message": "Call added successfully",
        "call_id": call_log.call_id if call_log else str(call_session.id)[:8],
        "session_id": str(call_session.id),
        "agent_id": str(agent.id),
        "phone_number": phone_number,
        "campaign_id": campaign_id,
        "call_type": call_type
    }

async def _handle_get_calls(
    source: str,
    campaign_id: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Handle get_calls function - retrieve call logs"""
    
    # Build query
    query = db.query(CallLog).join(CallSession)
    
    if campaign_id:
        query = query.join(Agent).filter(Agent.name.ilike(f"%{campaign_id}%"))
    
    # Get recent calls (last 100)
    calls = query.order_by(CallLog.created_at.desc()).limit(100).all()
    
    call_list = []
    for call in calls:
        call_list.append({
            "call_id": call.call_id,
            "phone_number": call.customer_phone_number,
            "call_type": call.call_type,
            "status": call.success_evaluation or "unknown",
            "start_time": call.start_time.isoformat() if call.start_time else None,
            "duration": call.duration,
            "cost": call.cost,
            "ended_reason": call.ended_reason,
            "notes": call.notes
        })
    
    return {
        "status": "SUCCESS",
        "message": f"Retrieved {len(call_list)} calls",
        "calls": call_list,
        "total": len(call_list)
    }

async def _handle_get_stats(
    source: str,
    campaign_id: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Handle get_stats function - get call statistics"""
    
    # Build query
    query = db.query(CallLog).join(CallSession)
    
    if campaign_id:
        query = query.join(Agent).filter(Agent.name.ilike(f"%{campaign_id}%"))
    
    # Calculate statistics
    total_calls = query.count()
    successful_calls = query.filter(CallLog.success_evaluation == "success").count()
    failed_calls = query.filter(CallLog.success_evaluation == "fail").count()
    
    # Get cost statistics
    cost_result = query.with_entities(func.sum(CallLog.cost)).scalar()
    total_cost = float(cost_result) if cost_result else 0.0
    
    # Get duration statistics
    duration_result = query.with_entities(func.avg(CallLog.duration)).scalar()
    avg_duration = float(duration_result) if duration_result else 0.0
    
    return {
        "status": "SUCCESS",
        "message": "Statistics retrieved successfully",
        "stats": {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            "total_cost": total_cost,
            "average_duration": avg_duration,
            "campaign_id": campaign_id
        }
    }

def _authenticate_api_user(username: str, password: str) -> bool:
    """
    Authenticate API user
    In a real implementation, you would check against a database of API users
    For now, we'll use a simple hardcoded check
    """
    # Simple authentication - replace with proper API key management
    valid_users = {
        "apiuser": "apipass",
        "admin": "admin123",
        "test": "test123"
    }
    
    return valid_users.get(username) == password

# Additional VICIdial-compatible endpoints

@router.post("/api.php")
async def vicidial_api_post(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle POST requests to VICIdial API"""
    
    form_data = await request.form()
    
    source = form_data.get("source", "")
    user = form_data.get("user", "")
    pass_ = form_data.get("pass", "")
    function = form_data.get("function", "")
    phone_number = form_data.get("phone_number")
    campaign_id = form_data.get("campaign_id")
    call_type = form_data.get("call_type", "outbound")
    notes = form_data.get("notes")
    
    # Redirect to GET handler
    return await vicidial_api(
        request=request,
        source=source,
        user=user,
        pass_=pass_,
        function=function,
        phone_number=phone_number,
        campaign_id=campaign_id,
        call_type=call_type,
        notes=notes,
        db=db
    )

@router.get("/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "SUCCESS",
        "message": "VICIdial-compatible API is running",
        "version": "1.0.0",
        "supported_functions": ["add_call", "get_calls", "get_stats"],
        "timestamp": datetime.now().isoformat()
    }
