"""
Scheduled Calls API endpoints with Monday.com integration (per-user boards).
All tenants of a user share the same board, identified by tenant_id column in items.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Request, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime, timezone
import uuid
from app.api.deps import get_db, require_tenant, get_optional_tenant_user
from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async
from app.models.user import User
from app.models.agent import Agent
from app.models.call_session import CallSession
from app.schemas.scheduled_call import CSVUploadResponse, BoardInfoResponse, DeleteBoardItemsResponse, SingleCallRequest, SingleCallResponse
from app.services.scheduled_call_service import ScheduledCallService
from app.services.monday_service import MondayService
from app.services.transcript_service import transcript_service
from app.services.agent_service import agent_service
from app.services.model_service import ModelService
from app.services.call_session_service import call_session_service
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse
from typing import Optional, Dict, Any
import re

router = APIRouter()

scheduled_call_service = ScheduledCallService()
model_service = ModelService()


async def analyze_call_transcript_internal(
    db: Session,
    call_session: CallSession,
    user: User
) -> Optional[Dict[str, Any]]:
    """
    Internal helper function to analyze a call transcript.
    Returns analysis dict or None if analysis fails.
    """
    try:
        # Get transcript messages
        transcript_messages = transcript_service.get_messages_by_session(db, call_session.id)
        
        if not transcript_messages:
            return None
        
        # Format transcript for analysis
        transcript_text = ""
        for msg in transcript_messages:
            role_label = "Agent" if msg.role == "agent" else "Customer"
            transcript_text += f"{role_label}: {msg.message}\n"
        
        # Get agent and model info
        agent = None
        agent_prompt = None
        preferred_model = None
        
        if call_session.agent_id:
            try:
                agent = agent_service.get_agent_by_id(db, call_session.agent_id, call_session.tenant_id)
                if agent:
                    if agent.system_prompt:
                        agent_prompt = agent.system_prompt
                    elif agent.model and agent.model.system_prompt:
                        agent_prompt = agent.model.system_prompt
                    
                    if agent and agent.model:
                        preferred_model = agent.model.model_name
            except Exception as e:
                print(f"⚠️ Could not get agent/model info: {e}")
        
        # Fallback models
        fallback_models = [
            preferred_model,
            "gemini-2.0-flash",
            "llama-3.3-70b-versatile",
            "gpt-4o-mini"
        ]
        fallback_models = [m for m in fallback_models if m]
        
        # Find available model
        model = None
        for model_name in fallback_models:
            try:
                model = model_service.get_model_by_name(db, model_name)
                if model:
                    break
            except Exception:
                continue
        
        if not model:
            return None
        
        # Get API key
        current_api_key = None
        if model.api_key:
            from app.core.security import decrypt_api_key
            current_api_key = decrypt_api_key(model.api_key)
        
        # Create prompts
        summary_prompt = f"""
        Analyze this call transcript and provide a brief summary in 2-3 sentences.
        
        Call Transcript:
        {transcript_text}
        
        Provide only:
        - Brief call overview
        - Main topic/issue
        - Outcome/resolution
        
        Keep it concise and to the point.
        """
        
        sentiment_prompt = f"""
        Analyze the sentiment of this call transcript and provide a brief assessment.
        
        Call Transcript:
        {transcript_text}
        
        Provide only:
        - Overall sentiment (positive/negative/neutral)
        - Sentiment score (0-100)
        - Customer satisfaction level (high/medium/low)
        
        Keep it brief and concise.
        """
        
        recommendations_prompt = f"""
You are a friendly colleague reviewing this call. Based on the transcript and agent's instructions, provide helpful, friendly recommendations in a casual, conversational English tone.

Call Transcript:
{transcript_text}

Agent's Instructions/Purpose:
{agent_prompt if agent_prompt else "No specific instructions provided. Use general best practices for customer service calls."}

IMPORTANT - Write recommendations in a FRIENDLY, CONVERSATIONAL English tone:
- Use friendly, conversational language
- Be helpful and suggestive, not strict or commanding
- Keep it casual and friendly - like a colleague giving friendly advice
- Each recommendation should be 2-3 sentences

Provide 2-4 friendly recommendations in this format:
1. [Your friendly suggestion]. [Brief reason or context - 2-3 sentences total]
2. [Next helpful suggestion]. [Brief reason - 2-3 sentences total]

Keep it conversational and warm. 2-3 sentences per recommendation in friendly English.
"""
        
        fit_score_prompt = f"""
Evaluate how well this call aligned with the agent's purpose and instructions. Provide a fit score out of 10.

Call Transcript:
{transcript_text}

Agent's Instructions/Purpose:
{agent_prompt if agent_prompt else "No specific instructions provided. General customer service call."}

Evaluate the call's alignment with the agent's purpose and provide a fit score (0-10).

Format your response as:
Fit Score: [number 0-10]
Brief Explanation: [One line explaining the score - keep it concise]
"""
        
        # Helper function to generate analysis text
        def generate_analysis_text(current_model, current_api_key, prompt: str, max_tokens: int = 200):
            provider_name = (current_model.provider.name or "").strip().lower()
            
            if provider_name in ("gemini", "google", "google-ai", "google ai", "gemini-1.5-flash", "gemini-2.0-flash"):
                from app.services.gemini_service import GeminiService
                service = GeminiService()
                return service.generate_text(
                    prompt=prompt,
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            elif provider_name in ("openai", "gpt", "gpt-4o-mini", "gpt-4o", "gpt-4"):
                from app.services.openai_service import OpenAIService
                service = OpenAIService()
                return service.generate_text(
                    prompt=prompt,
                    system_prompt="You are an AI assistant that analyzes call transcripts.",
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            elif provider_name in ("groq", "llama", "llama-3.3-70b-versatile"):
                from app.services.groq_service import GroqService
                service = GroqService()
                return service.generate_text(
                    prompt=prompt,
                    system_prompt="You are an AI assistant that analyzes call transcripts.",
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")
        
        # Generate analysis
        summary_result = None
        sentiment_result = None
        recommendations_result = None
        fit_score_result = None
        
        try:
            summary_result = generate_analysis_text(model, current_api_key, summary_prompt, max_tokens=200)
            sentiment_result = generate_analysis_text(model, current_api_key, sentiment_prompt, max_tokens=150)
            
            if agent_prompt:
                try:
                    recommendations_result = generate_analysis_text(
                        model, current_api_key, recommendations_prompt, max_tokens=300
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate recommendations: {e}")
                
                try:
                    fit_score_result = generate_analysis_text(
                        model, current_api_key, fit_score_prompt, max_tokens=150
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate fit score: {e}")
        except Exception as e:
            print(f"⚠️ Error generating analysis: {e}")
            return None
        
        if not summary_result or not sentiment_result:
            return None
        
        # Prepare analysis data
        analysis_data = {
            "summary": summary_result.get("content", "").strip() if summary_result else "",
            "sentiment": sentiment_result.get("content", "").strip() if sentiment_result else ""
        }
        
        # Parse recommendations
        if recommendations_result:
            recommendations_text = recommendations_result.get("content", "").strip()
            recommendations_list = []
            
            lines = recommendations_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    recommendations_list.append(match.group(1).strip())
                elif line.startswith('- ') or line.startswith('* '):
                    recommendations_list.append(line[2:].strip())
                elif len(line) > 20 and not recommendations_list:
                    recommendations_list.append(line)
            
            if not recommendations_list:
                recommendations_list = [recommendations_text]
            
            analysis_data["recommendations"] = recommendations_list
            analysis_data["recommendations_text"] = recommendations_text
        
        # Parse fit score
        fit_score = None
        fit_score_explanation = None
        if fit_score_result:
            try:
                fit_score_text = fit_score_result.get("content", "").strip()
                score_match = re.search(r'Fit Score:\s*(\d+)', fit_score_text, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    if 0 <= score <= 10:
                        fit_score = score
                    elif 0 <= score <= 100:
                        fit_score = round(score / 10)
                    
                    explanation_match = re.search(r'Brief Explanation:\s*(.+?)(?:\n|$)', fit_score_text, re.IGNORECASE | re.DOTALL)
                    if explanation_match:
                        fit_score_explanation = explanation_match.group(1).strip()
                
                if fit_score is None:
                    numbers = re.findall(r'\b([0-9]|10)\b', fit_score_text)
                    for num in numbers:
                        score = int(num)
                        if 0 <= score <= 10:
                            fit_score = score
                            break
            except Exception as e:
                print(f"⚠️ Failed to parse fit score: {e}")
        
        if fit_score is not None:
            analysis_data["fit_score"] = fit_score
            if fit_score_explanation:
                analysis_data["fit_score_explanation"] = fit_score_explanation
        
        return {
            "analysis": analysis_data,
            "model_used": model.model_name,
            "transcript_message_count": len(transcript_messages)
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing transcript for call session {call_session.id}: {e}")
        return None


@router.post("", response_model=SuccessResponse[CSVUploadResponse])
async def upload_scheduled_calls_csv(
    file: UploadFile = File(..., description="CSV file with scheduled calls"),
    agent_id: str = Query(..., description="Agent ID to use for all calls in this CSV (required)"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to create scheduled calls in Monday.com board.
    
    **CSV Format (2 columns only):**
    ```
    phone_number,call_time_utc
    ```
    
    **Required:**
    - Select agent before upload (agent_id query parameter - required)
    - CSV with phone_number and call_time_utc only
    
    **Required Columns:**
    - `phone_number`: Phone number to call (e.g., +1234567890)
    - `call_time_utc`: Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS
    
    **Note:** `tenant_id` and `user_id` are automatically taken from your logged-in session.
    All calls in this CSV will use the selected agent.
    
    **Example CSV:**
    ```csv
    phone_number,call_time_utc
    +1234567890,2024-12-02T14:30:00Z
    +0987654321,2024-12-02T14:31:00Z
    +1234567892,2024-12-02T14:32:00Z
    ```
    
    **Flow:**
    1. Select agent from dropdown
    2. Upload CSV (2 columns: phone_number, call_time_utc)
    3. Backend parses CSV and validates data
    4. Creates items in the user's Monday.com board (status: "Pending", tenant_id stored in column)
    5. n8n cron (every 1 min) detects new items
    6. n8n waits until call_time_utc
    7. n8n calls backend `/voice/call/initiate`
    8. n8n updates Monday.com status ("Called" or "Failed")
    
    **Data storage:** CSV rows live only in Monday.com. The backend stores one board
    record per user (shared by all their tenants). Items are identified by tenant_id column.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Validate and verify agent_id (REQUIRED)
        try:
            agent_uuid = uuid.UUID(agent_id)
            # Verify agent exists and belongs to tenant
            agent = db.query(Agent).filter(
                and_(
                    Agent.id == agent_uuid,
                    Agent.tenant_id == user.current_tenant_id,
                    Agent.is_deleted == False
                )
            ).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found or doesn't belong to tenant")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        result = await scheduled_call_service.parse_csv_and_send_to_monday(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            csv_content=csv_content,
            default_agent_id=agent_uuid  # Pass selected agent (required)
        )

        message = (
            f"Processed {result.total_rows} rows: {result.successful_rows} added to Monday.com, "
            f"{result.failed_rows} failed. Board URL: {result.board_url}"
        )

        return create_success_response(result, message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@router.post("/single-call", response_model=SuccessResponse[SingleCallResponse])
async def create_single_scheduled_call(
    agent_id: str = Query(..., description="Agent ID (UUID)"),
    phone_number: str = Query(..., description="Phone number to call (e.g., +1234567890)"),
    call_time_utc: str = Query(..., description="Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Create a single scheduled call in Monday.com board.
    
    **Query Parameters:**
    - `agent_id`: Agent ID (UUID)
    - `phone_number`: Phone number to call (e.g., +1234567890)
    - `call_time_utc`: Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS
    
    **Flow:**
    1. Validates agent exists and belongs to tenant
    2. Generates unique batch_id for this single call
    3. Creates item in user's Monday.com board (status: "Pending", batch_id stored)
    4. n8n cron detects new item and triggers call at scheduled time
    5. When call completes (Called/Failed), n8n will send email for this batch
    
    **Note:** `tenant_id` and `user_id` are automatically taken from logged-in session.
    """
    try:
        # Parse agent_id
        try:
            agent_uuid = uuid.UUID(agent_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        result = await scheduled_call_service.create_single_scheduled_call(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            phone_number=phone_number,
            agent_id=agent_uuid,
            call_time_utc=call_time_utc
        )
        
        return create_success_response(
            SingleCallResponse(**result),
            result["message"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scheduled call: {str(e)}")


@router.get("/board", response_model=SuccessResponse[BoardInfoResponse])
async def get_board_url(user: User = Depends(require_tenant), db: Session = Depends(get_db)):
    """
    Retrieve the Monday.com board URL for the current user.
    All tenants of this user share the same board.
    """
    board_record = scheduled_call_service.get_board_for_user(db, user.id)
    if not board_record:
        raise HTTPException(status_code=404, detail="No scheduled calls board found for this user")

    data = BoardInfoResponse(
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, "Scheduled calls board retrieved")


@router.delete("/board/items", response_model=SuccessResponse[DeleteBoardItemsResponse])
async def clear_board_items(user: User = Depends(require_tenant), db: Session = Depends(get_db)):
    """
    Remove all items belonging to the current tenant from the user's Monday.com board.
    Only items with matching tenant_id are deleted, keeping other tenants' items intact.
    """
    board_record, deleted = scheduled_call_service.clear_board_items(
        db, 
        user.id,  # user_id
        user.current_tenant_id  # tenant_id for filtering
    )
    data = DeleteBoardItemsResponse(
        items_deleted=deleted,
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, f"Deleted {deleted} item(s) for current tenant from the board")


@router.get("/batch/{batch_id}/analysis", response_model=SuccessResponse[dict])
async def get_batch_analysis(
    batch_id: str,
    http_request: Request,
    tenant_id: Optional[str] = Query(None, description="Tenant ID (required when using webhook secret)"),
    user_id: Optional[str] = Query(None, description="User ID (required when using webhook secret)"),
    user: Optional[User] = Depends(get_optional_tenant_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis data for a completed batch.
    
    **Authentication:** 
    - JWT token (default) - user and tenant from token
    - OR X-N8N-Webhook-Secret header - provide tenant_id and user_id as query params
    
    **Query Parameters (for n8n webhook):**
    - `tenant_id` (str, optional): Required when using webhook secret
    - `user_id` (str, optional): Required when using webhook secret
    
    **Returns comprehensive analysis including:**
    - Total scheduled calls (from CSV)
    - Total calls made vs pending
    - Successfully called vs failed calls
    - Total and average call duration
    - Call times and details for each call
    - Success/failure rates
    - Total cost
    - LLM transcript analysis for each call
    - Current timestamp (report generation time)
    - user_email (for n8n to send email)
    
    **Matching Logic:**
    - First tries to match by call_session_id from Monday.com items (most accurate)
    - Falls back to phone number matching if call_session_id not available
    
    **Note:** n8n workflow should:
    1. Check batch completion on Monday.com
    2. Wait 10 minutes
    3. Call this endpoint with webhook secret + tenant_id + user_id
    4. Use returned data to send email
    """
    try:
        # Verify authentication: either JWT token OR webhook secret
        is_webhook = await verify_n8n_webhook_secret_async(http_request)
        
        if is_webhook:
            # Webhook authentication - get tenant_id and user_id from query params
            if not tenant_id or not user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tenant_id and user_id are required as query parameters when using webhook secret"
                )
            try:
                tenant_uuid = uuid.UUID(tenant_id)
                user_uuid = uuid.UUID(user_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid UUID format for tenant_id or user_id"
                )
            
            # Get user from database
            user = db.query(User).filter(User.id == user_uuid).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user.current_tenant_id = tenant_uuid
        else:
            # JWT authentication - get from user token
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: JWT token or n8n webhook secret"
                )
        
        # Get user's board
        board_record = scheduled_call_service.get_board_for_user(db, user.id)
        if not board_record:
            raise HTTPException(status_code=404, detail="Board not found for user")
        
        # Get column map
        column_map = MondayService.ensure_required_columns(board_record.monday_board_id)
        
        # Fetch all items from Monday.com with this batch_id and tenant_id
        items = MondayService.get_items_by_batch_id(
            board_id=board_record.monday_board_id,
            batch_id=batch_id,
            tenant_id=str(user.current_tenant_id),
            column_map=column_map
        )
        
        if not items:
            raise HTTPException(status_code=404, detail=f"No items found for batch_id: {batch_id}")
        
        # Total scheduled calls (from Monday.com items)
        total_scheduled = len(items)
        
        # Extract call_session_ids and phone numbers from items
        call_session_ids = []
        phone_numbers = []
        
        for item in items:
            phone_number = item.get("name", "").strip()
            if phone_number:
                phone_numbers.append(phone_number)
            
            # Extract call_session_id from column values
            for col_val in item.get("column_values", []):
                if col_val.get("id") == column_map.get("call_session_id"):
                    session_id = col_val.get("text", "").strip()
                    if session_id:
                        try:
                            call_session_ids.append(uuid.UUID(session_id))
                        except ValueError:
                            pass
                    break
        
        # Fetch call sessions - prefer call_session_id, fallback to phone_number
        if call_session_ids:
            # Use call_session_id for accurate matching
            call_sessions = db.query(CallSession).filter(
                and_(
                    CallSession.id.in_(call_session_ids),
                    CallSession.tenant_id == user.current_tenant_id
                )
            ).all()
        else:
            # Fallback: match by phone number
            call_sessions = db.query(CallSession).filter(
                and_(
                    CallSession.to_number.in_(phone_numbers),
                    CallSession.tenant_id == user.current_tenant_id
                )
            ).all()
        
        # Calculate statistics
        total_calls_made = len(call_sessions)  # Actually made calls
        called_count = len([cs for cs in call_sessions if cs.status == "completed"])
        failed_count = len([cs for cs in call_sessions if cs.status in ["failed", "busy"]])
        pending_count = total_scheduled - total_calls_made  # Items that never got called
        
        total_duration = sum(cs.duration or 0 for cs in call_sessions)
        avg_duration = total_duration / total_calls_made if total_calls_made > 0 else 0
        
        # Prepare call details with transcript analysis
        call_details = []
        for cs in call_sessions:
            call_detail = {
                "call_session_id": str(cs.id),
                "phone_number": cs.to_number,
                "start_time": cs.start_time.isoformat() if cs.start_time else None,
                "end_time": cs.end_time.isoformat() if cs.end_time else None,
                "duration_seconds": cs.duration,
                "duration_formatted": f"{(cs.duration or 0) // 60}m {(cs.duration or 0) % 60}s" if cs.duration else "0s",
                "status": cs.status,
                "success_evaluation": cs.success_evaluation,
                "ended_reason": cs.ended_reason,
                "cost": float(cs.cost) if cs.cost else 0.0
            }
            
            # Add transcript analysis if call is completed and has transcript
            if cs.status == "completed":
                try:
                    transcript_analysis = await analyze_call_transcript_internal(
                        db=db,
                        call_session=cs,
                        user=user
                    )
                    if transcript_analysis:
                        call_detail["transcript_analysis"] = transcript_analysis.get("analysis")
                        call_detail["analysis_model_used"] = transcript_analysis.get("model_used")
                        call_detail["transcript_message_count"] = transcript_analysis.get("transcript_message_count", 0)
                    else:
                        call_detail["transcript_analysis"] = None
                        call_detail["transcript_message_count"] = 0
                except Exception as e:
                    print(f"⚠️ Failed to analyze transcript for call {cs.id}: {e}")
                    call_detail["transcript_analysis"] = None
                    call_detail["transcript_message_count"] = 0
            else:
                call_detail["transcript_analysis"] = None
                call_detail["transcript_message_count"] = 0
            
            call_details.append(call_detail)
        
        # Analysis summary
        analysis = {
            "batch_id": batch_id,
            "user_email": user.email,
            "current_time": datetime.now(timezone.utc).isoformat(),  # Report generation time
            "total_scheduled": total_scheduled,  # Total calls scheduled in CSV
            "total_calls_made": total_calls_made,  # Actually made calls
            "called": called_count,  # Successfully called
            "failed": failed_count,  # Failed calls
            "pending": pending_count,  # Never called (still Pending)
            "successful_calls": called_count,  # Alias for compatibility
            "failed_calls": failed_count,  # Alias for compatibility
            "total_duration_seconds": total_duration,
            "total_duration_formatted": f"{total_duration // 60}m {total_duration % 60}s",
            "average_duration_seconds": int(avg_duration),
            "average_duration_formatted": f"{int(avg_duration) // 60}m {int(avg_duration) % 60}s",
            "success_rate_percent": round((called_count / total_calls_made * 100) if total_calls_made > 0 else 0, 2),
            "failure_rate_percent": round((failed_count / total_calls_made * 100) if total_calls_made > 0 else 0, 2),
            "total_cost": round(sum(float(cs.cost or 0) for cs in call_sessions), 2),
            "call_details": call_details
        }
        
        return create_success_response(analysis, "Batch analysis retrieved successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch analysis: {str(e)}")
 