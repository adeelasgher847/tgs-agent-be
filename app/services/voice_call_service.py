from datetime import datetime, timezone
import uuid
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.call_session import CallSession
from app.models.user import User
from app.schemas.twilio import (
    CallInitiateRequest,
    CallInitiateResponse,
    CallInitiateErrorResponse,
)
from app.schemas.base import SuccessResponse
from app.services.agent_service import agent_service
from app.services.call_session_service import call_session_service
from app.services.credit_service import credit_service
from app.services.phone_number_service import phone_number_service
from app.services.twilio_service import twilio_service
from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async
from app.utils.response import create_success_response
from app.routers.general_websocket import broadcast_call_status_update


async def initiate_call(
    call_request: CallInitiateRequest,
    http_request: Request,
    user: Optional[User],
    db: Session,
) -> SuccessResponse[CallInitiateResponse] | JSONResponse:
    """
    Behavior-preserving extraction of the original `initiate_call` route logic.
    """
    try:
        # Verify authentication: either JWT token OR webhook secret
        is_webhook = await verify_n8n_webhook_secret_async(http_request)

        if is_webhook:
            # Webhook authentication - get tenant_id and user_id from request body
            if not call_request.tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tenant_id is required in request body when using webhook secret",
                )
            try:
                tenant_uuid = uuid.UUID(call_request.tenant_id)
                user_uuid = (
                    uuid.UUID(call_request.user_id)
                    if call_request.user_id
                    else None
                )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid UUID format for tenant_id or user_id",
                )
            tenant_id_filter = tenant_uuid
            user_id_filter = user_uuid
        else:
            # JWT authentication - get from user token
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: JWT token or n8n webhook secret",
                )
            tenant_id_filter = user.current_tenant_id
            user_id_filter = user.id

        # Validate agent exists in database
        try:
            agent_id = uuid.UUID(call_request.agentId)
            agent = agent_service.get_agent_by_id(db, agent_id, tenant_id_filter)
        except (ValueError, HTTPException):
            raise HTTPException(
                status_code=404, detail=f"Agent {call_request.agentId} not found"
            )

        # Validate phone number format
        if not twilio_service.validate_phone_number(call_request.userPhoneNumber):
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format. Must start with +",
            )

        # Check credits before initiating call
        if not agent.model:
            raise HTTPException(
                status_code=400, detail="Agent does not have a model configured"
            )

        model_name = agent.model.model_name
        has_sufficient, current_credits, required_credits = (
            credit_service.has_sufficient_credits(
                db=db,
                tenant_id=tenant_id_filter,
                model_name=model_name,
                estimated_minutes=1,  # Check for at least 1 minute
            )
        )

        if not has_sufficient:
            logger.warning(
                "❌ Insufficient credits: %s < %s", current_credits, required_credits
            )
            error_message = (
                "Insufficient credits to initiate call. Current balance: "
                f"{current_credits} credits, Required: {required_credits} credits. "
                f"Model: {model_name}"
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=error_message,
            )

        logger.info(
            "✅ Credit check passed: %s credits available, %s required for model %s",
            current_credits,
            required_credits,
            model_name,
        )

        # Get phone number and credentials - Priority: User Selected > Agent Assigned > Env
        from app.models.phone_number import PhoneNumber

        phone_number_obj = None
        from_number: Optional[str] = None
        use_custom_credentials = False
        account_sid: Optional[str] = None
        auth_token: Optional[str] = None

        # Priority 1: Check if user explicitly selected a phone number (VAPI style)
        if call_request.phone_number_id:
            try:
                phone_number_uuid = uuid.UUID(call_request.phone_number_id)
                phone_number_obj = phone_number_service.get_phone_number_by_id(
                    db=db,
                    phone_number_id=phone_number_uuid,
                    tenant_id=tenant_id_filter,
                )
                if phone_number_obj and phone_number_obj.status == "active":
                    logger.info(
                        "✅ Using user selected phone number: %s (ID: %s)",
                        phone_number_obj.phone_number,
                        phone_number_uuid,
                    )
                elif phone_number_obj and phone_number_obj.status != "active":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Phone number {call_request.phone_number_id} "
                            "is not active."
                        ),
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=(
                            f"Phone number {call_request.phone_number_id} "
                            "not found in your account."
                        ),
                    )
            except HTTPException:
                raise
            except (ValueError, Exception) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid phone_number_id format: {str(e)}",
                )

        # Priority 2: Check if agent has assigned phone number in DB
        if not phone_number_obj and agent.id:
            phone_number_obj = (
                db.query(PhoneNumber)
                .filter(
                    PhoneNumber.assistant_id == agent.id,
                    PhoneNumber.tenant_id == tenant_id_filter,
                    PhoneNumber.status == "active",
                )
                .first()
            )
            if phone_number_obj:
                logger.info(
                    "✅ Using agent's assigned phone number: %s",
                    phone_number_obj.phone_number,
                )

        # Use selected phone number with credentials if available
        if (
            phone_number_obj
            and phone_number_obj.twilio_account_sid
            and phone_number_obj.twilio_auth_token
        ):
            from_number = phone_number_obj.phone_number
            from app.core.security import decrypt_api_key

            account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
            auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
            use_custom_credentials = True
            logger.info(
                "✅ Using DB phone number: %s with custom credentials", from_number
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No phone number found. Please create and assign a phone number "
                    "in your account before making calls."
                ),
            )

        # Get base URL for webhooks
        base_url = settings.WEBHOOK_BASE_URL

        # Create call session first so we can include the ID in webhook URLs
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user_id_filter,
            agent_id=agent.id,
            tenant_id=tenant_id_filter,
            twilio_call_sid="",  # Will be updated after call is made
            from_number=from_number,
            to_number=call_request.userPhoneNumber,
            call_type="outbound",
        )

        webhook_url = (
            f"{base_url}/api/v1/voice/gather/streaming?"
            f"agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        )
        status_callback_url = (
            f"{base_url}/api/v1/voice/webhook/call-events?"
            f"agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        )

        logger.info("Making call with webhook_url: %s", webhook_url)
        logger.info("Making call with status_callback_url: %s", status_callback_url)

        # Optional WebSocket broadcast
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiating",
                metadata={
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.userPhoneNumber,
                    "from_number": from_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info("✅ WebSocket: Call initiating event sent")
        except Exception as e:  # pragma: no cover - non-critical
            logger.warning("⚠️ WebSocket broadcast failed (non-critical): %s", e)

        # Make call with appropriate credentials
        if use_custom_credentials:
            call = twilio_service.make_call_with_credentials(
                to_number=call_request.userPhoneNumber,
                from_number=from_number,
                webhook_url=webhook_url,
                status_callback_url=status_callback_url,
                account_sid=account_sid,
                auth_token=auth_token,
            )
        else:
            call = twilio_service.make_call(
                to_number=call_request.userPhoneNumber,
                from_number=from_number,
                webhook_url=webhook_url,
                status_callback_url=status_callback_url,
            )
        logger.info("✅ Call initiated successfully")

        # Update call session with Twilio SID
        call_session.twilio_call_sid = call.sid
        db.commit()
        logger.info(
            "✅ Updated call session %s with Twilio SID: %s",
            call_session.id,
            call.sid,
        )

        # Broadcast call initiated event AFTER Twilio confirms
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiated",
                metadata={
                    "call_sid": call.sid,
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.userPhoneNumber,
                    "from_number": from_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(
                "✅ Call initiated event sent for session %s", call_session.id
            )
        except Exception as e:  # pragma: no cover - non-critical
            logger.warning("⚠️ Failed to send call initiated event: %s", e)

        # Generate call ID
        call_id = f"call_{call.sid[-8:]}"

        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = (
            call_request.call_session_id_field_id
            or call_request.call_session_id_column_id
        )

        return create_success_response(
            CallInitiateResponse(
                callId=call_id,
                twilioCallSid=call.sid,
                callSessionId=str(call_session.id),
                status="initiated",
                board_id=call_request.board_id,
                monday_item_id=call_request.monday_item_id,
                status_column_id=call_request.status_column_id,
                call_session_id_column_id=call_request.call_session_id_column_id,
                crm_container_id=crm_container_id,
                crm_item_id=crm_item_id,
                status_field_id=status_field_id,
                call_session_id_field_id=call_session_id_field_id,
                crm_type=call_request.crm_type,
            ),
            "Call initiated successfully",
        )

    except HTTPException as e:
        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = (
            call_request.call_session_id_field_id
            or call_request.call_session_id_column_id
        )

        error_response = CallInitiateErrorResponse(
            detail=e.detail,
            board_id=call_request.board_id,
            monday_item_id=call_request.monday_item_id,
            status_column_id=call_request.status_column_id,
            call_session_id_column_id=call_request.call_session_id_column_id,
            crm_container_id=crm_container_id,
            crm_item_id=crm_item_id,
            status_field_id=status_field_id,
            call_session_id_field_id=call_session_id_field_id,
            crm_type=call_request.crm_type,
        )
        return JSONResponse(
            status_code=e.status_code,
            content=error_response.dict(exclude_none=True),
        )
    except Exception as e:  # pragma: no cover - defensive
        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = (
            call_request.call_session_id_field_id
            or call_request.call_session_id_column_id
        )

        error_response = CallInitiateErrorResponse(
            detail=str(e),
            board_id=call_request.board_id,
            monday_item_id=call_request.monday_item_id,
            status_column_id=call_request.status_column_id,
            call_session_id_column_id=call_request.call_session_id_column_id,
            crm_container_id=crm_container_id,
            crm_item_id=crm_item_id,
            status_field_id=status_field_id,
            call_session_id_field_id=call_session_id_field_id,
            crm_type=call_request.crm_type,
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict(exclude_none=True),
        )

