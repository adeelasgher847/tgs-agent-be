"""
Credit Service Module
Handles credit deduction, validation, and monitoring for voice calls
"""

from sqlalchemy.orm import Session
from app.models.tenant import Tenant
from app.models.agent import Agent
from app.models.call_session import CallSession
from typing import Optional, Dict, Any
import uuid
import asyncio
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class CreditService:
    """Service for managing tenant credits and call billing"""
    
    # Credit costs per minute based on model (deducted every 30 seconds)
    MODEL_CREDIT_COSTS = {
        "gemini-2.0-flash": 4,  # 8 credits per minute, 4 every 30 seconds
        "gemini-2.0-flash-exp": 4,  # 8 credits per minute, 4 every 30 seconds
        "gpt-4o-mini": 5,  # 10 credits per minute, 5 every 30 seconds
        "gpt-4o": 7,  # 15 credits per minute, 7 every 30 seconds (rounded down)
        "gpt-4": 10,  # 20 credits per minute, 10 every 30 seconds
        "llama-3.3-70b-versatile": 6,  # 12 credits per minute, 6 every 30 seconds
        "default": 5  # 10 credits per minute, 5 every 30 seconds
    }
    
    # How often to deduct credits (in seconds)
    DEDUCTION_INTERVAL = 30  # Deduct every 30 seconds (changed from 60)
    
    def __init__(self):
        self._active_monitors = {}  # {call_session_id: task}
    
    def get_credit_cost_for_model(self, model_name: str) -> int:
        """
        Get credit cost per 30-second interval for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Credits per 30 seconds (half of per-minute cost)
        """
        # Check for exact match first
        if model_name in self.MODEL_CREDIT_COSTS:
            return self.MODEL_CREDIT_COSTS[model_name]
        
        # Check for partial match (e.g., "gemini-2.0-flash" in "gemini-2.0-flash-thinking-exp")
        for key, cost in self.MODEL_CREDIT_COSTS.items():
            if key.lower() in model_name.lower():
                return cost
        
        # Return default cost
        return self.MODEL_CREDIT_COSTS["default"]
    
    def get_tenant_credits(self, db: Session, tenant_id: uuid.UUID) -> int:
        """
        Get current credit balance for a tenant
        
        Args:
            db: Database session
            tenant_id: Tenant UUID
            
        Returns:
            Current credit balance
        """
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return 0
        return tenant.credits or 0
    
    def has_sufficient_credits(
        self, 
        db: Session, 
        tenant_id: uuid.UUID, 
        model_name: str,
        estimated_minutes: int = 1
    ) -> tuple[bool, int, int]:
        """
        Check if tenant has sufficient credits for a call
        Now requires credits > 0 (not >= required amount for 1 minute)
        
        Args:
            db: Database session
            tenant_id: Tenant UUID
            model_name: Name of the model being used
            estimated_minutes: Estimated call duration in minutes (used for calculation)
            
        Returns:
            Tuple of (has_sufficient, current_credits, required_credits)
        """
        current_credits = self.get_tenant_credits(db, tenant_id)
        cost_per_minute = self.get_credit_cost_for_model(model_name)
        required_credits = cost_per_minute * estimated_minutes
        
        # Change: Only require credits > 0 (call will end when reaching 0 during monitoring)
        return (current_credits > 0, current_credits, required_credits)
    
    def deduct_credits(
        self, 
        db: Session, 
        tenant_id: uuid.UUID, 
        amount: int,
        call_session_id: Optional[uuid.UUID] = None,
        description: str = None
    ) -> tuple[bool, int]:
        """
        Deduct credits from tenant account
        
        Args:
            db: Database session
            tenant_id: Tenant UUID
            amount: Amount of credits to deduct
            call_session_id: Optional call session ID for tracking
            description: Optional description of the deduction
            
        Returns:
            Tuple of (success, remaining_credits)
        """
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            logger.error(f"Tenant {tenant_id} not found")
            return (False, 0)
        
        current_credits = tenant.credits or 0
        
        # Check if we have sufficient credits to deduct
        if current_credits <= 0:
            logger.warning(f"Insufficient credits for tenant {tenant_id}: {current_credits} <= 0")
            return (False, current_credits)
        
        # Deduct credits but never allow negative balance
        new_credits = max(0, current_credits - amount)
        tenant.credits = new_credits
        db.commit()
        db.refresh(tenant)
        
        # If credits reached 0, signal that deduction was the last one
        credits_exhausted = (new_credits == 0 and current_credits > 0)
        
        logger.info(
            f"Deducted {amount} credits from tenant {tenant_id}. "
            f"Remaining: {tenant.credits}. "
            f"Call: {call_session_id}. "
            f"Description: {description}"
        )
        
        # Return success=True if credits available, False if exhausted
        return (not credits_exhausted, tenant.credits)
    
    async def start_credit_monitoring(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        agent_id: uuid.UUID
    ):
        """
        Start monitoring and deducting credits for an active call
        
        Args:
            db: Database session
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            agent_id: Agent UUID
        """
        if str(call_session_id) in self._active_monitors:
            logger.warning(f"Credit monitor already active for call {call_session_id}")
            return
        
        # Get agent and model information
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent or not agent.model:
            logger.error(f"Agent {agent_id} or model not found")
            return
        
        model_name = agent.model.model_name
        credit_cost_per_minute = self.get_credit_cost_for_model(model_name)
        
        logger.info(
            f"Starting credit monitor for call {call_session_id}. "
            f"Model: {model_name}, Cost: {credit_cost_per_minute} credits/min"
        )
        
        # Create monitoring task
        task = asyncio.create_task(
            self._monitor_and_deduct_credits(
                db, call_session_id, tenant_id, model_name, credit_cost_per_minute
            )
        )
        self._active_monitors[str(call_session_id)] = task
    
    async def _monitor_and_deduct_credits(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credit_cost_per_minute: int
    ):
        """
        Background task to monitor and deduct credits during call
        
        Args:
            db: Database session
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            model_name: Model name
            credit_cost_per_minute: Credits to deduct per minute
        """
        try:
            deduction_count = 0
            
            while True:
                # Wait for the deduction interval
                await asyncio.sleep(self.DEDUCTION_INTERVAL)
                
                # Check if call is still active
                call_session = db.query(CallSession).filter(
                    CallSession.id == call_session_id
                ).first()
                
                if not call_session:
                    logger.info(f"Call session {call_session_id} not found, stopping monitor")
                    break
                
                # ✅ Allow credits to deduct for "answered", "active", and "in-progress" statuses
                if call_session.status not in ["active", "in-progress", "answered"]:
                    logger.info(f"Call {call_session_id} status is {call_session.status}, stopping monitor")
                    break
                
                # Deduct credits (every 30 seconds)
                deduction_count += 1
                success, remaining_credits = self.deduct_credits(
                    db=db,
                    tenant_id=tenant_id,
                    amount=credit_cost_per_minute,
                    call_session_id=call_session_id,
                    description=f"Call 30s interval {deduction_count} - Model: {model_name}"
                )
                
                if not success:
                    # Insufficient credits - end the call
                    logger.warning(
                        f"Insufficient credits for call {call_session_id}. "
                        f"Ending call. Remaining credits: {remaining_credits}"
                    )
                    
                    # Update call session status
                    call_session.status = "completed"
                    call_session.end_time = datetime.now(timezone.utc)
                    call_session.ended_reason = "Insufficient credits"
                    
                    if call_session.start_time:
                        duration = (call_session.end_time - call_session.start_time).total_seconds()
                        call_session.duration = int(duration)
                    
                    db.commit()
                    
                    # Try to end the Twilio call
                    try:
                        await self._end_twilio_call(call_session.twilio_call_sid)
                    except Exception as e:
                        logger.error(f"Error ending Twilio call: {e}")
                    
                    # Broadcast call ended event
                    try:
                        from app.routers.general_websocket import broadcast_call_status_update
                        await broadcast_call_status_update(
                            call_session_id=str(call_session_id),
                            status="completed",
                            metadata={
                                "reason": "insufficient_credits",
                                "message": "Call ended due to insufficient credits",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error broadcasting call end event: {e}")
                    
                    break
                
                logger.info(
                    f"Call {call_session_id}: Deducted {credit_cost_per_minute} credits. "
                    f"Remaining: {remaining_credits}"
                )
        
        except Exception as e:
            logger.error(f"Error in credit monitoring for call {call_session_id}: {e}")
        
        finally:
            # Clean up monitor
            if str(call_session_id) in self._active_monitors:
                del self._active_monitors[str(call_session_id)]
            logger.info(f"Credit monitor stopped for call {call_session_id}")
    
    async def _end_twilio_call(self, twilio_call_sid: str):
        """
        End a Twilio call
        
        Args:
            twilio_call_sid: Twilio call SID
        """
        if not twilio_call_sid:
            return
        
        try:
            from app.services.twilio_service import twilio_service
            twilio_service.end_call(twilio_call_sid)
            logger.info(f"Successfully ended Twilio call {twilio_call_sid}")
        except Exception as e:
            logger.error(f"Error ending Twilio call {twilio_call_sid}: {e}")
    
    def stop_credit_monitoring(self, call_session_id: uuid.UUID):
        """
        Stop monitoring credits for a call
        
        Args:
            call_session_id: Call session UUID
        """
        call_id_str = str(call_session_id)
        if call_id_str in self._active_monitors:
            task = self._active_monitors[call_id_str]
            task.cancel()
            del self._active_monitors[call_id_str]
            logger.info(f"Stopped credit monitor for call {call_session_id}")
    
    def get_active_monitors(self) -> Dict[str, Any]:
        """Get list of active credit monitors"""
        return {
            call_id: {"active": not task.done()}
            for call_id, task in self._active_monitors.items()
        }


# Global instance
credit_service = CreditService()

