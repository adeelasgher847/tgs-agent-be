"""
Credit Service Module
Vapi-style real-time billing: Per-second accurate credit deduction
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
import math

logger = logging.getLogger(__name__)


class CreditService:
    """
    Service for managing tenant credits and call billing
    Vapi-style: Real-time per-second billing with accurate duration tracking
    """
    
    # ============================================================================
    # MODEL CREDIT COSTS (Per Minute) - Structured for easy addition
    # ============================================================================
    # Add new models here with their per-minute credit cost
    MODEL_CREDIT_COSTS_PER_MINUTE: Dict[str, int] = {
        # Gemini Models
        "gemini-2.0-flash": 8,
        "gemini-2.0-flash-exp": 8,
        "gemini-1.5-flash": 8,
        "gemini-1.5-pro": 12,
        
        # OpenAI Models
        "gpt-4o-mini": 12,
        "gpt-4o": 15,
        "gpt-4": 20,
        "gpt-3.5-turbo": 8,
        
        # Groq Models
        "llama-3.3-70b-versatile": 10,
        "llama-3.1-70b-versatile": 10,
        "llama-3.1-8b-instant": 8,
        
        # Default fallback
        "default": 10
    }
    
    # Monitoring interval (check every N seconds for real-time accuracy)
    MONITORING_INTERVAL = 10  # Check every 10 seconds (Vapi-style real-time)
    
    def __init__(self):
        self._active_monitors = {}  # {call_session_id: task}
        self._call_start_times = {}  # {call_session_id: start_time}
        self._last_deduction_time = {}  # {call_session_id: last_deduction_timestamp}
        self._accumulated_seconds = {}  # {call_session_id: accumulated_seconds}
    
    def get_credit_cost_per_minute(self, model_name: str) -> int:
        """
        Get credit cost per minute for a specific model
        
        Args:
            model_name: Name of the model (e.g., "gemini-2.0-flash", "gpt-4o-mini")
            
        Returns:
            Credits per minute
        """
        # Normalize model name (lowercase, strip whitespace)
        model_name = model_name.lower().strip()
        
        # Check for exact match first
        if model_name in self.MODEL_CREDIT_COSTS_PER_MINUTE:
            return self.MODEL_CREDIT_COSTS_PER_MINUTE[model_name]
        
        # Check for partial match (e.g., "gemini-2.0-flash" in "gemini-2.0-flash-thinking-exp")
        for key, cost in self.MODEL_CREDIT_COSTS_PER_MINUTE.items():
            if key != "default" and key in model_name:
                logger.info(f"Matched model '{model_name}' to '{key}' with cost {cost} credits/min")
                return cost
        
        # Return default cost if no match found
        logger.warning(f"Model '{model_name}' not found in pricing, using default cost")
        return self.MODEL_CREDIT_COSTS_PER_MINUTE["default"]
    
    def calculate_credits_for_duration(self, duration_seconds: float, credits_per_minute: int) -> int:
        """
        Calculate credits for a specific duration (Vapi-style: per-second with rounding)
        
        Args:
            duration_seconds: Duration in seconds (can be fractional)
            credits_per_minute: Credits per minute for the model
            
        Returns:
            Credits to deduct (rounded up to ensure fair billing)
        """
        if duration_seconds <= 0:
            return 0
        
        # Vapi-style: Calculate per-second cost
        credits_per_second = credits_per_minute / 60.0
        total_credits = duration_seconds * credits_per_second
        
        # Round up to nearest integer (industry standard - charge for partial seconds)
        return math.ceil(total_credits)
    
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
        
        Args:
            db: Database session
            tenant_id: Tenant UUID
            model_name: Name of the model being used
            estimated_minutes: Estimated call duration in minutes (for calculation)
            
        Returns:
            Tuple of (has_sufficient, current_credits, required_credits)
        """
        current_credits = self.get_tenant_credits(db, tenant_id)
        credits_per_minute = self.get_credit_cost_per_minute(model_name)
        required_credits = credits_per_minute * estimated_minutes
        
        # Require credits > 0 (call will end when reaching 0 during monitoring)
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
        
        # ✅ IMMEDIATE DATABASE UPDATE - Commit right away
        db.commit()
        db.refresh(tenant)
        
        # Check if credits reached 0 after deduction
        credits_exhausted = (new_credits == 0 and current_credits > 0)
        
        logger.info(
            f"✅ Deducted {amount} credits from tenant {tenant_id}. "
            f"Remaining: {tenant.credits} (updated in DB). "
            f"Call: {call_session_id}. "
            f"Description: {description}"
        )
        
        # Return success=False if credits exhausted (call should end immediately)
        return (not credits_exhausted, tenant.credits)
    
    async def start_credit_monitoring(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        agent_id: uuid.UUID
    ):
        """
        Start monitoring and deducting credits for an active call (Vapi-style real-time)
        
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
        credits_per_minute = self.get_credit_cost_per_minute(model_name)
        
        # Initialize tracking for this call
        call_start_time = datetime.now(timezone.utc)
        call_id_str = str(call_session_id)
        self._call_start_times[call_id_str] = call_start_time
        self._last_deduction_time[call_id_str] = call_start_time
        self._accumulated_seconds[call_id_str] = 0.0
        
        logger.info(
            f"Starting credit monitor for call {call_session_id}. "
            f"Model: {model_name}, Cost: {credits_per_minute} credits/min "
            f"(Vapi-style per-second billing)"
        )
        
        # Create monitoring task
        task = asyncio.create_task(
            self._monitor_and_deduct_credits(
                db, call_session_id, tenant_id, model_name, credits_per_minute
            )
        )
        self._active_monitors[call_id_str] = task
    
    async def _monitor_and_deduct_credits(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credits_per_minute: int
    ):
        """
        Background task to monitor and deduct credits during call
        Vapi-style: Real-time per-second billing with accurate duration tracking
        
        Args:
            db: Database session
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            model_name: Model name
            credits_per_minute: Credits per minute for the model
        """
        call_id_str = str(call_session_id)
        
        try:
            while True:
                # Wait for monitoring interval (real-time checks)
                await asyncio.sleep(self.MONITORING_INTERVAL)
                
                # Check if call is still active
                call_session = db.query(CallSession).filter(
                    CallSession.id == call_session_id
                ).first()
                
                if not call_session:
                    logger.info(f"Call session {call_session_id} not found, stopping monitor")
                    break
                
                # Only deduct for active call statuses
                if call_session.status not in ["active", "in-progress", "answered"]:
                    # Call ended - do final deduction for remaining time
                    await self._finalize_call_credits(
                        db, call_session_id, tenant_id, model_name, credits_per_minute, call_session
                    )
                    break
                
                # Calculate duration since last deduction
                current_time = datetime.now(timezone.utc)
                last_deduction = self._last_deduction_time.get(call_id_str)
                
                if not last_deduction:
                    last_deduction = self._call_start_times.get(call_id_str, current_time)
                    self._last_deduction_time[call_id_str] = last_deduction
                
                # Calculate seconds since last deduction
                elapsed_seconds = (current_time - last_deduction).total_seconds()
                
                if elapsed_seconds > 0:
                    # Check current credits before attempting deduction
                    current_tenant_credits = self.get_tenant_credits(db, tenant_id)
                    if current_tenant_credits <= 0:
                        # Credits already finished - end call immediately
                        logger.warning(
                            f"⛔ Credits already finished for call {call_session_id}. "
                            f"Current credits: {current_tenant_credits}. Ending call immediately."
                        )
                        call_session.status = "completed"
                        call_session.end_time = datetime.now(timezone.utc)
                        call_session.ended_reason = "Insufficient credits"
                        if call_session.start_time:
                            duration = (call_session.end_time - call_session.start_time).total_seconds()
                            call_session.duration = int(duration)
                        db.commit()
                        try:
                            await self._end_twilio_call(call_session.twilio_call_sid)
                        except Exception as e:
                            logger.error(f"Error ending Twilio call: {e}")
                        break
                    
                    # Accumulate seconds
                    accumulated = self._accumulated_seconds.get(call_id_str, 0.0)
                    accumulated += elapsed_seconds
                    self._accumulated_seconds[call_id_str] = accumulated
                    
                    # Calculate credits for accumulated time (Vapi-style: per-second)
                    credits_to_deduct = self.calculate_credits_for_duration(accumulated, credits_per_minute)
                    
                    if credits_to_deduct > 0:
                        # ✅ Deduct accumulated credits (updates DB immediately)
                        success, remaining_credits = self.deduct_credits(
                            db=db,
                            tenant_id=tenant_id,
                            amount=credits_to_deduct,
                            call_session_id=call_session_id,
                            description=f"Call duration: {accumulated:.1f}s - Model: {model_name}"
                        )
                        
                        if not success or remaining_credits <= 0:
                            # ✅ CREDITS FINISHED - END CALL IMMEDIATELY
                            logger.warning(
                                f"⛔ Credits finished for call {call_session_id}. "
                                f"Remaining credits: {remaining_credits}. Ending call immediately."
                            )
                            
                            # Update call session status immediately
                            call_session.status = "completed"
                            call_session.end_time = datetime.now(timezone.utc)
                            call_session.ended_reason = "Insufficient credits"
                            
                            if call_session.start_time:
                                duration = (call_session.end_time - call_session.start_time).total_seconds()
                                call_session.duration = int(duration)
                            
                            # ✅ IMMEDIATE DATABASE UPDATE
                            db.commit()
                            
                            # Try to end the Twilio call immediately
                            try:
                                await self._end_twilio_call(call_session.twilio_call_sid)
                                logger.info(f"✅ Twilio call ended immediately due to insufficient credits")
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
                                        "remaining_credits": remaining_credits,
                                        "timestamp": datetime.now(timezone.utc).isoformat()
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error broadcasting call end event: {e}")
                            
                            # Stop monitoring immediately
                            break
                        
                        # Reset accumulated time after successful deduction
                        self._accumulated_seconds[call_id_str] = 0.0
                        self._last_deduction_time[call_id_str] = current_time
                        
                        logger.info(
                            f"Call {call_session_id}: Deducted {credits_to_deduct} credits "
                            f"for {accumulated:.1f}s duration. Remaining: {remaining_credits}"
                        )
        
        except asyncio.CancelledError:
            logger.info(f"Credit monitor cancelled for call {call_session_id}")
        except Exception as e:
            logger.error(f"Error in credit monitoring for call {call_session_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final deduction for any remaining time
            try:
                call_session = db.query(CallSession).filter(
                    CallSession.id == call_session_id
                ).first()
                if call_session:
                    await self._finalize_call_credits(
                        db, call_session_id, tenant_id, model_name, credits_per_minute, call_session
                    )
            except Exception as e:
                logger.error(f"Error in final credit deduction: {e}")
            
            # Clean up monitor and tracking
            if call_id_str in self._active_monitors:
                del self._active_monitors[call_id_str]
            if call_id_str in self._call_start_times:
                del self._call_start_times[call_id_str]
            if call_id_str in self._last_deduction_time:
                del self._last_deduction_time[call_id_str]
            if call_id_str in self._accumulated_seconds:
                del self._accumulated_seconds[call_id_str]
            
            logger.info(f"Credit monitor stopped for call {call_session_id}")
    
    async def _finalize_call_credits(
        self,
        db: Session,
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credits_per_minute: int,
        call_session: CallSession
    ):
        """
        Finalize credits for call end - deduct any remaining accumulated time (Vapi-style)
        """
        call_id_str = str(call_session_id)
        accumulated = self._accumulated_seconds.get(call_id_str, 0.0)
        
        if accumulated > 0:
            # Calculate final credits for remaining time
            final_credits = self.calculate_credits_for_duration(accumulated, credits_per_minute)
            
            if final_credits > 0:
                success, remaining_credits = self.deduct_credits(
                    db=db,
                    tenant_id=tenant_id,
                    amount=final_credits,
                    call_session_id=call_session_id,
                    description=f"Final call duration: {accumulated:.1f}s - Model: {model_name}"
                )
                
                logger.info(
                    f"Call {call_session_id}: Final deduction of {final_credits} credits "
                    f"for {accumulated:.1f}s. Remaining: {remaining_credits}"
                )
    
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
