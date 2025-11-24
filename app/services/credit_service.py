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
import time
from app.services.pricing_service import pricing_service

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
    DEDUCTION_INTERVAL = 1  # Deduct every 1 second (Vapi style - per-second deduction)
    
    def __init__(self):
        self._active_monitors = {}  # {call_session_id: task}
    
    def get_credit_cost_for_model(self, model_name: str) -> float:
        """
        Get credit cost per minute for a specific model using pricing_service
        Converts USD pricing to credits: $1 = 10 credits (1 credit = $0.10)
        
        Args:
            model_name: Name of the model
            
        Returns:
            Credits per minute (float for precision)
        """
        try:
            # Get pricing from pricing_service (includes LLM + Twilio)
            pricing = pricing_service.get_pricing_for_model(
                model_name=model_name,
                include_twilio=True,  # Include Twilio cost
                eleven_plan=None,  # TTS cost handled separately if needed
                tts_turbo=False
            )
            
            total_cost_per_minute_usd = pricing.get("total_cost_per_minute")
            
            if total_cost_per_minute_usd is None or total_cost_per_minute_usd == 0:
                # Fallback to default if model not found in pricing_service
                logger.warning(f"Model {model_name} not found in pricing_service, using default")
                return 10.0  # Default: 10 credits/min = $1/min
            
            # Convert USD to credits: $1 = 10 credits, so multiply by 10
            credits_per_minute = total_cost_per_minute_usd * 10.0
            
            logger.info(
                f"Model {model_name}: ${total_cost_per_minute_usd:.6f}/min = {credits_per_minute:.2f} credits/min"
            )
            
            return credits_per_minute
            
        except Exception as e:
            logger.error(f"Error getting pricing for model {model_name}: {e}")
            # Fallback to default
            return 10.0  # Default: 10 credits/min
    
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
        cost_per_minute = self.get_credit_cost_for_model(model_name)  # Now returns float
        required_credits = int(cost_per_minute * estimated_minutes)  # Convert to int for comparison
        
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
        # Get credit cost per minute from pricing_service (now returns float)
        credit_cost_per_minute = self.get_credit_cost_for_model(model_name)
        
        logger.info(
            f"Starting credit monitor for call {call_session_id}. "
            f"Model: {model_name}, Cost: {credit_cost_per_minute:.2f} credits/min "
            f"({credit_cost_per_minute/60.0:.4f} credits/sec)"
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
        db: Session,  # Initial session (will create new ones)
        call_session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        model_name: str,
        credit_cost_per_minute: float
    ):
        """
        Background task to monitor and deduct credits during call
        NOW WITH PER-SECOND DEDUCTION AND REAL-TIME UPDATES (Vapi Style)
        
        CRITICAL: Creates its own DB sessions to avoid session expiration
        
        Args:
            db: Database session (initial, will create new ones)
            call_session_id: Call session UUID
            tenant_id: Tenant UUID
            model_name: Model name
            credit_cost_per_minute: Credits to deduct per minute (float)
        """
        # Initialize variables before try block (accessible in finally)
        accumulated_credits = 0.0  # Fractional credits accumulated
        total_deducted = 0.0  # Total credits deducted (for summary)
        
        try:
            from app.routers.general_websocket import (
                broadcast_credit_update,
                broadcast_balance_zero,
                broadcast_call_summary,
                broadcast_topup_needed
            )
            from app.db.session import SessionLocal  # Import for new sessions
            
            # Calculate per-second credit cost (float for precision)
            per_second_credits = credit_cost_per_minute / 60.0  # e.g., 10/60 = 0.1667
            
            # Track fractional credits in memory (not in DB)
            last_topup_warning = 0.0  # Track last top-up warning to avoid spam
            
            logger.info(
                f"Starting per-second credit monitor for call {call_session_id}. "
                f"Model: {model_name}, Cost: {credit_cost_per_minute:.2f} credits/min "
                f"({per_second_credits:.4f} credits/sec)"
            )
            
            while True:
                # Wait for 1 second
                await asyncio.sleep(self.DEDUCTION_INTERVAL)
                
                # CRITICAL: Create new DB session for each iteration (avoids session expiration)
                db = SessionLocal()
                try:
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
                        # Call ended normally - send final summary
                        if call_session.status == "completed":
                            try:
                                if call_session.start_time and call_session.end_time:
                                    duration = (call_session.end_time - call_session.start_time).total_seconds()
                                else:
                                    duration = 0
                                
                                # Deduct any remaining accumulated credits
                                if accumulated_credits >= 1.0:
                                    credits_to_deduct = int(accumulated_credits)
                                    self.deduct_credits(
                                        db=db,
                                        tenant_id=tenant_id,
                                        amount=credits_to_deduct,
                                        call_session_id=call_session_id,
                                        description=f"Final call deduction - Model: {model_name}"
                                    )
                                    total_deducted += float(credits_to_deduct)
                                
                                await broadcast_call_summary(
                                    call_session_id=str(call_session_id),
                                    total_cost=round(total_deducted, 2),
                                    duration_sec=int(duration),
                                    components={
                                        "platform": round(total_deducted, 2),
                                        "model": model_name
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error sending final summary: {e}")
                        break
                    
                    # Get current balance from DB (integer)
                    current_db_credits = self.get_tenant_credits(db, tenant_id)
                    
                    # Calculate effective balance (DB credits - accumulated fractional)
                    effective_balance = float(current_db_credits) - accumulated_credits
                    
                    # Accumulate per-second credits
                    accumulated_credits += per_second_credits
                    total_deducted += per_second_credits
                    
                    # If accumulated >= 1 credit, deduct from DB
                    if accumulated_credits >= 1.0:
                        credits_to_deduct = int(accumulated_credits)  # Convert to int for DB
                        accumulated_credits -= float(credits_to_deduct)  # Keep remainder
                        
                        # Deduct from DB (using existing method - NO CHANGES to this method)
                        success, remaining_db_credits = self.deduct_credits(
                            db=db,
                            tenant_id=tenant_id,
                            amount=credits_to_deduct,  # Integer amount
                            call_session_id=call_session_id,
                            description=f"Call per-second accumulation - Model: {model_name}"
                        )
                        
                        if not success:
                            # Balance reached zero - end call
                            logger.warning(
                                f"Balance reached zero for call {call_session_id}. "
                                f"Total deducted: {total_deducted:.2f}"
                            )
                            
                            # Send balance zero event
                            try:
                                await broadcast_balance_zero(
                                    call_session_id=str(call_session_id),
                                    metadata={
                                        "total_deducted": round(total_deducted, 2),
                                        "model_name": model_name
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error broadcasting balance zero: {e}")
                            
                            # Update call session status
                            call_session.status = "completed"
                            call_session.end_time = datetime.now(timezone.utc)
                            call_session.ended_reason = "Insufficient credits"
                            
                            if call_session.start_time:
                                duration = (call_session.end_time - call_session.start_time).total_seconds()
                                call_session.duration = int(duration)
                            
                            db.commit()
                            
                            # Send final call summary
                            try:
                                await broadcast_call_summary(
                                    call_session_id=str(call_session_id),
                                    total_cost=round(total_deducted, 2),
                                    duration_sec=int(duration) if call_session.start_time else 0,
                                    components={
                                        "platform": round(total_deducted, 2),
                                        "model": model_name
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error broadcasting call summary: {e}")
                            
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
                        
                        # Update effective balance after DB deduction
                        effective_balance = float(remaining_db_credits) - accumulated_credits
                    else:
                        # No DB deduction yet, but update effective balance
                        effective_balance = float(current_db_credits) - accumulated_credits
                    
                    # Send real-time credit update via WebSocket (every second)
                    try:
                        await broadcast_credit_update(
                            call_session_id=str(call_session_id),
                            remaining_credits=effective_balance,  # Show fractional balance
                            metadata={
                                "per_second_cost": round(per_second_credits, 4),
                                "total_deducted": round(total_deducted, 4),
                                "accumulated": round(accumulated_credits, 4),
                                "db_credits": current_db_credits,
                                "model_name": model_name
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error broadcasting credit update: {e}")
                    
                    # Check if balance is low and send top-up reminder (once per 30 seconds to avoid spam)
                    now = time.time()
                    if effective_balance < 5.0 and effective_balance > 0 and (now - last_topup_warning) > 30:
                        try:
                            await broadcast_topup_needed(
                                call_session_id=str(call_session_id),
                                remaining_credits=effective_balance
                            )
                            last_topup_warning = now
                        except Exception as e:
                            logger.error(f"Error broadcasting top-up needed: {e}")
                    
                    logger.debug(
                        f"Call {call_session_id}: Accumulated {accumulated_credits:.4f}, "
                        f"Effective balance: {effective_balance:.4f}, "
                        f"Total deducted: {total_deducted:.4f}"
                    )
                
                finally:
                    # CRITICAL: Close DB session after each iteration
                    db.close()
        
        except Exception as e:
            logger.error(f"Error in credit monitoring for call {call_session_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final cleanup: Deduct any remaining accumulated credits
            try:
                db = SessionLocal()
                try:
                    call_session = db.query(CallSession).filter(
                        CallSession.id == call_session_id
                    ).first()
                    
                    if call_session and accumulated_credits >= 1.0:
                        credits_to_deduct = int(accumulated_credits)
                        self.deduct_credits(
                            db=db,
                            tenant_id=tenant_id,
                            amount=credits_to_deduct,
                            call_session_id=call_session_id,
                            description=f"Final accumulated credits - Model: {model_name}"
                        )
                        total_deducted += float(credits_to_deduct)
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Error in final cleanup: {e}")
            
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

