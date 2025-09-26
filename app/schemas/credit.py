from pydantic import BaseModel, Field
from typing import Optional
import uuid

class CreditBalanceResponse(BaseModel):
    """Response model for credit balance information"""
    tenant_id: uuid.UUID
    credit_balance: int
    plan_credits: int  # Credits from current plan
    plan_name: str

class CreditUsageRequest(BaseModel):
    """Request model for using credits"""
    amount: int = Field(..., ge=1, description="Number of credits to use")
    description: Optional[str] = Field(None, description="Description of the usage")

class CreditPurchaseRequest(BaseModel):
    """Request model for purchasing credits"""
    amount: int = Field(..., ge=1, description="Number of credits to purchase")
    payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")
