import stripe
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.models.subscription import Subscription
import uuid
from datetime import datetime

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeService:
    
    @staticmethod
    def create_customer(tenant: Tenant, email: str) -> str:
        """Create a Stripe customer for a tenant"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=tenant.name,
                metadata={
                    'tenant_id': str(tenant.id),
                    'tenant_name': tenant.name
                }
            )
            return customer.id
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create Stripe customer: {str(e)}")
    
    @staticmethod
    def create_checkout_session(
        tenant_id: str,
        plan_id: str,
        success_url: str,
        cancel_url: str,
        db: Session,
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a Stripe checkout session for subscription"""
        try:
            # Get plan details using the database session
            plan = db.query(Plan).filter(Plan.id == plan_id).first()
            if not plan:
                raise Exception("Plan not found")
            
            if not plan.stripe_price_id:
                raise Exception("No Stripe price ID configured for this plan")
            
            session_params = {
                'payment_method_types': ['card'],
                'line_items': [{
                    'price': plan.stripe_price_id,
                    'quantity': 1,
                }],
                'mode': 'subscription',
                'success_url': success_url,
                'cancel_url': cancel_url,
                'metadata': {
                    'tenant_id': tenant_id,
                    'plan_id': plan_id
                },
                'subscription_data': {
                    'metadata': {
                        'tenant_id': tenant_id,
                        'plan_id': plan_id
                    }
                }
            }
            
            # Only add customer if we have an existing customer_id
            # In subscription mode, customers are created automatically if not provided
            if customer_id:
                session_params['customer'] = customer_id
            
            session = stripe.checkout.Session.create(**session_params)
            return {
                'session_id': session.id,
                'url': session.url
            }
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create checkout session: {str(e)}")
    
    @staticmethod
    def create_portal_session(customer_id: str, return_url: str) -> Dict[str, Any]:
        """Create a Stripe customer portal session"""
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return {
                'url': session.url
            }
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create portal session: {str(e)}")
    
    @staticmethod
    def get_subscription(subscription_id: str) -> Dict[str, Any]:
        """Get subscription details from Stripe"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get subscription: {str(e)}")
    
    @staticmethod
    def cancel_subscription(subscription_id: str, at_period_end: bool = True) -> Dict[str, Any]:
        """Cancel a Stripe subscription"""
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to cancel subscription: {str(e)}")
    
    @staticmethod
    def update_subscription_plan(subscription_id: str, new_price_id: str) -> Dict[str, Any]:
        """Update subscription to a different plan"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            stripe.Subscription.modify(
                subscription_id,
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_price_id,
                }],
                proration_behavior='create_prorations'
            )
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to update subscription: {str(e)}")
    
    @staticmethod
    def construct_webhook_event(payload: bytes, sig_header: str) -> Dict[str, Any]:
        """Construct and verify webhook event"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
            return event
        except ValueError as e:
            raise Exception(f"Invalid payload: {str(e)}")
        except stripe.error.SignatureVerificationError as e:
            raise Exception(f"Invalid signature: {str(e)}")
    
    @staticmethod
    def handle_checkout_completed(event_data: Dict[str, Any], db: Session) -> None:
        """Handle checkout.session.completed event"""
        session = event_data['data']['object']
        tenant_id = session['metadata']['tenant_id']
        plan_id = session['metadata']['plan_id']
        customer_id = session['customer']
        subscription_id = session['subscription']
        
        # Get or create subscription record
        subscription = db.query(Subscription).filter(
            Subscription.tenant_id == tenant_id
        ).first()
        
        if not subscription:
            subscription = Subscription(
                tenant_id=tenant_id,
                plan_id=plan_id,
                stripe_subscription_id=subscription_id,
                stripe_customer_id=customer_id,
                status='active'
            )
            db.add(subscription)
        else:
            subscription.stripe_subscription_id = subscription_id
            subscription.stripe_customer_id = customer_id
            subscription.plan_id = plan_id
            subscription.status = 'active'
        
        db.commit()
    
    @staticmethod
    def handle_invoice_paid(event_data: Dict[str, Any], db: Session) -> None:
        """Handle invoice.paid event"""
        invoice = event_data['data']['object']
        subscription_id = invoice['subscription']
        
        if subscription_id:
            subscription = db.query(Subscription).filter(
                Subscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = 'active'
                db.commit()
    
    @staticmethod
    def handle_invoice_payment_failed(event_data: Dict[str, Any], db: Session) -> None:
        """Handle invoice.payment_failed event"""
        invoice = event_data['data']['object']
        subscription_id = invoice['subscription']
        
        if subscription_id:
            subscription = db.query(Subscription).filter(
                Subscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = 'past_due'
                db.commit()
    
    @staticmethod
    def handle_subscription_updated(event_data: Dict[str, Any], db: Session) -> None:
        """Handle customer.subscription.updated event"""
        stripe_subscription = event_data['data']['object']
        subscription_id = stripe_subscription['id']
        
        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = stripe_subscription['status']
            subscription.current_period_start = datetime.fromtimestamp(
                stripe_subscription['current_period_start']
            )
            subscription.current_period_end = datetime.fromtimestamp(
                stripe_subscription['current_period_end']
            )
            subscription.cancel_at_period_end = stripe_subscription['cancel_at_period_end']
            
            if stripe_subscription['canceled_at']:
                subscription.canceled_at = datetime.fromtimestamp(
                    stripe_subscription['canceled_at']
                )
            
            db.commit()
    
    @staticmethod
    def handle_subscription_deleted(event_data: Dict[str, Any], db: Session) -> None:
        """Handle customer.subscription.deleted event"""
        stripe_subscription = event_data['data']['object']
        subscription_id = stripe_subscription['id']
        
        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = 'canceled'
            subscription.canceled_at = datetime.now()
            db.commit()
