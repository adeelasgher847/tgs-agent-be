import stripe
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.user import User
import uuid
from datetime import datetime, timedelta

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeService:
    
    @staticmethod
    def create_customer(tenant: Tenant, email: str, user: Optional[User] = None) -> str:
        """Create a Stripe customer for a tenant"""
        try:
            customer_data = {
                'email': email,
                'name': tenant.name,
                'metadata': {
                    'tenant_id': str(tenant.id),
                    'tenant_name': tenant.name
                }
            }
            
            # Add user information if available
            if user:
                customer_data['metadata']['user_id'] = str(user.id)
                customer_data['metadata']['user_name'] = f"{user.first_name} {user.last_name}".strip()
            
            customer = stripe.Customer.create(**customer_data)
            return customer.id
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create Stripe customer: {str(e)}")
    
    @staticmethod
    def get_customer(customer_id: str) -> Dict[str, Any]:
        """Get customer details from Stripe"""
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return customer
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get customer: {str(e)}")
    
    @staticmethod
    def update_customer(customer_id: str, **kwargs) -> Dict[str, Any]:
        """Update customer information"""
        try:
            customer = stripe.Customer.modify(customer_id, **kwargs)
            return customer
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to update customer: {str(e)}")
    
    @staticmethod
    def get_customer_payment_methods(customer_id: str) -> List[Dict[str, Any]]:
        """Get all payment methods for a customer"""
        try:
            payment_methods = stripe.PaymentMethod.list(
                customer=customer_id,
                type='card'
            )
            return payment_methods.data
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get payment methods: {str(e)}")
    
    @staticmethod
    def set_default_payment_method(customer_id: str, payment_method_id: str) -> Dict[str, Any]:
        """Set default payment method for a customer"""
        try:
            customer = stripe.Customer.modify(
                customer_id,
                invoice_settings={
                    'default_payment_method': payment_method_id
                }
            )
            return customer
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to set default payment method: {str(e)}")
    
    @staticmethod
    def detach_payment_method(payment_method_id: str) -> Dict[str, Any]:
        """Detach a payment method from a customer"""
        try:
            payment_method = stripe.PaymentMethod.detach(payment_method_id)
            return payment_method
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to detach payment method: {str(e)}")
    
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
                'mode': 'payment',
                'success_url': f"{success_url}?session_id={{CHECKOUT_SESSION_ID}}",
                'cancel_url': cancel_url,
                'metadata': {
                    'tenant_id': tenant_id,
                    'plan_id': plan_id
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
    def pause_subscription(subscription_id: str) -> Dict[str, Any]:
        """Pause a subscription"""
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                pause_collection={
                    'behavior': 'void'
                }
            )
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to pause subscription: {str(e)}")
    
    @staticmethod
    def resume_subscription(subscription_id: str) -> Dict[str, Any]:
        """Resume a paused subscription"""
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                pause_collection=None
            )
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to resume subscription: {str(e)}")
    
    @staticmethod
    def get_upcoming_invoice(customer_id: str) -> Dict[str, Any]:
        """Get upcoming invoice for a customer"""
        try:
            invoice = stripe.Invoice.upcoming(customer=customer_id)
            return invoice
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get upcoming invoice: {str(e)}")
    
    @staticmethod
    def get_invoices(customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent invoices for a customer"""
        try:
            invoices = stripe.Invoice.list(
                customer=customer_id,
                limit=limit
            )
            return invoices.data
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get invoices: {str(e)}")
    
    @staticmethod
    def create_usage_record(subscription_item_id: str, quantity: int, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Create a usage record for metered billing"""
        try:
            usage_record = stripe.UsageRecord.create(
                subscription_item=subscription_item_id,
                quantity=quantity,
                timestamp=timestamp or int(datetime.now().timestamp())
            )
            return usage_record
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create usage record: {str(e)}")
    
    @staticmethod
    def get_usage_records(subscription_item_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get usage records for a subscription item"""
        try:
            usage_records = stripe.UsageRecord.list(
                subscription_item=subscription_item_id,
                limit=limit
            )
            return usage_records.data
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to get usage records: {str(e)}")
    
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
        tenant_id = session.get("metadata", {}).get("tenant_id")
        user_id = session.get("metadata", {}).get("user_id")
        user_email = session.get("metadata", {}).get("user_email")
        plan_id = session.get("metadata", {}).get("plan_id")
        amount = float(session.get("metadata", {}).get("amount", 0))
        purchase_type = session.get("metadata", {}).get("purchase_type", "credit_purchase")

        print(f"STRIPE WEBHOOK DEBUG: Processing checkout completed")
        print(f"STRIPE WEBHOOK DEBUG: Tenant ID: {tenant_id}")
        print(f"STRIPE WEBHOOK DEBUG: User ID: {user_id}")
        print(f"STRIPE WEBHOOK DEBUG: User Email: {user_email}")
        print(f"STRIPE WEBHOOK DEBUG: Plan ID: {plan_id}")
        print(f"STRIPE WEBHOOK DEBUG: Amount: {amount}")
        print(f"STRIPE WEBHOOK DEBUG: Purchase type: {purchase_type}")
        print(f"STRIPE WEBHOOK DEBUG: Session ID: {session.get('id')}")
        print(f"STRIPE WEBHOOK DEBUG: Payment status: {session.get('payment_status')}")

        if not tenant_id:
            print("STRIPE WEBHOOK ERROR: Tenant ID not found in checkout session metadata")
            return

        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            print(f"STRIPE WEBHOOK ERROR: Tenant with ID {tenant_id} not found")
            return

        # Get user who made the payment
        user = None
        if user_id:
            from app.models.user import User
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                print(f"STRIPE WEBHOOK DEBUG: Found user - ID: {user.id}, Email: {user.email}")
            else:
                print(f"STRIPE WEBHOOK WARNING: User with ID {user_id} not found")

        print(f"STRIPE WEBHOOK DEBUG: Found tenant - ID: {tenant.id}")
        print(f"STRIPE WEBHOOK DEBUG: Current credits: {tenant.credits}")
        print(f"STRIPE WEBHOOK DEBUG: Current status: {tenant.status}")

        # Check if payment is actually completed
        if session.get('payment_status') != 'paid':
            print(f"STRIPE WEBHOOK DEBUG: Payment not completed, status: {session.get('payment_status')}")
            return

        # Calculate credits based on payment amount: $1 = 10 credits
        # Add credits for both credit purchases and plan purchases
        credits_to_add = int(amount * 10)
        tenant.credits = (tenant.credits or 0) + credits_to_add
        print(f"STRIPE WEBHOOK DEBUG: Added {credits_to_add} credits to tenant {tenant.id} (purchase type: {purchase_type})")
        
        # Update tenant status to active
        tenant.status = 'active'
        
        # Update subscription ID if available (for subscription purchases)
        if session.get('subscription'):
            tenant.stripe_subscription_id = session.get('subscription')
            print(f"STRIPE WEBHOOK DEBUG: Updated subscription ID: {session.get('subscription')}")
        
        # Update customer ID if available (for both credit and subscription purchases)
        if session.get('customer'):
            tenant.stripe_customer_id = session.get('customer')
            print(f"STRIPE WEBHOOK DEBUG: Updated customer ID: {session.get('customer')}")
        
        # For plan purchases, create or update subscription record
        if purchase_type == 'plan_purchase' and plan_id:
            # Get plan details
            plan = db.query(Plan).filter(Plan.id == plan_id).first()
            if plan:
                print(f"STRIPE WEBHOOK DEBUG: Processing plan purchase for: {plan.name}")
                
                # Create or update subscription record
                from app.models.subscription import Subscription
                existing_subscription = db.query(Subscription).filter(
                    Subscription.tenant_id == tenant_id
                ).first()
                
                if existing_subscription:
                    # Update existing subscription
                    existing_subscription.plan_id = plan_id
                    existing_subscription.stripe_customer_id = session.get('customer')
                    existing_subscription.stripe_session_id = session.get('id')
                    existing_subscription.status = 'paid'  # Set status to paid
                    existing_subscription.current_period_start = datetime.now()
                    existing_subscription.current_period_end = datetime.now() + timedelta(days=30)  # 30 days from now
                    print(f"STRIPE WEBHOOK DEBUG: Updated existing subscription for tenant {tenant_id} with status 'paid'")
                else:
                    # Create new subscription
                    new_subscription = Subscription(
                        tenant_id=tenant_id,
                        plan_id=plan_id,
                        stripe_customer_id=session.get('customer'),
                        stripe_session_id=session.get('id'),
                        status='paid',  # Set status to paid
                        current_period_start=datetime.now(),
                        current_period_end=datetime.now() + timedelta(days=30)  # 30 days from now
                    )
                    db.add(new_subscription)
                    print(f"STRIPE WEBHOOK DEBUG: Created new subscription for tenant {tenant_id} with status 'paid'")

        # Update user information if available
        if user:
            # You can add user-specific updates here
            # For example, update user's last payment date, total spent, etc.
            print(f"STRIPE WEBHOOK DEBUG: Updating user {user.id} information")
            
            # Example: Update user's last payment information
            # user.last_payment_date = datetime.now()
            # user.total_payments = (user.total_payments or 0) + amount
            # You can add more user-specific fields as needed
            
            print(f"STRIPE WEBHOOK DEBUG: User {user.id} updated successfully")

        # Commit changes
        db.commit()
        db.refresh(tenant)

        print(f"STRIPE WEBHOOK SUCCESS: Updated tenant {tenant.id}")
        print(f"STRIPE WEBHOOK SUCCESS: New credits: {tenant.credits}")
        print(f"STRIPE WEBHOOK SUCCESS: New status: {tenant.status}")
        print(f"STRIPE WEBHOOK SUCCESS: Subscription ID: {tenant.stripe_subscription_id}")
        
        # Log subscription details if created/updated
        if purchase_type == 'plan_purchase' and plan_id:
            subscription = db.query(Subscription).filter(Subscription.tenant_id == tenant_id).first()
            if subscription:
                print(f"STRIPE WEBHOOK SUCCESS: Subscription created/updated - Plan ID: {subscription.plan_id}, Status: {subscription.status}")
                print(f"STRIPE WEBHOOK SUCCESS: Subscription period: {subscription.current_period_start} to {subscription.current_period_end}")
        
        if user:
            print(f"STRIPE WEBHOOK SUCCESS: User {user.id} ({user.email}) payment processed")
        else:
            print(f"STRIPE WEBHOOK SUCCESS: Payment processed (user info not available)")

    # Note: Idempotency is now handled at the endpoint level using in-memory store in deps