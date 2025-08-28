#!/usr/bin/env python3
"""
Test script to initiate a Twilio call and verify webhook functionality
"""

from twilio.rest import Client
from app.core.config import settings

def test_twilio_call():
    # Initialize Twilio client
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    
    try:
        # Make a test call
        call = client.calls.create(
            to='+1234567890',  # Replace with your test number
            from_=settings.TWILIO_PHONE_NUMBER,
            url='https://0141aa697006.ngrok-free.app/voice/webhook/voice-init?agentId=test-agent',
            status_callback='https://0141aa697006.ngrok-free.app/api/v1/voice/webhook/call-events',
            status_callback_method='POST',
            status_callback_event=['initiated', 'ringing', 'answered', 'completed']
        )
        
        print(f"Call initiated successfully!")
        print(f"Call SID: {call.sid}")
        print(f"Call Status: {call.status}")
        print(f"From: {call.from_}")
        print(f"To: {call.to}")
        
        return call.sid
        
    except Exception as e:
        print(f"Error initiating call: {e}")
        return None

if __name__ == "__main__":
    test_twilio_call()
