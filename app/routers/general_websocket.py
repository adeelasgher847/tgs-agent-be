"""
General WebSocket Router
Real-time events for general application monitoring - allows frontend to receive live updates
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
import uuid
import json
import asyncio
from datetime import datetime, timezone

from app.api.deps import get_db, require_tenant, get_current_user_jwt
from app.models.user import User
from app.models.call_session import CallSession
from app.services.call_session_service import call_session_service

router = APIRouter(
    tags=["General WebSocket"],
    responses={404: {"description": "Not found"}},
)

class GeneralWebSocketManager:
    """Manages WebSocket connections for general application events"""
    
    def __init__(self):
        # List of all active WebSocket connections
        self.active_connections: List[WebSocket] = []
        # Map of websocket -> user info for cleanup
        self.websocket_to_user: Dict[WebSocket, Dict[str, Any]] = {}
        # Map of websocket -> subscribed event types
        self.websocket_subscriptions: Dict[WebSocket, List[str]] = {}
        # Global metadata
        self.global_metadata: Dict[str, Any] = {
            "total_connections": 0,
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
    
    async def connect(self, websocket: WebSocket, user_id: str = None, user_info: Dict[str, Any] = None):
        """Connect a WebSocket for general event monitoring"""
        await websocket.accept()
        
        # Add to active connections
        self.active_connections.append(websocket)
        
        # Store user info
        user_data = {
            "user_id": user_id,
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "user_info": user_info or {}
        }
        self.websocket_to_user[websocket] = user_data
        
        # Initialize subscriptions (default to all events)
        self.websocket_subscriptions[websocket] = ["all"]
        
        # Update global metadata
        self.global_metadata["total_connections"] = len(self.active_connections)
        self.global_metadata["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        print(f"✅ WebSocket connected for general monitoring. Total connections: {len(self.active_connections)}")
        print(f"✅ WebSocket manager ID: {id(self)}")
        
        # Send initial connection confirmation
        await self.send_to_websocket(websocket, {
            "type": "connection_established",
            "message": "Connected to general event monitoring",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_info": user_data,
            "global_metadata": self.global_metadata,
            "available_event_types": [
                "call_status_update",
                "transcript_update", 
                "call_ended",
                "call_event",
                "system_notification"
            ]
        })
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.websocket_to_user:
            del self.websocket_to_user[websocket]
        
        if websocket in self.websocket_subscriptions:
            del self.websocket_subscriptions[websocket]
        
        # Update global metadata
        self.global_metadata["total_connections"] = len(self.active_connections)
        self.global_metadata["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_websocket(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"❌ Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: dict, event_type: str = None):
        """Broadcast a message to all connected WebSockets"""
        print(f"📡 Broadcasting to {len(self.active_connections)} connections")
        print(f"📡 Message type: {message.get('type', 'unknown')}")
        print(f"📡 Event type: {event_type}")
        print(f"📡 WebSocket manager ID: {id(self)}")
        
        if len(self.active_connections) == 0:
            print(f"⚠️ No active WebSocket connections to broadcast to!")
            return
        
        disconnected_websockets = []
        messages_sent = 0
        
        for websocket in self.active_connections:
            try:
                # Check if websocket is subscribed to this event type
                subscriptions = self.websocket_subscriptions.get(websocket, ["all"])
                print(f"📡 WebSocket subscriptions: {subscriptions}")
                
                if "all" in subscriptions or (event_type and event_type in subscriptions):
                    await websocket.send_text(json.dumps(message))
                    messages_sent += 1
                    print(f"✅ Message sent to WebSocket {id(websocket)}")
                else:
                    print(f"⚠️ WebSocket {id(websocket)} not subscribed to {event_type}")
            except Exception as e:
                print(f"❌ Error broadcasting to WebSocket: {e}")
                disconnected_websockets.append(websocket)
        
        print(f"📡 Total messages sent: {messages_sent}")
        
        # Clean up disconnected websockets
        for websocket in disconnected_websockets:
            self.disconnect(websocket)
        
        # Update last activity
        self.global_metadata["last_activity"] = datetime.now(timezone.utc).isoformat()
    
    async def broadcast_to_user(self, user_id: str, message: dict, event_type: str = None):
        """Broadcast a message to all WebSockets for a specific user"""
        user_websockets = [
            ws for ws, user_data in self.websocket_to_user.items()
            if user_data.get("user_id") == user_id
        ]
        
        print(f"📡 Broadcasting to {len(user_websockets)} connections for user {user_id}")
        
        for websocket in user_websockets:
            try:
                subscriptions = self.websocket_subscriptions.get(websocket, ["all"])
                if "all" in subscriptions or (event_type and event_type in subscriptions):
                    await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"❌ Error broadcasting to user WebSocket: {e}")
                self.disconnect(websocket)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections"""
        return {
            "total_connections": len(self.active_connections),
            "global_metadata": self.global_metadata,
            "user_connections": len(self.websocket_to_user),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global WebSocket manager instance
websocket_manager = GeneralWebSocketManager()

@router.websocket("/ws")
async def general_websocket(
    websocket: WebSocket,
    token: str = None,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for general application event monitoring
    
    This endpoint allows frontends to connect and receive real-time updates
    about various application events without being tied to a specific call session.
    
    Message Types Received:
    - ping: Keep-alive ping
    - subscribe: Subscribe to specific event types (e.g., ["call_status_update", "transcript_update"])
    - unsubscribe: Unsubscribe from event types
    
    Message Types Sent:
    - connection_established: Initial connection confirmation
    - call_status_update: Call status changed (ringing, answered, completed, etc.)
    - transcript_update: New transcript messages added
    - call_ended: Call session ended
    - call_event: General call events
    - system_notification: System-wide notifications
    - error: Error occurred
    - pong: Response to ping
    
    Args:
        websocket: WebSocket connection
        db: Database session
    """
    try:
        # Validate JWT token for WebSocket connections (with fallback for testing)
        user_id = None
        user_info = None
        
        if token:
            try:
                # Decode JWT token to get user info
                from app.core.security import verify_token
                token_data = verify_token(token)
                if token_data:
                    user_id = token_data.get("user_id")
                    user_info = {
                        "email": token_data.get("email"),
                        "tenant_id": token_data.get("tenant_id")
                    }
                    print(f"✅ JWT token validated for user: {user_id}")
                else:
                    raise Exception("Invalid token")
            except Exception as e:
                print(f"⚠️ JWT token validation failed: {e}")
                # For testing, allow connection without valid token
                user_id = "test_user"
                user_info = {"email": "test@example.com", "tenant_id": "test_tenant"}
        else:
            # For testing, allow connection without token
            user_id = "test_user"
            user_info = {"email": "test@example.com", "tenant_id": "test_tenant"}
            print("⚠️ No JWT token provided - using test user")
        
        # Connect to general monitoring
        await websocket_manager.connect(websocket, user_id, user_info)
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "ping":
                    await websocket_manager.send_to_websocket(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                elif message_type == "subscribe":
                    event_types = message.get("event_types", ["all"])
                    websocket_manager.websocket_subscriptions[websocket] = event_types
                    await websocket_manager.send_to_websocket(websocket, {
                        "type": "subscription_updated",
                        "subscribed_to": event_types,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                elif message_type == "unsubscribe":
                    event_types = message.get("event_types", [])
                    current_subscriptions = websocket_manager.websocket_subscriptions.get(websocket, ["all"])
                    new_subscriptions = [sub for sub in current_subscriptions if sub not in event_types]
                    websocket_manager.websocket_subscriptions[websocket] = new_subscriptions
                    await websocket_manager.send_to_websocket(websocket, {
                        "type": "subscription_updated",
                        "subscribed_to": new_subscriptions,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                elif message_type == "get_stats":
                    stats = websocket_manager.get_connection_stats()
                    await websocket_manager.send_to_websocket(websocket, {
                        "type": "connection_stats",
                        "stats": stats,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                else:
                    await websocket_manager.send_to_websocket(websocket, {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager.send_to_websocket(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                print(f"❌ Error handling WebSocket message: {e}")
                await websocket_manager.send_to_websocket(websocket, {
                    "type": "error",
                    "message": f"Internal error: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

# Broadcast functions for use by other parts of the application
async def broadcast_call_status_update(call_session_id: str, status: str, metadata: dict = None):
    """Broadcast call status update to all connected clients"""
    print(f"🔔 broadcast_call_status_update called: session={call_session_id}, status={status}")
    print(f"🔔 Active connections: {len(websocket_manager.active_connections)}")
    
    message = {
        "type": "call_status_update",
        "call_session_id": call_session_id,
        "status": status,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        await websocket_manager.broadcast_to_all(message, "call_status_update")
        print(f"✅ broadcast_call_status_update completed successfully")
    except Exception as e:
        print(f"❌ broadcast_call_status_update failed: {e}")
        import traceback
        traceback.print_exc()

async def broadcast_transcript_update(call_session_id: str, transcript: list, new_messages: list = None):
    """Broadcast transcript update to all connected clients"""
    print(f"💬 broadcast_transcript_update called: session={call_session_id}, new_messages={len(new_messages or [])}")
    print(f"💬 Active connections: {len(websocket_manager.active_connections)}")
    
    message = {
        "type": "transcript_update",
        "call_session_id": call_session_id,
        "transcript": transcript,
        "new_messages": new_messages or [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        await websocket_manager.broadcast_to_all(message, "transcript_update")
        print(f"✅ broadcast_transcript_update completed successfully")
    except Exception as e:
        print(f"❌ broadcast_transcript_update failed: {e}")
        import traceback
        traceback.print_exc()

async def broadcast_call_ended(call_session_id: str, reason: str, final_data: dict = None):
    """Broadcast call ended event to all connected clients"""
    message = {
        "type": "call_ended",
        "call_session_id": call_session_id,
        "reason": reason,
        "final_data": final_data or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket_manager.broadcast_to_all(message, "call_ended")

async def broadcast_call_event(call_session_id: str, event_type: str, event_data: dict = None):
    """Broadcast general call event to all connected clients"""
    message = {
        "type": "call_event",
        "call_session_id": call_session_id,
        "event_type": event_type,
        "event_data": event_data or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket_manager.broadcast_to_all(message, "call_event")

async def broadcast_system_notification(notification_type: str, message: str, metadata: dict = None):
    """Broadcast system notification to all connected clients"""
    notification = {
        "type": "system_notification",
        "notification_type": notification_type,
        "message": message,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket_manager.broadcast_to_all(notification, "system_notification")

@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return websocket_manager.get_connection_stats()

@router.post("/test-broadcast")
async def test_broadcast():
    """Test endpoint to send a test message to all connected WebSocket clients"""
    try:
        await broadcast_system_notification(
            notification_type="test",
            message="This is a test broadcast message!",
            metadata={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        return {"status": "success", "message": "Test broadcast sent", "connections": len(websocket_manager.active_connections)}
    except Exception as e:
        return {"status": "error", "message": str(e), "connections": len(websocket_manager.active_connections)}

@router.get("/test-page")
async def get_test_page():
    """Serve the custom HTML test page"""
    try:
        import os
        # Get the current directory and navigate to the HTML file
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        html_file_path = os.path.join(current_dir, "call_session_websocket_test.html")
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content, media_type="text/html")
    except Exception as e:
        return HTMLResponse(f"Error loading test page: {str(e)}", status_code=500)

@router.post("/test-call-events")
async def test_call_events():
    """Test endpoint to simulate a complete call flow for testing WebSocket events"""
    try:
        test_call_session_id = "test-call-123"
        
        # Simulate call flow events
        events = [
            ("initiating", "Call is being initiated..."),
            ("initiated", "Call has been initiated"),
            ("ringing", "Call is ringing..."),
            ("in-progress", "Call is now in progress"),
            ("completed", "Call has ended")
        ]
        
        for status, message in events:
            await broadcast_call_status_update(
                call_session_id=test_call_session_id,
                status=status,
                metadata={
                    "test": True,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            # Small delay between events
            import asyncio
            await asyncio.sleep(0.5)
        
        # Simulate transcript events
        await broadcast_transcript_update(
            call_session_id=test_call_session_id,
            transcript=[
                {"role": "agent", "message": "Hello! This is a test agent. How can I help you?", "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "client", "message": "Hi, this is a test call!", "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "agent", "message": "Great! This is working perfectly.", "timestamp": datetime.now(timezone.utc).isoformat()}
            ],
            new_messages=[
                {"role": "agent", "message": "Hello! This is a test agent. How can I help you?", "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "client", "message": "Hi, this is a test call!", "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "agent", "message": "Great! This is working perfectly.", "timestamp": datetime.now(timezone.utc).isoformat()}
            ]
        )
        
        return {
            "status": "success", 
            "message": "Test call events sent", 
            "connections": len(websocket_manager.active_connections),
            "events_sent": len(events) + 1  # +1 for transcript
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "connections": len(websocket_manager.active_connections)}
