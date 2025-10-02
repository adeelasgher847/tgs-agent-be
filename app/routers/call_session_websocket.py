"""
Call Session WebSocket Router
Real-time events for call sessions - allows frontend to receive live updates
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
    tags=["Call Session WebSocket"],
    responses={404: {"description": "Not found"}},
)

class CallSessionWebSocketManager:
    """Manages WebSocket connections for call session events"""
    
    def __init__(self):
        # Map of call_session_id -> List of WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Map of websocket -> call_session_id for cleanup
        self.websocket_to_session: Dict[WebSocket, str] = {}
        # Map of call_session_id -> session metadata
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, call_session_id: str, user_id: str = None):
        """Connect a WebSocket to a call session"""
        await websocket.accept()
        
        if call_session_id not in self.active_connections:
            self.active_connections[call_session_id] = []
        
        self.active_connections[call_session_id].append(websocket)
        self.websocket_to_session[websocket] = call_session_id
        
        # Initialize session metadata if not exists
        if call_session_id not in self.session_metadata:
            self.session_metadata[call_session_id] = {
                "connected_users": set(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "event_count": 0
            }
        
        if user_id:
            self.session_metadata[call_session_id]["connected_users"].add(user_id)
        
        # Send connection confirmation
        await self.send_to_session(call_session_id, {
            "type": "connection_established",
            "call_session_id": call_session_id,
            "message": "Connected to call session events",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connected_users": len(self.session_metadata[call_session_id]["connected_users"])
        })
        
        print(f"✅ WebSocket connected to call session {call_session_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket from its call session"""
        if websocket in self.websocket_to_session:
            call_session_id = self.websocket_to_session[websocket]
            
            # Remove from active connections
            if call_session_id in self.active_connections:
                if websocket in self.active_connections[call_session_id]:
                    self.active_connections[call_session_id].remove(websocket)
                
                # Clean up empty session
                if not self.active_connections[call_session_id]:
                    del self.active_connections[call_session_id]
                    if call_session_id in self.session_metadata:
                        del self.session_metadata[call_session_id]
            
            del self.websocket_to_session[websocket]
            print(f"❌ WebSocket disconnected from call session {call_session_id}")
    
    async def send_to_session(self, call_session_id: str, message: dict):
        """Send a message to all WebSockets connected to a call session"""
        print(f"📡 send_to_session called for {call_session_id}")
        print(f"📡 Message type: {message.get('type', 'unknown')}")
        print(f"📡 All active sessions: {list(self.active_connections.keys())}")
        print(f"📡 Total active connections: {sum(len(conns) for conns in self.active_connections.values())}")
        
        if call_session_id not in self.active_connections:
            print(f"❌ No active connections for session {call_session_id}")
            print(f"❌ Available sessions: {list(self.active_connections.keys())}")
            return
        
        connection_count = len(self.active_connections[call_session_id])
        print(f"📡 Sending to {connection_count} connections for session {call_session_id}")
        
        # Update session metadata
        if call_session_id in self.session_metadata:
            self.session_metadata[call_session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
            self.session_metadata[call_session_id]["event_count"] += 1
        
        # Send to all connected WebSockets
        disconnected_websockets = []
        for i, websocket in enumerate(self.active_connections[call_session_id]):
            try:
                message_json = json.dumps(message)
                await websocket.send_text(message_json)
                print(f"✅ Message sent to WebSocket #{i+1} for session {call_session_id}")
                print(f"📤 Message content: {message_json[:100]}...")
            except Exception as e:
                print(f"❌ Error sending message to WebSocket #{i+1}: {e}")
                disconnected_websockets.append(websocket)
        
        # Clean up disconnected WebSockets
        for websocket in disconnected_websockets:
            self.disconnect(websocket)
    
    async def send_to_websocket(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    def get_session_info(self, call_session_id: str) -> Dict[str, Any]:
        """Get information about a call session's WebSocket connections"""
        if call_session_id not in self.active_connections:
            return {"connected_count": 0, "users": [], "last_activity": None}
        
        metadata = self.session_metadata.get(call_session_id, {})
        return {
            "connected_count": len(self.active_connections[call_session_id]),
            "users": list(metadata.get("connected_users", [])),
            "last_activity": metadata.get("last_activity"),
            "event_count": metadata.get("event_count", 0)
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active call sessions"""
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.active_connections.keys()
        }

# Global WebSocket manager instance
websocket_manager = CallSessionWebSocketManager()

@router.websocket("/ws/{call_session_id}")
async def call_session_websocket(
    websocket: WebSocket,
    call_session_id: str,
    token: str = None,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time call session events
    
    This endpoint allows frontends to connect to a specific call session
    and receive real-time updates about call events, status changes,
    transcript updates, and other call-related information.
    
    Message Types Received:
    - ping: Keep-alive ping
    - subscribe: Subscribe to specific event types
    - unsubscribe: Unsubscribe from event types
    
    Message Types Sent:
    - connection_established: Initial connection confirmation
    - call_status_update: Call status changed (ringing, answered, completed, etc.)
    - transcript_update: New transcript messages added
    - call_metadata_update: Call metadata updated
    - call_ended: Call session ended
    - error: Error occurred
    - pong: Response to ping
    
    Args:
        websocket: WebSocket connection
        call_session_id: UUID of the call session to monitor
        db: Database session
    """
    try:
        # Validate call session exists
        try:
            session_uuid = uuid.UUID(call_session_id)
            call_session = db.query(CallSession).filter(CallSession.id == session_uuid).first()
            if not call_session:
                await websocket.close(code=4004, reason="Call session not found")
                return
        except ValueError:
            await websocket.close(code=4000, reason="Invalid call session ID")
            return
        
        # Validate JWT token for WebSocket connections
        user_id = None
        if token:
            try:
                from app.core.security import verify_token
                from app.models.user import User
                
                print(f"🔐 Validating JWT token: {token[:20]}...")
                
                # Verify the JWT token
                payload = verify_token(token)
                if payload:
                    user_id = payload.get("sub")
                    print(f"✅ WebSocket authenticated for user: {user_id}")
                    
                    # Verify user exists and has access to this call session
                    user = db.query(User).filter(User.id == user_id).first()
                    if not user:
                        print(f"❌ User not found: {user_id}")
                        await websocket.close(code=4001, reason="User not found")
                        return
                        
                    print(f"✅ User found: {user.email}, current_tenant: {user.current_tenant_id}")
                    print(f"✅ Call session tenant: {call_session.tenant_id}")
                    
                    # Check if user has access to this call session (same tenant)
                    if call_session.tenant_id != user.current_tenant_id:
                        print(f"❌ Tenant mismatch: user tenant {user.current_tenant_id} != call session tenant {call_session.tenant_id}")
                        await websocket.close(code=4003, reason="Access denied to this call session")
                        return
                        
                else:
                    print(f"❌ Invalid JWT token payload")
                    await websocket.close(code=4001, reason="Invalid token")
                    return
            except Exception as e:
                print(f"❌ WebSocket authentication error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.close(code=4001, reason="Authentication failed")
                return
        else:
            print(f"❌ No JWT token provided")
            await websocket.close(code=4001, reason="Token required")
            return
        
        # Connect to the call session
        print(f"🔌 Connecting WebSocket to call session: {call_session_id}")
        print(f"🔌 User ID: {user_id}")
        print(f"🔌 Call session status: {call_session.status}")
        print(f"🔌 Call session tenant: {call_session.tenant_id}")
        
        await websocket_manager.connect(websocket, call_session_id, user_id)
        print(f"🔌 WebSocket connected successfully to session: {call_session_id}")
        print(f"🔌 Total active connections: {len(websocket_manager.active_connections)}")
        print(f"🔌 Active sessions: {list(websocket_manager.active_connections.keys())}")
        
        # Send initial call session data
        await websocket_manager.send_to_websocket(websocket, {
            "type": "call_session_data",
            "call_session_id": call_session_id,
            "data": {
                "id": str(call_session.id),
                "status": call_session.status,
                "start_time": call_session.start_time.isoformat() if call_session.start_time else None,
                "end_time": call_session.end_time.isoformat() if call_session.end_time else None,
                "duration": call_session.duration,
                "call_type": call_session.call_type,
                "from_number": call_session.from_number,
                "to_number": call_session.to_number,
                "assistant_phone_number": call_session.assistant_phone_number,
                "customer_phone_number": call_session.customer_phone_number,
                "recording_url": call_session.recording_url,
                "transcript": call_session.call_transcript or [],
                "metadata": call_session.call_metadata or {}
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                await handle_websocket_message(websocket, call_session_id, message_data, db)
                
            except WebSocketDisconnect:
                print(f"🔌 WebSocket disconnected for session {call_session_id}")
                break
            except json.JSONDecodeError:
                await websocket_manager.send_to_websocket(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                await websocket_manager.send_to_websocket(websocket, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
    except Exception as e:
        print(f"Call session WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

async def handle_websocket_message(websocket: WebSocket, call_session_id: str, message_data: dict, db: Session):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message_data.get("type")
        
        if message_type == "ping":
            await websocket_manager.send_to_websocket(websocket, {
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            print(f"💓 Heartbeat received from session {call_session_id}")
        elif message_type == "subscribe":
            # Handle subscription to specific event types
            event_types = message_data.get("event_types", [])
            await websocket_manager.send_to_websocket(websocket, {
                "type": "subscription_confirmed",
                "event_types": event_types,
                "message": f"Subscribed to {len(event_types)} event types",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        elif message_type == "unsubscribe":
            # Handle unsubscription from event types
            event_types = message_data.get("event_types", [])
            await websocket_manager.send_to_websocket(websocket, {
                "type": "unsubscription_confirmed",
                "event_types": event_types,
                "message": f"Unsubscribed from {len(event_types)} event types",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        elif message_type == "get_session_info":
            # Send current session information
            session_info = websocket_manager.get_session_info(call_session_id)
            await websocket_manager.send_to_websocket(websocket, {
                "type": "session_info",
                "call_session_id": call_session_id,
                "info": session_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            await websocket_manager.send_to_websocket(websocket, {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
    except Exception as e:
        await websocket_manager.send_to_websocket(websocket, {
            "type": "error",
            "message": f"Error handling message: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# Event broadcasting functions for use by other services
async def broadcast_call_status_update(call_session_id: str, status: str, metadata: dict = None):
    """Broadcast call status update to all connected WebSockets"""
    print(f"🔔 BROADCASTING call status update: {status} for session {call_session_id}")
    print(f"🔔 Active connections: {list(websocket_manager.active_connections.keys())}")
    print(f"🔔 Connected to this session: {call_session_id in websocket_manager.active_connections}")
    
    await websocket_manager.send_to_session(call_session_id, {
        "type": "call_status_update",
        "call_session_id": call_session_id,
        "status": status,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

async def broadcast_transcript_update(call_session_id: str, transcript: list, new_messages: list = None):
    """Broadcast transcript update to all connected WebSockets"""
    print(f"🔔 BROADCASTING transcript update for session {call_session_id}")
    print(f"🔔 Active connections: {list(websocket_manager.active_connections.keys())}")
    print(f"🔔 Connected to this session: {call_session_id in websocket_manager.active_connections}")
    
    await websocket_manager.send_to_session(call_session_id, {
        "type": "transcript_update",
        "call_session_id": call_session_id,
        "transcript": transcript,
        "new_messages": new_messages or [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

async def broadcast_call_metadata_update(call_session_id: str, metadata: dict):
    """Broadcast call metadata update to all connected WebSockets"""
    await websocket_manager.send_to_session(call_session_id, {
        "type": "call_metadata_update",
        "call_session_id": call_session_id,
        "metadata": metadata,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

async def broadcast_call_ended(call_session_id: str, reason: str = None, final_data: dict = None):
    """Broadcast call ended event to all connected WebSockets"""
    print(f"🔔 BROADCASTING call ended event for session {call_session_id}")
    print(f"🔔 Active connections: {list(websocket_manager.active_connections.keys())}")
    print(f"🔔 Connected to this session: {call_session_id in websocket_manager.active_connections}")
    
    await websocket_manager.send_to_session(call_session_id, {
        "type": "call_ended",
        "call_session_id": call_session_id,
        "reason": reason,
        "final_data": final_data or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

async def broadcast_call_event(call_session_id: str, event_type: str, event_data: dict):
    """Broadcast a custom call event to all connected WebSockets"""
    await websocket_manager.send_to_session(call_session_id, {
        "type": "call_event",
        "call_session_id": call_session_id,
        "event_type": event_type,
        "event_data": event_data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@router.get("/sessions/active")
async def get_active_sessions(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get information about all active call session WebSocket connections"""
    return {
        "active_sessions": websocket_manager.get_all_sessions(),
        "total_connections": sum(
            len(connections) for connections in websocket_manager.active_connections.values()
        ),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.get("/debug/connections")
async def debug_connections():
    """Debug endpoint to check WebSocket connections"""
    return {
        "active_connections": len(websocket_manager.active_connections),
        "active_sessions": list(websocket_manager.active_connections.keys()),
        "session_metadata": websocket_manager.session_metadata,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.post("/debug/test-broadcast/{call_session_id}")
async def test_broadcast(call_session_id: str):
    """Test endpoint to manually trigger a broadcast"""
    try:
        await broadcast_call_status_update(
            call_session_id=call_session_id,
            status="test",
            metadata={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        return {
            "success": True,
            "message": f"Test broadcast sent to session {call_session_id}",
            "active_connections": len(websocket_manager.active_connections.get(call_session_id, [])),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/sessions/{call_session_id}/info")
async def get_session_info(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get information about a specific call session's WebSocket connections"""
    try:
        session_uuid = uuid.UUID(call_session_id)
        
        # Verify user has access to this call session
        call_session = db.query(CallSession).filter(
            CallSession.id == session_uuid,
            CallSession.tenant_id == user.current_tenant_id
        ).first()
        
        if not call_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Call session not found or access denied"
            )
        
        session_info = websocket_manager.get_session_info(call_session_id)
        return {
            "call_session_id": call_session_id,
            "info": session_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid call session ID"
        )

@router.get("/test", response_class=HTMLResponse)
async def get_websocket_test_page():
    """Serve the WebSocket test page"""
    try:
        with open("/Users/macbookpro/Desktop/branch-code/tgs-agent-be/call_session_websocket_test.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test page not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load test page: {str(e)}"
        )
