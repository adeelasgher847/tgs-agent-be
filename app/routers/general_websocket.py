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
        
        disconnected_websockets = []
        
        for websocket in self.active_connections:
            try:
                # Check if websocket is subscribed to this event type
                subscriptions = self.websocket_subscriptions.get(websocket, ["all"])
                if "all" in subscriptions or (event_type and event_type in subscriptions):
                    await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"❌ Error broadcasting to WebSocket: {e}")
                disconnected_websockets.append(websocket)
        
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
                from app.core.security import decode_access_token
                token_data = decode_access_token(token)
                user_id = token_data.get("sub")
                user_info = {
                    "email": token_data.get("email"),
                    "tenant_id": token_data.get("tenant_id")
                }
                print(f"✅ JWT token validated for user: {user_id}")
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
    message = {
        "type": "call_status_update",
        "call_session_id": call_session_id,
        "status": status,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket_manager.broadcast_to_all(message, "call_status_update")

async def broadcast_transcript_update(call_session_id: str, transcript: list, new_messages: list = None):
    """Broadcast transcript update to all connected clients"""
    message = {
        "type": "transcript_update",
        "call_session_id": call_session_id,
        "transcript": transcript,
        "new_messages": new_messages or [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await websocket_manager.broadcast_to_all(message, "transcript_update")

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

@router.get("/test", response_class=HTMLResponse)
async def websocket_test_page():
    """Serve a test page for WebSocket connections"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Event Monitor</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5;
                color: #333;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .header h1 { margin: 0; font-size: 2em; }
            .header p { margin: 10px 0 0 0; opacity: 0.9; }
            
            .section { 
                margin: 0; 
                padding: 20px; 
                border-bottom: 1px solid #eee; 
            }
            .section:last-child { border-bottom: none; }
            .section h3 { 
                margin: 0 0 15px 0; 
                color: #333;
                font-size: 1.2em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 5px;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }
            
            input, button, select { 
                margin: 5px; 
                padding: 10px; 
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            input[type="text"], input[type="password"] { 
                min-width: 300px; 
            }
            button { 
                background: #667eea; 
                color: white; 
                border: none; 
                cursor: pointer; 
                transition: background 0.3s;
                font-weight: 500;
            }
            button:hover { background: #5a6fd8; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            button.success { background: #28a745; }
            button.success:hover { background: #218838; }
            
            .status { 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px; 
                font-weight: bold;
                text-align: center;
            }
            .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                border-left: 4px solid #667eea;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
            }
            
            .events-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            
            .events-panel {
                background: #f8f9fa;
                border-radius: 5px;
                overflow: hidden;
            }
            .events-header {
                background: #667eea;
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .events { 
                height: 500px; 
                overflow-y: auto; 
                padding: 10px; 
                background: white;
            }
            .event { 
                margin: 8px 0; 
                padding: 12px; 
                border-left: 4px solid #007bff; 
                background: white;
                border-radius: 0 5px 5px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .event:hover { transform: translateX(5px); }
            .event.call_status_update { border-left-color: #17a2b8; }
            .event.transcript_update { border-left-color: #28a745; }
            .event.call_ended { border-left-color: #dc3545; }
            .event.call_event { border-left-color: #ffc107; }
            .event.system_notification { border-left-color: #6f42c1; }
            .event.connection_established { border-left-color: #20c997; }
            .event.error { border-left-color: #dc3545; background: #fff5f5; }
            
            .event-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .event-type {
                font-weight: bold;
                color: #667eea;
                text-transform: uppercase;
                font-size: 0.8em;
                letter-spacing: 1px;
            }
            .event-time {
                color: #666;
                font-size: 0.8em;
            }
            .event-content {
                font-size: 0.9em;
                line-height: 1.4;
            }
            .event-details {
                margin-top: 8px;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.8em;
                color: #666;
                max-height: 100px;
                overflow-y: auto;
            }
            
            .transcript-panel {
                background: #f8f9fa;
                border-radius: 5px;
                overflow: hidden;
            }
            .transcript-header {
                background: #28a745;
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .transcript { 
                height: 500px; 
                overflow-y: auto; 
                padding: 15px; 
                background: white;
            }
            .transcript-entry {
                margin: 10px 0;
                padding: 12px;
                border-radius: 8px;
                position: relative;
            }
            .transcript-entry.agent {
                background: #e3f2fd;
                margin-left: 20px;
                border-left: 4px solid #2196f3;
            }
            .transcript-entry.client {
                background: #f3e5f5;
                margin-right: 20px;
                border-left: 4px solid #9c27b0;
            }
            .transcript-role {
                font-weight: bold;
                font-size: 0.8em;
                text-transform: uppercase;
                margin-bottom: 5px;
                opacity: 0.7;
            }
            .transcript-message {
                font-size: 0.95em;
                line-height: 1.4;
            }
            .transcript-time {
                font-size: 0.7em;
                opacity: 0.6;
                margin-top: 5px;
            }
            
            .filters {
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            .filter-group {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .filter-group label {
                font-size: 0.9em;
                color: #666;
            }
            
            .pulse {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .badge {
                display: inline-block;
                padding: 2px 8px;
                font-size: 0.7em;
                border-radius: 10px;
                color: white;
                font-weight: bold;
            }
            .badge.connected { background: #28a745; }
            .badge.disconnected { background: #dc3545; }
            .badge.reconnecting { background: #ffc107; color: #333; }
            
            @media (max-width: 768px) {
                .events-container {
                    grid-template-columns: 1fr;
                }
                .controls {
                    flex-direction: column;
                    align-items: stretch;
                }
                input[type="text"], input[type="password"] {
                    min-width: auto;
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Real-time Event Monitor</h1>
                <p>Monitor all application events in real-time without specifying call session IDs</p>
            </div>
            
            <div class="section">
                <h3>🔌 Connection Settings</h3>
                <div class="controls">
                    <div>
                        <label>JWT Token:</label><br>
                        <input type="password" id="token" placeholder="Enter JWT token (optional for testing)" style="width: 100%;">
                        <small style="color: #666;">Token will be stored in localStorage for this session</small>
                    </div>
                    <div>
                        <label>WebSocket URL:</label><br>
                        <input type="text" id="wsUrl" value="ws://localhost:8000/api/v1/general/ws" readonly style="width: 100%;">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>🎮 Controls</h3>
                <div class="controls">
                    <button id="connectBtn" onclick="connect()" class="success">🔗 Connect</button>
                    <button id="disconnectBtn" onclick="disconnect()" disabled class="danger">❌ Disconnect</button>
                    <button onclick="clearEvents()">🗑️ Clear Events</button>
                    <button onclick="getStats()">📊 Get Stats</button>
                    <button onclick="clearTranscript()">🗑️ Clear Transcript</button>
                </div>
            </div>
            
            <div class="section">
                <h3>📊 Connection Status</h3>
                <div id="status" class="status disconnected">
                    <span class="badge disconnected">Disconnected</span> Ready to connect
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="reconnectAttempts">0</div>
                        <div class="stat-label">Reconnection Attempts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="totalEvents">0</div>
                        <div class="stat-label">Total Events</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="transcriptEntries">0</div>
                        <div class="stat-label">Transcript Entries</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="activeCalls">0</div>
                        <div class="stat-label">Active Calls</div>
                    </div>
                </div>
                <div style="text-align: center; color: #666; margin-top: 10px;">
                    Last Activity: <span id="lastActivity">Never</span>
                </div>
            </div>
            
            <div class="section">
                <h3>🎯 Event Subscriptions</h3>
                <div class="filters">
                    <div class="filter-group">
                        <input type="checkbox" id="subAll" checked>
                        <label for="subAll">All Events</label>
                    </div>
                    <div class="filter-group">
                        <input type="checkbox" id="subCallStatus">
                        <label for="subCallStatus">Call Status</label>
                    </div>
                    <div class="filter-group">
                        <input type="checkbox" id="subTranscript">
                        <label for="subTranscript">Transcript</label>
                    </div>
                    <div class="filter-group">
                        <input type="checkbox" id="subCallEnded">
                        <label for="subCallEnded">Call Ended</label>
                    </div>
                    <div class="filter-group">
                        <input type="checkbox" id="subCallEvent">
                        <label for="subCallEvent">Call Events</label>
                    </div>
                    <div class="filter-group">
                        <input type="checkbox" id="subSystem">
                        <label for="subSystem">System</label>
                    </div>
                    <button onclick="updateSubscriptions()">Update Subscriptions</button>
                </div>
            </div>
            
            <div class="section">
                <h3>📡 Real-time Monitoring</h3>
                <div class="events-container">
                    <div class="events-panel">
                        <div class="events-header">
                            <span>📋 Events Log</span>
                            <span id="eventCount">0 events</span>
                        </div>
                        <div id="events" class="events">
                            <div style="text-align: center; color: #666; padding: 50px;">
                                <div style="font-size: 3em; margin-bottom: 20px;">📡</div>
                                <div>No events yet. Click "Connect" to start monitoring.</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="transcript-panel">
                        <div class="transcript-header">
                            <span>💬 Live Transcript</span>
                            <span id="transcriptCount">0 entries</span>
                        </div>
                        <div id="transcript" class="transcript">
                            <div style="text-align: center; color: #666; padding: 50px;">
                                <div style="font-size: 3em; margin-bottom: 20px;">💬</div>
                                <div>No transcript entries yet. Start a call to see conversation.</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            let totalEvents = 0;
            let transcriptEntries = 0;
            let activeCalls = new Set();
            let currentTranscript = [];
            
            // Load token from localStorage
            window.onload = function() {
                const savedToken = localStorage.getItem('websocket_token');
                if (savedToken) {
                    document.getElementById('token').value = savedToken;
                }
            };
            
            function connect() {
                const token = document.getElementById('token').value;
                const wsUrl = document.getElementById('wsUrl').value;
                
                // Save token to localStorage
                if (token) {
                    localStorage.setItem('websocket_token', token);
                }
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    addEvent('Already connected', 'error');
                    return;
                }
                
                ws = new WebSocket(wsUrl + (token ? `?token=${encodeURIComponent(token)}` : ''));
                
                ws.onopen = function(event) {
                    addEvent('Connected to general WebSocket', 'connection_established');
                    updateStatus('Connected', 'connected');
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    reconnectAttempts = 0;
                    updateStats();
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                        document.getElementById('lastActivity').textContent = new Date().toLocaleString();
                    } catch (e) {
                        addEvent(`Received (raw): ${event.data}`, 'error');
                    }
                };
                
                ws.onclose = function(event) {
                    addEvent(`Connection closed: ${event.code} - ${event.reason}`, 'error');
                    updateStatus('Disconnected', 'disconnected');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    
                    // Auto-reconnect logic
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        updateStats();
                        addEvent(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`, 'error');
                        setTimeout(connect, 2000);
                    }
                };
                
                ws.onerror = function(error) {
                    addEvent(`WebSocket error: ${error}`, 'error');
                };
            }
            
            function handleMessage(data) {
                totalEvents++;
                updateStats();
                
                // Handle different message types
                switch(data.type) {
                    case 'connection_established':
                        addEvent('Connection established successfully', 'connection_established');
                        break;
                        
                    case 'transcript_update':
                        handleTranscriptUpdate(data);
                        addEvent(`Transcript update for call ${data.call_session_id}`, 'transcript_update');
                        break;
                        
                    case 'call_status_update':
                        handleCallStatusUpdate(data);
                        addEvent(`Call status: ${data.status} for call ${data.call_session_id}`, 'call_status_update');
                        break;
                        
                    case 'call_ended':
                        handleCallEnded(data);
                        addEvent(`Call ended: ${data.reason} for call ${data.call_session_id}`, 'call_ended');
                        break;
                        
                    case 'call_event':
                        addEvent(`Call event: ${data.event_type} for call ${data.call_session_id}`, 'call_event');
                        break;
                        
                    case 'system_notification':
                        addEvent(`System notification: ${data.message}`, 'system_notification');
                        break;
                        
                    case 'subscription_updated':
                        addEvent(`Subscriptions updated: ${data.subscribed_to.join(', ')}`, 'success');
                        break;
                        
                    case 'connection_stats':
                        updateConnectionStats(data.stats);
                        break;
                        
                    default:
                        addEvent(`Unknown event type: ${data.type}`, 'error');
                }
            }
            
            function handleTranscriptUpdate(data) {
                if (data.new_messages && data.new_messages.length > 0) {
                    data.new_messages.forEach(message => {
                        currentTranscript.push(message);
                        transcriptEntries++;
                        addTranscriptEntry(message);
                    });
                }
                updateStats();
            }
            
            function handleCallStatusUpdate(data) {
                if (data.status === 'in-progress') {
                    activeCalls.add(data.call_session_id);
                } else if (data.status === 'completed' || data.status === 'failed') {
                    activeCalls.delete(data.call_session_id);
                }
                updateStats();
            }
            
            function handleCallEnded(data) {
                activeCalls.delete(data.call_session_id);
                updateStats();
            }
            
            function addTranscriptEntry(message) {
                const transcriptEl = document.getElementById('transcript');
                
                // Remove placeholder if it exists
                const placeholder = transcriptEl.querySelector('.placeholder');
                if (placeholder) {
                    placeholder.remove();
                }
                
                const entryEl = document.createElement('div');
                entryEl.className = `transcript-entry ${message.role}`;
                
                const roleIcon = message.role === 'agent' ? '🤖' : '👤';
                const roleName = message.role === 'agent' ? 'Agent' : 'Client';
                
                entryEl.innerHTML = `
                    <div class="transcript-role">${roleIcon} ${roleName}</div>
                    <div class="transcript-message">${message.message}</div>
                    <div class="transcript-time">${new Date(message.timestamp).toLocaleString()}</div>
                `;
                
                transcriptEl.appendChild(entryEl);
                transcriptEl.scrollTop = transcriptEl.scrollHeight;
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function updateStatus(status, className) {
                const statusEl = document.getElementById('status');
                const badgeClass = className === 'connected' ? 'connected' : 'disconnected';
                statusEl.innerHTML = `<span class="badge ${badgeClass}">${status}</span> ${status === 'Connected' ? 'Monitoring events' : 'Ready to connect'}`;
                statusEl.className = `status ${className}`;
            }
            
            function addEvent(message, type = '') {
                const eventsEl = document.getElementById('events');
                
                // Remove placeholder if it exists
                const placeholder = eventsEl.querySelector('.placeholder');
                if (placeholder) {
                    placeholder.remove();
                }
                
                const eventEl = document.createElement('div');
                eventEl.className = `event ${type}`;
                
                const time = new Date().toLocaleString();
                const typeIcon = getEventIcon(type);
                
                eventEl.innerHTML = `
                    <div class="event-header">
                        <span class="event-type">${typeIcon} ${type.replace('_', ' ')}</span>
                        <span class="event-time">${time}</span>
                    </div>
                    <div class="event-content">${message}</div>
                `;
                
                eventsEl.appendChild(eventEl);
                eventsEl.scrollTop = eventsEl.scrollHeight;
                
                // Update event count
                document.getElementById('eventCount').textContent = `${totalEvents} events`;
            }
            
            function getEventIcon(type) {
                const icons = {
                    'connection_established': '🔗',
                    'transcript_update': '💬',
                    'call_status_update': '📞',
                    'call_ended': '📴',
                    'call_event': '📋',
                    'system_notification': '🔔',
                    'error': '❌',
                    'success': '✅'
                };
                return icons[type] || '📡';
            }
            
            function updateStats() {
                document.getElementById('reconnectAttempts').textContent = reconnectAttempts;
                document.getElementById('totalEvents').textContent = totalEvents;
                document.getElementById('transcriptEntries').textContent = transcriptEntries;
                document.getElementById('activeCalls').textContent = activeCalls.size;
                document.getElementById('transcriptCount').textContent = `${transcriptEntries} entries`;
            }
            
            function updateConnectionStats(stats) {
                document.getElementById('totalEvents').textContent = stats.total_connections || 0;
            }
            
            function clearEvents() {
                document.getElementById('events').innerHTML = `
                    <div class="placeholder" style="text-align: center; color: #666; padding: 50px;">
                        <div style="font-size: 3em; margin-bottom: 20px;">📡</div>
                        <div>Events cleared. Waiting for new events...</div>
                    </div>
                `;
                totalEvents = 0;
                updateStats();
            }
            
            function clearTranscript() {
                document.getElementById('transcript').innerHTML = `
                    <div class="placeholder" style="text-align: center; color: #666; padding: 50px;">
                        <div style="font-size: 3em; margin-bottom: 20px;">💬</div>
                        <div>Transcript cleared. Waiting for new conversation...</div>
                    </div>
                `;
                currentTranscript = [];
                transcriptEntries = 0;
                updateStats();
            }
            
            function getStats() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'get_stats' }));
                } else {
                    addEvent('Not connected', 'error');
                }
            }
            
            function updateSubscriptions() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const subscriptions = [];
                    
                    if (document.getElementById('subAll').checked) {
                        subscriptions.push('all');
                    } else {
                        if (document.getElementById('subCallStatus').checked) subscriptions.push('call_status_update');
                        if (document.getElementById('subTranscript').checked) subscriptions.push('transcript_update');
                        if (document.getElementById('subCallEnded').checked) subscriptions.push('call_ended');
                        if (document.getElementById('subCallEvent').checked) subscriptions.push('call_event');
                        if (document.getElementById('subSystem').checked) subscriptions.push('system_notification');
                    }
                    
                    ws.send(JSON.stringify({ 
                        type: 'subscribe', 
                        event_types: subscriptions 
                    }));
                    
                    addEvent(`Updated subscriptions: ${subscriptions.join(', ')}`, 'success');
                } else {
                    addEvent('Not connected', 'error');
                }
            }
            
            // Handle subscription checkboxes
            document.getElementById('subAll').addEventListener('change', function() {
                const checkboxes = ['subCallStatus', 'subTranscript', 'subCallEnded', 'subCallEvent', 'subSystem'];
                checkboxes.forEach(id => {
                    document.getElementById(id).disabled = this.checked;
                    if (this.checked) {
                        document.getElementById(id).checked = false;
                    }
                });
            });
        </script>
    </body>
    </html>
    """)

@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return websocket_manager.get_connection_stats()
