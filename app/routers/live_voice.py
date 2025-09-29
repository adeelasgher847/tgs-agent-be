"""
Live Voice Router - Talk to Assistant Feature
Real-time voice conversation with agents using browser microphone/speakers
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import uuid
import json
import asyncio
import time
import base64
import io
from datetime import datetime

from app.api.deps import get_db, require_tenant
from app.models.agent import Agent
from app.models.user import User
from app.models.call_session import CallSession
from app.schemas.base import SuccessResponse
from app.services.agent_service import agent_service
from app.services.call_session_service import call_session_service
from app.services.openai_service import openai_service
from app.utils.response import create_success_response

router = APIRouter(
    tags=["Live Voice - Talk to Assistant"],
    responses={404: {"description": "Not found"}},
)

@router.get("/test-interface", 
           summary="Get Test Interface",
           description="Serve the main test interface HTML page for testing Talk to Assistant feature",
           response_class=HTMLResponse)
async def get_test_interface():
    """
    Serve the test interface HTML page
    
    Returns a comprehensive testing interface that allows users to:
    - View all available agents
    - Test voice conversation capabilities
    - Monitor system status
    - Access individual agent test pages
    """
    try:
        with open("/Users/macbookpro/Desktop/branch-code/tgs-agent-be/talk_to_assistant_test.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, media_type="text/html")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test interface file not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load test interface: {str(e)}"
        )

@router.get("/", 
           summary="Live Voice Home",
           description="Redirect to the test interface for Talk to Assistant feature",
           response_class=HTMLResponse)
async def get_live_voice_home():
    """
    Redirect to the test interface
    
    This endpoint provides a landing page that automatically redirects
    users to the main test interface for the Talk to Assistant feature.
    """
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Talk to Assistant - Live Voice</title>
        <meta http-equiv="refresh" content="0; url=/api/v1/live-voice/test-interface">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: white;
                color: #333;
                padding: 40px;
                border-radius: 20px;
                max-width: 500px;
                margin: 0 auto;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 { margin-bottom: 20px; }
            p { margin-bottom: 20px; }
            a { 
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 10px;
                font-weight: bold;
            }
            a:hover { background: #5a6fd8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Talk to Assistant</h1>
            <p>Redirecting to the test interface...</p>
            <p>If you're not redirected automatically, <a href="/api/v1/live-voice/test-interface">click here</a></p>
        </div>
    </body>
    </html>
    """, media_type="text/html")

# Active live voice sessions
class LiveVoiceManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))

    def create_session(self, session_id: str, agent_data: dict, user_data: dict):
        self.session_data[session_id] = {
            "agent_data": agent_data,
            "user_data": user_data,
            "messages": [],
            "start_time": datetime.now().isoformat(),
            "is_speaking": False,
            "is_listening": False
        }

manager = LiveVoiceManager()


@router.websocket("/live/{agent_id}")
async def live_voice_conversation(
    websocket: WebSocket,
    agent_id: str,
    db: Session = Depends(get_db)
):
    """
    Live voice conversation with an agent - real-time speech
    
    This WebSocket endpoint enables real-time voice conversation:
    - Accepts speech text from client
    - Processes with OpenAI AI models
    - Returns AI responses for text-to-speech
    - Handles connection management and error handling
    
    Message Types:
    - speech_text: Client sends transcribed speech
    - ping: Keep-alive ping
    - pong: Keep-alive response
    
    Response Types:
    - connection_established: Initial connection confirmation
    - speech_received: Confirmation of received speech
    - agent_thinking: AI is processing the message
    - agent_response: AI response ready for TTS
    - error: Error occurred during processing
    
    Args:
        websocket: WebSocket connection
        agent_id: UUID of the agent to talk to
        db: Database session
    """
    session_id = str(uuid.uuid4())
    
    try:
        # Validate agent exists
        try:
            agent_uuid = uuid.UUID(agent_id)
            agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
            if not agent:
                await websocket.close(code=4004, reason="Agent not found")
                return
        except ValueError:
            await websocket.close(code=4000, reason="Invalid agent ID")
            return

        # Accept WebSocket connection
        await manager.connect(websocket, session_id)
        
        # Create session data with proper system prompt that includes agent name
        agent_name = agent.name if agent.name else "AI Assistant"
        agent_language = agent.language if agent.language else "English"
        
        # Always ensure agent name is included in system prompt
        if agent.system_prompt:
            agent_system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Always respond as {agent_name} - introduce yourself naturally when appropriate

Custom instructions:
{agent.system_prompt}

Guidelines for phone conversations:
- Speak naturally and conversationally like a real person
- Be friendly and professional
- Give complete, helpful answers
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be helpful and try to answer questions thoroughly
- Remember you are {agent_name} - introduce yourself naturally when appropriate

Always respond as {agent_name}, a real person, not as any kind of system or tool."""
        else:
            agent_system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Be friendly, professional, and conversational like a real person

Guidelines for phone conversations:
- Speak naturally and conversationally
- Be friendly and professional
- Give complete, helpful answers
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be helpful and try to answer questions thoroughly
- Remember you are {agent_name} - introduce yourself naturally when appropriate

Always respond as {agent_name}, a real person, not as any kind of system or tool."""
        
        manager.create_session(session_id, {
            "agent_id": str(agent.id),
            "agent_name": agent.name,
            "agent_voice_type": agent.voice_type,
            "agent_system_prompt": agent_system_prompt
        }, {
            "user_id": "anonymous",  # For now, can be enhanced with auth
            "tenant_id": str(agent.tenant_id)
        })
        
        # Create call session in database for tracking
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=uuid.uuid4(),  # Anonymous user for now
            agent_id=agent.id,
            tenant_id=agent.tenant_id,
            twilio_call_sid=f"live_voice_{session_id}",
            from_number="browser_mic",
            to_number="agent_voice",
            call_type="web",
            assistant_phone_number="web_agent",
            customer_phone_number="browser_mic"
        )
        
        # Send initial connection message
        await manager.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "agent_name": agent.name,
            "agent_voice_type": agent.voice_type,
            "message": f"Connected to {agent.name}. You can now start talking!"
        })
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                await handle_live_voice_message(session_id, message_data, db)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except Exception as e:
        print(f"Live voice WebSocket error: {e}")
    finally:
        # Update call session status when session ends
        try:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, f"live_voice_{session_id}")
            if call_session:
                call_session_service.update_call_session_status(
                    db=db,
                    session_id=call_session.id,
                    status="completed",
                    ended_reason="WebSocket disconnected",
                    success_evaluation="success"
                )
                print(f"Updated live voice call session {call_session.id} to completed")
        except Exception as e:
            print(f"Error updating call session on disconnect: {e}")
        
        manager.disconnect(session_id)


async def handle_live_voice_message(session_id: str, message_data: dict, db: Session):
    """
    Handle incoming live voice messages
    """
    try:
        message_type = message_data.get("type")
        
        if message_type == "audio_data":
            await handle_audio_input(session_id, message_data, db)
        elif message_type == "speech_text":
            await handle_speech_text(session_id, message_data, db)
        elif message_type == "start_listening":
            await handle_start_listening(session_id, message_data)
        elif message_type == "stop_listening":
            await handle_stop_listening(session_id, message_data)
        elif message_type == "ping":
            await manager.send_message(session_id, {"type": "pong"})
        else:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
            
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error handling message: {str(e)}"
        })


async def handle_audio_input(session_id: str, message_data: dict, db: Session):
    """
    Handle raw audio data from microphone
    """
    try:
        # In a real implementation, you would:
        # 1. Process the audio data (base64 encoded)
        # 2. Convert to text using speech-to-text service
        # 3. Process with AI
        # 4. Convert response to speech
        # 5. Send audio back to client
        
        audio_data = message_data.get("audio", "")
        if not audio_data:
            return
        
        # For now, we'll simulate processing
        await manager.send_message(session_id, {
            "type": "audio_processing",
            "message": "Processing your speech..."
        })
        
        # Simulate speech-to-text result
        simulated_text = "Hello, I can hear you speaking!"
        
        await handle_speech_text(session_id, {
            "type": "speech_text",
            "text": simulated_text
        }, db)
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error processing audio: {str(e)}"
        })


async def handle_speech_text(session_id: str, message_data: dict, db: Session):
    """
    Handle transcribed speech text
    """
    try:
        if session_id not in manager.session_data:
            raise ValueError("Session not found")
        
        session_data = manager.session_data[session_id]
        agent_data = session_data["agent_data"]
        
        user_text = message_data.get("text", "")
        
        if not user_text:
            return
        
        # Add user message to session history
        user_message = {
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now().isoformat(),
            "type": "speech"
        }
        session_data["messages"].append(user_message)
        
        # Send acknowledgment
        await manager.send_message(session_id, {
            "type": "speech_received",
            "user_text": user_text
        })
        
        # Process with AI
        await process_with_ai_live(session_id, user_text, session_data, db)
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error processing speech: {str(e)}"
        })


async def handle_start_listening(session_id: str, message_data: dict):
    """
    Handle start listening event
    """
    try:
        if session_id in manager.session_data:
            manager.session_data[session_id]["is_listening"] = True
        
        await manager.send_message(session_id, {
            "type": "listening_started",
            "message": "I'm listening..."
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error starting listening: {str(e)}"
        })


async def handle_stop_listening(session_id: str, message_data: dict):
    """
    Handle stop listening event
    """
    try:
        if session_id in manager.session_data:
            manager.session_data[session_id]["is_listening"] = False
        
        await manager.send_message(session_id, {
            "type": "listening_stopped",
            "message": "I stopped listening."
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error stopping listening: {str(e)}"
        })


async def process_with_ai_live(session_id: str, user_input: str, session_data: dict, db: Session):
    """
    Process user input with AI and send voice response
    """
    try:
        agent_data = session_data["agent_data"]
        
        # Get conversation history for context
        conversation_history = []
        for msg in session_data["messages"][-10:]:  # Last 10 messages for context
            if msg["role"] in ["user", "assistant"]:
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Send thinking indicator
        await manager.send_message(session_id, {
            "type": "agent_thinking",
            "message": f"{agent_data['agent_name']} is thinking..."
        })
        
        # Process with OpenAI
        try:
            openai_response = openai_service.process_agent_conversation(
                user_input=user_input,
                agent_system_prompt=agent_data["agent_system_prompt"] or "You are a helpful assistant.",
                conversation_history=conversation_history[:-1]  # Exclude current message
            )
            
            ai_response_text = openai_response["response"]
            response_time = openai_response["response_time"]
            
        except Exception as e:
            print(f"Error processing with OpenAI: {e}")
            ai_response_text = "I'm sorry, but I'm having trouble processing your request right now. Please try again."
            response_time = 0
        
        # Add AI response to session history
        assistant_message = {
            "role": "assistant",
            "content": ai_response_text,
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "type": "speech"
        }
        session_data["messages"].append(assistant_message)
        
        # Send AI response as text (client will convert to speech)
        await manager.send_message(session_id, {
            "type": "agent_response",
            "agent_name": agent_data["agent_name"],
            "message": ai_response_text,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "voice_type": agent_data["agent_voice_type"],
            "should_speak": True  # Tell client to speak this response
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error processing with AI: {str(e)}"
        })


@router.get("/talk/{agent_id}",
           summary="Talk to Assistant (Authenticated)",
           description="Get the Talk to Assistant page for a specific agent with authentication",
           response_class=HTMLResponse)
async def get_talk_to_assistant_page(
    agent_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get the "Talk to Assistant" page - live voice conversation interface (requires auth)
    
    This endpoint provides a complete voice conversation interface for authenticated users:
    - Real-time WebSocket connection
    - Speech-to-text conversion
    - AI processing with OpenAI
    - Text-to-speech responses
    - Requires valid authentication and tenant access
    
    Args:
        agent_id: UUID of the agent to talk to
        user: Authenticated user (from JWT token)
        db: Database session
        
    Returns:
        HTML page with voice conversation interface
    """
    try:
        # Validate agent exists
        try:
            agent_uuid = uuid.UUID(agent_id)
            agent = agent_service.get_agent_by_id(db, agent_uuid, user.current_tenant_id)
        except (ValueError, HTTPException):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Create the live voice conversation interface
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Talk to {agent.name}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{ 
                    background: white; 
                    border-radius: 20px; 
                    padding: 40px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    max-width: 500px;
                    width: 90%;
                    text-align: center;
                }}
                .agent-info {{ 
                    margin-bottom: 30px; 
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 15px;
                }}
                .agent-avatar {{ 
                    width: 80px; 
                    height: 80px; 
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    border-radius: 50%; 
                    margin: 0 auto 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 32px;
                    color: white;
                }}
                .agent-name {{ 
                    font-size: 24px; 
                    font-weight: 600; 
                    color: #333;
                    margin-bottom: 5px;
                }}
                .agent-voice {{ 
                    color: #666; 
                    font-size: 14px;
                }}
                .voice-controls {{ 
                    margin: 30px 0; 
                }}
                .talk-button {{ 
                    width: 120px; 
                    height: 120px; 
                    border-radius: 50%; 
                    border: none; 
                    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                    color: white; 
                    font-size: 18px; 
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
                    margin: 0 auto 20px;
                    display: block;
                }}
                .talk-button:hover {{ 
                    transform: scale(1.05);
                    box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4);
                }}
                .talk-button:active {{ 
                    transform: scale(0.95);
                }}
                .talk-button.listening {{ 
                    background: linear-gradient(45deg, #00b894, #00a085);
                    animation: pulse 1.5s infinite;
                }}
                .talk-button.speaking {{ 
                    background: linear-gradient(45deg, #fdcb6e, #e17055);
                    animation: pulse 1.5s infinite;
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.1); }}
                    100% {{ transform: scale(1); }}
                }}
                .status {{ 
                    margin: 20px 0; 
                    padding: 15px; 
                    border-radius: 10px; 
                    font-weight: 500;
                }}
                .status.connected {{ background: #d4edda; color: #155724; }}
                .status.connecting {{ background: #fff3cd; color: #856404; }}
                .status.error {{ background: #f8d7da; color: #721c24; }}
                .status.listening {{ background: #cce5ff; color: #004085; }}
                .status.speaking {{ background: #ffeaa7; color: #6c5ce7; }}
                .conversation {{ 
                    margin: 20px 0; 
                    max-height: 200px; 
                    overflow-y: auto; 
                    text-align: left;
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                }}
                .message {{ 
                    margin: 10px 0; 
                    padding: 10px; 
                    border-radius: 8px; 
                }}
                .user-message {{ 
                    background: #e3f2fd; 
                    text-align: right; 
                }}
                .agent-message {{ 
                    background: #f3e5f5; 
                }}
                .system-message {{ 
                    background: #fff3e0; 
                    font-style: italic; 
                    text-align: center;
                }}
                .controls {{ 
                    margin-top: 20px; 
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                }}
                .control-btn {{ 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 25px; 
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }}
                .connect-btn {{ 
                    background: #667eea; 
                    color: white; 
                }}
                .disconnect-btn {{ 
                    background: #ff6b6b; 
                    color: white; 
                }}
                .control-btn:hover {{ 
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .control-btn:disabled {{ 
                    background: #ccc; 
                    cursor: not-allowed;
                    transform: none;
                    box-shadow: none;
                }}
                .hidden {{ display: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="agent-info">
                    <div class="agent-avatar">🤖</div>
                    <div class="agent-name">{agent.name}</div>
                    <div class="agent-voice">Voice: {agent.voice_type or 'Default'}</div>
                </div>
                
                <div id="status" class="status connecting">Connecting...</div>
                
                <div class="voice-controls">
                    <button id="talkButton" class="talk-button" onclick="toggleListening()" disabled>
                        🎤<br>Talk
                    </button>
                </div>
                
                <div id="conversation" class="conversation hidden">
                    <div class="message system-message">Conversation started with {agent.name}</div>
                </div>
                
                <div class="controls">
                    <button id="connectBtn" class="control-btn connect-btn" onclick="connect()">Connect</button>
                    <button id="disconnectBtn" class="control-btn disconnect-btn" onclick="disconnect()" disabled>Disconnect</button>
                </div>
            </div>
            
            <script>
                let ws = null;
                let isConnected = false;
                let isListening = false;
                let isSpeaking = false;
                let recognition = null;
                let synthesis = null;
                
                // Initialize speech recognition
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    
                    recognition.onstart = function() {{
                        isListening = true;
                        updateTalkButton('listening', '🔴<br>Listening...');
                        updateStatus('listening', 'Listening...');
                    }};
                    
                    recognition.onresult = function(event) {{
                        const transcript = event.results[0][0].transcript;
                        sendSpeechText(transcript);
                    }};
                    
                    recognition.onerror = function(event) {{
                        console.error('Speech recognition error:', event.error);
                        isListening = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    recognition.onend = function() {{
                        isListening = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                }} else {{
                    console.warn('Speech recognition not supported');
                }}
                
                // Initialize speech synthesis
                if ('speechSynthesis' in window) {{
                    synthesis = window.speechSynthesis;
                }}
                
                function connect() {{
                    const wsUrl = `ws://localhost:8001/api/v1/live-voice/live/{agent_id}`;
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {{
                        isConnected = true;
                        updateStatus('connected', 'Connected! Click the button to start talking');
                        document.getElementById('connectBtn').disabled = true;
                        document.getElementById('disconnectBtn').disabled = false;
                        document.getElementById('talkButton').disabled = false;
                        document.getElementById('conversation').classList.remove('hidden');
                    }};
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                    }};
                    
                    ws.onclose = function(event) {{
                        isConnected = false;
                        updateStatus('error', 'Disconnected');
                        document.getElementById('connectBtn').disabled = false;
                        document.getElementById('disconnectBtn').disabled = true;
                        document.getElementById('talkButton').disabled = true;
                    }};
                    
                    ws.onerror = function(error) {{
                        console.error('WebSocket error:', error);
                        updateStatus('error', 'Connection error');
                    }};
                }}
                
                function disconnect() {{
                    if (ws) {{
                        ws.close();
                    }}
                }}
                
                function toggleListening() {{
                    if (!recognition) {{
                        alert('Speech recognition not supported in this browser');
                        return;
                    }}
                    
                    if (isListening) {{
                        recognition.stop();
                    }} else {{
                        recognition.start();
                    }}
                }}
                
                function sendSpeechText(text) {{
                    if (!isConnected) return;
                    
                    ws.send(JSON.stringify({{
                        type: 'speech_text',
                        text: text
                    }}));
                    
                    addMessage('user', text);
                }}
                
                function handleMessage(data) {{
                    switch(data.type) {{
                        case 'connection_established':
                            addMessage('system', data.message);
                            break;
                        case 'speech_received':
                            // Visual feedback that speech was received
                            break;
                        case 'agent_thinking':
                            updateStatus('connected', data.message);
                            break;
                        case 'agent_response':
                            addMessage('agent', data.message);
                            
                            // Speak the response
                            if (data.should_speak && synthesis) {{
                                speakText(data.message, data.voice_type);
                            }}
                            break;
                        case 'listening_started':
                            updateStatus('listening', data.message);
                            break;
                        case 'listening_stopped':
                            updateStatus('connected', data.message);
                            break;
                        case 'error':
                            addMessage('system', 'Error: ' + data.message);
                            break;
                        case 'pong':
                            console.log('Pong received');
                            break;
                    }}
                }}
                
                function speakText(text, voiceType) {{
                    if (!synthesis) return;
                    
                    isSpeaking = true;
                    updateTalkButton('speaking', '🔊<br>Speaking...');
                    updateStatus('speaking', '{agent.name} is speaking...');
                    
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = 0.9;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    
                    // Try to set voice type if available
                    if (voiceType) {{
                        const voices = synthesis.getVoices();
                        const preferredVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes(voiceType.toLowerCase()) ||
                            voice.lang.includes('en')
                        );
                        if (preferredVoice) {{
                            utterance.voice = preferredVoice;
                        }}
                    }}
                    
                    utterance.onend = function() {{
                        isSpeaking = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    utterance.onerror = function(event) {{
                        console.error('Speech synthesis error:', event.error);
                        isSpeaking = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    synthesis.speak(utterance);
                }}
                
                function addMessage(sender, text) {{
                    const conversation = document.getElementById('conversation');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{sender}}-message`;
                    messageDiv.innerHTML = `<strong>${{sender.charAt(0).toUpperCase() + sender.slice(1)}}:</strong> ${{text}}`;
                    conversation.appendChild(messageDiv);
                    conversation.scrollTop = conversation.scrollHeight;
                }}
                
                function updateStatus(type, message) {{
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = `status ${{type}}`;
                    statusDiv.textContent = message;
                }}
                
                function updateTalkButton(state, text) {{
                    const button = document.getElementById('talkButton');
                    button.className = `talk-button ${{state}}`;
                    button.innerHTML = text;
                }}
                
                // Auto-connect on page load
                window.addEventListener('load', function() {{
                    connect();
                }});
                
                // Ping every 30 seconds to keep connection alive
                setInterval(function() {{
                    if (isConnected && ws) {{
                        ws.send(JSON.stringify({{type: 'ping'}}));
                    }}
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, media_type="text/html")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate talk page: {str(e)}"
        )

@router.get("/test-talk/{agent_id}",
           summary="Test Talk to Assistant (No Auth)",
           description="Get the Talk to Assistant page for a specific agent without authentication",
           response_class=HTMLResponse)
async def get_talk_to_assistant_page_test(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the "Talk to Assistant" page - live voice conversation interface (NO AUTH REQUIRED)
    
    This endpoint provides a complete voice conversation interface for testing agents:
    - Real-time WebSocket connection
    - Speech-to-text conversion
    - AI processing with OpenAI
    - Text-to-speech responses
    - No authentication required for testing
    
    Args:
        agent_id: UUID of the agent to talk to
        
    Returns:
        HTML page with voice conversation interface
    """
    try:
        # Validate agent exists (without tenant check for testing)
        try:
            agent_uuid = uuid.UUID(agent_id)
            agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Invalid agent ID: {agent_id}"
            )
        
        # Create the live voice conversation interface
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Talk to {agent.name}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{ 
                    background: white; 
                    border-radius: 20px; 
                    padding: 40px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    max-width: 500px;
                    width: 90%;
                    text-align: center;
                }}
                .agent-info {{ 
                    margin-bottom: 30px; 
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 15px;
                }}
                .agent-avatar {{ 
                    width: 80px; 
                    height: 80px; 
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    border-radius: 50%; 
                    margin: 0 auto 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 32px;
                    color: white;
                }}
                .agent-name {{ 
                    font-size: 24px; 
                    font-weight: 600; 
                    color: #333;
                    margin-bottom: 5px;
                }}
                .agent-voice {{ 
                    color: #666; 
                    font-size: 14px;
                }}
                .voice-controls {{ 
                    margin: 30px 0; 
                }}
                .talk-button {{ 
                    width: 120px; 
                    height: 120px; 
                    border-radius: 50%; 
                    border: none; 
                    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                    color: white; 
                    font-size: 18px; 
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
                    margin: 0 auto 20px;
                    display: block;
                }}
                .talk-button:hover {{ 
                    transform: scale(1.05);
                    box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4);
                }}
                .talk-button:active {{ 
                    transform: scale(0.95);
                }}
                .talk-button.listening {{ 
                    background: linear-gradient(45deg, #00b894, #00a085);
                    animation: pulse 1.5s infinite;
                }}
                .talk-button.speaking {{ 
                    background: linear-gradient(45deg, #fdcb6e, #e17055);
                    animation: pulse 1.5s infinite;
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.1); }}
                    100% {{ transform: scale(1); }}
                }}
                .status {{ 
                    margin: 20px 0; 
                    padding: 15px; 
                    border-radius: 10px; 
                    font-weight: 500;
                }}
                .status.connected {{ background: #d4edda; color: #155724; }}
                .status.connecting {{ background: #fff3cd; color: #856404; }}
                .status.error {{ background: #f8d7da; color: #721c24; }}
                .status.listening {{ background: #cce5ff; color: #004085; }}
                .status.speaking {{ background: #ffeaa7; color: #6c5ce7; }}
                .conversation {{ 
                    margin: 20px 0; 
                    max-height: 200px; 
                    overflow-y: auto; 
                    text-align: left;
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                }}
                .message {{ 
                    margin: 10px 0; 
                    padding: 10px; 
                    border-radius: 8px; 
                }}
                .user-message {{ 
                    background: #e3f2fd; 
                    text-align: right; 
                }}
                .agent-message {{ 
                    background: #f3e5f5; 
                }}
                .system-message {{ 
                    background: #fff3e0; 
                    font-style: italic; 
                    text-align: center;
                }}
                .controls {{ 
                    margin-top: 20px; 
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                }}
                .control-btn {{ 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 25px; 
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }}
                .connect-btn {{ 
                    background: #667eea; 
                    color: white; 
                }}
                .disconnect-btn {{ 
                    background: #ff6b6b; 
                    color: white; 
                }}
                .control-btn:hover {{ 
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .control-btn:disabled {{ 
                    background: #ccc; 
                    cursor: not-allowed;
                    transform: none;
                    box-shadow: none;
                }}
                .hidden {{ display: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="agent-info">
                    <div class="agent-avatar">🤖</div>
                    <div class="agent-name">{agent.name}</div>
                    <div class="agent-voice">Voice: {agent.voice_type or 'Default'}</div>
                </div>
                
                <div id="status" class="status connecting">Connecting...</div>
                
                <div class="voice-controls">
                    <button id="talkButton" class="talk-button" onclick="toggleListening()" disabled>
                        🎤<br>Talk
                    </button>
                </div>
                
                <div id="conversation" class="conversation hidden">
                    <div class="message system-message">Conversation started with {agent.name}</div>
                </div>
                
                <div class="controls">
                    <button id="connectBtn" class="control-btn connect-btn" onclick="connect()">Connect</button>
                    <button id="disconnectBtn" class="control-btn disconnect-btn" onclick="disconnect()" disabled>Disconnect</button>
                </div>
            </div>
            
            <script>
                let ws = null;
                let isConnected = false;
                let isListening = false;
                let isSpeaking = false;
                let recognition = null;
                let synthesis = null;
                
                // Initialize speech recognition
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    
                    recognition.onstart = function() {{
                        isListening = true;
                        updateTalkButton('listening', '🔴<br>Listening...');
                        updateStatus('listening', 'Listening...');
                    }};
                    
                    recognition.onresult = function(event) {{
                        const transcript = event.results[0][0].transcript;
                        sendSpeechText(transcript);
                    }};
                    
                    recognition.onerror = function(event) {{
                        console.error('Speech recognition error:', event.error);
                        isListening = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    recognition.onend = function() {{
                        isListening = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                }} else {{
                    console.warn('Speech recognition not supported');
                }}
                
                // Initialize speech synthesis
                if ('speechSynthesis' in window) {{
                    synthesis = window.speechSynthesis;
                }}
                
                function connect() {{
                    const wsUrl = `ws://localhost:8001/api/v1/live-voice/live/{agent_id}`;
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function(event) {{
                        isConnected = true;
                        updateStatus('connected', 'Connected! Click the button to start talking');
                        document.getElementById('connectBtn').disabled = true;
                        document.getElementById('disconnectBtn').disabled = false;
                        document.getElementById('talkButton').disabled = false;
                        document.getElementById('conversation').classList.remove('hidden');
                    }};
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                    }};
                    
                    ws.onclose = function(event) {{
                        isConnected = false;
                        updateStatus('error', 'Disconnected');
                        document.getElementById('connectBtn').disabled = false;
                        document.getElementById('disconnectBtn').disabled = true;
                        document.getElementById('talkButton').disabled = true;
                    }};
                    
                    ws.onerror = function(error) {{
                        console.error('WebSocket error:', error);
                        updateStatus('error', 'Connection error');
                    }};
                }}
                
                function disconnect() {{
                    if (ws) {{
                        ws.close();
                    }}
                }}
                
                function toggleListening() {{
                    if (!recognition) {{
                        alert('Speech recognition not supported in this browser');
                        return;
                    }}
                    
                    if (isListening) {{
                        recognition.stop();
                    }} else {{
                        recognition.start();
                    }}
                }}
                
                function sendSpeechText(text) {{
                    if (!isConnected) return;
                    
                    ws.send(JSON.stringify({{
                        type: 'speech_text',
                        text: text
                    }}));
                    
                    addMessage('user', text);
                }}
                
                function handleMessage(data) {{
                    switch(data.type) {{
                        case 'connection_established':
                            addMessage('system', data.message);
                            break;
                        case 'speech_received':
                            // Visual feedback that speech was received
                            break;
                        case 'agent_thinking':
                            updateStatus('connected', data.message);
                            break;
                        case 'agent_response':
                            addMessage('agent', data.message);
                            
                            // Speak the response
                            if (data.should_speak && synthesis) {{
                                speakText(data.message, data.voice_type);
                            }}
                            break;
                        case 'listening_started':
                            updateStatus('listening', data.message);
                            break;
                        case 'listening_stopped':
                            updateStatus('connected', data.message);
                            break;
                        case 'error':
                            addMessage('system', 'Error: ' + data.message);
                            break;
                        case 'pong':
                            console.log('Pong received');
                            break;
                    }}
                }}
                
                function speakText(text, voiceType) {{
                    if (!synthesis) return;
                    
                    isSpeaking = true;
                    updateTalkButton('speaking', '🔊<br>Speaking...');
                    updateStatus('speaking', '{agent.name} is speaking...');
                    
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = 0.9;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    
                    // Try to set voice type if available
                    if (voiceType) {{
                        const voices = synthesis.getVoices();
                        const preferredVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes(voiceType.toLowerCase()) ||
                            voice.lang.includes('en')
                        );
                        if (preferredVoice) {{
                            utterance.voice = preferredVoice;
                        }}
                    }}
                    
                    utterance.onend = function() {{
                        isSpeaking = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    utterance.onerror = function(event) {{
                        console.error('Speech synthesis error:', event.error);
                        isSpeaking = false;
                        updateTalkButton('idle', '🎤<br>Talk');
                        updateStatus('connected', 'Ready to talk');
                    }};
                    
                    synthesis.speak(utterance);
                }}
                
                function addMessage(sender, text) {{
                    const conversation = document.getElementById('conversation');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{sender}}-message`;
                    messageDiv.innerHTML = `<strong>${{sender.charAt(0).toUpperCase() + sender.slice(1)}}:</strong> ${{text}}`;
                    conversation.appendChild(messageDiv);
                    conversation.scrollTop = conversation.scrollHeight;
                }}
                
                function updateStatus(type, message) {{
                    const statusDiv = document.getElementById('status');
                    statusDiv.className = `status ${{type}}`;
                    statusDiv.textContent = message;
                }}
                
                function updateTalkButton(state, text) {{
                    const button = document.getElementById('talkButton');
                    button.className = `talk-button ${{state}}`;
                    button.innerHTML = text;
                }}
                
                // Auto-connect on page load
                window.addEventListener('load', function() {{
                    connect();
                }});
                
                // Ping every 30 seconds to keep connection alive
                setInterval(function() {{
                    if (isConnected && ws) {{
                        ws.send(JSON.stringify({{type: 'ping'}}));
                    }}
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, media_type="text/html")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate talk page: {str(e)}"
        )
