from typing import List, Dict, Any, Optional
from twilio.twiml.voice_response import VoiceResponse
import uuid


class VoiceAgentManager:
    """Manages voice agent logic for handling calls"""
    
    def __init__(self):
        self.active_agents = {}
        self.call_agents = {}
    
    def register_agent(self, agent_id: str, capabilities: List[str] = None) -> bool:
        """Register an agent for handling calls"""
        self.active_agents[agent_id] = {
            'capabilities': capabilities or ['voice', 'general'],
            'status': 'available',
            'current_call': None
        }
        return True
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self.active_agents.get(agent_id)
    
    def assign_call_to_agent(self, call_sid: str, agent_id: str) -> bool:
        """Assign a call to a specific agent"""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.call_agents[call_sid] = agent_id
        self.active_agents[agent_id]['current_call'] = call_sid
        self.active_agents[agent_id]['status'] = 'busy'
        return True
    
    def release_agent_from_call(self, call_sid: str) -> bool:
        """Release an agent from a call"""
        if call_sid in self.call_agents:
            agent_id = self.call_agents[call_sid]
            if agent_id in self.active_agents:
                self.active_agents[agent_id]['current_call'] = None
                self.active_agents[agent_id]['status'] = 'available'
            del self.call_agents[call_sid]
            return True
        return False
    
    def generate_agent_response(self, agent_id: str, call_data: Dict[str, Any]) -> str:
        """Generate TwiML response based on agent logic"""
        agent = self.get_agent(agent_id)
        if not agent:
            return self._generate_default_response()
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Agent-specific greeting
        response.say(f"Hello! This is agent {agent_id} speaking. How can I help you today?", voice="alice")
        
        # Add gather to collect user input
        gather = response.gather(
            input='speech',
            timeout=10,
            speech_timeout='auto',
            action=f'/api/v1/voice/gather?agentId={agent_id}',
            method='POST'
        )
        gather.say("Please tell me how I can assist you.", voice="alice")
        
        # Fallback if no input
        response.say("I didn't catch that. Let me transfer you to a human agent.", voice="alice")
        response.redirect('/api/v1/voice/transfer')
        
        return str(response)
    
    def _generate_default_response(self) -> str:
        """Generate default TwiML response"""
        response = VoiceResponse()
        response.say("Thank you for calling. An agent will be with you shortly.", voice="alice")
        response.pause(length=2)
        response.say("Please hold while we connect you.", voice="alice")
        return str(response)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get overall agent status"""
        return {
            "agents": self.active_agents,
            "call_assignments": self.call_agents,
            "total_agents": len(self.active_agents),
            "busy_agents": len([a for a in self.active_agents.values() if a['status'] == 'busy'])
        }


# Initialize global voice agent manager
voice_agent_manager = VoiceAgentManager()

# Register some sample agents
voice_agent_manager.register_agent("agent_12345", ["voice", "support", "sales"])
voice_agent_manager.register_agent("agent_67890", ["voice", "technical", "billing"])
