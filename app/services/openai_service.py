"""
OpenAI Service Module
Handles all OpenAI-related operations including text generation and chat completions
"""

from app.core.config import settings
from app.core.openai_client import get_openai_client, get_async_openai_client
from typing import List, Dict, Any
import time

# GPT-5.x is a reasoning-model family (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.2,
# gpt-5.4, ...). Per OpenAI docs these do not support a customizable
# `temperature` on Chat Completions (only the default is accepted) and use
# `max_completion_tokens` instead of `max_tokens`. gpt-4.x / gpt-4o / o1 / o3
# etc. are untouched — this only applies to the "gpt-5" prefix.
_REASONING_MODEL_PREFIX = "gpt-5"

# `max_completion_tokens` on gpt-5.x is a SHARED budget covering invisible
# reasoning tokens + the visible completion. Voice agents default to a small
# conversational max_tokens (e.g. 100, see agent_runtime.resolve_llm_runtime's
# default) which is fine for gpt-4.x but on gpt-5.x can be entirely consumed
# by reasoning, yielding content="" + finish_reason="length" with no
# exception raised (silent dead air on a live call). Floor the budget so a
# low conversational setting can never starve the reasoning family. Applies
# ONLY to gpt-5* — every other model's max_tokens is passed through unchanged.
_REASONING_MODEL_MIN_COMPLETION_TOKENS = 1500


def _is_reasoning_model(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith(_REASONING_MODEL_PREFIX)


def _completion_params(model_name: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Build the model/temperature/max-tokens kwargs for a Chat Completions call.

    Isolated here so all OpenAI call sites in this service (streaming and
    non-streaming) apply the gpt-5.x parameter differences identically.
    """
    if _is_reasoning_model(model_name):
        # temperature intentionally omitted: gpt-5.x rejects/ignores non-default
        # temperature on Chat Completions.
        floored_max_tokens = max(int(max_tokens or 0), _REASONING_MODEL_MIN_COMPLETION_TOKENS)
        return {"model": model_name, "max_completion_tokens": floored_max_tokens}
    return {"model": model_name, "temperature": temperature, "max_tokens": max_tokens}


class OpenAIService:
    """Service class for handling OpenAI operations"""

    def __init__(self):
        self._clients = {}  # Store clients by API key
        self._async_clients = {}  # Store async clients by API key (streaming hot path)
        self._current_api_key = None

    def get_client(self, api_key: str = None):
        """Get or create OpenAI client with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.OPENAI_API_KEY

        if not key_to_use:
            raise Exception("OpenAI API key not found. Please provide an API key or set OPENAI_API_KEY in your config.")

        # Return existing client or create new one for this API key
        if key_to_use not in self._clients:
            self._clients[key_to_use] = get_openai_client(key_to_use)

        return self._clients[key_to_use]

    def get_async_client(self, api_key: str = None):
        """Get or create an AsyncOpenAI client with specific API key.

        Used for the streaming hot path (stream_text) so network I/O never
        blocks the event loop — the sync client used elsewhere in this
        service is intentionally left untouched for non-streaming call sites.
        """
        key_to_use = api_key or settings.OPENAI_API_KEY

        if not key_to_use:
            raise Exception("OpenAI API key not found. Please provide an API key or set OPENAI_API_KEY in your config.")

        if key_to_use not in self._async_clients:
            self._async_clients[key_to_use] = get_async_openai_client(key_to_use)

        return self._async_clients[key_to_use]

    def embed_text(
        self,
        text: str,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
    ) -> List[float]:
        """
        Generate an embedding vector for a single text input.
        """
        client = self.get_client(api_key)
        response = client.embeddings.create(
            model=model_name,
            input=text,
        )
        # OpenAI returns a list of data objects; we take the first embedding
        return list(response.data[0].embedding)
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     model_name: str = "gpt-3.5-turbo", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000,
                     api_key: str = None) -> Dict[str, Any]:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: The input prompt for text generation
            system_prompt: System prompt to set the context
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate content
            response = client.chat.completions.create(
                messages=messages,
                **_completion_params(model_name, temperature, max_tokens),
            )

            end_time = time.time()
            response_time = end_time - start_time

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            raise Exception(f"Error in OpenAI text generation: {str(e)}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       system_prompt: str = None, 
                       model_name: str = "gpt-3.5-turbo", 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       api_key: str = None) -> Dict[str, Any]:
        """
        Generate chat completion using OpenAI API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to use for the conversation
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages with system prompt
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            api_messages.extend(messages)
            
            # Generate chat completion
            response = client.chat.completions.create(
                messages=api_messages,
                **_completion_params(model_name, temperature, max_tokens),
            )

            end_time = time.time()
            response_time = end_time - start_time

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            raise Exception(f"Error in OpenAI chat completion: {str(e)}")
    
    async def stream_text(self, prompt: str, system_prompt: str = None,
                          model_name: str = "gpt-3.5-turbo",
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          api_key: str = None):
        """
        Stream text from OpenAI as it's generated (async generator).
        Yields text chunks as they arrive.
        
        Args:
            prompt: The input prompt
            system_prompt: System prompt to set the context
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Yields:
            Text chunks as strings
        """
        try:
            # Get async client instance with specific API key — this is the
            # streaming hot path, so we must never block the event loop with
            # a sync HTTP call / sync SSE iteration here (see AsyncOpenAI).
            client = self.get_async_client(api_key)

            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Stream response
            stream = await client.chat.completions.create(
                messages=messages,
                stream=True,
                **_completion_params(model_name, temperature, max_tokens),
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"Error in OpenAI streaming: {str(e)}")
    
    def text_to_speech(self, text: str, voice: str = "alloy", 
                      model: str = "tts-1", output_format: str = "mp3",
                      api_key: str = None) -> bytes:
        """
        Convert text to speech using OpenAI TTS API
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use
            output_format: Output format (mp3, opus, aac, flac)
            api_key: Model-specific API key (optional)
            
        Returns:
            Audio data as bytes
        """
        try:
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format
            )
            
            return response.content
            
        except Exception as e:
            raise Exception(f"Error in OpenAI text-to-speech: {str(e)}")
    
    def process_agent_conversation(self, user_input: str, agent_system_prompt: str,
                                 conversation_history: List[Dict[str, str]] = None,
                                 model_name: str = "gpt-3.5-turbo",
                                 temperature: float = 0.7,
                                 max_tokens: int = 100,
                                 api_key: str = None) -> Dict[str, Any]:
        """
        Process a conversation turn with an agent

        Args:
            user_input: User's speech input (transcribed text)
            agent_system_prompt: Agent's system prompt
            conversation_history: Previous conversation messages
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)

        Returns:
            Dictionary with agent response and metadata
        """
        start_time = time.time()

        # Prepare messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})

        # Get response from OpenAI
        response = self.chat_completion(
            messages=messages,
            system_prompt=agent_system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        response_time = time.time() - start_time
        
        return {
            "response": response["content"],
            "response_time": response_time,
            "usage": response["usage"],
            "model": response["model"]
        }

# Create a singleton instance
openai_service = OpenAIService()
