from typing import Optional

from app.core.logger import logger


def get_gather_language(agent) -> str:
    """Get language code for Twilio Gather based on agent language."""
    if not agent or not getattr(agent, "language", None):
        return "en-US"

    # Map agent language to Twilio supported languages
    language_map = {
        "en": "en-US",
        "es": "es-ES",
        "hi": "hi-IN",
        "ar": "ar-SA",
        "zh": "zh-CN",
        "ur": "ur-PK",
    }

    return language_map.get(agent.language, "en-US")


def get_agent_voice(agent) -> str:
    """Get the appropriate Twilio voice based on agent's voice type and language."""
    if not agent:
        return "Polly.Joanna"  # Default female voice

    # Get voice type and language from agent
    voice_type: Optional[str] = getattr(agent, "voice_type", None)
    language: Optional[str] = getattr(agent, "language", None)

    # Voice mapping based on language and gender using correct Twilio voice names
    voice_map = {
        # English voices
        "en": {
            "male": "Polly.Matthew",
            "female": "Polly.Joanna",
        },
        # Spanish voices
        "es": {
            "male": "Polly.Miguel",
            "female": "Polly.Penelope",
        },
        # Hindi voices
        "hi": {
            "male": "Polly.Aditi",
            "female": "Polly.Aditi",
        },
        # Arabic voices
        "ar": {
            "male": "Polly.Zeina",
            "female": "Polly.Zeina",
        },
        # Chinese voices
        "zh": {
            "male": "Polly.Zhiyu",
            "female": "Polly.Zhiyu",
        },
        # Urdu voices
        "ur": {
            "male": "Polly.Aditi",
            "female": "Polly.Aditi",
        },
    }

    # Default to English if language not specified
    if not language:
        language = "en"

    # Default to female if voice type not specified
    if not voice_type:
        voice_type = "female"

    selected_voice = voice_map.get(language, voice_map["en"]).get(
        voice_type, "Polly.Joanna"
    )

    logger.debug(
        "🎤 Agent voice selection: language=%s, voice_type=%s, selected_voice=%s",
        language,
        voice_type,
        selected_voice,
    )

    return selected_voice

