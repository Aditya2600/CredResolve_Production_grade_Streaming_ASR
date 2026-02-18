from typing import Any, Dict, Optional

# Default voice mapping
# You can extend this map based on available TTS voices
VOICE_MAP = {
    "hi": "hi-IN-female-1",
    "en": "en-IN-female-1",
    "te": "te-IN-female-1",
    "ta": "ta-IN-female-1",
    "mr": "mr-IN-female-1",
    "bn": "bn-IN-female-1",
    "gu": "gu-IN-female-1",
    "kn": "kn-IN-female-1",
    "ml": "ml-IN-female-1",
    "od": "od-IN-female-1",
    "pa": "pa-IN-female-1",
}

DEFAULT_VOICE = "hi-IN-female-1"

def get_voice_for_language(language: str) -> str:
    """
    Returns the specific TTS voice ID for a given language code.
    Fallback to DEFAULT_VOICE if language is unknown.
    """
    lang = (language or "").strip().lower()
    return VOICE_MAP.get(lang, DEFAULT_VOICE)

def create_tts_payload(
    text: str,
    language: str,
    language_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a payload suitable for downstream TTS services.
    
    Args:
        text: The text to synthesize.
        language: The resolved language code (e.g., 'hi', 'en').
        language_source: Metadata about how the language was resolved.
    
    Returns:
        A dictionary containing the text, language, assigned voice, and metadata.
    """
    voice = get_voice_for_language(language)
    return {
        "text": text,
        "language": language,
        "voice": voice,
        "meta": {
            "language_source": language_source or "unknown"
        }
    }
