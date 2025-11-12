"""
TTS Text Preprocessing Middleware

Simple text preprocessing for natural TTS output:
- Normalize abbreviations (Dr. → Doctor)
- Normalize numbers and dates
- Add natural pauses
- Wrap in SSML with prosody and emphasis

Usage:
    from app.middleware.tts_preprocessing_middleware import preprocess_for_tts
    
    ssml_text = preprocess_for_tts("Dr. Smith called at 3:00 p.m.")
    # Returns: <speak>Doctor Smith called at 3, 00, P M.</speak>
"""

import re
from typing import Optional


def normalize_abbreviations(text: str) -> str:
    """Normalize common abbreviations for better pronunciation."""
    abbreviations = {
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Missus',
        r'\bMs\.': 'Miss',
        r'\bProf\.': 'Professor',
        r'\betc\.': 'et cetera',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\ba\.m\.': 'A M',
        r'\bA\.M\.': 'A M',
        r'\bp\.m\.': 'P M',
        r'\bP\.M\.': 'P M',
    }
    
    result = text
    for abbr, full in abbreviations.items():
        result = re.sub(abbr, full, result, flags=re.IGNORECASE)
    
    return result


def normalize_numbers(text: str) -> str:
    """Normalize numbers and currency for better pronunciation."""
    result = text
    
    # Currency: $100 → "100 dollars"
    result = re.sub(r'\$(\d+)', r'\1 dollars', result)
    
    # Percentages: 25% → "25 percent"
    result = re.sub(r'(\d+)%', r'\1 percent', result)
    
    # Phone numbers: add commas for pauses
    result = re.sub(r'(\d{3})[-.]?(\d{3})[-.]?(\d{4})', r'\1, \2, \3', result)
    
    # Dates: 12/25/2024 → "12, 25, 2024"
    result = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1, \2, \3', result)
    
    # Times: 3:30 → "3, 30"
    result = re.sub(r'(\d{1,2}):(\d{2})', r'\1, \2', result)
    
    return result


def add_natural_pauses(text: str) -> str:
    """Add natural pauses using punctuation."""
    result = text
    
    # Add pause after introductory words
    intro_words = ['Well', 'Actually', 'However', 'Therefore', 'Meanwhile']
    for word in intro_words:
        result = re.sub(rf'\b{word}\b(?!,)', f'{word},', result, flags=re.IGNORECASE)
    
    # Convert ellipsis to comma (for pause)
    result = re.sub(r'\.{3,}', ',', result)
    
    # Clean up duplicate punctuation
    result = re.sub(r'([.!?;,]){2,}', r'\1', result)
    
    # Fix spacing
    result = re.sub(r'\s+([.!?;,])', r'\1', result)
    result = re.sub(r'([.!?;,])([A-Za-z])', r'\1 \2', result)
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


def detect_emotion_prosody(sentence: str, punct: str = "") -> tuple[str, str, str]:
    """
    Detect emotion from text and return appropriate prosody settings.
    
    Returns:
        (rate, pitch, volume) tuple
        
    Examples:
        "How are you?" → ("102%", "+2st", "medium")  # Question
        "Amazing!" → ("108%", "+3st", "loud")        # Excitement
        "I'm sorry" → ("88%", "-2st", "soft")        # Sadness
    """
    import random
    
    text = sentence.lower()
    
    # EXCITEMENT: ! or words like "amazing", "great", "awesome"
    excitement_words = ["amazing", "awesome", "great", "wonderful", "fantastic", "excellent", "love", "excited"]
    if punct == "!" or any(word in text for word in excitement_words):
        return (
            random.choice(["105%", "108%", "110%"]),  # Faster
            random.choice(["+2st", "+3st"]),           # Higher pitch
            "loud"
        )
    
    # QUESTION: ? or question words
    question_words = ["how", "what", "when", "where", "why", "who", "can", "could", "would", "should"]
    if punct == "?" or any(text.startswith(word) for word in question_words):
        return (
            random.choice(["100%", "102%", "105%"]),  # Slightly faster
            random.choice(["+1st", "+2st"]),           # Higher pitch at end
            "medium"
        )
    
    # SADNESS/APOLOGY: words like "sorry", "sad", "unfortunately"
    sad_words = ["sorry", "sad", "unfortunately", "apologize", "regret", "disappointed"]
    if any(word in text for word in sad_words):
        return (
            random.choice(["85%", "88%", "90%"]),     # Slower
            random.choice(["-2st", "-1st"]),           # Lower pitch
            "soft"
        )
    
    # EMPHASIS: ALL CAPS words or important words
    if any(word.isupper() and len(word) > 1 for word in sentence.split()):
        return (
            random.choice(["98%", "100%", "102%"]),   # Normal to slightly fast
            random.choice(["0st", "+1st"]),            # Normal to slightly higher
            "loud"
        )
    
    # NEUTRAL: default natural variation
    return (
        random.choice(["95%", "98%", "100%", "102%"]),  # Natural variation
        random.choice(["-1st", "0st", "+1st"]),          # Subtle pitch variation
        "medium"
    )


def add_emphasis_tags(text: str) -> str:
    """
    Add emphasis tags to ALL CAPS words.
    
    Example:
        "This is VERY important" → "This is <emphasis level='strong'>VERY</emphasis> important"
    """
    # Find ALL CAPS words (2+ letters)
    def replace_caps(match):
        word = match.group(1)
        return f'<emphasis level="strong">{word.title()}</emphasis>'
    
    return re.sub(r'\b([A-Z]{2,})\b', replace_caps, text)


def wrap_in_ssml(text: str, add_prosody: bool = True, add_emotion: bool = True) -> str:
    """
    Wrap text in SSML with prosody, emotion, and breaks.
    
    NEW: Emotion detection for natural speech!
    - Questions → higher pitch
    - Excitement → faster + louder
    - Sadness → slower + softer
    - Emphasis → loud + emphasis tags
    
    Example:
        Input: "Amazing! How can I help you?"
        Output: <speak>
                  <prosody rate="108%" pitch="+3st" volume="loud">Amazing!</prosody>
                  <break time="200ms"/>
                  <prosody rate="102%" pitch="+2st" volume="medium">How can I help you?</prosody>
                </speak>
    """
    if not text or not text.strip():
        return ""
    
    ssml = '<speak>\n'
    
    # Split by sentences
    sentences = re.split(r'([.!?])', text)
    
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if not sentence:
            continue
        
        # Add emphasis tags to CAPS words
        if add_emotion:
            sentence = add_emphasis_tags(sentence)
        
        # Add prosody with emotion detection
        if add_prosody and add_emotion:
            # Detect emotion and get prosody settings
            rate, pitch, volume = detect_emotion_prosody(sentence, punct)
            ssml += f'  <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{sentence}{punct}</prosody>\n'
        elif add_prosody:
            # Basic prosody without emotion
            import random
            rate = random.choice(["95%", "98%", "100%", "102%"])
            ssml += f'  <prosody rate="{rate}">{sentence}{punct}</prosody>\n'
        else:
            ssml += f'  {sentence}{punct}\n'
        
        # Add break after sentences (longer for questions/excitement)
        if punct in ['!', '?']:
            ssml += '  <break time="250ms"/>\n'  # Longer pause for emotion
        elif punct == '.':
            ssml += '  <break time="200ms"/>\n'
        elif punct == ',':
            ssml += '  <break time="100ms"/>\n'
    
    # Add remaining text
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        last = sentences[-1].strip()
        if add_emotion:
            last = add_emphasis_tags(last)
        if add_prosody and add_emotion:
            rate, pitch, volume = detect_emotion_prosody(last, "")
            ssml += f'  <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{last}</prosody>\n'
        elif add_prosody:
            import random
            rate = random.choice(["95%", "98%", "100%", "102%"])
            ssml += f'  <prosody rate="{rate}">{last}</prosody>\n'
        else:
            ssml += f'  {last}\n'
    
    ssml += '</speak>'
    
    return ssml


def preprocess_for_tts(
    text: str,
    use_ssml: bool = True,
    add_prosody: bool = True,
    add_emotion: bool = True,
    normalize: bool = True
) -> str:
    """
    Main preprocessing function - use this!
    
    Complete pipeline:
    1. Normalize abbreviations (Dr. → Doctor)
    2. Normalize numbers and currency
    3. Add natural pauses
    4. Detect emotions and apply prosody
    5. Wrap in SSML (optional)
    
    NEW: Emotion Detection!
    - Questions → higher pitch (+2st)
    - Excitement → faster + louder (110% rate)
    - Sadness → slower + softer (85% rate)
    - Emphasis (CAPS) → loud + emphasis tags
    
    Args:
        text: Raw text from LLM
        use_ssml: Wrap result in SSML tags
        add_prosody: Add prosody variations
        add_emotion: Enable emotion detection (NEW!)
        normalize: Enable normalization (abbreviations, numbers)
    
    Returns:
        Preprocessed text (with SSML + emotion if enabled)
    
    Examples:
        >>> preprocess_for_tts("Amazing! How are you?")
        '<speak>
          <prosody rate="108%" pitch="+3st" volume="loud">Amazing!</prosody>
          <break time="250ms"/>
          <prosody rate="102%" pitch="+2st" volume="medium">How are you?</prosody>
        </speak>'
        
        >>> preprocess_for_tts("I'm sorry about that")
        '<speak>
          <prosody rate="88%" pitch="-2st" volume="soft">I\'m sorry about that</prosody>
        </speak>'
        
        >>> preprocess_for_tts("This is VERY important")
        '<speak>
          <prosody rate="100%" pitch="+1st" volume="loud">This is <emphasis level="strong">Very</emphasis> important</prosody>
        </speak>'
    """
    if not text or not text.strip():
        return ""
    
    result = text.strip()
    
    # Step 1: Normalize abbreviations and numbers
    if normalize:
        result = normalize_abbreviations(result)
        result = normalize_numbers(result)
    
    # Step 2: Add natural pauses
    result = add_natural_pauses(result)
    
    # Step 3: Wrap in SSML with emotion detection
    if use_ssml:
        result = wrap_in_ssml(result, add_prosody=add_prosody, add_emotion=add_emotion)
    
    return result


# ============================================
# Quick helpers for common use cases
# ============================================

def quick_clean(text: str) -> str:
    """
    Quick text cleaning without SSML (fast path).
    Use for simple phrases or cached content.
    """
    if not text:
        return ""
    
    result = text.strip()
    
    # Quick fixes only
    result = normalize_abbreviations(result)
    result = re.sub(r'\.{3,}', ',', result)  # ... → ,
    result = re.sub(r'([.!?;,]){2,}', r'\1', result)  # !!! → !
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


def add_emphasis(text: str, word: str) -> str:
    """
    Add emphasis to a specific word in SSML.
    
    Example:
        >>> add_emphasis("This is important", "important")
        'This is <emphasis level="strong">important</emphasis>'
    """
    pattern = rf'\b{re.escape(word)}\b'
    return re.sub(pattern, f'<emphasis level="strong">{word}</emphasis>', text)


def add_break(text: str, duration_ms: int = 200) -> str:
    """
    Add a break tag at the end of text.
    
    Example:
        >>> add_break("Hello there", 300)
        'Hello there <break time="300ms"/>'
    """
    return f'{text} <break time="{duration_ms}ms"/>'

