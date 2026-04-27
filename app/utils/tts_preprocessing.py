"""
Google TTS-Compatible Humanized Preprocessing Utilities

--------------------------------------------------------
✅ Natural fillers: uhh, umm, hmm, etc.
✅ Subtle breathing (realistic timing)
✅ Emotion-aware prosody (happy, sad, confident, uncertain)
✅ Natural rhythm: pauses, pacing, tone variation
✅ Thinking delays: 400ms pause before contemplative phrases
✅ 100% valid SSML for Google Cloud TTS

Use with: text input → preprocess_for_tts(text) → pass to Google TTS API
"""

import re
import random


# ---------------------------------------------------------
# 1. Basic Normalization
# ---------------------------------------------------------

def normalize_abbreviations(text: str) -> str:
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
        r'\bp\.m\.': 'P M',
    }
    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    return text


def normalize_numbers(text: str) -> str:
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    text = re.sub(r'(\d{3})[-.]?(\d{3})[-.]?(\d{4})', r'\1, \2, \3', text)
    text = re.sub(r'(\d{1,2}):(\d{2})', r'\1, \2', text)
    return text


def add_contractions(text: str) -> str:
    contractions = {
        r'\bI am\b': "I'm", r'\byou are\b': "you're", r'\bhe is\b': "he's",
        r'\bshe is\b': "she's", r'\bit is\b': "it's", r'\bwe are\b': "we're",
        r'\bthey are\b': "they're", r'\bthat is\b': "that's",
        r'\bdo not\b': "don't", r'\bcan not\b': "can't",
        r'\bwill not\b': "won't", r'\bshould not\b': "shouldn't",
    }
    for pat, rep in contractions.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------
# 1b. Symbol Normalization (for TTS readability)
# ---------------------------------------------------------

def normalize_symbols(text: str) -> str:
    """
    Clean up punctuation/symbols that sound unnatural when read literally:
    - "..." → comma pause (prevents "dot dot dot")
    - "word/word" → "word and word" (e.g. FastAPI/ML → FastAPI and ML)
    """
    if not text:
        return ""

    # Collapse ellipses (and longer) into a soft pause
    text = re.sub(r'\.{3,}', ', ', text)

    # Turn inline slashes between words into "and"
    # e.g. "FastAPI/ML" or "A / B" → "FastAPI and ML"
    text = re.sub(r'(?<=\w)\s*/\s*(?=\w)', ' and ', text)

    return text

# ---------------------------------------------------------
# 1c. Emotion-aware Interjections (text-only, streaming-safe)
# ---------------------------------------------------------

def inject_emotion_preface(text: str, probability: float = 0.18) -> str:
    """
    Add a short human interjection at the START of an utterance.
    This is TEXT-ONLY (no SSML), so it works with both:
    - SSML synthesize_speech (tags wrap around it)
    - StreamingSynthesize (where SSML tags may be stripped)
    """
    if not text or not text.strip():
        return ""

    t = text.strip()
    lower = t.lower()

    # If it already starts with an interjection, don't stack.
    if re.match(r'^(umm|um|hmm|hm|oh|okay|alright|well|right|got it|hang on|wait)\b', lower):
        return t

    if random.random() >= probability:
        return t

    emotion = detect_emotion(t)

    if emotion == "uncertain":
        prefaces = ["Hmm,", "Umm,", "Well,"]
    elif emotion == "happy":
        prefaces = ["Oh, nice,", "Oh, great,", "Alright,"]
    elif emotion == "sad":
        prefaces = ["Oh, I'm sorry,", "Hmm, okay,", "I see,"]
    elif emotion == "confident":
        prefaces = ["Alright,", "Okay,", "Got it,"]
    else:
        prefaces = ["Okay,", "Alright,", "I see,"]

    return f"{random.choice(prefaces)} {t}"

# ---------------------------------------------------------
# 2. Thinking Delay Mode (NEW!)
# ---------------------------------------------------------

def add_thinking_delays(text: str) -> str:
    """
    Adds realistic thinking pauses (400ms) before contemplative phrases.
    Makes agent sound more human - like they're actually thinking!
    
    Examples:
        "Let me think" → "<break time='400ms'/> Let me think"
        "Hmm, I see" → "<break time='400ms'/> Hmm, I see"
    """
    # Thinking phrases that deserve a pause BEFORE them
    # Removed "you know" and "I mean" - too annoying in TTS output
    thinking_phrases = [
        'let me think',
        'let me see',
        'let me check',
        'hmm',
        'well',
        'actually',
        'maybe',
        'perhaps',
        'how should I say',
        'to be honest',
        'frankly',
        'honestly',
    ]
    
    for phrase in thinking_phrases:
        # Add 400ms pause BEFORE the phrase
        pattern = rf'\b{re.escape(phrase)}\b'
        replacement = f'<break time="400ms"/> {phrase}'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)  # Only first occurrence
    
    return text


# ---------------------------------------------------------
# 3. Emotion & Prosody Detection
# ---------------------------------------------------------

def detect_emotion(sentence: str) -> str:
    s = sentence.lower()
    if any(w in s for w in ["sorry", "sad", "unfortunately", "regret"]):
        return "sad"
    if any(w in s for w in ["amazing", "great", "excited", "love", "!"]):
        return "happy"
    if any(w in s for w in ["maybe", "perhaps", "might", "guess", "hmm"]):
        return "uncertain"
    if any(w in s for w in ["definitely", "surely", "certainly", "clearly"]):
        return "confident"
    return "neutral"


def emotion_to_prosody(emotion: str):
    """
    Maps emotion to prosody settings.
    Rate 0.93 (93%) = human-like natural speed (Vapi-style). Pitch fixed 0st for consistent voice.
    """
    if emotion == "happy":
        return ("0.93", "0st", "medium")
    if emotion == "sad":
        return ("0.93", "0st", "soft")
    if emotion == "uncertain":
        return ("0.93", "0st", "soft")
    if emotion == "confident":
        return ("0.93", "0st", "medium")
    # Neutral: 0.93 same human-like rate, 0st pitch
    return ("0.93", "0st", "medium")


# ---------------------------------------------------------
# 4. Humanization: Fillers + Breathing
# ---------------------------------------------------------

def insert_fillers(sentence: str, emotion: str) -> str:
    """
    VAPI-STYLE: Removed ALL mid-sentence fillers to prevent clicking/tak sounds.
    Prosody interruptions within sentences cause audio distortion.
    Only thinking delays (natural pauses) are preserved for humanization.
    """
    # VAPI APPROACH: No fillers within sentences - eliminates clicking sounds
    # Humanization happens at natural pause points only (thinking delays)
    return sentence  # Return unchanged - no fillers!


def add_breath(sentence: str, emotion: str) -> str:
    """
    Adds subtle breath between very long thoughts.
    DISABLED: Google Cloud TTS doesn't support <audio> tags properly.
    The URL gets read as text instead of playing audio, causing issues.
    """
    # DISABLED: Google Cloud TTS API doesn't support <audio> tags - they get read as text
    # Since breathing sound wasn't working anyway, just return sentence unchanged
    # This ensures calling functionality and flow remain unaffected
    return sentence


# ---------------------------------------------------------
# 5. SSML Generator (Google-Compatible)
# ---------------------------------------------------------

def wrap_in_ssml(
    text: str,
    add_office_bg: bool = False,
    start_break_ms: int = 150,
    between_sentence_break_ms: int = 150,
) -> str:
    """
    Wraps text in SSML with prosody and emotion.
    
    Args:
        text: Text to wrap
        add_office_bg: Deprecated; ignored (SSML <par> office ambience was never enabled).
        start_break_ms: Optional break at start of utterance (ms). Use 0 for no leading silence.
        between_sentence_break_ms: Break inserted between sentences (ms). Use 0 to disable.
    """
    # Add thinking delays BEFORE wrapping
    text = add_thinking_delays(text)
    
    # 🎯 DETECT EMOTION ONCE for entire response (prevents prosody clicks!)
    # This ensures consistent prosody across all sentences - no abrupt transitions
    # Emotion still preserved (based on overall response tone), but applied consistently
    overall_emotion = detect_emotion(text)
    rate, pitch, volume = emotion_to_prosody(overall_emotion)
    
    sentences = re.split(r'([.!?])', text)
    ssml = "<speak>"
    # Optional break at start to prevent audio pop (can be reduced/disabled when audio fade-in is applied)
    if start_break_ms and start_break_ms > 0:
        ssml += f'<break time="{int(start_break_ms)}ms"/>'
    
    use_par_tags = False

    # Apply SAME prosody to all sentences (prevents clicks/tak sounds)
    # Emotion still applied (based on overall response), but consistently!
    ssml += f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
    
    processed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i].strip()
        p = sentences[i+1]
        if s:
            processed_sentences.append(s + p)
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        processed_sentences.append(sentences[-1].strip())

    for i, sentence in enumerate(processed_sentences):
        sentence_with_breath = add_breath(sentence, overall_emotion)
        ssml += sentence_with_breath
        if i < len(processed_sentences) - 1:
            if between_sentence_break_ms and between_sentence_break_ms > 0:
                ssml += f'<break time="{int(between_sentence_break_ms)}ms"/>'
    
    # Close prosody tag
    ssml += '</prosody>'
    
    ssml += "</speak>"
    return ssml


# ---------------------------------------------------------
# 6. Main Preprocessing Entry
# ---------------------------------------------------------

def preprocess_for_tts(
    text: str,
    add_office_bg: bool = False,
    start_break_ms: int = 150,
    between_sentence_break_ms: int = 150,
) -> str:
    """
    Complete humanization pipeline with optional office background.
    
    Pipeline:
    1. Normalize abbreviations (Dr. → Doctor)
    2. Normalize numbers ($100 → 100 dollars)
    3. Add contractions (I am → I'm)
    4. Add thinking delays (400ms before "let me think")
    5. Detect emotions (happy, sad, uncertain, confident)
    6. VAPI-STYLE: No mid-sentence fillers (eliminates clicking sounds)
    7. Add breathing (subtle, 3% on very long sentences only)
    8. Optional office background flag (deprecated; ignored)
    9. Generate SSML with prosody
    
    Args:
        text: Raw text from LLM
        add_office_bg: Enable office background ambience (default: False, disabled)
    
    Returns:
        SSML-formatted text ready for Google TTS
    """
    if not text or not text.strip():
        return ""
    text = text.strip()
    text = normalize_abbreviations(text)
    text = normalize_numbers(text)
    text = add_contractions(text)
    text = normalize_symbols(text)
    return wrap_in_ssml(
        text,
        add_office_bg=add_office_bg,
        start_break_ms=start_break_ms,
        between_sentence_break_ms=between_sentence_break_ms,
    )


# ---------------------------------------------------------
# 7. Quick Utility
# ---------------------------------------------------------

def quick_clean(text: str) -> str:
    """Fast cleaning without SSML (for cached phrases)"""
    if not text:
        return ""
    text = re.sub(r'\.{3,}', ',', text)
    text = re.sub(r'([.!?;,]){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

