"""
Google TTS-Compatible Humanized Preprocessing Middleware

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
    thinking_phrases = [
        'let me think',
        'let me see',
        'let me check',
        'hmm',
        'well',
        'actually',
        'you know',
        'I mean',
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
    Base speed: 0.95 (5% slower than normal for clarity)
    """
    if emotion == "happy":
        return ("1.00", "+2st", "medium")  # Normal speed, higher pitch (was 1.05)
    if emotion == "sad":
        return ("0.88", "-1st", "soft")    # 12% slower, gentle (was 0.93)
    if emotion == "uncertain":
        return ("0.91", "-1st", "soft")    # 9% slower, hesitant (was 0.96)
    if emotion == "confident":
        return ("0.97", "+1st", "medium")  # 3% slower, clear (was 1.02)
    # Neutral: 92-96% range (was 97-101%)
    return (str(round(random.uniform(0.92, 0.96), 2)), random.choice(["-1st", "0st", "+1st"]), "medium")


# ---------------------------------------------------------
# 4. Humanization: Fillers + Breathing
# ---------------------------------------------------------

def insert_fillers(sentence: str, emotion: str) -> str:
    """
    Adds natural fillers ("uhh", "umm", etc.) based on emotion and context.
    """
    if emotion in ["happy", "confident"]:
        return sentence  # confident speech = fewer fillers

    fillers = ['uhh', 'umm', 'hmm', 'you know', 'I mean']
    start_words = ['Well', 'So', 'Actually', 'Maybe', 'Perhaps']

    # Start filler (thinking style)
    if any(sentence.strip().startswith(w) for w in start_words):
        if random.random() < 0.4:
            filler = random.choice(fillers)
            return f'<prosody rate="0.95" pitch="-1st">{filler}</prosody> <break time="70ms"/> {sentence}'

    # Mid-sentence filler (light)
    if random.random() < 0.1 and len(sentence.split()) > 6:
        words = sentence.split()
        insert_at = random.randint(3, len(words) - 2)
        words.insert(insert_at, f'<prosody rate="0.95" pitch="-1st">{random.choice(fillers)}</prosody> <break time="60ms"/>')
        return ' '.join(words)

    return sentence


def add_breath(sentence: str, emotion: str) -> str:
    """
    Adds subtle breath between long thoughts.
    """
    if len(sentence.split()) < 12:
        return sentence

    if emotion in ["sad", "neutral", "uncertain"]:
        if random.random() < 0.07:  # small chance
            sentence += ' <audio src="https://actions.google.com/sounds/v1/human_voices/breath.ogg" soundLevel="-34dB"/>'
    return sentence


# ---------------------------------------------------------
# 5. SSML Generator (Google-Compatible)
# ---------------------------------------------------------

def wrap_in_ssml(text: str, add_office_bg: bool = True) -> str:
    """
    Wraps text in SSML with optional office background ambience.
    
    Args:
        text: Text to wrap
        add_office_bg: Add subtle office background sounds (30% chance)
    """
    # Add thinking delays BEFORE wrapping
    text = add_thinking_delays(text)
    
    sentences = re.split(r'([.!?])', text)
    ssml = "<speak>\n"
    
    # Add subtle office background ambience (30% chance - not every call)
    if add_office_bg and random.random() < 0.30:
        # Very subtle office ambience throughout the response
        ssml += '  <par>\n'
        ssml += '    <media soundLevel="-38dB">\n'
        ssml += '      <audio src="https://actions.google.com/sounds/v1/ambiences/office_ambience.ogg"/>\n'
        ssml += '    </media>\n'
        ssml += '    <media>\n'
        # Main speech will be inside this media tag
        use_par_tags = True
    else:
        use_par_tags = False

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""

        if not sentence:
            continue

        emotion = detect_emotion(sentence)
        rate, pitch, volume = emotion_to_prosody(emotion)

        # Add realism
        sentence = insert_fillers(sentence, emotion)
        sentence = add_breath(sentence, emotion)

        ssml += f'  <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{sentence}{punct}</prosody>\n'
        ssml += '  <break time="200ms"/>\n'

    # Close par/media tags if office background was added
    if use_par_tags:
        ssml += '    </media>\n'
        ssml += '  </par>\n'
    
    ssml += "</speak>"
    return ssml


# ---------------------------------------------------------
# 6. Main Preprocessing Entry
# ---------------------------------------------------------

def preprocess_for_tts(text: str, add_office_bg: bool = False) -> str:
    """
    Complete humanization pipeline with optional office background.
    
    Pipeline:
    1. Normalize abbreviations (Dr. → Doctor)
    2. Normalize numbers ($100 → 100 dollars)
    3. Add contractions (I am → I'm)
    4. Add thinking delays (400ms before "let me think")
    5. Detect emotions (happy, sad, uncertain, confident)
    6. Add fillers (uhh, umm - context-aware)
    7. Add breathing (subtle, 7% on long sentences)
    8. Add office background (DISABLED by default, can enable if needed)
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
    return wrap_in_ssml(text, add_office_bg=add_office_bg)


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
