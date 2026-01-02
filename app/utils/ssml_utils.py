"""
SSML and text processing utilities for TTS.
Handles SSML tag manipulation, text cleaning, and smart chunking.
"""

import re
import random
import sys


def strip_ssml_tags(text: str) -> str:
    """
    Remove all SSML tags from text, keeping only the actual text content.
    Used for saving clean text to transcript.
    Handles both complete and incomplete SSML tags.
    """
    if not text:
        return ""
    
    # Remove complete SSML tags (<tag>content</tag> or <tag/>)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove incomplete SSML tags (tags without closing >, like <break time="150ms)
    text = re.sub(r'<[^>]*', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def add_natural_ssml(text: str, use_ssml: bool = True, add_breaths: bool = True, add_fillers: bool = True, add_boundary_pause: bool = False) -> str:
    """
    Add SSML tags and natural speech elements for human-like delivery (Vapi-style).
    
    Features:
    1. SSML <prosody> tags with varied rate (95-105%) and pitch (±2st)
    2. <break> tags for natural pauses (100-200ms)
    3. <audio> tags for breath sounds between sentences (Google hosted)
    4. Hesitation fillers WITH breaks: "Hmm <break time='120ms'/> text"
    5. Boundary fillers with breaks for smooth chunk transitions
    
    Args:
        text: Input text
        use_ssml: Enable SSML tags
        add_breaths: Add breath pauses between sentences
        add_fillers: Add occasional fillers (uh, hmm)
        add_boundary_pause: Add natural pause/filler at chunk boundary
        
    Returns:
        Enhanced text with SSML and natural elements
    """
    if not text or not text.strip():
        return ""
    
    cleaned = text.strip()
    
    # If SSML not requested, just clean and return
    if not use_ssml:
        return clean_text_for_tts(cleaned)
    
    # Add hesitation fillers WITH breaks (Vapi-style - exact implementation)
    if add_fillers and random.random() < 0.15:
        # Hesitation patterns with breaks for natural pauses
        hesitations = [
            'Hmm <break time="120ms"/> ',
            'Uh <break time="100ms"/> ',
            'Well <break time="150ms"/> ',
            'Let me see <break time="180ms"/> ',
            'Umm <break time="130ms"/> ',
        ]
        cleaned = random.choice(hesitations) + cleaned
    
    # Wrap in SSML speak tag
    ssml = '<speak>'
    
    # Split into sentences for breath insertion
    sentences = re.split(r'([.!?;])', cleaned)
    
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i+1] if i+1 < len(sentences) else ""
        
        if not sentence:
            continue
        
        # Add prosody variation (Vapi-style - subtle speed and pitch changes)
        rate_variation = random.choice(["95%", "98%", "100%", "102%", "105%"])  # Vapi-style range
        pitch_variation = random.choice(["-1st", "0st", "+1st", "+2st"])  # Include +2st like Vapi
        
        ssml += f'<prosody rate="{rate_variation}" pitch="{pitch_variation}">'
        ssml += sentence + punct
        ssml += '</prosody>'
        
        # Add natural breath/pause after sentences (Vapi-style with audio!)
        if add_breaths and punct in ['.', '!', '?', ';']:
            # Vary break duration for naturalness
            break_time = random.choice(["150ms", "180ms", "200ms"])
            ssml += f'<break time="{break_time}"/>'
            
            # Add breath audio file for natural pauses (low volume, subtle)
            # DISABLED: Google Cloud TTS doesn't support <audio> tags properly.
            # The URL gets read as text instead of playing audio, causing issues.
            # if random.random() < 0.15:  # 15% chance - very subtle
            #     BREATH_AUDIO = "https://actions.google.com/sounds/v1/human_voices/breath.ogg"
            #     ssml += f'<audio src="{BREATH_AUDIO}" soundLevel="-10dB"/>'  # Quieter breath
    
    # Add remaining text (if any)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        ssml += sentences[-1]
    
    # Add boundary filler for smooth chunk transitions - ALWAYS (100%)
    # Critical: This eliminates tak-tak distortion between chunks
    if add_boundary_pause:
        # ALWAYS add boundary connector (100% - not random!) for seamless audio
        # Boundary fillers with breaks for natural thinking pauses
        boundary_fillers = [
            ' <break time="80ms"/><prosody rate="90%" pitch="-1st">uhh</prosody>',
            ' <break time="90ms"/><prosody rate="88%" pitch="-2st">umm</prosody>',
            ' <break time="70ms"/><prosody rate="92%" pitch="0st">uh</prosody>',
            ' <break time="100ms"/><prosody rate="85%" pitch="-1st">hmm</prosody>',
        ]
        chosen_filler = random.choice(boundary_fillers)
        ssml += chosen_filler
        print(f"🔗 Added boundary filler: {chosen_filler[:50]}")
        sys.stdout.flush()
    
    ssml += '</speak>'
    
    return ssml


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS to prevent reading punctuation marks aloud.
    Removes duplicate punctuation and normalizes spacing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text safe for TTS
    """
    if not text or not text.strip():
        return ""
    
    cleaned = text.strip()
    
    # Remove multiple punctuation (e.g., "!!!" -> "!", "..." -> ".")
    cleaned = re.sub(r'([.!?;,—:])\1+', r'\1', cleaned)
    
    # Remove standalone punctuation marks that get read as words
    cleaned = re.sub(r'\s+([.!?;,—:])\s+', r'\1 ', cleaned)
    
    # Ensure proper spacing after punctuation
    cleaned = re.sub(r'([.!?;,—:])([A-Za-z])', r'\1 \2', cleaned)
    
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()


def smart_chunk_text(text: str, max_words: int = 15) -> tuple[str, str]:
    """
    Smart text chunking that splits at natural pauses for smoother speech.
    Prefers splitting at sentence boundaries to maintain natural flow.
    
    Args:
        text: Text to split
        max_words: Maximum words in prefix chunk
        
    Returns:
        (prefix, suffix) tuple
    """
    if not text or not text.strip():
        return "", ""
    
    text = text.strip()
    words = text.split()
    
    # If text is short enough, return as-is
    if len(words) <= max_words:
        return text, ""
    
    # Try to split at sentence boundaries (., !, ?)
    sentence_endings = ['. ', '! ', '? ']
    best_split = None
    
    for ending in sentence_endings:
        parts = text.split(ending)
        if len(parts) > 1:
            prefix_candidate = parts[0] + ending.strip()
            prefix_words = len(prefix_candidate.split())
            
            # Use this split if it's within our word limit
            if prefix_words <= max_words and prefix_words > max_words * 0.5:
                best_split = (prefix_candidate, text[len(prefix_candidate):].strip())
                break
    
    # If no good sentence split, try comma split
    if not best_split and ', ' in text:
        parts = text.split(', ', 1)
        prefix_candidate = parts[0] + ','
        prefix_words = len(prefix_candidate.split())
        
        if prefix_words <= max_words and prefix_words > 5:
            best_split = (prefix_candidate, parts[1].strip())
    
    # Fallback: split at word count
    if not best_split:
        prefix = " ".join(words[:max_words])
        suffix = " ".join(words[max_words:])
        best_split = (prefix, suffix)
    
    return best_split

