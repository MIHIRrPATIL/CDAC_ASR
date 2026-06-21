import os
import hashlib
from gtts import gTTS

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "tts")

os.makedirs(CACHE_DIR, exist_ok=True)

def generate_reference_audio(text: str, slow: bool = False) -> str:
    """Generates an MP3 file of the text using gTTS and returns the absolute file path."""
    text_hash = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    filename = f"{text_hash}_{'slow' if slow else 'normal'}.mp3"
    filepath = os.path.join(CACHE_DIR, filename)
    
    # If already cached, return the path
    if os.path.exists(filepath):
        return filepath
        
    # Generate reference audio
    tts = gTTS(text=text, lang="en", slow=slow)
    tts.save(filepath)
    return filepath
