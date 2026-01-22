import numpy as np

def preprocess_audio(audio):
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def detect_language(text):
    # Bangla Unicode range
    for ch in text:
        if '\u0980' <= ch <= '\u09FF':
            return "bn"
    return "en"
