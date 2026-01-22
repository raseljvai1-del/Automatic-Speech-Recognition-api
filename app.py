from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import io
import librosa

from model import run_inference
from audio import preprocess_audio, detect_language

app = FastAPI(title="ASR API")

# CORS (for Streamlit / browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXT = [".wav", ".mp3", ".flac"]
MAX_DURATION = 30  # seconds


@app.get("/")
def home():
    return {"message": "ASR API is running"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    start_time = time.time()

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Read audio bytes
    audio_bytes = await file.read()

    # Load audio safely
    try:
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio file")

    # Validate duration
    duration = librosa.get_duration(y=audio, sr=sr)
    if duration > MAX_DURATION:
        raise HTTPException(status_code=400, detail="Audio too long")

    # Preprocess
    audio = preprocess_audio(audio)

    # ASR inference
    transcription = run_inference(audio)

    # Language detection
    language = detect_language(transcription)

    # Latency
    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "transcription": transcription,
        "language": language,
        "latency_ms": latency_ms
    }
