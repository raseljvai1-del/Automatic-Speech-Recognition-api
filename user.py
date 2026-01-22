import streamlit as st
import requests

st.set_page_config(page_title="ASR Demo", layout="centered")
st.title("🎤 Bangla–English ASR Demo")

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac"]
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe"):
        with st.spinner("Processing..."):
            response = requests.post(
                "http://127.0.0.1:8000/transcribe",
                files={"file": audio_file}
            )

        if response.status_code == 200:
            data = response.json()
            st.success("Transcription completed")
            st.write("📝 **Text:**", data["transcription"])
            st.write("🌐 **Language:**", data["language"])
            st.write("⚡ **Latency (ms):**", data["latency_ms"])
        else:
            st.error(response.text)
