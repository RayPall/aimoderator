# audio_upload_whisper_segmenter_live.py
"""Streamlit app that supports two modes:
1. **Upload** – user uploads an audio file, script sends it to Whisper and then to Make to obtain bullet‑points.
2. **Live (WebRTC)** – captures audio from the browser (mic or virtual loop‑back) via streamlit‑webrtc, buffers N seconds, sends to Whisper, and displays bullet‑points in near‑real‑time.

Requirements (requirements.txt):
    streamlit==1.35.0
    streamlit-webrtc==0.46.0
    av>=14.4.0               # wheels pro Py≥3.13
    ffmpeg-python~=0.2.0
    openai==1.25.0
    requests>=2.31.0

Optional packages.txt (APT):
    ffmpeg                   # binárka FFmpegu – dev balíčky nejsou nutné
"""
from __future__ import annotations

import io
import threading
import time
from typing import List

import av                    # PyAV – povinné pro streamlit‑webrtc
import requests
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEGMENT_SEC = 60                 # výchozí délka minutového segmentu
WHISPER_MODEL = "whisper-1"      # OpenAI Whisper model name
MAKE_WEBHOOK_URL = st.secrets.get("MAKE_WEBHOOK_URL", "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Whisper + Make helpers
# ---------------------------------------------------------------------------

def whisper_transcribe(audio_bytes: bytes, filename: str = "live.wav") -> str:
    """Send WAV/MP3 bytes to OpenAI Whisper and return plain‑text transcript."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    resp = client.audio.transcriptions.create(model=WHISPER_MODEL, file=files["file"])
    return resp.text  # type: ignore[attr-defined]


def post_to_make(transcript: str) -> List[str]:
    """Send transcript to Make scenario; expect list of bullet‑points back."""
    if not MAKE_WEBHOOK_URL:
        return ["(Make webhook URL not configured in Streamlit secrets)"]
    r = requests.post(MAKE_WEBHOOK_URL, json={"transcript": transcript}, timeout=30)
    r.raise_for_status()
    return r.json().get("bullets", [])

# ---------------------------------------------------------------------------
# Live audio processing class
# ---------------------------------------------------------------------------

class LiveAudioProcessor(AudioProcessorBase):
    """Collect raw PCM frames for SEGMENT_SEC, then hand off for transcription."""

    def __init__(self):
        self._buffer = bytearray()
        self._last_sent = time.time()
        self.sample_rate = 48000  # streamlit‑webrtc default

    def recv_audio(self, frame: av.AudioFrame):  # type: ignore[override]
        pcm = frame.to_ndarray().tobytes()
        self._buffer.extend(pcm)
        now = time.time()
        if now - self._last_sent >= SEGMENT_SEC:
            wav_bytes = _pcm_to_wav(bytes(self._buffer), self.sample_rate)
            threading.Thread(target=_process_and_update, args=(wav_bytes,), daemon=True).start()
            self._buffer.clear()
            self._last_sent = now
        return frame  # pass‑through

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16‑bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _process_and_update(wav_bytes: bytes) -> None:
    """Background: send to Whisper, then Make, then update Streamlit state."""
    transcript = whisper_transcribe(wav_bytes)
    bullets = post_to_make(transcript)
    st.session_state.setdefault("bullets", []).extend(bullets)
    st.session_state.setdefault("transcripts", []).append(transcript)
    st.experimental_rerun()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    global SEGMENT_SEC  # declare before first use inside this function

    st.title("📝 AI Moderátor – audio ➜ bullet‑points")

    mode = st.sidebar.radio("Režim", ["Upload", "Live (WebRTC)"])
    SEGMENT_SEC = st.sidebar.slider("Interval odesílání (s)", 15, 120, SEGMENT_SEC, 5)

    if mode == "Upload":
        _upload_ui()
    else:
        _live_ui()


def _upload_ui() -> None:
    uploaded = st.file_uploader("Nahraj audio soubor", type=["wav", "mp3", "m4a"])
    if uploaded and st.button("Přepsat a vytvořit bullet‑points"):
        with st.spinner("⏳ Odesílám do Whisperu…"):
            transcript = whisper_transcribe(uploaded.read(), uploaded.name)
            bullets = post_to_make(transcript)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))


def _live_ui() -> None:
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    webrtc_streamer(
        key="live_audio",
        mode=WebRtcMode.SENDRECV,  # enum, ne string!
        audio_processor_factory=LiveAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.subheader("Živé bullet‑points")
    bullets_ui = st.empty()
    bullets = st.session_state.get("bullets", [])
    if bullets:
        bullets_ui.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        bullets_ui.info("Čekám na první segment…")


if __name__ == "__main__":
    main()
