# audio_upload_whisper_segmenter_live.py
"""Streamlit app that supports two modes:
1. **Upload** – user uploads an audio file, script sends it to Whisper and then to Make to obtain bullet‑points.
2. **Live (WebRTC)** – captures audio from the browser (mic or virtual loop‑back) via streamlit‑webrtc, buffers 60 s, sends to Whisper, displays bullet‑points in real‑time.

Requirements (put in requirements.txt):
    streamlit==1.35.0
    streamlit-webrtc==0.46.0
    av>=14.4.0               # wheel for Py ≥3.13, no compile step
    ffmpeg-python~=0.2.0
    openai==1.25.0
    requests>=2.31.0

Optional packages.txt:
    ffmpeg                   # runtime binary; dev headers nejsou nutné

If `PYAV_LOGGING` env var is set to "off", PyAV disables its logging.
"""
from __future__ import annotations

import io
import queue
import threading
import time
from pathlib import Path
from typing import Any, Deque, List

import av  # PyAV ‑ required by streamlit‑webrtc
import requests
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEGMENT_SEC = 60          # how long we buffer live audio before sending
WHISPER_MODEL = "whisper-1"  # OpenAI Whisper model name
MAKE_WEBHOOK_URL = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def whisper_transcribe(audio_bytes: bytes, filename: str = "live.wav") -> str:
    """Send raw WAV/MP3 bytes to OpenAI Whisper and return transcription text."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    response = client.audio.transcriptions.create(model=WHISPER_MODEL, file=files["file"])
    return response.text  # type: ignore[attr-defined]


def post_to_make(transcript: str) -> List[str]:
    """Send transcript to Make scenario via webhook; receive bullet‑points list."""
    if not MAKE_WEBHOOK_URL:
        return ["(Make webhook URL not configured)"]
    resp = requests.post(MAKE_WEBHOOK_URL, json={"transcript": transcript}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("bullets", [])


# ---------------------------------------------------------------------------
# Live audio processor
# ---------------------------------------------------------------------------
class LiveAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self._buffer: bytearray = bytearray()
        self._last_sent = time.time()
        self.sample_rate = 48000  # streamlit‑webrtc default for browser mic

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:  # type: ignore[override]
        pcm = frame.to_ndarray().tobytes()
        self._buffer.extend(pcm)
        now = time.time()
        if now - self._last_sent >= SEGMENT_SEC:
            # Convert raw PCM to WAV in‑memory
            wav_bytes = _pcm_to_wav(bytes(self._buffer), self.sample_rate)
            threading.Thread(target=_process_and_update, args=(wav_bytes,), daemon=True).start()
            self._buffer.clear()
            self._last_sent = now
        return frame  # pass‑through so user hears themself if needed


# ---------------------------------------------------------------------------
# Internal helpers
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
    """Background thread: send to Whisper + Make, then update UI."""
    txt = whisper_transcribe(wav_bytes)
    bullets = post_to_make(txt)
    # Append to session state list; trigger rerun
    st.session_state.setdefault("bullets", []).extend(bullets)
    st.session_state.setdefault("transcripts", []).append(txt)
    st.experimental_rerun()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📝 AI Moderátor – audio→bullet‑points")

    mode = st.sidebar.radio("Režim", ["Upload", "Live (WebRTC)"])
    interval = st.sidebar.slider("Interval odesílání (s)", 15, 120, SEGMENT_SEC, 5)
    global SEGMENT_SEC
    SEGMENT_SEC = interval

    if mode == "Upload":
        _upload_ui()
    else:
        _live_ui()


def _upload_ui() -> None:
    uploaded = st.file_uploader("Nahraj audio soubor", type=["wav", "mp3", "m4a"])
    if uploaded and st.button("Přepsat a vytvořit bullet‑points"):
        with st.spinner("Odesílám do Whisperu…"):
            transcript = whisper_transcribe(uploaded.read(), uploaded.name)
            bullets = post_to_make(transcript)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))


def _live_ui() -> None:
    st.markdown("Klikni **Allow** pro mikrofon / virtual loop‑back.")
    webrtc_ctx = webrtc_streamer(
        key="live_audio",
        mode=WebRtcMode.SENDRECV,  # enum, not string!
        audio_processor_factory=LiveAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.subheader("Živé bullet‑points")
    bullets_area = st.empty()
    bullets = st.session_state.get("bullets", [])
    if bullets:
        bullets_area.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        bullets_area.info("Čekám na první minutový segment…")


if __name__ == "__main__":
    main()
