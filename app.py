# audio_upload_whisper_segmenter_live.py
"""
Streamlit app with two modes:
1. **Upload** – user uploads an audio file, the app sends it to OpenAI Whisper → Make webhook → shows bullet‑points.
2. **Live (WebRTC)** – captures microphone / virtual loop‑back audio in browser, buffers a user‑set interval, sends each chunk to Whisper, forwards the transcript to Make, and streams the bullet‑points live.

## Runtime requirements
Put this into **requirements.txt**
```
streamlit==1.35.0
streamlit-webrtc==0.46.0
av>=14.4.0              # wheels available for Py 3.13
ffmpeg-python~=0.2.0
openai==1.25.0
requests>=2.31.0
```
Optional minimal **packages.txt** for Streamlit Cloud / Docker base image:
```
ffmpeg                 # only the binary, no ‑dev headers needed
```
If you run on Streamlit Cloud you may omit `runtime.txt`; if you keep it, `python-3.13.x` works fine with AV ≥ 14.
"""
from __future__ import annotations

import io
import threading
import time
from typing import List

import av                     # PyAV – required by streamlit‑webrtc
import requests
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ---------------------------------------------------------------------------
# CONFIG (override in .streamlit/secrets.toml or Streamlit Cloud Secrets UI)
# ---------------------------------------------------------------------------
SEGMENT_SEC = 60                          # default chunk length in seconds
WHISPER_MODEL = "whisper-1"              # OpenAI Whisper model name
MAKE_WEBHOOK_URL = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# OpenAI Whisper + Make helpers
# ---------------------------------------------------------------------------

def whisper_transcribe(audio_bytes: bytes, filename: str = "live.wav") -> str:
    """Send WAV/MP3 bytes to Whisper and return transcript text."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(filename, io.BytesIO(audio_bytes), "audio/wav"),  # type: ignore[arg-type]
    )
    return resp.text  # type: ignore[attr-defined]


def post_to_make(transcript: str) -> List[str]:
    """POST transcript to Make scenario → expect JSON { bullets: [...] }."""
    if not MAKE_WEBHOOK_URL:
        return ["(⚠️ MAKE_WEBHOOK_URL chybí v Streamlit secrets.)"]
    r = requests.post(MAKE_WEBHOOK_URL, json={"transcript": transcript}, timeout=30)
    r.raise_for_status()
    return r.json().get("bullets", [])

# ---------------------------------------------------------------------------
# Live audio processor
# ---------------------------------------------------------------------------

class LiveAudioProcessor(AudioProcessorBase):
    """Collect raw PCM frames and every SEGMENT_SEC send them for processing."""

    def __init__(self):
        self.buffer = bytearray()
        self.last_sent = time.time()
        self.sample_rate = 48000  # streamlit‑webrtc default

    def recv_audio(self, frame: av.AudioFrame):  # type: ignore[override]
        self.buffer.extend(frame.to_ndarray().tobytes())
        if time.time() - self.last_sent >= SEGMENT_SEC:
            wav_bytes = _pcm_to_wav(bytes(self.buffer), self.sample_rate)
            threading.Thread(
                target=_process_and_update, args=(wav_bytes,), daemon=True
            ).start()
            self.buffer.clear()
            self.last_sent = time.time()
        return frame  # pass through unchanged

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)         # 16‑bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _process_and_update(wav_bytes: bytes) -> None:
    """Background worker: Whisper → Make → push result into session state."""
    transcript = whisper_transcribe(wav_bytes)
    bullets = post_to_make(transcript)
    st.session_state.setdefault("bullets", []).extend(bullets)
    st.session_state.setdefault("transcripts", []).append(transcript)
    # trigger UI refresh
    st.experimental_rerun()

# ---------------------------------------------------------------------------
# Streamlit UI building blocks
# ---------------------------------------------------------------------------

def main() -> None:
    global SEGMENT_SEC  # declare early

    st.title("📝 AI Moderátor – audio ➜ bullet‑points")

    mode = st.sidebar.radio("Režim", ["Upload", "Live (WebRTC)"])
    SEGMENT_SEC = st.sidebar.slider("Interval odesílání (s)", 15, 180, SEGMENT_SEC, 5)

    if mode == "Upload":
        _upload_ui()
    else:
        _live_ui()


def _upload_ui() -> None:
    uploaded = st.file_uploader("Nahraj audio soubor", type=["wav", "mp3", "m4a"])
    if uploaded and st.button("Přepsat a vytvořit bullet‑points"):
        with st.spinner("⏳ Odesílám do Whisperu…"):
            transcript = whisper_transcribe(uploaded.read(), uploaded.name)
            bullets = post_to_make(transcript)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))


def _live_ui() -> None:
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    ctx = webrtc_streamer(
        key="live_audio",
        mode=WebRtcMode.SENDONLY,  # nepotřebujeme přijímat video/audio zpět
        audio_processor_factory=LiveAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={  # veřejný STUN server
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    # Debug info (optional):
    st.caption(f"WebRTC state: {ctx.state}")

    # Live bullet‑points output
    st.subheader("Živé bullet‑points")
    bullets_box = st.empty()
    bullets = st.session_state.get("bullets", [])
    if bullets:
        bullets_box.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        bullets_box.info("Čekám na první segment…")


if __name__ == "__main__":
    main()
