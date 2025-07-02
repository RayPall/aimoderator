# audio_upload_whisper_segmenter_live.py
"""
Streamlit app
1. **Upload** – user uploads an audio file → Whisper → Make webhook → bullet‑points.
2. **Live (WebRTC)** – captures mic / virtual cable audio in browser, slices every `SEGMENT_SEC`, sends to Whisper → Make, streams bullet‑points live.

New in this version
-------------------
* **Live audio level indicator** – realtime RMS bar that turns green when sound is detected.
* Hard dependency on `streamlit-extras` removed; if absent, the app still runs.
"""
from __future__ import annotations

import asyncio, io, time, threading, queue
from typing import List

import av
import numpy as np
import openai
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ------------------------------------------------------------------------------
# OPTIONAL refresh helper -------------------------------------------------------
# ------------------------------------------------------------------------------
try:
    from streamlit_extras.st_autorefresh import st_autorefresh  # type: ignore
except ModuleNotFoundError:  # fallback – noop stub
    def st_autorefresh(*_, **__):
        return None

# ------------------------------------------------------------------------------
# CONFIG (fill these two in Secrets) -------------------------------------------
# ------------------------------------------------------------------------------
SEGMENT_SEC       = 60
WHISPER_MODEL     = "whisper-1"
MAKE_WEBHOOK_URL  = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY", "")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# global queues communicated between threads and main UI
result_q: "queue.Queue[List[str]]" = queue.Queue()
level_q:  "queue.Queue[float]"    = queue.Queue(maxsize=5)

# ------------------------------------------------------------------------------
# Helper functions --------------------------------------------------------------
# ------------------------------------------------------------------------------

def whisper_transcribe(wav_bytes: bytes) -> str:
    resp = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("live.wav", io.BytesIO(wav_bytes), "audio/wav")  # type: ignore[arg-type]
    )
    return resp.text  # type: ignore[attr-defined]

def post_to_make(transcript: str) -> List[str]:
    if not MAKE_WEBHOOK_URL:
        return ["(⚠️ MAKE_WEBHOOK_URL není v Secrets)"]
    r = requests.post(MAKE_WEBHOOK_URL, json={"transcript": transcript}, timeout=30)
    r.raise_for_status()
    return r.json().get("bullets", [])

def pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    import wave, io as _io
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return buf.getvalue()

# ------------------------------------------------------------------------------
# Audio processor ---------------------------------------------------------------
# ------------------------------------------------------------------------------

class LiveAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self._buf: bytearray = bytearray()
        self._last = time.time()
        self._rate = 48000  # streamlit-webrtc default

    def recv_audio(self, frame: "av.AudioFrame"):  # type: ignore[override]
        pcm16 = frame.to_ndarray().tobytes()
        self._buf.extend(pcm16)

        # --- compute RMS for level indicator ---
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples))) / 32768.0)  # 0‒1
        try:
            level_q.put_nowait(rms)
        except queue.Full:
            pass

        # --- segment every SEGMENT_SEC seconds ---
        if time.time() - self._last >= SEGMENT_SEC:
            wav = pcm_to_wav(bytes(self._buf), self._rate)
            threading.Thread(target=self._whisper_worker, args=(wav,), daemon=True).start()
            self._buf.clear()
            self._last = time.time()

        return frame  # passthrough

    # background whisper + Make
    def _whisper_worker(self, wav: bytes) -> None:
        try:
            txt = whisper_transcribe(wav)
            bullets = post_to_make(txt)
            result_q.put(bullets)
        except Exception as e:
            result_q.put([f"❌ {e}"])

# ------------------------------------------------------------------------------
# UI ----------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AI Moderátor", page_icon="📝", layout="centered")
    st.title("📝 AI Moderátor – audio ➜ bullet‑points")

    mode = st.sidebar.radio("Režim", ["Upload", "Live"], index=1)
    seg = st.sidebar.slider("Interval segmentu (s)", 15, 180, SEGMENT_SEC, 5)
    globals()["SEGMENT_SEC"] = seg

    if mode == "Upload":
        upload_ui()
    else:
        live_ui()

# --------------------------------------------

def upload_ui() -> None:
    f = st.file_uploader("Nahraj audio", type=["wav", "mp3", "m4a"])
    if f and st.button("Zpracovat"):
        with st.spinner("Whisper → Make…"):
            txt = whisper_transcribe(f.read())
            bullets = post_to_make(txt)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))

# --------------------------------------------

def live_ui() -> None:
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    ctx = webrtc_streamer(
        key="live_audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=LiveAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": "turn:global.relay.metered.ca:80", "username": "global", "credential": "global"},
            ]
        },
    )

    # --- live widgets ---
    level_placeholder = st.empty()
    bullet_container = st.container()

    if "bullets" not in st.session_state:
        st.session_state["bullets"] = []

    # gentle refresh every 700 ms if autorefresh available
    st_autorefresh(interval=700, key="__auto")

    # pull queued results
    try:
        while True:
            st.session_state["bullets"].extend(result_q.get_nowait())
    except queue.Empty:
        pass

    # pull latest level (keep last)
    level = 0.0
    try:
        while True:
            level = level_q.get_nowait()
    except queue.Empty:
        pass

    # render level bar (simple text‑based)
    bar_len = int(level * 20)
    bar = "🟩" * bar_len + "▫️" * (20 - bar_len)
    level_placeholder.markdown(f"**Úroveň audia:** {bar}")

    # render bullets
    bullet_container.subheader("Živé bullet‑points")
    bullets: List[str] = st.session_state["bullets"]
    if bullets:
        bullet_container.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        bullet_container.info("Čekám na první segment…")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
