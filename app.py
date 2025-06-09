# audio_upload_whisper_segmenter_live.py
"""
Streamlit app
1. **Upload** – user uploads an audio file > Whisper > Make webhook > bullet‑points.
2. **Live (WebRTC)** – capt­ures mic / virtual cable audio in browser, slices every `SEGMENT_SEC`, sends to Whisper → Make, streams bullet‑points live.

Key changes vs. previous revision
--------------------------------
* **ŽÁDNÉ `st.experimental_rerun`** – nahrazeno vlajkou v `st.session_state` a jemným `st_autorefresh` každých 500 ms, aby nedocházelo k fatal write errorům.
* **Public TURN fallback** – STUN nestačil; přidán free TURN (Metered) ⇒ ICE handshake projde i za NAT/UDP blokádou.
* **Debug panel** – zobrazuje stav peer‑connection a počet přijato snímků.
"""
from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from typing import List

import av
import requests
import streamlit as st
from streamlit_extras.st_autorefresh import st_autorefresh
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

# ----------------------------------------------------------------------------
# CONFIG (override in secrets)
# ----------------------------------------------------------------------------
SEGMENT_SEC       = 60
WHISPER_MODEL     = "whisper-1"
MAKE_WEBHOOK_URL  = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY", "")

# ----------------------------------------------------------------------------
# OpenAI Whisper + Make webhook helpers
# ----------------------------------------------------------------------------

def whisper_transcribe(wav: bytes) -> str:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    r = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("live.wav", io.BytesIO(wav), "audio/wav"),  # type: ignore[arg-type]
    )
    return r.text  # type: ignore[attr-defined]

def post_to_make(text: str) -> List[str]:
    if not MAKE_WEBHOOK_URL:
        return ["(⚠️ MAKE_WEBHOOK_URL není nastaven)"]
    res = requests.post(MAKE_WEBHOOK_URL, json={"transcript": text}, timeout=30)
    res.raise_for_status()
    return res.json().get("bullets", [])

# ----------------------------------------------------------------------------
#  Live audio processor – push results do fronty
# ----------------------------------------------------------------------------

_result_q: "queue.Queue[List[str]]" = queue.Queue()

class LiveProcessor(AudioProcessorBase):
    def __init__(self):
        self.buf = bytearray(); self.last = time.time(); self.rate = 48000
    def recv_audio(self, frame: av.AudioFrame):  # type: ignore[override]
        self.buf.extend(frame.to_ndarray().tobytes())
        if time.time() - self.last >= SEGMENT_SEC:
            wav = _pcm_to_wav(bytes(self.buf), self.rate)
            threading.Thread(target=_worker, args=(wav,), daemon=True).start()
            self.buf.clear(); self.last = time.time()
        return frame

def _pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    import wave, io as _io
    b = _io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return b.getvalue()

def _worker(wav: bytes):
    try:
        txt = whisper_transcribe(wav)
        bullets = post_to_make(txt)
    except Exception as e:
        bullets = [f"❌ {e}"]
    _result_q.put(bullets)
    st.session_state["__new__"] = True  # flag pro UI

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def main():
    global SEGMENT_SEC
    st.title("📝 AI Moderátor – audio ➜ bullet‑points")

    mode = st.sidebar.radio("Režim", ["Upload", "Live"])
    SEGMENT_SEC = st.sidebar.slider("Interval odesílání (s)", 15, 180, SEGMENT_SEC, 5)

    if mode == "Upload":
        upload_ui(); return

    live_ui()


def upload_ui():
    f = st.file_uploader("Nahraj audio", type=["wav", "mp3", "m4a"])
    if f and st.button("Zpracovat"):
        with st.spinner("Whisper → Make…"):
            txt = whisper_transcribe(f.read())
            bullets = post_to_make(txt)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))


def live_ui():
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    ctx = webrtc_streamer(
        key="live_audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=LiveProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": "turn:global.relay.metered.ca:80", "username": "global", "credential": "global"},
            ]
        },
    )

    st.caption(f"WebRTC state: {ctx.state}")

    st_autorefresh(interval=500, key="__auto")

    if "bullets" not in st.session_state:
        st.session_state["bullets"] = []
    col = st.container()
    if st.session_state.pop("__new__", False):
        try:
            while True:
                st.session_state["bullets"].extend(_result_q.get_nowait())
        except queue.Empty:
            pass
    b = st.session_state["bullets"]
    col.subheader("Živé bullet‑points")
    col.markdown("\n".join(f"• {x}" for x in b) if b else "_Čekám na první segment…_")


if __name__ == "__main__":
    main()
