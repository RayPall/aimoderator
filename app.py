# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
• Streamlit (upload / mikrofon) ↦ Whisper (transkript)
• transkript POST na Make webhook (token, transcript, existing)
• Make odpoví JSON polem nových bodů → hned přidáme na flipchart
• Diagnostické okno, čisté vlákno bez „missing ScriptRunContext!“

Secrets
-------
OPENAI_API_KEY     = "sk-..."
WEBHOOK_OUT_TOKEN  = "out-token"    # posíláme Make
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import threading
import wave
from typing import List

import numpy as np
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ────────────────── CONFIG ────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160  # kolik sekund audia pošleme najednou do Whisper

MAKE_WEBHOOK_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN  = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

logging.basicConfig(level=logging.INFO)

# ───────── SESSION STATE INIT ─────────────────────────────────────────────
def _init_state():
    ss = st.session_state
    ss.setdefault("flip_points",        [])
    ss.setdefault("transcript_buffer",  "")
    ss.setdefault("audio_buffer",       [])
    ss.setdefault("status",             "🟡 Čekám na mikrofon…")
    ss.setdefault("upload_processed",   False)
    ss.setdefault("audio_stop_event",   None)
    ss.setdefault("runner_thread",      None)
_init_state()

# ───────── STATUS HELPERS ─────────────────────────────────────────────────
def set_status(txt: str):
    if st.session_state.status != txt:
        st.session_state.status = txt

@contextlib.contextmanager
def status_ctx(running: str, done: str | None = None):
    prev = st.session_state.status
    set_status(running)
    try:
        yield
    finally:
        set_status(done or prev)

# ───────── WHISPER SAFE CALL ──────────────────────────────────────────────
def whisper_safe(file_like, label: str) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=file_like, language="cs"
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper error (%s)", label)
        set_status(f"❌ Chyba Whisperu ({label})")
        st.error(
            f"❌ Whisper API error ({exc.__class__.__name__}). "
            "Zkontroluj kredit nebo formát."
        )
        return None

# ───────── SEND TO MAKE & GET POINTS ──────────────────────────────────────
def call_make(transcript: str, existing: list[str]) -> list[str]:
    payload = {
        "token": WEBHOOK_OUT_TOKEN,
        "transcript": transcript,
        "existing": existing,
    }
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            logging.error("Make response is not list: %s", data)
            set_status("⚠️ Neplatná odpověď Make")
            return []
        return [str(p).strip() for p in data if str(p).strip()]
    except Exception as exc:
        logging.exception("Chyba HTTP volání Make")
        set_status("⚠️ Chyba volání Make")
        st.error(f"❌ HTTP chyba při volání Make: {exc}")
        return []

# ───────── FLIPCHART RENDER ──────────────────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart li {opacity:0; transform:translateY(8px); animation:fadeIn 0.45s forwards;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

def render_flipchart():
    st.markdown(STYLES, unsafe_allow_html=True)
    pts = st.session_state.flip_points
    if not pts:
        st.info("Čekám na první shrnutí…")
        return
    st.markdown(
        "<ul class='flipchart'>" +
        "".join(f"<li style='animation-delay:{i*0.1}s'>{p}</li>"
                for i, p in enumerate(pts)) +
        "</ul>",
        unsafe_allow_html=True,
    )

# ───────── PCM → WAV ─────────────────────────────────────────────────────
def pcm_to_wav(frames: list[bytes], sr: int = 48000) -> bytes:
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()

# ───────── UI LAYOUT ─────────────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator", layout="wide")
tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# === TAB 1 – Ovládání =====================================================
with tabs[0]:
    st.header("Nastavení a vstup zvuku")

    # ---- Upload ---------------------------------------------------------
    up = st.file_uploader("▶️ Testovací WAV/MP3", type=["wav", "mp3", "m4a"])
    if up is not None and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            txt = whisper_safe(up, "upload")
        if txt:
            with status_ctx("📤 Make…", "🟢 Čekám na Make odpověď"):
                new_pts = call_make(txt, st.session_state.flip_points)
            st.session_state.flip_points.extend(
                [p for p in new_pts if p not in st.session_state.flip_points]
            )
            st.session_state.upload_processed = True

    # ---- Živý mikrofon --------------------------------------------------
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    # stop staré vlákno při rerunu
    if (old := st.session_state.runner_thread) and old.is_alive():
        st.session_state.audio_stop_event.set()
        old.join(timeout=2)

    stop_evt = threading.Event()
    st.session_state.audio_stop_event = stop_evt

    async def audio_pipeline(ctx, stop_event):
        SR = 48000
        target = AUDIO_BATCH_SECONDS * SR * 2  # bytes v dávce

        while not stop_event.is_set():
            if not ctx.audio_receiver:
                set_status("🟡 Čekám na mikrofon…")
                await asyncio.sleep(0.1)
                continue

            set_status("🔴 Zachytávám audio…")
            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(
                f.to_ndarray().tobytes() for f in frames
            )

            if sum(len(b) for b in st.session_state.audio_buffer) < target:
                await asyncio.sleep(0.05)
                continue

            wav = pcm_to_wav(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()

            with status_ctx("🟣 Whisper (mic)…"):
                tr = whisper_safe(io.BytesIO(wav), "mikrofon")
            if not tr:
                await asyncio.sleep(1)
                continue

            st.session_state.transcript_buffer += " " + tr
            if len(st.session_state.transcript_buffer.split()) >= 325:
                with status_ctx("📤 Make…", "🟢 Čekám Make odpověď"):
                    new_pts = call_make(
                        st.session_state.transcript_buffer,
                        st.session_state.flip_points,
                    )
                st.session_state.flip_points.extend(
                    [p for p in new_pts if p not in st.session_state.flip_points]
                )
                st.session_state.transcript_buffer = ""

            await asyncio.sleep(0.05)

    t = threading.Thread(
        target=lambda c=webrtc_ctx, e=stop_evt: asyncio.run(audio_pipeline(c, e)),
        daemon=True,
    )
    add_script_run_ctx(t)
    t.start()
    st.session_state.runner_thread = t

    # ---- Sidebar diagnostika -------------------------------------------
    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav")
    st.sidebar.write(st.session_state.status)
    st.sidebar.write("Audio thread:", t.is_alive())

# === TAB 2 – Flipchart ====================================================
with tabs[1]:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
