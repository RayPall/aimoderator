# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon → OpenAI Whisper → Make → Flipchart
---------------------------------------------------------------

Tok
---
1. Upload nebo živý mikrofon se odešle na OpenAI Whisper ⇒ text.
2. Přepis + již existující body pošleme na Make webhook (MAKE_OUT_WEBHOOK_URL).
3. Scénář v Make vytvoří **nové** odrážky a zavolá náš webhook
   https://aimoderator.streamlit.app:8000/make s JSONem:
   {
     "token": "in-token",
     "points": ["NADPIS\\n- detail", "…"]
   }
4. Webhook body přidá do flipchartu.

Secrets (Streamlit → Secrets pane / .toml)
------------------------------------------
OPENAI_API_KEY = "sk-…"          # klíč k Whisperu
WEBHOOK_OUT_TOKEN = "out-token"  # volitelné – posíláme Make
WEBHOOK_IN_TOKEN  = "in-token"   # musí se shodovat s Make → Webhook
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
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ─────────── CONFIG ──────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160  # kolik sekund audia posíláme do Whisperu najednou

# — Make webhooky a tokeny —
MAKE_OUT_WEBHOOK_URL = (
    "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
)  # kam ODESÍLÁME přepis
WEBHOOK_OUT_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")
WEBHOOK_IN_TOKEN: str = st.secrets.get("WEBHOOK_IN_TOKEN", "in-token")

WEBHOOK_PORT = 8000  # náš příchozí FastAPI server

logging.basicConfig(level=logging.INFO)

# ─────────── SESSION STATE INIT ──────────────────────────────────────────
if "flip_points" not in st.session_state:
    st.session_state.flip_points: List[str] = []
if "transcript_buffer" not in st.session_state:
    st.session_state.transcript_buffer = ""
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer: list[bytes] = []
if "status" not in st.session_state:
    st.session_state.status = "🟡 Čekám na mikrofon…"
if "upload_processed" not in st.session_state:
    st.session_state.upload_processed = False
# vlákenné entity
if "audio_stop_event" not in st.session_state:
    st.session_state.audio_stop_event: threading.Event | None = None
if "runner_thread" not in st.session_state:
    st.session_state.runner_thread: threading.Thread | None = None
if "webhook_thread" not in st.session_state:
    st.session_state.webhook_thread: threading.Thread | None = None

# ─────────── STATUS HELPERS ──────────────────────────────────────────────
def set_status(txt: str) -> None:
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

# ─────────── WHISPER SAFE CALL ───────────────────────────────────────────
def whisper_safe(file_like, label: str) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=file_like,
            language="cs",
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper error (%s)", label)
        set_status(f"❌ Chyba Whisperu ({label})")
        st.error(
            f"❌ Whisper API vrátilo chybu ({exc.__class__.__name__}). "
            "Zkontroluj kredit nebo formát souboru."
        )
        return None

# ─────────── SEND TRANSCRIPT TO MAKE ─────────────────────────────────────
def send_to_make(transcript: str, existing: list[str]) -> None:
    payload = {
        "token": WEBHOOK_OUT_TOKEN,
        "transcript": transcript,
        "existing": existing,
    }
    try:
        requests.post(MAKE_OUT_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as exc:
        logging.exception("Nepodařilo se odeslat na Make: %s", exc)
        set_status("⚠️ Nelze odeslat na Make")

# ─────────── INBOUND FASTAPI WEBHOOK ─────────────────────────────────────
class InPayload(BaseModel):
    token:  str       = Field(..., description="Sdílený tajný token")
    points: list[str] = Field(..., description="Pole nových odrážek")

api = FastAPI()

@api.post("/make")
async def ingest(payload: InPayload):
    if payload.token != WEBHOOK_IN_TOKEN:
        raise HTTPException(401, "Bad token")

    new_pts = [p for p in payload.points if p not in st.session_state.flip_points]
    st.session_state.flip_points.extend(new_pts)
    set_status("✅ Body doručeny z Make")
    logging.info("Webhook přidal %d bodů", len(new_pts))
    return {"added": len(new_pts), "total": len(st.session_state.flip_points)}

def start_webhook_server():
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=WEBHOOK_PORT, log_level="warning")

if st.session_state.webhook_thread is None:
    t_web = threading.Thread(target=start_webhook_server, daemon=True, name="webhook")
    add_script_run_ctx(t_web)  # zamezí 'missing ScriptRunContext'
    t_web.start()
    st.session_state.webhook_thread = t_web

# ─────────── CSS + RENDER FLIPCHART ──────────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart li {opacity:0; transform:translateY(8px); animation:fadeIn 0.45s forwards;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

def render_flipchart() -> None:
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

# ─────────── PCM → WAV HELPER ───────────────────────────────────────────
def pcm_to_wav(frames: list[bytes], sr: int = 48000) -> bytes:
    if not frames:
        return b""
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()

# ─────────── STREAMLIT UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator", layout="wide")
tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ========== TAB 1 – Ovládání ============================================
with tabs[0]:
    st.header("Nastavení a vstup zvuku")

    # --- Upload souboru --------------------------------------------------
    uploaded = st.file_uploader(
        "▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )

    if uploaded is not None and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            txt = whisper_safe(uploaded, "upload")
        if txt:
            with status_ctx("📤 Odesílám Make…", done="🟢 Čekám na Flipchart"):
                send_to_make(txt, st.session_state.flip_points)
            st.session_state.upload_processed = True

    # --- Živý mikrofon ---------------------------------------------------
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    # Ukonči staré audio vlákno (při rerunu)
    if (old := st.session_state.runner_thread) and old.is_alive():
        st.session_state.audio_stop_event.set()
        old.join(timeout=2)

    stop_evt = threading.Event()
    st.session_state.audio_stop_event = stop_evt

    async def audio_pipeline(ctx, stop_event: threading.Event):
        SR = 48000
        target_bytes = AUDIO_BATCH_SECONDS * SR * 2  # 16-bit mono bytes

        while not stop_event.is_set():
            if not ctx.audio_receiver:
                set_status("🟡 Čekám na mikrofon…")
                await asyncio.sleep(0.1)
                continue

            set_status("🔴 Zachytávám audio…")
            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(f.to_ndarray().tobytes() for f in frames)

            if sum(len(b) for b in st.session_state.audio_buffer) < target_bytes:
                await asyncio.sleep(0.05)
                continue

            wav_bytes = pcm_to_wav(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()

            with status_ctx("🟣 Whisper (mikrofon)…"):
                tr = whisper_safe(io.BytesIO(wav_bytes), "mikrofon")
            if not tr:
                await asyncio.sleep(1)
                continue

            st.session_state.transcript_buffer += " " + tr
            if len(st.session_state.transcript_buffer.split()) >= 325:
                with status_ctx("📤 Odesílám Make…", done="🟢 Čekám na Flipchart"):
                    send_to_make(
                        st.session_state.transcript_buffer,
                        st.session_state.flip_points,
                    )
                st.session_state.transcript_buffer = ""

            await asyncio.sleep(0.05)

        set_status("⏹️ Audio pipeline ukončena")

    t_audio = threading.Thread(
        target=lambda c=webrtc_ctx, e=stop_evt: asyncio.run(audio_pipeline(c, e)),
        daemon=True,
        name="audio-runner",
    )
    add_script_run_ctx(t_audio)
    t_audio.start()
    st.session_state.runner_thread = t_audio

    # --- Sidebar diagnostika --------------------------------------------
    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav")
    st.sidebar.write(st.session_state.status)
    st.sidebar.write("Audio vlákno běží:", t_audio.is_alive())
    st.sidebar.write("Webhook vlákno běží:",
                     st.session_state.webhook_thread.is_alive())

# ========== TAB 2 – Flipchart ============================================
with tabs[1]:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
