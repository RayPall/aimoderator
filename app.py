# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon → OpenAI Whisper → ChatGPT → živý Flipchart
---------------------------------------------------------------------------

Funkce
------
• Živý mikrofon i testovací upload souboru  
• Automatické shrnování řeči do odrážek (flipchart)  
• Diagnostické okno se stavem zpracování + zachytávání chyb API  
• Vlákno s audio pipeline se korektně ukončí při každém rerunu (žádné
  'missing ScriptRunContext!' varování)
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
import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ────────────────────── CONFIG ──────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160            # kolik sekund audia posíláme najednou

logging.basicConfig(level=logging.INFO)

# ──────────────────── SESSION STATE INIT ───────────────────────────────────
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
# vlákno + stop_event pro pipeline
if "audio_stop_event" not in st.session_state:
    st.session_state.audio_stop_event: threading.Event | None = None
if "runner_thread" not in st.session_state:
    st.session_state.runner_thread: threading.Thread | None = None

# ───────────────────── STATUS HELPERS ──────────────────────────────────────
def set_status(s: str) -> None:
    """Přepíše status jen pokud se změnil – šetří log & render."""
    if st.session_state.status != s:
        st.session_state.status = s

@contextlib.contextmanager
def prev_status_ctx(running: str, done: str | None = None):
    """Dočasně nastaví stav; po ★yield★ vrátí předchozí (nebo `done`)."""
    prev = st.session_state.status
    set_status(running)
    try:
        yield
    finally:
        set_status(done or prev)

def whisper_safe_call(*, file_like, label: str) -> str | None:
    """
    Zavolá Whisper. Při chybě zaloguje, informuje uživatele a vrátí None.
    `label` → upload / mikrofon – zobrazí se v diagnostice.
    """
    try:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=file_like,
            language="cs",
        ).text
    except OpenAIError as e:
        # plný traceback do logu
        logging.exception("Whisper API error (%s)", label)
        set_status(f"❌ Chyba Whisperu ({label})")
        st.error(
            "❌ Whisper API vrátilo chybu.\n"
            "• Zkontroluj formát/velikost souboru a kredit účtu.\n\n"
            f"Typ chyby: **{e.__class__.__name__}**"
        )
        return None

# ───────────────────────── CSS ─────────────────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart li {opacity:0; transform:translateY(8px); animation:fadeIn 0.45s forwards;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

# ─────────────────────── HELPERS ───────────────────────────────────────────
def render_flipchart() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    pts = st.session_state.flip_points
    if not pts:
        st.info("Čekám na první shrnutí…")
        return
    st.markdown(
        "<ul class='flipchart'>" +
        "".join(
            f"<li style='animation-delay:{i*0.1}s'>{p}</li>"
            for i, p in enumerate(pts)
        ) +
        "</ul>",
        unsafe_allow_html=True,
    )

def pcm_frames_to_wav(frames: list[bytes], sample_rate: int = 48000) -> bytes:
    """Sloučí PCM rámce na WAV (mono, 16-bit)."""
    if not frames:
        return b""
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()

PROMPT = """
Jsi zkušený moderátor workshopu FWB Summit 2025 …
Tvým úkolem je shrnovat projev do klíčových myšlenek:
NADPIS
- bod
- bod
Vracíš pouze NOVÉ body jako JSON pole.
"""

def summarise_new_points(text: str, existing: list[str]) -> list[str]:
    msgs = [
        {"role": "system",    "content": PROMPT},
        {"role": "user",      "content": text},
        {"role": "assistant", "content": json.dumps(existing, ensure_ascii=False)},
    ]
    raw = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", temperature=0.2, messages=msgs
    ).choices[0].message.content
    try:
        pts = json.loads(raw)
        if not isinstance(pts, list):
            raise ValueError
        return [p.strip() for p in pts if p.strip()]
    except Exception:
        # fallback – prosté rozparsování odrážek
        return [ln.lstrip("-• ").strip() for ln in raw.splitlines() if ln.strip()]

# ─────────────────────── UI LAYOUT ─────────────────────────────────────────
st.set_page_config(page_title="AI Flipchart", layout="wide")
tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ========== TAB 1: Ovládání ==================================================
with tabs[0]:
    st.header("Nastavení a vstup zvuku")

    # —— TESTOVACÍ UPLOAD ————————————————————————————————
    uploaded = st.file_uploader(
        "▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )

    if uploaded is
