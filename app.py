# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
• Upload / mikrofon  →  OpenAI Whisper  – přepis
• Přepis POST na Make webhook (token, transcript, existing)
• Make vrací pole NOVÝCH bodů → okamžitě zobrazíme
• Flipchart: NADPIS tučně, podbody s "•", funguje i když Make vrátí:
    1) víceřádkově:  NADPIS\n- detail
    2) jednořádkově: NADPIS - detail - detail
• WebRTC = SENDONLY  → žádná ozvěna do sluchátek
"""

from __future__ import annotations
import asyncio, contextlib, io, logging, re, threading, wave
from typing import List

import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ───────────── CONFIG ────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets")
    st.stop()

client             = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160

MAKE_WEBHOOK_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN  = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

logging.basicConfig(level=logging.INFO)

# ───────────── SESSION STATE INIT ────────────────────────────────────────
def _init():
    ss = st.session_state
    ss.setdefault("flip_points", [])
    ss.setdefault("transcript_buffer", "")
    ss.setdefault("audio_buffer", [])
    ss.setdefault("status", "🟡 Čekám na mikrofon…")
    ss.setdefault("upload_processed", False)
    ss.setdefault("audio_stop_event", None)
    ss.setdefault("runner_thread", None)
_init()

# ───────────── STATUS HELPERS ────────────────────────────────────────────
def set_status(txt: str):  # minimalizuje rerender
    if st.session_state.status != txt:
        st.session_state.status = txt

@contextlib.contextmanager
def status_ctx(running: str, done: str | None = None):
    prev = st.session_state.status
    set_status(running)
    try: yield
    finally: set_status(done or prev)

# ───────────── WHISPER SAFE CALL ─────────────────────────────────────────
def whisper_safe(file_like, lbl: str) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=file_like, language="cs"
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper error (%s)", lbl)
        set_status(f"❌ Chyba Whisperu ({lbl})")
        st.error(f"❌ Whisper API: {exc.__class__.__name__}")
        return None

# ───────────── CALL MAKE ────────────────────────────────────────────────
def call_make(transcript: str, existing: list[str]) -> list[str]:
    try:
        r = requests.post(
            MAKE_WEBHOOK_URL,
            json={"token": WEBHOOK_OUT_TOKEN,
                  "transcript": transcript,
                  "existing": existing},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            logging.error("Make response není list: %s", data)
            set_status("⚠️ Neplatná odpověď Make")
            return []
        return [str(p).strip() for p in data if str(p).strip()]
    except Exception as exc:
        logging.exception("HTTP Make error")
        set_status("⚠️ Chyba Make")
        st.error(exc)
        return []

# ───────────── FLIPCHART RENDER ─────────────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart > li {opacity:0; transform:translateY(8px);
                   animation:fadeIn 0.45s forwards; margin-bottom:1.2rem;}

ul.flipchart strong {display:block; font-weight:700; margin-bottom:0.4rem;}
ul.flipchart ul    {margin:0 0 0 1.2rem; padding-left:0;}
ul.flipchart ul li {list-style-type:disc; margin-left:1rem; margin-bottom:0.2rem;}

@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

DASH_SPLIT   = re.compile(r"\s+-\s+")      # mezery, pomlčka, mezery
STRIP_CHARS  = "-–—• "                     # ascii, en-dash, em-dash, bullet, mezera

def format_point(raw: str) -> str:
    """
    Převod vstupu na HTML:
       • NADPIS\n- detail
       • NADPIS - detail - detail
    """
    raw = raw.strip()
    # Rozděl na řádky / části
    if "\n" in raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        parts = DASH_SPLIT.split(raw)
        lines = [parts[0]] + [f"- {p}" for p in parts[1:]]

    if not lines: return ""

    heading, *details = lines
    heading_html = f"<strong>{heading.upper()}</strong>"

    if not details:
        return heading_html

    items = "".join(
        f"<li>{d.lstrip(STRIP_CHARS)}</li>"
        for d in details
    )
    return f"{heading_html}<ul>{items}</ul>"

def render_flipchart():
    st.markdown(STYLES, unsafe_allow_html=True)
    pts = st.session_state.flip_points
    if not pts:
        st.info("Čekám na první shrnutí…")
        return
    html = "<ul class='flipchart'>" + "".join(
        f"<li style='animation-delay:{i*0.1}s'>{format_point(p)}</li>"
        for i, p in enumerate(pts)
    ) + "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ─
