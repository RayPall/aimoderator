# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon → OpenAI Whisper → Make webhook → živý Flipchart
---------------------------------------------------------------------------

Co dělá
-------
• Živý mikrofon i testovací upload souboru (WAV/MP3/M4A)  
• OpenAI Whisper → text přepisu  
• Přepis + stávající body odešle na Make webhook  
  (https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7)  
  ➜ Scénář v Make vrátí **JSON pole nových odrážek**  
• Ty se animovaně doplní na Flipchart  
• Diagnostické okno se stavem zpracování, robustní zachytávání chyb  
• Vlákno audio-pipeline se korektně restartuje při každém rerunu
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
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ─────────────────────── CONFIG ────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

MAKE_WEBHOOK_URL = (
    "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
)  # <- zde změň, pokud bude jiný

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160  # kolik sekund audia pošleme v jedné dávce do Whisperu

logging.basicConfig(level=logging.INFO)

# ─────────────────── SESSION STATE INIT ────────────────────────────────────
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
    if st.session_state.status != s:
        st.session_state.status = s

@contextlib.contextmanager
def status_ctx(running: str, done: str | None = None):
    prev = st.session_state.status
    set_status(running)
    try:
        yield
    finally:
        set_status(done or prev)

def whisper_safe(file_like, label: str) -> str | None:
    """Volá Whisper; při chybě zaloguje, zobrazí a vrátí None."""
    try:
        return (
            client.audio.transcriptions.create(
                model="whisper-1", file=file_like, language="cs"
            ).text
        )
    except OpenAIError as e:
        logging.exception("Whisper API error (%s)", label)
        set_status(f"❌ Chyba Whisperu ({label})")
        st.error(
            "❌ Whisper API vrátilo chybu – zkontroluj formát/velikost souboru "
            "a kredit účtu.\n\n"
            f"Typ chyby: **{e.__class__.__name__}**"
        )
        return None

def call_make_webhook(transcript: str, existing: list[str]) -> list[str]:
    """Pošle JSON na Make, vrátí list nových bodů (nebo prázdný)."""
    payload = {"transcript": transcript, "existing": existing}
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            logging.error("Webhook nevrátil list: %s", data)
            set_status("⚠️ Neplatná odpověď webhooku")
            return []
        return [str(p).strip() for p in data if str(p).strip()]
    except Exception as e:
        logging.exception("Chyba při volání Make webhooku")
        set_status("❌ Chyba webhooku")
        st.error(f"❌ Chyba při volání Make scénáře: {e}")
        return []

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

def pcm_frames_to_wav(frames: list[bytes], sr: int = 48000) -> bytes:
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

# ───────────────────── UI LAYOUT ───────────────────────────────────────────
st.set_page_config(page_title="AI Flipchart", layout="wide")
tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ========== TAB 1: Ovládání ================================================
with tabs[0]:
    st.header("Nastavení a vstup zvuku")

    # —— TESTOVACÍ UPLOAD ——————————————————————————
    uploaded = st.file_uploader(
        "▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )
    if uploaded is not None and not st.session_state.upload_processed:
        with status_ctx("🟣 Odesílám soubor do Whisper…"):
            st.info("⏳ Zpracovávám nahraný soubor…")
            transcription = whisper_safe(uploaded, label="upload")
            if transcription:
                with status_ctx("🧠 Volám Make webhook…"):
                    new_pts = call_make_webhook(
                        transcript=transcription,
                        existing=st.session_state.flip_points,
                    )
                    st.session_state.flip_points.extend(new_pts)
                st.success("✅ Soubor zpracován – přepněte na záložku Flipchart")
                st.session_state.upload_processed = True

    # —— ŽIVÝ MIKROFON ———————————————————————————
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    # — zastav předchozí vlákno (při rerunu) —
    old_t = st.session_state.runner_thread
    if old_t and old_t.is_alive():
        st.session_state.audio_stop_event.set()
        old_t.join(timeout=2)

    stop_evt = threading.Event()
    st.session_state.audio_stop_event = stop_evt

    async def pipeline_runner(ctx, stop_event: threading.Event):
        SAMPLE_RATE = 48000
        bytes_per_sec = SAMPLE_RATE * 2
        target = AUDIO_BATCH_SECONDS * bytes_per_sec

        while not stop_event.is_set():
            if not ctx.audio_receiver:
                set_status("🟡 Čekám na mikrofon…")
                await asyncio.sleep(0.1)
                continue

            set_status("🔴 Zachytávám audio…")
            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(
                fr.to_ndarray().tobytes() for fr in frames
            )
            if sum(len(b) for b in st.session_state.audio_buffer) < target:
                await asyncio.sleep(0.05)
                continue

            wav_bytes = pcm_frames_to_wav(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()

            set_status("🟣 Odesílám do Whisper…")
            transcription = whisper_safe(io.BytesIO(wav_bytes), label="mikrofon")
            if not transcription:
                await asyncio.sleep(1)
                continue

            st.session_state.transcript_buffer += " " + transcription

            if len(st.session_state.transcript_buffer.split()) >= 325:
                set_status("🧠 Volám Make webhook…")
                new_pts = call_make_webhook(
                    transcript=st.session_state.transcript_buffer,
                    existing=st.session_state.flip_points,
                )
                st.session_state.flip_points.extend(
                    [p for p in new_pts if p not in st.session_state.flip_points]
                )
                st.session_state.transcript_buffer = ""

            set_status("🟢 Čekám na další dávku…")
            await asyncio.sleep(0.05)

        set_status("⏹️ Audio vlákno ukončeno")

    t = threading.Thread(
        target=lambda c=webrtc_ctx, e=stop_evt: asyncio.run(pipeline_runner(c, e)),
        daemon=True,
        name="audio-runner",
    )
    add_script_run_ctx(t)
    t.start()
    st.session_state.runner_thread = t

    # —— SIDEBAR DIAGNOSTIKA ————————————————————
    st.sidebar.header("ℹ️ Stav aplikace")
    st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav zpracování")
    st.sidebar.write(st.session_state.get("status", "❔ Neznámý stav"))
    st.sidebar.write("Vlákno žije:", t.is_alive())

# ========== TAB 2: Flipchart ===============================================
with tabs[1]:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
