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

    if uploaded is not None and not st.session_state.upload_processed:
        with prev_status_ctx("🟣 Odesílám soubor do Whisper…"):
            st.info("⏳ Zpracovávám nahraný soubor…")
            transcription = whisper_safe_call(file_like=uploaded, label="upload")
            if transcription is not None:
                with prev_status_ctx("🧠 Generuji shrnutí (soubor)…"):
                    new_pts = summarise_new_points(
                        transcription, st.session_state.flip_points
                    )
                    st.session_state.flip_points.extend(new_pts)
                st.success("✅ Soubor zpracován – přepněte na záložku Flipchart")
                st.session_state.upload_processed = True

    # —— ŽIVÝ MIKROFON ————————————————————————————————
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    # --- zastavení předchozího vlákna (pokud bylo) --------------------------
    old_thread = st.session_state.runner_thread
    if old_thread is not None and old_thread.is_alive():
        st.session_state.audio_stop_event.set()
        old_thread.join(timeout=2)

    # --- nový stop_event + vlákno ------------------------------------------
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
            transcription = whisper_safe_call(
                file_like=io.BytesIO(wav_bytes), label="mikrofon"
            )
            if transcription is None:
                await asyncio.sleep(1)
                continue

            st.session_state.transcript_buffer += " " + transcription

            if len(st.session_state.transcript_buffer.split()) >= 325:
                set_status("🧠 Generuji shrnutí…")
                new_pts = summarise_new_points(
                    st.session_state.transcript_buffer,
                    st.session_state.flip_points,
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

    # —— SIDEBAR DIAGNOSTIKA ————————————————————————————
    st.sidebar.header("ℹ️ Stav aplikace")
    st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav zpracování")
    st.sidebar.write(st.session_state.get("status", "❔ Neznámý stav"))
    st.sidebar.write("Vlákno žije:", t.is_alive())

# ========== TAB 2: Flipchart =================================================
with tabs[1]:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
