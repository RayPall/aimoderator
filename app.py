# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon → OpenAI Whisper → ChatGPT → živý Flipchart
---------------------------------------------------------------------------

Novinky
=======
1. Animované vplynutí (fade-in) bodů
2. Záložky (tabs)
3. Diagnostický panel se stavem v sidebaru
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import threading
import wave
from typing import List

import numpy as np
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ─────────── CONFIG ───────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160

# ─────────── SESSION STATE INIT ───────────
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

# ─────────── STATUS HELPERS ───────────
def set_status(s: str) -> None:
    """Bezpečně zapíše stav (jen pokud se opravdu změnil)."""
    if st.session_state.status != s:
        st.session_state.status = s

@contextlib.contextmanager
def prev_status_ctx(running: str, done: str | None = None):
    """Dočasně nastaví stav, po blokovém kódu vrátí původní."""
    previous = st.session_state.status
    set_status(running)
    try:
        yield
    finally:
        set_status(done or previous)

# ─────────── CSS (fade-in + fullscreen) ───────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart li {opacity:0; transform:translateY(8px); animation:fadeIn 0.45s forwards;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

# ─────────── HELPERS ───────────
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

def pcm_frames_to_wav(frames: list[bytes], sample_rate: int = 48000) -> bytes:
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
Tvým úkolem je shrnovat projev do klíčových myšlenek …
"""

def summarise_new_points(text: str, existing: list[str]) -> list[str]:
    msgs = [
        {"role": "system", "content": PROMPT},
        {"role": "user",    "content": text},
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
        return [ln.lstrip("-• ").strip() for ln in raw.splitlines() if ln.strip()]

# ─────────── STREAMLIT UI ───────────
st.set_page_config(page_title="AI Flipchart", layout="wide")
tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ========== TAB 1: Ovládání ==========
with tabs[0]:
    st.header("Nastavení a vstup zvuku")
    uploaded = st.file_uploader(
        "▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=False,
    )

    # ---------- TESTOVACÍ / UPLOAD VĚTEV ----------
    if uploaded is not None:
        if not st.session_state.upload_processed:
            with prev_status_ctx("🟣 Odesílám soubor do Whisper…",
                                 done="✅ Soubor zpracován"):
                st.info("⏳ Zpracovávám nahraný soubor…")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=uploaded, language="cs"
                ).text
                with prev_status_ctx("🧠 Generuji shrnutí (soubor)…"):
                    new_pts = summarise_new_points(
                        transcription, st.session_state.flip_points
                    )
                    st.session_state.flip_points.extend(new_pts)
                st.session_state.upload_processed = True
        else:
            st.success("✅ Soubor už byl zpracován – přepněte na záložku Flipchart")

    # ---------- LIVE MICROPHONE ----------
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    async def pipeline_runner():
        SAMPLE_RATE = 48000
        bytes_per_sec = SAMPLE_RATE * 2
        target = AUDIO_BATCH_SECONDS * bytes_per_sec
        while True:
            if not webrtc_ctx.audio_receiver:
                set_status("🟡 Čekám na mikrofon…")
                await asyncio.sleep(0.1)
                continue

            set_status("🔴 Zachytávám audio…")
            frames = await webrtc_ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(fr.to_ndarray().tobytes() for fr in frames)

            if sum(len(b) for b in st.session_state.audio_buffer) < target:
                await asyncio.sleep(0.05)
                continue

            wav_bytes = pcm_frames_to_wav(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()
            set_status("🟣 Odesílám do Whisper…")
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=io.BytesIO(wav_bytes), language="cs"
            ).text
            st.session_state.transcript_buffer += " " + transcription

            if len(st.session_state.transcript_buffer.split()) >= 325:
                set_status("🧠 Generuji shrnutí…")
                new_pts = summarise_new_points(
                    st.session_state.transcript_buffer, st.session_state.flip_points
                )
                st.session_state.flip_points.extend(
                    [p for p in new_pts if p not in st.session_state.flip_points]
                )
                st.session_state.transcript_buffer = ""
            set_status("🟢 Čekám na další dávku…")
            await asyncio.sleep(0.05)

    if "runner_created" not in st.session_state:
        t = threading.Thread(
            target=lambda: asyncio.run(pipeline_runner()),
            daemon=True, name="audio-runner",
        )
        add_script_run_ctx(t)
        t.start()
        st.session_state.runner_created = True

    # ---------- SIDEBAR ----------
    st.sidebar.header("ℹ️ Stav aplikace")
    st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav zpracování")
    st.sidebar.write(st.session_state.get("status", "❔ Neznámý stav"))

# ========== TAB 2: Flipchart ==========
with tabs[1]:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
