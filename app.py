# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
•  Mikrofon / upload ⭢ OpenAI Whisper  
•  Výsledek ⭢ Make (webhook) → dostaneme pole bodů  
•  Flipchart: NADPIS tučně, podbody s „•“  
•  Sidebar: diagnostika + indikátor „🎙️ Zvuk přijímán / 🟡 Žádný zvuk“  
•  Tester mikrofonu: nahraj 3 s, WAV se rovnou přehraje
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import queue as q
import re
import threading
import time
import wave
from typing import List

import numpy as np
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ─────────── CONFIG ─────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets"); st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_BLOCK_SEC       = 5          # délka WAV bloku posílaného do Whisper
RECEIVER_SIZE_FRAMES  = 1024       # fronta streamlit-webrtc (~20 s)
MAKE_WEBHOOK_URL      = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN     = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

logging.basicConfig(level=logging.INFO)

# ─────────── SESSION STATE INIT ─────────────────────────────────────────
s = st.session_state
defaults = {
    "flip_points": [],            # zobrazené body
    "transcript_buffer": "",      # surový buffer slov
    "audio_buffer": [],           # PCM rámce (list[bytes])
    "status": "🟡 Čekám na mikrofon…",
    "upload_processed": False,
    "audio_stop_event": None,
    "runner_thread": None,
    "last_frame_time": 0.0,       # čas posledního doručeného audio-rámce
    "test_wav": b"",              # WAV z testeru
}
for k, v in defaults.items():
    s.setdefault(k, v)

def set_status(txt: str) -> None:
    s.status = txt

@contextlib.contextmanager
def status_ctx(running: str, done: str | None = None):
    prev = s.status
    set_status(running)
    try:
        yield
    finally:
        set_status(done or prev)

# ─────────── OpenAI Whisper wrappers ────────────────────────────────────
def whisper_safe(buf: io.BytesIO | io.BufferedReader, label: str) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=buf, language="cs"
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper error (%s)", label)
        st.error(f"❌ Whisper API: {exc}")
        return None

# ─────────── Make webhook call ──────────────────────────────────────────
def call_make(transcript: str, existing: list[str]) -> list[str]:
    payload = {"token": WEBHOOK_OUT_TOKEN,
               "transcript": transcript,
               "existing": existing}
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        return [str(p).strip() for p in data if str(p).strip()] if isinstance(data, list) else []
    except Exception as exc:
        logging.exception("Make")
        st.error(exc)
        return []

# ─────────── Flipchart render ───────────────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style:none; padding-left:0;}
ul.flipchart>li {opacity:0; transform:translateY(8px);
                 animation:fadeIn .45s forwards; margin-bottom:1.2rem;}
ul.flipchart strong {display:block; font-weight:700; margin-bottom:.4rem;}
ul.flipchart ul {margin:0 0 0 1.2rem; padding-left:0;}
ul.flipchart ul li {list-style:disc; margin-left:1rem; margin-bottom:.2rem;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:.5rem;}
</style>
"""
DASH_SPLIT = re.compile(r"\s+-\s+");  STRIP_CHARS = "-–—• "

def fmt_point(raw: str) -> str:
    raw = raw.strip()
    if "\n" in raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        parts = DASH_SPLIT.split(raw)
        lines = [parts[0]] + [f"- {p}" for p in parts[1:]]
    if not lines:
        return ""
    heading, *details = lines
    heading_html = f"<strong>{heading.upper()}</strong>"
    if not details:
        return heading_html
    items = "".join(f"<li>{d.lstrip(STRIP_CHARS)}</li>" for d in details)
    return f"{heading_html}<ul>{items}</ul>"

def render_flipchart() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    if not s.flip_points:
        st.info("Čekám na shrnutí…")
        return
    html = "<ul class='flipchart'>" + "".join(
        f"<li style='animation-delay:{i*0.1}s'>{fmt_point(p)}</li>"
        for i, p in enumerate(s.flip_points)
    ) + "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ─────────── Audio utils ────────────────────────────────────────────────
def pcm_to_wav(frames: list[bytes], sr: int = 48000) -> bytes:
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()

# ─────────── Layout ─────────────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator", layout="wide")
tab_ctrl, tab_flip = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# === TAB 1 – Ovládání ====================================================
with tab_ctrl:
    st.header("Nahrávání / Upload")

    # -------- Upload souboru --------
    up = st.file_uploader("▶️ WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
    if up and not s.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            txt = whisper_safe(up, "upload")
        if txt:
            with status_ctx("📤 Make…", "🟢 Čekám Make"):
                new_pts = call_make(txt, s.flip_points)
            s.flip_points.extend(p for p in new_pts if p not in s.flip_points)
            s.upload_processed = True

    # -------- Živý mikrofon ---------
    st.subheader("🎤 Mikrofon (živý)")
    webrtc_ctx = webrtc_streamer(
        key="mic-main",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=RECEIVER_SIZE_FRAMES,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )

    # Stop předchozí audio-pipeline při rerunu
    if (old := s.runner_thread) and old.is_alive():
        s.audio_stop_event.set(); old.join(timeout=2)

    stop_evt = threading.Event(); s.audio_stop_event = stop_evt
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    async def reader(ctx):
        sr = 48000
        target = AUDIO_BLOCK_SEC * sr * 2
        buf: list[bytes] = []
        while not stop_evt.is_set():
            if not ctx.audio_receiver:
                await asyncio.sleep(0.05); continue
            try:
                frames = ctx.audio_receiver.get_frames(timeout=1)  # BEZ await
            except q.Empty:
                continue

            if frames:
                s.last_frame_time = time.time()
                set_status("🔴 Zachytávám audio…")

            buf.extend(f.to_ndarray().tobytes() for f in frames)
            if sum(map(len, buf)) >= target:
                wav = pcm_to_wav(buf); buf.clear()
                try:
                    queue.put_nowait(wav)
                except asyncio.QueueFull:
                    _ = queue.get_nowait(); queue.put_nowait(wav)

    async def worker():
        while not stop_evt.is_set():
            wav = await queue.get()
            set_status("🟣 Whisper …")
            txt = await asyncio.to_thread(
                whisper_safe, io.BytesIO(wav), "mic"
            )
            if txt:
                s.transcript_buffer += " " + txt
                if len(s.transcript_buffer.split()) >= 325:
                    set_status("📤 Make…")
                    new = await asyncio.to_thread(
                        call_make, s.transcript_buffer, s.flip_points
                    )
                    s.flip_points.extend(
                        p for p in new if p not in s.flip_points
                    )
                    s.transcript_buffer = ""
            queue.task_done()

    async def pipeline():
        await asyncio.gather(reader(webrtc_ctx), worker())

    thr = threading.Thread(
        target=lambda: asyncio.run(pipeline()),
        daemon=True,
        name="audio-pipe",
    )
    add_script_run_ctx(thr); thr.start(); s.runner_thread = thr

    # -------- Mikrofonní tester --------
    st.markdown("---")
    with st.expander("🧪 Test mikrofonu (nahraj 3 s a přehraj)"):
        test_ctx = webrtc_streamer(
            key="mic-test",
            mode=WebRtcMode.SENDONLY,          # pouze odesíláme
            audio_receiver_size=256,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"audio": True, "video": False},
        )

        if st.button("🎙️ Nahrát 3 s"):
            if not test_ctx.audio_receiver:
                st.warning("Mikrofon ještě nepřipraven"); st.stop()

            frames: list[bytes] = []
            start = time.time()
            with st.spinner("Nahrávám…"):
                while time.time() - start < 3:
                    try:
                        batch = test_ctx.audio_receiver.get_frames(timeout=1)
                        frames.extend(f.to_ndarray().tobytes() for f in batch)
                    except q.Empty:
                        pass

            test_ctx.stop()  # korektně ukončit widget
            s.test_wav = pcm_to_wav(frames)
            st.success("Hotovo – přehrajte níže")

        if s.test_wav:
            st.audio(s.test_wav, format="audio/wav")

    # -------- Sidebar diagnostika ------
    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body:", len(s.flip_points))
    st.sidebar.write("Slov v bufferu:", len(s.transcript_buffer.split()))
    if time.time() - s.last_frame_time < 2:
        st.sidebar.success("🎙️ Zvuk přijímán")
    else:
        st.sidebar.warning("🟡 Žádný zvuk")
    st.sidebar.write("Stav:", s.status)
    st.sidebar.write("Audio thread alive:", thr.is_alive())

# === TAB 2 – Flipchart ===================================================
with tab_flip:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
