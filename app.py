# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
•  Mikrofon / upload ⭢ OpenAI Whisper – transkript  
•  Transkript POST na Make (token, transcript, existing)  
•  Make vrátí JSON pole nových bodů → hned zobrazíme  

UX
~~~
*  NADPIS tučně (velkými)  
*  Podbody se symbolem •  
*  Podporuje formáty  
       NADPIS\n- detail  
       NADPIS - detail - detail
~~~

Technika
--------
*  WebRTC = **SENDONLY** → žádná ozvěna do sluchátek  
*  Příjem audia (`reader`) a volání Whisper + Make (`worker`) běží
   paralelně → fronta nikdy nepřeteče  
*  `audio_receiver_size = 1024` (cca 20 s audia)  
*  Jeden WAV blok = **5 s** → rychlejší odezva
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import re
import threading
import wave
from typing import List

import numpy as np
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ───────────────────── CONFIG ───────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_BATCH_SECONDS  = 5        # délka WAV bloku
RECEIVER_SIZE_FRAMES = 1024     # fronta webrtc (≈ 20 s)
MAKE_WEBHOOK_URL     = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN    = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

logging.basicConfig(level=logging.INFO)

# ───────────────────── SESSION STATE ────────────────────────────────────
def _init_state() -> None:
    s = st.session_state
    s.setdefault("flip_points", [])
    s.setdefault("transcript_buffer", "")
    s.setdefault("audio_buffer", [])
    s.setdefault("status", "🟡 Čekám na mikrofon…")
    s.setdefault("upload_processed", False)
    s.setdefault("audio_stop_event", None)
    s.setdefault("runner_thread", None)
    s.setdefault("webrtc_ctx", None)

_init_state()

# ───────────────────── STATUS HELPERS ───────────────────────────────────
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

# ───────────────────── Whisper wrapper ──────────────────────────────────
def whisper_safe(buf: io.BytesIO, label: str) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
            language="cs",
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper error (%s)", label)
        set_status(f"❌ Chyba Whisperu ({label})")
        st.error(f"❌ Whisper API: {exc.__class__.__name__}")
        return None

# ───────────────────── Make volání ──────────────────────────────────────
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
            set_status("⚠️ Neplatná odpověď Make")
            return []
        return [str(p).strip() for p in data if str(p).strip()]
    except Exception as exc:
        logging.exception("HTTP Make error")
        set_status("⚠️ Chyba Make")
        st.error(exc)
        return []

# ───────────────────── Flipchart render ─────────────────────────────────
STYLES = """
<style>
ul.flipchart {list-style:none; padding-left:0;}
ul.flipchart>li{opacity:0;transform:translateY(8px);
                animation:fadeIn .45s forwards;margin-bottom:1.2rem;}
ul.flipchart strong{display:block;font-weight:700;margin-bottom:.4rem;}
ul.flipchart ul{margin:0 0 0 1.2rem;padding-left:0;}
ul.flipchart ul li{list-style:disc;margin-left:1rem;margin-bottom:.2rem;}
@keyframes fadeIn{to{opacity:1;transform:translateY(0);}}
.fullscreen header,.fullscreen #MainMenu,.fullscreen footer{visibility:hidden;}
.fullscreen .block-container{padding-top:.5rem;}
</style>
"""
DASH_SPLIT  = re.compile(r"\s+-\s+")
STRIP_CHARS = "-–—• "

def format_point(raw: str) -> str:
    raw = raw.strip()
    if "\n" in raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        parts = DASH_SPLIT.split(raw)
        lines = [parts[0]] + [f"- {p}" for p in parts[1:]]

    if not lines:
        return ""
    heading, *details = lines
    head_html = f"<strong>{heading.upper()}</strong>"
    if not details:
        return head_html
    items = "".join(f"<li>{d.lstrip(STRIP_CHARS)}</li>" for d in details)
    return f"{head_html}<ul>{items}</ul>"

def render_flipchart() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    pts = st.session_state.flip_points
    if not pts:
        st.info("Čekám na shrnutí…")
        return
    html = "<ul class='flipchart'>" + "".join(
        f"<li style='animation-delay:{i*0.1}s'>{format_point(p)}</li>"
        for i, p in enumerate(pts)
    ) + "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ───────────────────── Audio utils ──────────────────────────────────────
def pcm_to_wav(frames: list[bytes], sr: int = 48000) -> bytes:
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf.read()

# ───────────────────── UI LAYOUT ────────────────────────────────────────
st.set_page_config(page_title="AI Moderator", layout="wide")
tab_ctrl, tab_flip = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# === TAB OVLÁDÁNÍ =======================================================
with tab_ctrl:
    st.header("Nahrávání / Upload")

    up = st.file_uploader("▶️ Testovací WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
    if up and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            text = whisper_safe(up, "upload")
        if text:
            with status_ctx("📤 Make…", "🟢 Čekám Make"):
                new_pts = call_make(text, st.session_state.flip_points)
            st.session_state.flip_points.extend(
                p for p in new_pts if p not in st.session_state.flip_points
            )
            st.session_state.upload_processed = True

    st.subheader("🎤 Mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="mic-rec",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=RECEIVER_SIZE_FRAMES,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )
    st.session_state.webrtc_ctx = webrtc_ctx

    # Zastav předchozí audio vlákno při rerunu
    if (old := st.session_state.runner_thread) and old.is_alive():
        st.session_state.audio_stop_event.set()
        old.join(timeout=2)

    stop_evt = threading.Event()
    st.session_state.audio_stop_event = stop_evt
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    async def reader(ctx) -> None:
        sr = 48000
        target = AUDIO_BATCH_SECONDS * sr * 2  # 5 s v bajtech
        while not stop_evt.is_set():
            if not ctx.audio_receiver:
                await asyncio.sleep(0.05)
                continue

            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(
                f.to_ndarray().tobytes() for f in frames
            )

            if sum(map(len, st.session_state.audio_buffer)) >= target:
                wav = pcm_to_wav(st.session_state.audio_buffer)
                st.session_state.audio_buffer.clear()
                try:
                    queue.put_nowait(wav)
                except asyncio.QueueFull:
                    _ = queue.get_nowait()
                    queue.put_nowait(wav)

    async def worker() -> None:
        while not stop_evt.is_set():
            wav = await queue.get()
            set_status("🟣 Whisper (worker)…")
            txt = await asyncio.to_thread(whisper_safe, io.BytesIO(wav), "mic")
            if txt:
                st.session_state.transcript_buffer += " " + txt
                if len(st.session_state.transcript_buffer.split()) >= 325:
                    set_status("📤 Make…")
                    new = await asyncio.to_thread(
                        call_make,
                        st.session_state.transcript_buffer,
                        st.session_state.flip_points,
                    )
                    st.session_state.flip_points.extend(
                        p for p in new if p not in st.session_state.flip_points
                    )
                    st.session_state.transcript_buffer = ""
            queue.task_done()

    # ---------- SPUŠTĚNÍ PIPELINE (reader + worker) ----------------------
    async def pipeline():
        await asyncio.gather(reader(webrtc_ctx), worker())   # ❗ BEZ []

    thr = threading.Thread(
        target=lambda: asyncio.run(pipeline()),
        daemon=True,
        name="audio-pipeline"
    )
    add_script_run_ctx(thr)
    thr.start()
    st.session_state.runner_thread = thr

    # Sidebar diagnostika
    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav")
    st.sidebar.write(st.session_state.status)
    st.sidebar.write("Audio thread:", thr.is_alive())

# === TAB FLIPCHART ======================================================
with tab_flip:
    st.components.v1.html(
        "<script>document.body.classList.add('fullscreen');</script>", height=0
    )
    render_flipchart()
