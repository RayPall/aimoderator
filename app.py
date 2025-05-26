# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon ➜ OpenAI Whisper API ➜ ChatGPT ➜ živý „flipchart“
Kompatibilní se **streamlit-webrtc ≥ 0.52** (už bez ClientSettings) a s Python 3.13
na Streamlit Community Cloud.

Lokální spuštění
--------------------------------
1. `pip install -r requirements.txt`  
2. `.streamlit/secrets.toml` → `OPENAI_API_KEY = "sk-…"`  
3. `streamlit run ai_flipchart_streamlit_whisper_api.py`

`requirements.txt`
```
streamlit
streamlit-webrtc>=0.52
openai
soundfile
numpy
```
Na Streamlit Cloud doporučuji ještě `packages.txt` s jedním řádkem `ffmpeg`.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from typing import List

import numpy as np
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ───────────────────────── CONFIG ──────────────────────────
OPENAI_API_KEY: str | None = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secretů nebo env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160            # ≈ 2,5–3 min blok pro přepis

# ───────────────────────── UI LAYOUT ───────────────────────
st.set_page_config(page_title="AI Flipchart", layout="wide")
st.title("📋 AI Flipchart – FBW Summit 2025")

placeholder = st.empty()
if "flip_points" not in st.session_state:
    st.session_state.flip_points: List[str] = []
if "transcript_buffer" not in st.session_state:
    st.session_state.transcript_buffer = ""
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer: list[bytes] = []

# ───────────────────────── AUDIO CAPTURE ───────────────────
webrtc_ctx = webrtc_streamer(
    key="workshop-audio",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
    media_stream_constraints={"audio": True, "video": False},
)

# ───────────────────────── HELPERS ─────────────────────────

def pcm_frames_to_wav(frames: list[bytes], sample_rate: int = 48000) -> bytes:
    """Raw PCM int16 ➜ WAV container (in-memory)."""
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


def summarise_new_points(transcript: str, existing: list[str]) -> list[str]:
    """Vrátí *nové* odrážky (JSON pole) max 12 slov každá."""
    sys = (
        "Jsi moderátor českého workshopu. Z textu vyber NOVÉ klíčové myšlenky, "
        "každou max 12 slov, vrať JSON pole. Body, které už na flipchartu jsou, ignoruj."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": json.dumps(existing, ensure_ascii=False)},
    ]
    raw = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", temperature=0.2, messages=messages
    ).choices[0].message.content
    try:
        pts = json.loads(raw)
        if not isinstance(pts, list):
            raise ValueError
        return [p.strip() for p in pts if p.strip()]
    except Exception:
        return [ln.lstrip("-• ").strip() for ln in raw.splitlines() if ln.strip()]


async def pipeline_runner():
    """Loop: audio ➜ Whisper ➜ ChatGPT ➜ UI update."""
    SAMPLE_RATE = 48000
    bytes_per_sec = SAMPLE_RATE * 2  # int16 mono
    target_bytes = AUDIO_BATCH_SECONDS * bytes_per_sec

    while True:
        if not webrtc_ctx.audio_receiver:
            await asyncio.sleep(0.1)
            continue

        frames = await webrtc_ctx.audio_receiver.get_frames(timeout=1)
        st.session_state.audio_buffer.extend(fr.to_ndarray().tobytes() for fr in frames)

        if sum(len(b) for b in st.session_state.audio_buffer) < target_bytes:
            await asyncio.sleep(0.05)
            continue

        wav_bytes = pcm_frames_to_wav(st.session_state.audio_buffer)
        st.session_state.audio_buffer.clear()

        trans = client.audio.transcriptions.create(
            model="whisper-1",
            file=io.BytesIO(wav_bytes),
            response_format="text",
            language="cs",
        ).text.strip()

        if trans:
            st.session_state.transcript_buffer += " " + trans

        if len(st.session_state.transcript_buffer.split()) >= 325:  # ≈ 2,5 min řeči
            new_pts = summarise_new_points(
                st.session_state.transcript_buffer, st.session_state.flip_points
            )
            st.session_state.flip_points.extend(
                [p for p in new_pts if p not in st.session_state.flip_points]
            )
            st.session_state.transcript_buffer = ""

            with placeholder.container():
                st.subheader("Aktuální flipchart 📝")
                for p in st.session_state.flip_points:
                    st.markdown(f"- {p}")

        await asyncio.sleep(0.05)


# ───────────────────── spuštění background event-loopu ───────────────────────
if "runner_created" not in st.session_state:
    def _start_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(pipeline_runner())

    threading.Thread(target=_start_loop, daemon=True).start()
    st.session_state.runner_created = True

# ───────────────────── postranní panel / debug ───────────────────────
st.sidebar.header("ℹ️ Stav aplikace")
st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
st.sidebar.caption(
    "Aplikace běží, dokud je karta otevřená. Přepis + shrnutí přibude každých ≈ 10 s po batchi.")
