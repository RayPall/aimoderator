# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app, která každé ≈ 2,5 min vytáhne blok zvuku z mikrofonu,
pošle jej do OpenAI Whisper API k přepisu, nechá ChatGPT vyrobit
nové odrážky a ty v reálném čase zobrazí jako „flipchart“.

Lokální spuštění
----------------
1. pip install streamlit streamlit-webrtc openai soundfile numpy
2. Do .streamlit/secrets.toml (nebo proměnné prostředí) vložte:
   OPENAI_API_KEY = "sk-…"
3. streamlit run ai_flipchart_streamlit_whisper_api.py

Pro Streamlit Community Cloud: commitněte tento soubor + requirements.txt
a API klíč přidejte v sekci *Secrets*.
"""
from __future__ import annotations

import asyncio
import io
import json
import wave
from typing import List

import numpy as np
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import WebRtcMode, webrtc_streamer, ClientSettings

# ───────────────────────── CONFIG ──────────────────────────
OPENAI_API_KEY: str | None = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secretů nebo env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160            # ≈ 2,5–3 min @ 48 kHz

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
    client_settings=ClientSettings(media_stream_constraints={"audio": True,
                                                             "video": False}),
)

# ───────────────────────── HELPERS ─────────────────────────
def pcm_frames_to_wav(frames: list[bytes], sample_rate: int = 48000) -> bytes:
    """Raw PCM int16 → jednoduchý WAV v RAM."""
    if not frames:
        return b""
    audio_bytes = b"".join(frames)
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)           # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wav_io.seek(0)
        return wav_io.read()


def summarise_new_points(transcript: str,
                         existing: list[str]) -> list[str]:
    """ChatGPT vytáhne *nové* body, vrátí JSON pole."""
    sys = ("Jsi moderátor českého workshopu. Z následujícího textu vyber NOVÉ "
           "hlavní myšlenky, každou max 12 slov, vrať JSON pole řetězců. "
           "Body, které už na flipchartu jsou, ignoruj.")
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
        # fallback: každou řádku jako bod
        return [ln.lstrip("-• ").strip() for ln in raw.splitlines() if ln.strip()]


async def pipeline_runner():
    """Smyčka: audio ⇒ Whisper ⇒ ChatGPT ⇒ UI."""
    SAMPLE_RATE = 48000
    bytes_per_sec = SAMPLE_RATE * 2          # int16 mono
    target_bytes = AUDIO_BATCH_SECONDS * bytes_per_sec

    while True:
        if not webrtc_ctx.audio_receiver:
            await asyncio.sleep(0.1)
            continue

        frames = await webrtc_ctx.audio_receiver.get_frames(timeout=1)
        for fr in frames:
            st.session_state.audio_buffer.append(fr.to_ndarray().tobytes())

        if sum(len(b) for b in st.session_state.audio_buffer) < target_bytes:
            await asyncio.sleep(0.05)
            continue

        # ── přepis přes Whisper API ─────────────────────────────────────────
        wav_bytes = pcm_frames_to_wav(st.session_state.audio_buffer,
                                      sample_rate=SAMPLE_RATE)
        st.session_state.audio_buffer.clear()

        trans = client.audio.transcriptions.create(
            model="whisper-1",
            file=io.BytesIO(wav_bytes),
            response_format="text",
            language="cs",
        ).text.strip()

        if trans:
            st.session_state.transcript_buffer += " " + trans

        # ── shrnutí každých ≈ 2,5 min ───────────────────────────────────────
        if len(st.session_state.transcript_buffer.split()) >= 325:   # ~130 wpm
            new_pts = summarise_new_points(st.session_state.transcript_buffer,
                                           st.session_state.flip_points)
            st.session_state.flip_points.extend(
                [p for p in new_pts if p not in st.session_state.flip_points]
            )
            st.session_state.transcript_buffer = ""

            # aktualizace flipchartu
            with placeholder.container():
                st.subheader("Aktuální flipchart 📝")
                for p in st.session_state.flip_points:
                    st.markdown(f"- {p}")

        await asyncio.sleep(0.05)

# ─────────────────────── rozjezd background tasku ───────────────────────
if "runner_created" not in st.session_state:
    asyncio.create_task(pipeline_runner())
    st.session_state.runner_created = True

# ─────────────────────── postranní panel / debug ───────────────────────
st.sidebar.header("ℹ️ Stav aplikace")
st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
st.sidebar.caption("App běží, dokud necháte kartu otevřenou. "
                   "Latence ≈ 10 s po každém bloku + API.")
