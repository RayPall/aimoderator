# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web‑app: mikrofon ➜ OpenAI Whisper API ➜ ChatGPT ➜ živý „flipchart“
Nově podporuje i **offline test** – stačí nahrát WAV/MP3 soubor a aplikace jej
zpracuje, aniž byste potřebovali funkční WebRTC/mikrofon.

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
Na Streamlit Cloud přidejte `packages.txt` s jediným řádkem **`ffmpeg`**.
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
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets / env vars")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 160  # 2 ½–3 min blok mikrofonu
TEST_CHUNK_SEC = 30        # při testu rozsekáme nahraný soubor na 30 s kousky

# ───────────────────────── UI LAYOUT ───────────────────────
st.set_page_config(page_title="AI Flipchart", layout="wide")
st.title("📋 AI Flipchart – FBW Summit 2025")

# ▸ Upload pro test bez mikrofonu
uploaded = st.file_uploader("▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
                            type=["wav", "mp3", "m4a"], accept_multiple_files=False)

placeholder = st.empty()
if "flip_points" not in st.session_state:
    st.session_state.flip_points: List[str] = []
if "transcript_buffer" not in st.session_state:
    st.session_state.transcript_buffer = ""
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer: list[bytes] = []

# ───────────────────────── HELPERS ─────────────────────────

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


def summarise_new_points(text: str, existing: list[str]) -> list[str]:
    sys = (
        "Jsi moderátor českého workshopu. Z textu vyber NOVÉ klíčové myšlenky, "
        "každou max 12 slov, vrať JSON pole. Body, které už jsou na flipchartu, ignoruj."
    )
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": text},
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

# ───────────────────────── TEST: FILE UPLOAD ──────────────
if uploaded is not None:
    st.info("⏳ Zpracovávám nahraný soubor…")
    # Pošleme celý soubor do Whisper API (nebo po částech)
    # Pokud soubor > 25MB, OpenAI odmítne; pro demo předpokládáme krátký.
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=uploaded,
        
    new_pts = summarise_new_points(transcription, [])
    st.session_state.flip_points = new_pts
    # Vykresli flipchart
    with placeholder.container():
        st.subheader("Výsledek testu 📝")
        for p in new_pts:
            st.markdown(f"- {p}")
    st.stop()

# ───────────────────────── LIVE MICROPHONE MODUS ──────────
# (spustí se jen pokud nebyl upload)

# 1️⃣ Capture audio přes WebRTC
webrtc_ctx = webrtc_streamer(
    key="workshop-audio",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

async def pipeline_runner():
    SAMPLE_RATE = 48000
    bytes_per_sec = SAMPLE_RATE * 2
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

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=io.BytesIO(wav_bytes),
            response_format="text",
            language="cs",
        ).text.strip()

        if transcription:
            st.session_state.transcript_buffer += " " + transcription

        if len(st.session_state.transcript_buffer.split()) >= 325:
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

# ───────────────────── start background loop ──────────────
if "runner_created" not in st.session_state and uploaded is None:
    def _start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(pipeline_runner())

    threading.Thread(target=_start_loop, daemon=True).start()
    st.session_state.runner_created = True

# ───────────────────── sidebar debug ──────────────────────
st.sidebar.header("ℹ️ Stav aplikace")
st.sidebar.write("Body na flipchartu:", len(st.session_state.flip_points))
st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
st.sidebar.caption("Aplikace běží, dokud karta zůstává otevřená. Pro rychlý test nahrajte audio soubor.")
