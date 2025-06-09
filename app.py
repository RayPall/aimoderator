# audio_upload_whisper_segmenter_live.py
"""
Streamlit aplikace: 
1) **Upload režim** – uživatel nahraje audio soubor → segmentace FFmpeg → Whisper → Make → bullet‑points.
2) **Live režim (WebRTC)** – aplikace zachytává mikrofon / virtuální loop‑back v prohlížeči (streamlit‑webrtc), 
   každých 60 s odešle buffer do Whisper → Make → bullet‑points.

Vyžaduje:
    pip install streamlit streamlit-webrtc av openai ffmpeg-python requests

Pro systémový zvuk použijte virtuální zařízení (VB‑Audio, BlackHole, Loopback…)
a nastavte je jako vstupní mikrofon v prohlížeči.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import List

import av  # streamlit-webrtc závislost
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, webrtc_streamer

import openai

# ─────────── Nastavení API klíčů ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAKE_URL  = st.secrets.get("make_url", "")
MAKE_TOKEN = st.secrets.get("make_token", "")

openai.api_key = OPENAI_API_KEY

# ─────────── Helper: PCM → WAV (bytes) ─────────────────────────────────────
def pcm_to_wav(pcm_bytes: bytes, sr: int = 16_000) -> bytes:
    """Zabalí raw 16‑bit mono PCM do WAV kontejneru (in‑memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16‑bit
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ─────────── Whisper transkripce ───────────────────────────────────────────
def whisper_transcribe(audio_bytes: bytes, fname: str = "audio.wav") -> str:
    """Pošle audio bytes do Whisper‑1 (OpenAI) a vrátí text."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = fname
        resp = openai.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
        return resp  # type: ignore – OpenAI vrací str
    except Exception as e:
        st.error(f"❌ Whisper chyba: {e}")
        return ""


# ─────────── Make webhook → bullet‑points ──────────────────────────────────
def post_to_make(text: str) -> List[str]:
    """Odešle transkript do Make webhooku a očekává list bullet‑pointů."""
    if not text.strip():
        return []
    try:
        r = requests.post(
            MAKE_URL,
            json={"token": MAKE_TOKEN, "transcript": text, "existing": []},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"❌ Make webhook chyba: {e}")
        return []


# ─────────── Formátování bullet‑pointů ─────────────────────────────────────
DASH = re.compile(r"\s+-\s+")

def fmt_bullet(raw: str) -> str:
    """Nadpis (první řádek) tučně (uppercase), pod‑body jako <li>…</li>."""
    if "\n" in raw:
        parts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        items = DASH.split(raw.strip())
        parts = [items[0]] + [f"- {p}" for p in items[1:]]
    if not parts:
        return ""
    head, *det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det:
        return head_html
    det_html = "".join(f"<li>{d}</li>" for d in det)
    return f"{head_html}<ul>{det_html}</ul>"


# ─────────── Streamlit UI -----------------------------------------------------------------
st.set_page_config(page_title="AI moderátor – live bullet‑points", page_icon="🎙️")

st.title("🎙️ AI moderátor – přepis a bullet‑pointy v reálném čase")

MODE = st.sidebar.radio(
    "Režim",
    ("Live (WebRTC)", "Soubor upload"),
    index=0,
    help="Live režim zachytává mikrofon/systémový zvuk přes WebRTC. "
    "Upload režim umožní nahrát existující audio soubor.",
)

# Container pro dynamické bullet‑pointy
bullet_placeholder = st.empty()
if "bullets" not in st.session_state:
    st.session_state["bullets"] = []

def render_bullets():
    bullets = st.session_state.get("bullets", [])
    if bullets:
        bullet_placeholder.markdown(
            "<hr><h3>📌 Bullet‑pointy</h3><ul>"
            + "".join(f"<li>{fmt_bullet(b)}</li>" for b in bullets)
            + "</ul>",
            unsafe_allow_html=True,
        )

render_bullets()

# ─────────── Live režim (WebRTC) ───────────────────────────────────────────
if MODE == "Live (WebRTC)":
    st.info(
        "🔴 **Live**: Aplikace nahrává zvuk z vašeho mikrofonu "
        "(nebo virtuálního loop‑backu) a každou minutu ho odesílá do Whisperu."
    )

    SEGMENT_SEC = st.sidebar.slider("Interval odesílání (s)", 30, 120, 60, 5)
    SAMPLE_RATE = 16_000

    class LiveAudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.buffer = bytearray()
            self.last_sent = time.time()
            self.lock = threading.Lock()
            self.counter = 0

        def recv_audio(self, frame):
            pcm = frame.to_ndarray()  # int16 numpy array
            with self.lock:
                self.buffer.extend(pcm.tobytes())
                elapsed = time.time() - self.last_sent
                if elapsed >= SEGMENT_SEC:
                    pcm_bytes = bytes(self.buffer)
                    # Vyprázdnit buffer a posunout timestamp
                    self.buffer.clear()
                    self.last_sent = time.time()
                    # Asynchronně zpracovat segment
                    threading.Thread(
                        target=self._process_segment,
                        args=(pcm_bytes, self.counter),
                        daemon=True,
                    ).start()
                    self.counter += 1
            return frame  # passthrough

        def _process_segment(self, pcm_bytes: bytes, idx: int):
            wav_bytes = pcm_to_wav(pcm_bytes, sr=SAMPLE_RATE)
            transcript = whisper_transcribe(wav_bytes, fname=f"live_{idx:03d}.wav")
            bullets = post_to_make(transcript)
            if bullets:
                st.session_state.setdefault("bullets", []).extend(bullets)
                # UI refresh
                render_bullets()
                st.toast(f"📝 Přidáno {len(bullets)} bullet‑pointů", icon="✅")

    webrtc_streamer(
        key="live_audio",
        mode="SENDRECV",
        audio_processor_factory=LiveAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

# ─────────── Soubor upload režim (původní workflow) ────────────────────────
else:
    uploaded = st.file_uploader(
        "Nahrajte audio soubor (MP3/WAV/M4A)…",
        type=["mp3", "wav", "m4a"],
        accept_multiple_files=False,
    )

    if uploaded:
        import ffmpeg  # dynamický import – jen když je potřeba

        # 1) Uložit upload do temp
        raw_bytes = uploaded.read()
        size_mb = len(raw_bytes) / (1024 * 1024)
        st.write(f"Velikost nahrávky: **{size_mb:.1f}\u00a0MB**")

        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp_input.write(raw_bytes)
        tmp_input.flush()
        tmp_input_path = tmp_input.name
        tmp_input.close()

        # 2) Dočasný adresář pro 30 s segmenty
        tmp_dir = tempfile.mkdtemp(prefix="audio_chunks_")

        # 3) Spustit FFmpeg segmentaci
        with st.spinner("🔀 Segmentuji pomocí FFmpeg…"):
            (
                ffmpeg
                .input(tmp_input_path)
                .output(
                    str(Path(tmp_dir) / "chunk%03d.mp3"),
                    ar=16_000, ac=1, audio_bitrate="32k",
                    f="segment", segment_time=30, reset_timestamps=1,
                    loglevel="error"
                )
                .overwrite_output()
                .run()
            )

        # 4) Seřadit segmenty a poslat do Whisper
        chunks = sorted(Path(tmp_dir).glob("chunk*.mp3"))
        transcripts: List[str] = []
        prog = st.progress(0, "🎙️ Transkribuji…")
        for i, ch in enumerate(chunks, start=1):
            with ch.open("rb") as f:
                transcripts.append(whisper_transcribe(f.read(), ch.name))
            prog.progress(i / len(chunks))
        full_transcript = "\n".join(transcripts)

        # 5) Odeslat do Make
        st.success("✅ Transkripce hotová – odesílám do Make…")
        bullets = post_to_make(full_transcript)

        # 6) Zobrazit bullet‑points
        st.session_state.setdefault("bullets", []).extend(bullets)
        render_bullets()

        # 7) Úklid
        os.unlink(tmp_input_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
