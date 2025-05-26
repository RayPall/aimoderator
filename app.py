# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit web-app: mikrofon → OpenAI Whisper → ChatGPT → živý **Flipchart**
---------------------------------------------------------------------------
Novinky
=======
1. **Animované vplynutí (fade-in) bodů** – každá nová odrážka se objeví s jemným
   průletem/iz-fade-in.
2. **Záložky (tabs)** – nahoře jsou dvě karty:
      * **Ovládání** – zachytává audio / upload souboru.
      * **Flipchart** – fullscreen náhled (skryje záhlaví, menu i padding).

Lokální spuštění
----------------
```bash
pip install -r requirements.txt
streamlit run ai_flipchart_streamlit_whisper_api.py
```
`requirements.txt`
```
streamlit
streamlit-webrtc>=0.52
openai
soundfile
numpy
```
Na Streamlit Cloud navíc `packages.txt` s řádkem `ffmpeg`.
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

# ─────────── CSS (fade-in + fullscreen) ───────────
STYLES = """
<style>
ul.flipchart {list-style-type:none; padding-left:0;}
ul.flipchart li {opacity:0; transform:translateY(8px); animation:fadeIn 0.45s forwards;}
@keyframes fadeIn {to {opacity:1; transform:translateY(0);}}
/* Fullscreen */
.fullscreen header, .fullscreen #MainMenu, .fullscreen footer {visibility:hidden;}
.fullscreen .block-container {padding-top:0.5rem;}
</style>
"""

# ─────────── HELPERS ───────────

def render_flipchart() -> None:
    """Vykreslí flipchart s animací do st.container()"""
    st.markdown(STYLES, unsafe_allow_html=True)
    points = st.session_state.flip_points
    if not points:
        st.info("Čekám na první shrnutí…")
        return
    bullets = "<ul class='flipchart'>" + "".join(
        [f"<li style='animation-delay:{i*0.1}s'>{p}</li>" for i, p in enumerate(points)]
    ) + "</ul>"
    st.markdown(bullets, unsafe_allow_html=True)


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
Jsi zkušený moderátor workshopu FWB Summit 2025. Cholné setkání podnikatelských rodin, expertů, akademiků a politiků, kteří sdílí zkušenosti a formují budoucnost rodinného podnikání. Akce hostí světové i domácí osobnosti a nabízí unikátní prostor pro inspiraci, inovace a spolupráci.

Tvým úkolem je shrnovat projev do klíčových myšlenek. Myšlenky piš v následujícím formátu:
NADPIS MYŠLENKY
- detail 1
- detail 2
- detail 3
- atp.

Z textu vyber NOVÉ klíčové myšlenky. Vrať JSON pole. Body, které už jsou na flipchartu, ignoruj.
"""


def summarise_new_points(text: str, existing: list[str]) -> list[str]:
    msgs = [
        {"role": "system", "content": PROMPT},
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

# ─────────── STREAMLIT LAYOUT (tabs) ───────────
st.set_page_config(layout="wide", page_title="AI Flipchart")

tabs = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ========== Tab 1: OVLÁDÁNÍ ==========
with tabs[0]:
    st.header("Nastavení a vstup zvuku")
    uploaded = st.file_uploader("▶️ Nahrajte WAV/MP3 k otestování (max pár minut)",
