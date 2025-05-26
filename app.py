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
    bull
