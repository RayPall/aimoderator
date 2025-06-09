# audio_upload_whisper_segmenter_live.py
"""
Streamlit app
1) Upload – uživatel nahraje soubor → Whisper → Make → bullet-points
2) Live  – WebRTC stream mikrofonu, každých N s pošle chunk do Whisper a bullet-points zobrazí průběžně
"""
from __future__ import annotations

import asyncio, io, time, threading, queue
from typing import List

import av
import openai
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from streamlit_extras.st_autorefresh import st_autorefresh  # pip install streamlit-extras

# ------------------------------------------------------------------------------
# CONFIG ­– vyplň v Secrets
# ------------------------------------------------------------------------------
SEGMENT_SEC       = 60
WHISPER_MODEL     = "whisper-1"
MAKE_WEBHOOK_URL  = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY", "")

# ------------------------------------------------------------------------------
# OpenAI + Make
# ------------------------------------------------------------------------------

client = openai.OpenAI(api_key=OPENAI_AP
