# ai_moderator.py
"""
Mikrofon → Whisper → Make → Flipchart  (bez přetékání fronty)
------------------------------------------------------------
pip install streamlit streamlit-webrtc openai numpy requests
"""

import asyncio, io, queue, threading, time, wave, re
from typing import Deque
from collections import deque

import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ───── Nastavení ────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL      = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN    = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

SR            = 48_000          # Hz
BLOCK_SEC     = 5               # velikost WAV bloku pro Whisper
MAKE_PERIOD   = 60              # s
CAPACITY_PCM  = SR * 2 * BLOCK_SEC * 12   # ~1 min do dequ-pufru

# ───── Stav Streamlitu (minimal) ────────────────────────────────────────
s = st.session_state
s.setdefault("flip",        [])
s.setdefault("live_text",   "")
s.setdefault("last_send",   time.time())

# ───── Pomocné funkce ───────────────────────────────────────────────────
def pcm_to_wav(pcm_bytes: bytes, sr: int = SR) -> bytes:
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm_bytes)
        buf.seek(0); return buf.read()

def whisper(wav: bytes) -> str | None:
    try:
        return client.audio.transcriptions.create(model="whisper-1",
                                                  file=io.BytesIO(wav),
                                                  language="cs").text
    except OpenAIError as e:
        st.error(f"Whisper: {e}"); return None

def make_call(text: str, existing: list[str]) -> list[str]:
    try:
        r = requests.post(MAKE_URL, json={
            "token": MAKE_TOKEN,
            "transcript": text,
            "existing": existing,
        }, timeout=90)
        r.raise_for_status(); data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Make: {e}"); return []

# ───── Flipchart render ─────────────────────────────────────────────────
DASH = re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(p: str) -> str:
    parts = [ln.strip() for ln in p.splitlines() if ln.strip()] if "\n" in p \
            else [x if i==0 else f"- {x}" for i,x in enumerate(DASH.split(p.strip()))]
    head,*det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det: return head_html
    items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{items}</ul>"

def render_flip(points):
    css="<style>ul.f{list-style:none;padding-left:0;}ul.f>li{margin-bottom:1rem;}ul.f strong{display:block;margin-bottom:.3rem;}</style>"
    st.markdown(css,unsafe_allow_html=True)
    st.markdown("<ul class='f'>"+ "".join(fmt(p) for p in points)+"</ul>",unsafe_allow_html=True)

# ───── UI ­layout ───────────────────────────────────────────────────────
st.set_page_config("AI Moderator", layout="wide")
col_left, col_right = st.columns([1,2])

with col_left:
    st.header("🎤 Mikrofon")
    live_box = st.empty()
    # fronta pro příjem z callbacku
    frame_q: queue.Queue[bytes] = queue.Queue(maxsize=500)

    def audio_cb(f):
        try:
            frame_q.put_nowait(f.to_ndarray().tobytes())
        except queue.Full:
            pass                      # pokud nestíháme, starý dropneme
        return f

    webrtc_streamer( key="mic",
        mode=WebRtcMode.SENDONLY,
        in_audio_callback=audio_cb,          # ✓ okamžitě vybírá frame
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio":True,"video":False},
    )

with col_right:
    st.header("📝 Flipchart")
    flip_box = st.empty()
    render_flip(s.flip)

# ───── Background pipeline (vybírá frontu, volá Whisper & Make) ─────────
def backend():
    pcm_buf: Deque[bytes] = deque(maxlen=CAPACITY_PCM//2)  # 16-bit = 2 B
    bytes_per_block = BLOCK_SEC * SR * 2

    while True:
        try:
            pcm_buf.append(frame_q.get(timeout=1))
        except queue.Empty:
            pass

        if sum(map(len, pcm_buf)) >= bytes_per_block:
            chunk = b"".join(pcm_buf); pcm_buf.clear()
            wav = pcm_to_wav(chunk)
            text = whisper(wav)
            if text:
                s.live_text += " " + text
                live_box.text_area("Live transcript", s.live_text, height=220)

        if time.time() - s.last_send >= MAKE_PERIOD and s.live_text.strip():
            bullets = make_call(s.live_text, s.flip)
            s.flip.extend(b for b in bullets if b not in s.flip)
            flip_box.empty(); render_flip(s.flip)
            s.live_text, s.last_send = "", time.time()
            live_box.text_area("Live transcript", s.live_text, height=220)

thr = threading.Thread(target=backend, daemon=True, name="backend")
add_script_run_ctx(thr); thr.start()
