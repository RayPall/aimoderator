# ai_moderator.py
"""
Minimal — mikrofon → Whisper → Make → Flipchart
==============================================
• streamlit-webrtc (SENDONLY) přijímá audio
• každých 5 s blok → OpenAI Whisper
• průběžný transcript je v sidebaru („Live“)
• každou 1 min se buffer pošle do Make, vrácené body se zobrazí
"""

from __future__ import annotations
import asyncio, io, queue, threading, time, wave, logging, re
import numpy as np, requests, streamlit as st
from openai import OpenAI
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ═════ CONFIG ═══════════════════════════════════════════════════════════
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL        = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_TOKEN   = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

SAMPLE_RATE     = 48_000          # Hz
RECV_SIZE       = 1024            # fronta streamlit-webrtc
WAV_BLOCK_SEC   = 5               # posíláme po 5 s
MAKE_PERIOD     = 60              # každou minutu

logging.getLogger("streamlit.runtime.thread_util").setLevel(logging.ERROR)

# ═════ STATE ════════════════════════════════════════════════════════════
s = st.session_state
for k, v in {
    "flip":               [],
    "live_text":          "",
    "buf_pcm":            [],
    "last_make_sent":     time.time(),
    "runner_thread":      None,
    "stop_evt":           None,
}.items():
    s.setdefault(k, v)

# ═════ HELPERS ══════════════════════════════════════════════════════════
def pcm_to_wav(frames: list[bytes]) -> bytes:
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as b:
        with wave.open(b, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        b.seek(0); return b.read()

def whisper(wav: bytes) -> str:
    return client.audio.transcriptions.create(
        model="whisper-1", file=io.BytesIO(wav), language="cs"
    ).text

def call_make(text: str, existing: list[str]) -> list[str]:
    r = requests.post(MAKE_URL, json={
        "token": WEBHOOK_TOKEN,
        "transcript": text,
        "existing": existing,
    }, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

# ═════ FLIPCHART RENDER ═════════════════════════════════════════════════
DASH = re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(raw: str) -> str:
    if "\n" in raw:
        parts = [p.strip() for p in raw.splitlines() if p.strip()]
    else:
        items = DASH.split(raw.strip())
        parts = [items[0]] + [f"- {p}" for p in items[1:]]
    head,*det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det: return head_html
    lis = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{lis}</ul>"

def render_flip():
    css = "<style>ul.f{list-style:none;padding-left:0;}ul.f>li{margin-bottom:1rem;}ul.f strong{display:block;margin-bottom:.3rem;}</style>"
    st.markdown(css, unsafe_allow_html=True)
    if not s.flip:
        st.info("Čekám na bullet-pointy…"); return
    html = "<ul class='f'>" + "".join(fmt(p) for p in s.flip) + "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ═════ UI LAYOUT ════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Moderator", layout="wide")
col_left, col_right = st.columns([1,2])

with col_right:
    st.header("📝 Flipchart")
    render_flip()

with col_left:
    st.header("🎤 Mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="mic", mode=WebRtcMode.SENDONLY,
        audio_receiver_size=RECV_SIZE,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )
    st.subheader("Live transcript")
    st.text_area(" ", s.live_text, height=220, key="live_display")

# ═════ AUDIO PIPELINE (background thread) ═══════════════════════════════
if not s.runner_thread or not s.runner_thread.is_alive():
    stop_evt = threading.Event(); s.stop_evt = stop_evt
    async def pipeline():
        bytes_target = WAV_BLOCK_SEC * SAMPLE_RATE * 2
        while not stop_evt.is_set():
            if not webrtc_ctx.audio_receiver:
                await asyncio.sleep(0.05); continue
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                continue
            s.buf_pcm.extend(f.to_ndarray().tobytes() for f in frames)

            if len(s.buf_pcm) >= bytes_target:
                wav = pcm_to_wav(s.buf_pcm); s.buf_pcm.clear()
                text = await asyncio.to_thread(whisper, wav)
                s.live_text += " " + text
                st.session_state["live_display"] = s.live_text  # rerender widget

            if time.time() - s.last_make_sent >= MAKE_PERIOD and s.live_text.strip():
                bullets = await asyncio.to_thread(call_make, s.live_text, s.flip)
                s.flip.extend(p for p in bullets if p not in s.flip)
                s.live_text = ""; st.session_state["live_display"] = ""
                s.last_make_sent = time.time()
                st.experimental_rerun()   # refresh Flipchart

    thr = threading.Thread(target=lambda: asyncio.run(pipeline()), daemon=True)
    add_script_run_ctx(thr); thr.start(); s.runner_thread = thr
