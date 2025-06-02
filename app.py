# ai_moderator.py
"""
Minimalistická verze
====================
✓ Zachytává živý mikrofon (WebRTC SENDONLY)  
✓ Nepřetržitě přepisuje přes OpenAI Whisper (bloky 5 s)  
✓ Každých 60 s odešle sebraný přepis na Make; odpovědí jsou bullet-points  
✓ Flipchart okamžitě zobrazuje vrácené body  
✓ Lišta „Live Transcript“ ukazuje průběžný text

Požadavky (pip):
    streamlit
    streamlit-webrtc
    openai
    numpy
    requests
"""

from __future__ import annotations
import asyncio, io, queue, threading, time, re, wave
import numpy as np, streamlit as st, requests
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ────────── Nastavení ───────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL        = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_TOKEN   = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")
SAMPLE_RATE     = 48_000
RECEIVER_SIZE   = 1024          # ~20 s puffer
WHISPER_BLOCK_S = 5
MAKE_INTERVAL_S = 60

# ────────── Stav Streamlit ──────────────────────────────────────────────
s = st.session_state
for k, v in {
    "live_text":    "",          # průběžný text
    "flip_points":  [],
    "audio_buf":    [],          # PCM rámce
    "last_make":    time.time(),
}.items():
    s.setdefault(k, v)

# ────────── Pomocné funkce ──────────────────────────────────────────────
def pcm_to_wav(frames: list[bytes]) -> bytes:
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0); return buf.read()

def whisper_transcribe(wav: bytes) -> str | None:
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=io.BytesIO(wav), language="cs"
        ).text
    except OpenAIError as e:
        st.error(f"Whisper: {e}"); return None

def make_call(transcript: str) -> list[str]:
    r = requests.post(MAKE_URL, json={
        "token": WEBHOOK_TOKEN,
        "transcript": transcript,
        "existing": s.flip_points,
    }, timeout=90)
    try:
        r.raise_for_status(); data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Make error: {e}"); return []

# ────────── Flipchart formátování ───────────────────────────────────────
DASH = re.compile(r"\s+-\s+"); STRIP="-–—• "
def format_point(raw: str) -> str:
    parts = ([ln.strip() for ln in raw.splitlines() if ln.strip()]
             if "\n" in raw else
             [p if i==0 else f"- {p}" for i,p in enumerate(DASH.split(raw.strip()))])
    if not parts: return ""
    head,*det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det: return head_html
    items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{items}</ul>"

def render_flip():
    css = """<style>
        ul.fl{list-style:none;padding-left:0;}
        ul.fl>li{margin-bottom:1rem;}
        ul.fl strong{display:block;margin-bottom:.25rem;}
        </style>"""
    st.markdown(css,unsafe_allow_html=True)
    if not s.flip_points:
        st.info("Čekám na první body…"); return
    html = "<ul class='fl'>" + "".join(format_point(p) for p in s.flip_points) + "</ul>"
    st.markdown(html,unsafe_allow_html=True)

# ────────── UI rozložení ────────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator", layout="wide")
col_ctrl, col_flip = st.columns([1,2])

# === Sloupec 1: Ovládání + přepis =======================================
with col_ctrl:
    st.header("🎤 Mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="mic", mode=WebRtcMode.SENDONLY, audio_receiver_size=RECEIVER_SIZE,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio":True,"video":False},
    )
    transcript_box = st.empty()  # live přepis

# === Sloupec 2: Flipchart ==============================================
with col_flip:
    st.header("📝 Flipchart")
    flip_container = st.container()
    render_flip()

# ────────── Audio pipeline (v samostatném threadu) ──────────────────────
if "runner_thread" not in s or not s.runner_thread or not s.runner_thread.is_alive():
    stop_evt = threading.Event(); s.stop_evt = stop_evt

    async def pipeline():
        bytes_per_ms = SAMPLE_RATE * 2 // 1000
        target_bytes = WHISPER_BLOCK_S * 1000 * bytes_per_ms
        while not stop_evt.is_set():
            if not webrtc_ctx.audio_receiver:
                await asyncio.sleep(0.05); continue
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                continue
            s.audio_buf.extend(f.to_ndarray().tobytes() for f in frames)

            if len(b"".join(s.audio_buf)) >= target_bytes:
                wav = pcm_to_wav(s.audio_buf); s.audio_buf.clear()
                txt = await asyncio.to_thread(whisper_transcribe, wav)
                if txt:
                    s.live_text += " " + txt
                    transcript_box.text_area("Live transcript", s.live_text, height=200)

            # každou MINUTU odešleme na Make
            if time.time() - s.last_make >= MAKE_INTERVAL_S and s.live_text.strip():
                bullets = await asyncio.to_thread(make_call, s.live_text)
                s.flip_points.extend(p for p in bullets if p not in s.flip_points)
                flip_container.empty(); render_flip()
                s.live_text = ""; transcript_box.text_area("Live transcript", s.live_text, height=200)
                s.last_make = time.time()

    thr = threading.Thread(target=lambda: asyncio.run(pipeline()), daemon=True)
    add_script_run_ctx(thr); thr.start(); s.runner_thread = thr
