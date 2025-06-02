# ai_moderator.py
"""
Streamlit-webrtc ≥ 0.53  •  Mikrofon → Whisper → Make → Flipchart
"""

import asyncio, io, queue, threading, time, wave, re, logging
import numpy as np, streamlit as st, requests
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, ClientSettings

# ─── Nastavení ──────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
TOKEN      = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

SR, BLOCK_SEC, PERIOD = 48_000, 5, 60

logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

# ─── Stav ───────────────────────────────────────────────────────────────
s = st.session_state
for k, v in {"flip": [], "txt": "", "last_sent": time.time()}.items():
    s.setdefault(k, v)

# ─── Pomocné funkce ─────────────────────────────────────────────────────
pcm_to_wav = lambda pcm: (lambda b: (wave.open((io := io.BytesIO()).write(b) or io, "wb")
    .getparams() or io.seek(0) or io.read()))(
        (lambda w: w.setnchannels(1) or w.setsampwidth(2)
                  or w.setframerate(SR) or w.writeframes(pcm))(wave.open(io.BytesIO(), "wb")))  # compact hack

def whisper(wav: bytes) -> str:
    return client.audio.transcriptions.create(
        model="whisper-1", file=io.BytesIO(wav), language="cs"
    ).text

def to_make(text: str, old: list[str]) -> list[str]:
    r = requests.post(MAKE_URL, json={"token":TOKEN,"transcript":text,"existing":old}, timeout=90)
    r.raise_for_status(); d = r.json(); return d if isinstance(d, list) else []

# ─── Flipchart render ───────────────────────────────────────────────────
fmt = lambda p: "<strong>"+p.split("-")[0].upper()+"</strong>"+(
    "" if "-" not in p else "<ul><li>"+ "</li><li>".join(p.split("-")[1:]) +"</li></ul>"
)
def show_flip(): st.markdown("<ul>"+ "".join(f"<li>{fmt(p)}</li>" for p in s.flip)+"</ul>",unsafe_allow_html=True)

# ─── Layout ─────────────────────────────────────────────────────────────
st.set_page_config("AI Moderator", layout="wide")
c1, c2 = st.columns([1,2])

with c2:
    st.header("📝 Flipchart"); flip_box = st.empty(); show_flip()

with c1:
    st.header("🎤 Mikrofon")
    live_box = st.empty()

    # nový způsob – jen ClientSettings
    settings = ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        translation={"iceServers":[]},
        audio_receiver_size=1024,
    )

    frame_q: "queue.Queue[bytes]" = queue.Queue(maxsize=512)

    def audio_cb(frame):
        try: frame_q.put_nowait(frame.to_ndarray().tobytes())
        except queue.Full: pass
        return frame

    webrtc_streamer(
        key="mic",
        client_settings=settings,
        in_audio_frame_callback=audio_cb,   # nové jméno parametru
    )

# ─── Background vlákno — Whisper + Make ────────────────────────────────
def backend():
    buf = bytearray()
    bytes_block = SR*2*BLOCK_SEC
    while True:
        try: buf.extend(frame_q.get(timeout=1))
        except queue.Empty: pass
        if len(buf) >= bytes_block:
            wav, buf[:] = pcm_to_wav(buf[:bytes_block]), buf[bytes_block:]
            s.txt += " " + whisper(wav)
            live_box.text_area("Live", s.txt, height=200)

        if time.time()-s.last_sent > PERIOD and s.txt.strip():
            for p in to_make(s.txt, s.flip):
                if p not in s.flip: s.flip.append(p)
            s.txt, s.last_sent = "", time.time()
            live_box.text_area("Live", s.txt, height=200)
            flip_box.empty(); show_flip()

thr = threading.Thread(target=backend, daemon=True)
if "thr" not in s: s.thr = thr; threading.Thread(target=backend, daemon=True).start()
