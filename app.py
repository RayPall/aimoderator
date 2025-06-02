# ai_moderator.py
"""
Robustní verze – funguje se starými i novými verzemi streamlit-webrtc.
Mikrofon ⭢ Whisper ⭢ Make ⭢ Flipchart
"""

import asyncio, io, queue, threading, time, wave, re
import numpy as np, requests, streamlit as st
from openai import OpenAI

# ───────────── config ───────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL  = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
TOKEN     = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

SR, BLOCK_S, PERIOD_S = 48_000, 5, 60

# ───────────── load webrtc, detect variant ──────────────────────────────
try:  # nová API
    from streamlit_webrtc import webrtc_streamer, ClientSettings, WebRtcMode
    API = "new"
except ImportError:  # stará API
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    ClientSettings = None
    API = "old"

# ───────────── state ────────────────────────────────────────────────────
s = st.session_state
s.setdefault("flip", []); s.setdefault("txt", ""); s.setdefault("last", time.time())

# ───────────── helpers ─────────────────────────────────────────────────
def pcm_to_wav(pcm: bytes) -> bytes:
    with io.BytesIO() as b:
        with wave.open(b, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes(pcm)
        b.seek(0); return b.read()

def whisper(wav: bytes) -> str:
    return client.audio.transcriptions.create(model="whisper-1",
                                              file=io.BytesIO(wav),
                                              language="cs").text

def make(text: str) -> list[str]:
    r = requests.post(MAKE_URL, json={
        "token": TOKEN, "transcript": text, "existing": s.flip
    }, timeout=90)
    r.raise_for_status(); data = r.json()
    return data if isinstance(data, list) else []

fmt = lambda p: "<strong>"+p.split("-")[0].upper()+"</strong>"+(
    "" if "-" not in p else "<ul><li>"+"</li><li>".join(p.split("-")[1:])+"</li></ul>")

def show_flip(box):
    box.markdown("<ul>"+ "".join(f"<li>{fmt(pt)}</li>" for pt in s.flip)+"</ul>",
                 unsafe_allow_html=True)

# ───────────── UI ───────────────────────────────────────────────────────
st.set_page_config("AI Moderator", layout="wide")
left, right = st.columns([1,2])
with right:
    st.header("📝 Flipchart")
    flip_box = st.empty(); show_flip(flip_box)

with left:
    st.header("🎤 Mikrofon")
    live_box = st.empty()

    frame_q: "queue.Queue[bytes]" = queue.Queue(maxsize=768)

    def audio_cb(fr):
        try: frame_q.put_nowait(fr.to_ndarray().tobytes())
        except queue.Full: pass
        return fr

    try:
        if API == "new":
            settings = ClientSettings(
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
                audio_receiver_size=1024,
            )
            webrtc_ctx = webrtc_streamer(key="mic", client_settings=settings,
                                         in_audio_frame_callback=audio_cb)
        else:
            webrtc_ctx = webrtc_streamer(
                key="mic", mode=WebRtcMode.SENDONLY,
                rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
                in_audio_frame_callback=audio_cb,    # u velmi staré verze tento parametr chybí
            )
    except TypeError:
        # fallback na nejstarší API – úplně bez callbacku
        webrtc_ctx = webrtc_streamer(
            key="mic", mode=WebRtcMode.SENDONLY,
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        )
        API = "very_old"

# ───────────── backend thread ───────────────────────────────────────────
def backend():
    buf = bytearray(); target = SR*2*BLOCK_S

    while True:
        if API == "very_old":            # musíme tahat rovnou z ctx
            if webrtc_ctx.audio_receiver:
                try:
                    frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    buf.extend(f.to_ndarray().tobytes() for f in frames)
                except queue.Empty:
                    pass
        else:                            # přijímáme přes frontu
            try: buf.extend(frame_q.get(timeout=1))
            except queue.Empty: pass

        if len(buf) >= target:
            wav, buf[:] = pcm_to_wav(buf[:target]), buf[target:]
            s.txt += " " + whisper(wav)
            live_box.text_area("Live", s.txt, height=200)

        if time.time()-s.last >= PERIOD_S and s.txt.strip():
            for p in make(s.txt):
                if p not in s.flip: s.flip.append(p)
            s.txt, s.last = "", time.time()
            live_box.text_area("Live", s.txt, height=200)
            show_flip(flip_box)

thr = threading.Thread(target=backend, daemon=True)
if "thr" not in s: s.thr = thr; thr.start()
