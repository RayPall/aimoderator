# ai_moderator.py  ── funguje se starým i novým streamlit-webrtc
import asyncio, io, queue, threading, time, wave, re, logging
import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError

# --- detekce verze streamlit-webrtc ------------------------------------
try:
    from streamlit_webrtc import webrtc_streamer, ClientSettings, WebRtcMode
    NEW_API = True
except ImportError:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    ClientSettings = None
    NEW_API = False

# --- konfigurace --------------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
MAKE_URL  = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
TOKEN     = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

SR, BLOCK_SEC, PERIOD = 48_000, 5, 60
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

# --- state --------------------------------------------------------------
s = st.session_state
s.setdefault("flip", []); s.setdefault("txt", ""); s.setdefault("last", time.time())

# --- helpers ------------------------------------------------------------
pcm_to_wav = lambda b: (lambda buf: (
    wave.open(buf, "wb").setparams((1,2,SR,0,'NONE','NONE')) or
    wave.open(buf, "wb").writeframes(b) or buf.seek(0) or buf.read()
))(io.BytesIO())

def whisper(wav): return client.audio.transcriptions.create(
        model="whisper-1", file=io.BytesIO(wav), language="cs").text

def make(text, existing):
    r = requests.post(MAKE_URL, json={"token":TOKEN,"transcript":text,"existing":existing}, timeout=90)
    r.raise_for_status(); d=r.json(); return d if isinstance(d,list) else []

fmt = lambda p: "<strong>"+p.split("-")[0].upper()+"</strong>"+(
    "" if "-" not in p else "<ul><li>"+"</li><li>".join(p.split("-")[1:])+"</li></ul>")

def show_flip(): st.markdown("<ul>"+ "".join(f"<li>{fmt(p)}</li>" for p in s.flip)+"</ul>", unsafe_allow_html=True)

# --- UI -----------------------------------------------------------------
st.set_page_config("AI Moderator", layout="wide")
c1, c2 = st.columns([1,2])

with c2:
    st.header("📝 Flipchart"); flip_box = st.container(); show_flip()

with c1:
    st.header("🎤 Mikrofon")
    live_box = st.empty(); frame_q = queue.Queue(maxsize=512)

    def audio_cb(fr):                    # okamžitě vyzvedneme frame
        try: frame_q.put_nowait(fr.to_ndarray().tobytes())
        except queue.Full: pass
        return fr

    if NEW_API:
        settings = ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            audio_receiver_size=1024,
        )
        webrtc_streamer(key="mic", client_settings=settings,
                        in_audio_frame_callback=audio_cb)
    else:
        webrtc_streamer(
            key="mic", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024,
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
            in_audio_frame_callback=audio_cb,
        )

# --- background pipeline ------------------------------------------------
def backend():
    buf = bytearray(); target = SR*2*BLOCK_SEC
    while True:
        try: buf.extend(frame_q.get(timeout=1))
        except queue.Empty: pass
        if len(buf) >= target:
            wav, buf[:] = pcm_to_wav(buf[:target]), buf[target:]
            s.txt += " " + whisper(wav)
            live_box.text_area("Live", s.txt, height=200)
        if time.time()-s.last >= PERIOD and s.txt.strip():
            for p in make(s.txt, s.flip):
                if p not in s.flip: s.flip.append(p)
            s.txt, s.last = "", time.time()
            live_box.text_area("Live", s.txt, height=200)
            flip_box.empty(); show_flip()

thr = threading.Thread(target=backend, daemon=True)
if "thr" not in s: s.thr = thr; thr.start()
