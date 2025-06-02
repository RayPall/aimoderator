# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
• 60 s audio batch → Whisper → Make  
• TURN tcp/443 (openrelay.metered.ca) – forced relay  
• Viditelný stav v UI, žádná ozvěna
"""

from __future__ import annotations
import asyncio, contextlib, io, logging, re, threading, wave
import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer   # ⬅️ bez WebRtcState
# … --------------------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY – přidejte jej do Secrets"); st.stop()

client             = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS = 60
MAKE_WEBHOOK_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN  = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")
logging.basicConfig(level=logging.INFO)
# -----------------------------------------------------------------------
def _init():
    s = st.session_state
    s.setdefault("flip_points", [])
    s.setdefault("transcript_buffer", "")
    s.setdefault("audio_buffer", [])
    s.setdefault("status", "🟡 Čekám na mikrofon…")
    s.setdefault("upload_processed", False)
    s.setdefault("audio_stop_event", None)
    s.setdefault("runner_thread", None)
_init()
def set_status(t): st.session_state.status = t
@contextlib.contextmanager
def status_ctx(run, done=None):
    prev = st.session_state.status; set_status(run)
    try: yield
    finally: set_status(done or prev)
def whisper_safe(f_like, lbl):
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=f_like, language="cs").text
    except OpenAIError as exc:
        logging.exception("Whisper err"); st.error(exc); return None
def call_make(txt, exist):
    try:
        r = requests.post(MAKE_WEBHOOK_URL,
                          json={"token":WEBHOOK_OUT_TOKEN,"transcript":txt,"existing":exist},
                          timeout=90); r.raise_for_status(); d=r.json()
        return d if isinstance(d,list) else []
    except Exception as e:
        logging.exception("Make"); st.error(e); return []
# ---------- flipchart ----------
CSS="""<style>
ul.flipchart{list-style:none;padding-left:0;}
ul.flipchart>li{opacity:0;transform:translateY(8px);
 animation:fadeIn .45s forwards;margin-bottom:1.2rem;}
ul.flipchart strong{display:block;font-weight:700;margin-bottom:.4rem;}
ul.flipchart ul{margin:0 0 0 1.2rem;padding-left:0;}
ul.flipchart ul li{list-style:disc;margin-left:1rem;margin-bottom:.2rem;}
@keyframes fadeIn{to{opacity:1;transform:translateY(0);} }
.fullscreen header,.fullscreen #MainMenu,.fullscreen footer{visibility:hidden;}
.fullscreen .block-container{padding-top:.5rem;}
</style>"""
DASH=re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(raw:str):
    raw=raw.strip()
    if "\n" in raw: lines=[l.strip() for l in raw.splitlines() if l.strip()]
    else: parts=DASH.split(raw); lines=[parts[0]]+[f"- {p}" for p in parts[1:]]
    if not lines: return ""
    head,*det=lines; head=f"<strong>{head.upper()}</strong>"
    return head if not det else head+"<ul>"+"".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)+"</ul>"
def render():
    st.markdown(CSS,unsafe_allow_html=True)
    pts=st.session_state.flip_points
    if not pts: st.info("Čekám na první shrnutí…"); return
    st.markdown("<ul class=flipchart>"+ "".join(
        f"<li style='animation-delay:{i*0.1}s'>{fmt(p)}</li>" for i,p in enumerate(pts))+"</ul>",
        unsafe_allow_html=True)
def pcm_to_wav(fr,sr=48000):
    pcm=np.frombuffer(b"".join(fr),dtype=np.int16)
    with io.BytesIO() as b:
        with wave.open(b,"wb") as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr);wf.writeframes(pcm)
        b.seek(0); return b.read()
def run_async_forever(c):
    loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: loop.run_until_complete(c)
    finally:
        for t in asyncio.all_tasks(loop): t.cancel()
        loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop),return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens()); loop.close()
# ---------- UI ----------
st.set_page_config("AI Moderator",layout="wide")
tab1,tab2=st.tabs(["🛠 Ovládání","📝 Flipchart"])
with tab1:
    st.header("Nastavení a vstup zvuku")
    up=st.file_uploader("▶️ Testovací WAV/MP3/M4A",type=["wav","mp3","m4a"])
    if up and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            t=whisper_safe(up,"upload")
        if t:
            with status_ctx("📤 Odesílám do Make…","🟢 Čekám Make"):
                pts=call_make(t,[])
            st.session_state.flip_points.extend(p for p in pts if p not in st.session_state.flip_points)
            st.session_state.upload_processed=True
    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx=webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        sendback_audio=False,
        rtc_configuration={
            "iceTransportPolicy":"relay",
            "iceServers":[{"urls":"turn:openrelay.metered.ca:443?transport=tcp",
                           "username":"openrelayproject","credential":"openrelayproject"}]},
        media_stream_constraints={"audio":True,"video":False})
    # Stav WebRTC
    state=str(webrtc_ctx.state)
    if state=="WebRtcState.PLAYING" or state=="PLAYING":
        set_status(f"🔴 Nahrávám … (plním {AUDIO_BATCH_SECONDS}s dávku)")
    elif state=="WebRtcState.FAILED" or state=="FAILED":
        set_sta_
