# ai_flipchart_streamlit_whisper_api.py
from __future__ import annotations
import asyncio, contextlib, io, logging, re, threading, wave
import numpy as np, requests, streamlit as st
from typing import List
from openai import OpenAI, OpenAIError
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ───────── CONFIG ───────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY"); st.stop()

client                = OpenAI(api_key=OPENAI_API_KEY)
AUDIO_BATCH_SECONDS    = 20       # ⬅️ menší blok do Whisperu
MAKE_WEBHOOK_URL       = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN      = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

logging.basicConfig(level=logging.INFO)

# ───────── SESSION STATE ────────────────────────────────────────────────
def _init_state():
    s=st.session_state
    s.setdefault("flip_points", []); s.setdefault("transcript_buffer", "")
    s.setdefault("audio_buffer", []); s.setdefault("status","🟡 Čekám na mikrofon")
    s.setdefault("upload_processed", False)
    s.setdefault("audio_stop_event", None); s.setdefault("runner_thread", None)
_init_state()

def set_status(t:str): st.session_state.status=t
@contextlib.contextmanager
def status_ctx(run:str, done:str|None=None):
    p=st.session_state.status; set_status(run)
    try: yield
    finally: set_status(done or p)

# ───────── HELPERS ──────────────────────────────────────────────────────
def whisper_safe(buf, lbl:str)->str|None:
    try: return client.audio.transcriptions.create(model="whisper-1",file=buf,language="cs").text
    except OpenAIError as e:
        logging.exception("Whisper %s",lbl); set_status(f"❌ Whisper ({lbl})"); st.error(e); return None

def call_make(txt:str, exist:list[str])->list[str]:
    try:
        r=requests.post(MAKE_WEBHOOK_URL,json={"token":WEBHOOK_OUT_TOKEN,"transcript":txt,"existing":exist},timeout=90)
        r.raise_for_status(); data=r.json()
        return [str(p).strip() for p in data if str(p).strip()] if isinstance(data,list) else []
    except Exception as e:
        logging.exception("Make"); set_status("⚠️ Make"); st.error(e); return []

# ───────── RENDER FLIPCHART ─────────────────────────────────────────────
STYLES="""<style>
ul.flipchart{list-style:none;padding-left:0;}
ul.flipchart>li{opacity:0;transform:translateY(8px);animation:fadeIn .45s forwards;margin-bottom:1.2rem;}
ul.flipchart strong{display:block;font-weight:700;margin-bottom:.4rem;}
ul.flipchart ul{margin:0 0 0 1.2rem;padding-left:0;}
ul.flipchart ul li{list-style:disc;margin-left:1rem;margin-bottom:.2rem;}
@keyframes fadeIn{to{opacity:1;transform:translateY(0);}}
.fullscreen header,.fullscreen #MainMenu,.fullscreen footer{visibility:hidden;}
.fullscreen .block-container{padding-top:.5rem;}
</style>"""
DASH_SPLIT=re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(raw:str)->str:
    raw=raw.strip()
    lines=[ln.strip() for ln in raw.splitlines() if ln.strip()] if "\n" in raw else \
          [p if i==0 else f"- {p}" for i,p in enumerate(DASH_SPLIT.split(raw))]
    if not lines: return ""
    head,*det=lines; head=f"<strong>{head.upper()}</strong>"
    return head if not det else head+"<ul>"+"".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)+"</ul>"
def render(): 
    st.markdown(STYLES,unsafe_allow_html=True)
    pts=st.session_state.flip_points
    if not pts: st.info("Čekám na shrnutí…"); return
    st.markdown("<ul class='flipchart'>"+ "".join(f"<li style='animation-delay:{i*.1}s'>{fmt(p)}</li>" for i,p in enumerate(pts))+"</ul>",unsafe_allow_html=True)

def pcm_to_wav(frames,sr=48000)->bytes:
    pcm=np.frombuffer(b"".join(frames),dtype=np.int16)
    with io.BytesIO() as b: 
        with wave.open(b,"wb") as wf: wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr);wf.writeframes(pcm.tobytes())
        b.seek(0); return b.read()

# ───────── UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator",layout="wide")
tab_ctrl,tab_flip=st.tabs(["🛠 Ovládání","📝 Flipchart"])

with tab_ctrl:
    st.header("Nahrávání / Upload")
    up=st.file_uploader("▶️ WAV/MP3/M4A",type=["wav","mp3","m4a"])
    if up and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            txt=whisper_safe(up,"upload")
        if txt:
            with status_ctx("📤 Make…","🟢 Čekám Make"):
                new=call_make(txt,st.session_state.flip_points)
            st.session_state.flip_points.extend([p for p in new if p not in st.session_state.flip_points])
            st.session_state.upload_processed=True

    st.subheader("🎤 Mikrofon (SENDONLY)")
    webrtc_ctx=webrtc_streamer(
        key="workshop-audio",mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,                         # ⬅︎ větší fronta
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio":True,"video":False},
    )

    if (old:=st.session_state.runner_thread) and old.is_alive():
        st.session_state.audio_stop_event.set(); old.join(timeout=2)
    stop_evt=threading.Event(); st.session_state.audio_stop_event=stop_evt
    async def pipeline(ctx,ev):
        SR=48000; tgt=AUDIO_BATCH_SECONDS*SR*2
        while not ev.is_set():
            if not ctx.audio_receiver:
                set_status("🟡 Čekám mikrofon"); await asyncio.sleep(.1); continue
            frames=await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(f.to_ndarray().tobytes() for f in frames)
            if sum(map(len,st.session_state.audio_buffer))<tgt:
                await asyncio.sleep(.05); continue
            wav=pcm_to_wav(st.session_state.audio_buffer); st.session_state.audio_buffer.clear()
            with status_ctx("🟣 Whisper (mic)…"):
                tr=whisper_safe(io.BytesIO(wav),"mic")
            if tr: 
                st.session_state.transcript_buffer+=" "+tr
                if len(st.session_state.transcript_buffer.split())>=325:
                    with status_ctx("📤 Make…","🟢 Čekám Make"):
                        new=call_make(st.session_state.transcript_buffer,st.session_state.flip_points)
                    st.session_state.flip_points.extend([p for p in new if p not in st.session_state.flip_points])
                    st.session_state.transcript_buffer=""
        set_status("⏹️ Mikrofon stop")
    thr=threading.Thread(target=lambda c=webrtc_ctx,e=stop_evt: asyncio.run(pipeline(c,e)),daemon=True)
    add_script_run_ctx(thr); thr.start(); st.session_state.runner_thread=thr

    st.sidebar.header("ℹ️ Stav")
    st.sidebar.write("Body:",len(st.session_state.flip_points))
    st.sidebar.write("Buffer slov:",len(st.session_state.transcript_buffer.split()))
    st.sidebar.write("Status:",st.session_state.status)
    st.sidebar.write("Audio thread:",thr.is_alive())

with tab_flip:
    st.components.v1.html("<script>document.body.classList.add('fullscreen');</script>",height=0)
    render()
