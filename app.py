# ai_flipchart_streamlit_whisper_api.py
"""
Streamlit → Whisper → Make → Flipchart
--------------------------------------
• Mikrofon / upload ⭢ Whisper  • POST na Make  • Flipchart
• WebRTC SENDONLY, audio_receiver_size = 1024
"""

from __future__ import annotations
import asyncio, contextlib, io, logging, queue as q, re, threading, wave
from typing import List
import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ───── CONFIG ───────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: st.error("Chybí OPENAI_API_KEY"); st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_BLOCK_SEC, RECEIVER_SIZE = 5, 1024
MAKE_WEBHOOK_URL = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")
logging.basicConfig(level=logging.INFO)

# ───── SESSION STATE ────────────────────────────────────────────────────
def _init(): s=st.session_state; s.setdefault("flip_points",[]); s.setdefault("transcript_buffer","")
_init()
def set_status(t:str): st.session_state.status=t

@contextlib.contextmanager
def status_ctx(run:str,done:str|None=None):
    prev=st.session_state.get("status","")
    set_status(run);  yield;  set_status(done or prev)

# ───── HELPERS ──────────────────────────────────────────────────────────
def whisper_safe(buf, lbl): 
    try: return client.audio.transcriptions.create(model="whisper-1",file=buf,language="cs").text
    except OpenAIError as e: logging.exception("Whisper %s",lbl); st.error(e); return None
def call_make(txt, exist):
    try:
        r=requests.post(MAKE_WEBHOOK_URL,json={"token":WEBHOOK_OUT_TOKEN,"transcript":txt,"existing":exist},timeout=90)
        r.raise_for_status(); d=r.json(); 
        return [str(p).strip() for p in d if str(p).strip()] if isinstance(d,list) else []
    except Exception as e: logging.exception("Make"); st.error(e); return []

# ───── Flipchart render (stejné) ────────────────────────────────────────
STYLES="""<style>ul.flipchart{list-style:none;padding-left:0;} ul.flipchart>li{opacity:0;transform:translateY(8px);animation:fadeIn .45s forwards;margin-bottom:1.2rem;} ul.flipchart strong{display:block;font-weight:700;margin-bottom:.4rem;} ul.flipchart ul{margin:0 0 0 1.2rem;padding-left:0;} ul.flipchart ul li{list-style:disc;margin-left:1rem;margin-bottom:.2rem;} @keyframes fadeIn{to{opacity:1;transform:translateY(0);}} .fullscreen header,.fullscreen #MainMenu,.fullscreen footer{visibility:hidden;} .fullscreen .block-container{padding-top:.5rem;}</style>"""
DASH=re.compile(r"\s+-\s+"); STRIP="-–—• "
fmt=lambda r: (lambda h,*d:(f"<strong>{h.upper()}</strong>"+("" if not d else "<ul>"+ "".join(f"<li>{x.lstrip(STRIP)}</li>" for x in d)+"</ul>")))(*(([r.strip()]+[f"- {p}" for p in DASH.split(r.strip())[1:]]) if "\n" not in r else [ln.strip() for ln in r.splitlines() if ln.strip()]))
def render():
    st.markdown(STYLES,unsafe_allow_html=True)
    pts=st.session_state.flip_points
    if not pts: st.info("Čekám na shrnutí…"); return
    st.markdown("<ul class='flipchart'>"+"".join(f"<li style='animation-delay:{i*.1}s'>{fmt(p)}</li>" for i,p in enumerate(pts))+"</ul>",unsafe_allow_html=True)
def pcm_to_wav(fr,sr=48000):
    pcm=np.frombuffer(b"".join(fr),dtype=np.int16)
    with io.BytesIO() as b: 
        with wave.open(b,"wb") as wf: wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr);wf.writeframes(pcm.tobytes())
        b.seek(0); return b.read()

# ───── UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Moderator",layout="wide")
tab_ctrl,tab_flip=st.tabs(["🛠 Ovládání","📝 Flipchart"])

with tab_ctrl:
    st.header("Nahrávání / Upload")
    up=st.file_uploader("▶️ WAV/MP3/M4A",type=["wav","mp3","m4a"])
    if up and not st.session_state.get("upload_processed"):
        with status_ctx("🟣 Whisper (upload)…"):
            t=whisper_safe(up,"upload")
        if t:
            with status_ctx("📤 Make…","🟢 Čekám Make"):
                n=call_make(t,st.session_state.flip_points)
            st.session_state.flip_points.extend(p for p in n if p not in st.session_state.flip_points)
            st.session_state.upload_processed=True

    st.subheader("🎤 Mikrofon")
    ctx=webrtc_streamer(key="mic",mode=WebRtcMode.SENDONLY,audio_receiver_size=RECEIVER_SIZE,
                        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"audio":True,"video":False})
    if (old:=st.session_state.get("runner_thread")) and old.is_alive():
        st.session_state.audio_stop_event.set(); old.join(timeout=2)

    stop_evt=threading.Event(); st.session_state.audio_stop_event=stop_evt
    queue:asyncio.Queue[bytes]=asyncio.Queue(maxsize=8)

    async def reader(ctx):
        sr=48000; tgt=AUDIO_BLOCK_SEC*sr*2; buf:list[bytes]=[]
        while not stop_evt.is_set():
            if not ctx.audio_receiver: await asyncio.sleep(.05); continue
            try: frames=ctx.audio_receiver.get_frames(timeout=1)
            except q.Empty: continue
            if frames:
                set_status("🔴 Zachytávám audio…")
            buf.extend(f.to_ndarray().tobytes() for f in frames)
            if sum(map(len,buf))>=tgt:
                wav=pcm_to_wav(buf); buf.clear()
                try: queue.put_nowait(wav)
                except asyncio.QueueFull: _=queue.get_nowait(); queue.put_nowait(wav)

    async def worker():
        while not stop_evt.is_set():
            wav=await queue.get()
            set_status("🟣 Whisper …")
            t=await asyncio.to_thread(whisper_safe,io.BytesIO(wav),"mic")
            if t:
                st.session_state.transcript_buffer+=" "+t
                if len(st.session_state.transcript_buffer.split())>=325:
                    set_status("📤 Make…")
                    new=await asyncio.to_thread(call_make,st.session_state.transcript_buffer,st.session_state.flip_points)
                    st.session_state.flip_points.extend(p for p in new if p not in st.session_state.flip_points)
                    st.session_state.transcript_buffer=""
            queue.task_done()

    async def pipeline(): await asyncio.gather(reader(ctx), worker())

    thr=threading.Thread(target=lambda: asyncio.run(pipeline()),daemon=True,name="audio-pipe")
    add_script_run_ctx(thr); thr.start(); st.session_state.runner_thread=thr

    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body:",len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:",len(st.session_state.transcript_buffer.split()))
    st.sidebar.write("Stav:",st.session_state.get("status",""))
    st.sidebar.write("Thread alive:",thr.is_alive())

with tab_flip:
    st.components.v1.html("<script>document.body.classList.add('fullscreen');</script>",height=0)
    render()
