from __future__ import annotations
import asyncio, contextlib, io, logging, re, threading, time, wave
from typing import List

import numpy as np, requests, streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ─── optional auto-refresh (neshodí app, když chybí knihovna) ───────────
try:
    from streamlit_extras.app_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ModuleNotFoundError:
    _HAS_AUTOREFRESH = False
# ────────────────────────────────────────────────────────────────────────

# ─── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Chybí OPENAI_API_KEY"); st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_BATCH_SECONDS = 60
MAKE_WEBHOOK_URL  = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
WEBHOOK_OUT_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "out-token")

TURN_DOMAIN = st.secrets["METERED_TURN_DOMAIN"]     # e.g. "xyz.metered.live"
TURN_USER   = st.secrets["METERED_TURN_USER"]
TURN_PASS   = st.secrets["METERED_TURN_PASS"]

logging.basicConfig(level=logging.INFO)

# ─── SESSION STATE INIT ────────────────────────────────────────────────
def _init():
    ss = st.session_state
    ss.setdefault("flip_points", [])
    ss.setdefault("transcript_buffer", "")
    ss.setdefault("audio_buffer", [])
    ss.setdefault("status", "🟡 Čekám na mikrofon…")
    ss.setdefault("upload_processed", False)
    ss.setdefault("audio_stop_event", None)
    ss.setdefault("runner_thread", None)
    ss.setdefault("prev_bytes", 0)
    ss.setdefault("prev_ts", 0)
_init()

def set_status(txt): st.session_state.status = txt
@contextlib.contextmanager
def status_ctx(run, done=None):
    prev = st.session_state.status; set_status(run)
    try: yield
    finally: set_status(done or prev)

# ─── HELPERS: Whisper & Make ───────────────────────────────────────────
def whisper_safe(file_like, label):
    try:
        return client.audio.transcriptions.create(
            model="whisper-1", file=file_like, language="cs"
        ).text
    except OpenAIError as exc:
        logging.exception("Whisper %s", label); st.error(exc); return None

def call_make(text, existing):
    try:
        r = requests.post(MAKE_WEBHOOK_URL,
                          json={"token":WEBHOOK_OUT_TOKEN,
                                "transcript":text,"existing":existing},
                          timeout=90); r.raise_for_status()
        data = r.json(); return data if isinstance(data, list) else []
    except Exception as exc:
        logging.exception("Make"); st.error(exc); return []

# ─── Flipchart rendering ──────────────────────────────────────────────
CSS = """
<style>
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
def _fmt(raw:str):
    raw = raw.strip()
    if "\n" in raw:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    else:
        parts = DASH.split(raw)
        lines = [parts[0]] + [f"- {p}" for p in parts[1:]]
    if not lines: return ""
    head,*det = lines
    head = f"<strong>{head.upper()}</strong>"
    return head if not det else head + "<ul>" + "".join(
        f"<li>{d.lstrip(STRIP)}</li>" for d in det) + "</ul>"
def render_flipchart():
    st.markdown(CSS, unsafe_allow_html=True)
    pts = st.session_state.flip_points
    if not pts:
        st.info("Čekám na první shrnutí…"); return
    html = "<ul class='flipchart'>" + "".join(
        f"<li style='animation-delay:{i*0.1}s'>{_fmt(p)}</li>"
        for i,p in enumerate(pts)) + "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ─── Audio utils ───────────────────────────────────────────────────────
def pcm_to_wav(frames, sr=48000):
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm)
        buf.seek(0); return buf.read()

def run_async_forever(coro):
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: loop.run_until_complete(coro)
    finally:
        for t in asyncio.all_tasks(loop): t.cancel()
        loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop),
                                               return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens()); loop.close()

# ─── UI layout ─────────────────────────────────────────────────────────
st.set_page_config("AI Moderator", layout="wide")
tab1, tab2 = st.tabs(["🛠 Ovládání", "📝 Flipchart"])

# ===== TAB 1 – ovládání =================================================
with tab1:
    st.header("Nastavení a vstup zvuku")

    up = st.file_uploader("▶️ Testovací WAV/MP3/M4A", type=["wav","mp3","m4a"])
    if up and not st.session_state.upload_processed:
        with status_ctx("🟣 Whisper (upload)…"):
            txt = whisper_safe(up,"upload")
        if txt:
            with status_ctx("📤 Odesílám do Make…", "🟢 Čekám Make"):
                new_pts = call_make(txt, [])
            st.session_state.flip_points.extend(
                p for p in new_pts if p not in st.session_state.flip_points)
            st.session_state.upload_processed = True

    st.subheader("🎤 Živý mikrofon")
    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDRECV,
        sendback_audio=False,
        rtc_configuration={
            "iceTransportPolicy":"relay",
            "iceServers":[{
                "urls":[
                    f"turn:{TURN_DOMAIN}:80?transport=udp",
                    f"turn:{TURN_DOMAIN}:80?transport=tcp",
                    f"turns:{TURN_DOMAIN}:443?transport=tcp"],
                "username":TURN_USER,
                "credential":TURN_PASS}]
        },
        media_stream_constraints={"audio":True,"video":False})

    st.markdown(f"**Aktuální stav:** {st.session_state.status}")

    # ── Diagnostika (variant C) ────────────────────────────────────────
    with st.expander("📡 WebRTC diagnostika", expanded=False):
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=1000, key="webrtc_diag")
        else:
            st.caption("⚠️  streamlit-extras není v prostředí – panel se "
                       "neobnovuje automaticky.")

        st.write("**Context state:**", webrtc_ctx.state)

        pc = getattr(webrtc_ctx, "_pc", None)
        if pc:
            st.write("**ICE state:**", pc.iceConnectionState)
            st.write("**PC  state:**", pc.connectionState)

            # Rx bitrate (kbit/s) – synchronní volání getStats
            try:
                stats = asyncio.run(pc.getStats())
                inbound = [v for v in stats.values()
                           if v.type == "inbound-rtp" and v.kind == "audio"]
                if inbound:
                    br  = inbound[0].get("bytesReceived", 0)
                    ts  = inbound[0].get("timestamp",    0)
                    prev_br = st.session_state.prev_bytes
                    prev_ts = st.session_state.prev_ts
                    if ts and ts != prev_ts:
                        rate = 8*(br - prev_br)/(ts - prev_ts)   # kbit/s
                        st.write("**Rx bitrate:**", f"{rate:.1f} kbit/s")
                    st.session_state.prev_bytes, st.session_state.prev_ts = br, ts
            except Exception:
                pass
        else:
            st.info("PeerConnection zatím nevznikl – klikněte Start a "
                    "povolte mikrofon.")

        st.write("**Bytes v audio_bufferu:**",
                 sum(len(b) for b in st.session_state.audio_buffer))
    # ───────────────────────────────────────────────────────────────────

    # Restart audio pipeline při každém rerunu
    if (thr := st.session_state.runner_thread) and thr.is_alive():
        st.session_state.audio_stop_event.set(); thr.join(timeout=2)
    stop_evt = threading.Event(); st.session_state.audio_stop_event = stop_evt

    async def audio_pipeline(ctx, stop):
        SR=48000; target=AUDIO_BATCH_SECONDS*SR*2
        while not stop.is_set():
            if not ctx.audio_receiver:
                await asyncio.sleep(.1); continue
            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(f.to_ndarray().tobytes()
                                                 for f in frames)
            if sum(len(b) for b in st.session_state.audio_buffer) < target:
                await asyncio.sleep(.05); continue

            wav = pcm_to_wav(st.session_state.audio_buffer)
            st.session_state.audio_buffer.clear()
            with status_ctx("🟣 Whisper – zpracovávám…"):
                tr = whisper_safe(io.BytesIO(wav),"mic")
            if not tr: await asyncio.sleep(1); continue
            st.session_state.transcript_buffer += " " + tr

            if len(st.session_state.transcript_buffer.split()) >= 325:
                with status_ctx("📤 Odesílám do Make…","🟢 Čekám Make"):
                    pts = call_make(st.session_state.transcript_buffer,
                                    st.session_state.flip_points)
                st.session_state.flip_points.extend(
                    p for p in pts if p not in st.session_state.flip_points)
                st.session_state.transcript_buffer=""
            set_status(f"🔴 Nahrávám … (plním {AUDIO_BATCH_SECONDS}s dávku)")
            await asyncio.sleep(.05)
        set_status("⏹️ Audio pipeline ukončena")

    t = threading.Thread(
        target=lambda c=webrtc_ctx,e=stop_evt: run_async_forever(audio_pipeline(c,e)),
        daemon=True)
    add_script_run_ctx(t); t.start(); st.session_state.runner_thread = t

    # Sidebar
    st.sidebar.header("ℹ️ Diagnostika")
    st.sidebar.write("Body:", len(st.session_state.flip_points))
    st.sidebar.write("Slov v bufferu:", len(st.session_state.transcript_buffer.split()))
    st.sidebar.subheader("🧭 Stav"); st.sidebar.write(st.session_state.status)
    st.sidebar.write("Audio thread:", t.is_alive())

# ===== TAB 2 – Flipchart ===============================================
with tab2:
    fs = st.checkbox("🔲 Fullscreen zobrazení", key="fs")
    cmd = "add" if fs else "remove"
    st.components.v1.html(
        f"<script>document.body.classList.{cmd}('fullscreen');</script>", height=0)
    render_flipchart()
