# audio_upload_whisper_segmenter_live.py
"""
Streamlit app – **revamp podle whitphx demo**
==========================================
1. **Upload** – statické nahrání → Whisper → Make → bullet‑points.
2. **Live (WebRTC)** – mic / virtuální kabel zachycen přímo v UI vláknu, _bez_ autorefreshu či rerunů:
   * `webrtc_streamer` (SENDONLY)
   * v samostatném vlákně čteme `audio_receiver.get_frames()`
   * každých `SEGMENT_SEC` => WAV → Whisper → Make → přidá se do session_state → UI se aktualizuje přes `st.session_state.changed` + jemný `st.rerun()` **jen** po příchodu nových dat → žádné blikání.
   * Realtime RMS bar.
"""
from __future__ import annotations

import asyncio, io, time, threading, queue, wave, logging
from typing import List

import av, numpy as np, streamlit as st, openai, requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ----------------------------------------------------------------------------
# CONFIG ---------------------------------------------------------------------
# ----------------------------------------------------------------------------
SEGMENT_SEC       = 60
WHISPER_MODEL     = "whisper-1"
MAKE_WEBHOOK_URL  = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY", "")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------------
# Helper functions ------------------------------------------------------------
# ----------------------------------------------------------------------------

def pcm_to_wav(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def whisper_transcribe(wav_bytes: bytes) -> str:
    r = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("live.wav", io.BytesIO(wav_bytes), "audio/wav"),  # type: ignore[arg-type]
    )
    return r.text  # type: ignore[attr-defined]


def post_to_make(transcript: str) -> List[str]:
    if not MAKE_WEBHOOK_URL:
        return ["⚠️ MAKE_WEBHOOK_URL není v Secrets"]
    resp = requests.post(MAKE_WEBHOOK_URL, json={"transcript": transcript}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("bullets", [])

# ----------------------------------------------------------------------------
# Background worker -----------------------------------------------------------
# ----------------------------------------------------------------------------

def start_worker(ctx, placeholders):
    """Spustí thread, který tahá AudioFrame‑y a přenáší výsledky do session_state."""

    if "worker_running" in st.session_state:  # už běží
        return

    def _worker():
        audio_buf = bytearray()
        last      = time.time()
        sr        = 48000
        while ctx.state.playing:
            try:
                frames = ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                continue

            for frame in frames:
                if not isinstance(frame, av.AudioFrame):
                    continue
                pcm16 = frame.to_ndarray().tobytes()
                audio_buf.extend(pcm16)
                sr = frame.sample_rate or sr

                # RMS → level bar
                rms = float(np.sqrt(np.mean(np.square(np.frombuffer(pcm16, np.int16)))) / 32768)
                st.session_state["_audio_rms"] = rms

            # každých SEGMENT_SEC → Whisper → Make
            if time.time() - last >= SEGMENT_SEC and audio_buf:
                wav_bytes = pcm_to_wav(bytes(audio_buf), sr)
                audio_buf.clear(); last = time.time()
                try:
                    txt     = whisper_transcribe(wav_bytes)
                    bullets = post_to_make(txt)
                except Exception as e:  # network / API error
                    bullets = [f"❌ {e}"]
                st.session_state.setdefault("bullets", []).extend(bullets)
                st.experimental_rerun()  # jediný rerun – jen když máme nové bullet‑points

        st.session_state.pop("worker_running", None)

    threading.Thread(target=_worker, daemon=True).start()
    st.session_state["worker_running"] = True

# ----------------------------------------------------------------------------
# UI --------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    st.set_page_config("AI Moderátor", "📝", layout="centered")
    st.title("📝 AI Moderátor – audio ➜ bullet‑points")

    st.sidebar.header("Nastavení")
    mode = st.sidebar.radio("Režim", ["Upload", "Live"], index=1)
    seg  = st.sidebar.slider("Interval segmentu (s)", 15, 180, SEGMENT_SEC, 5)
    globals()["SEGMENT_SEC"] = seg

    if mode == "Upload":
        upload_ui()
    else:
        live_ui()

# --------------------------------------------

def upload_ui():
    f = st.file_uploader("Nahraj audio", type=["wav", "mp3", "m4a"])
    if f and st.button("Zpracovat"):
        with st.spinner("Whisper → Make…"):
            txt = whisper_transcribe(f.read())
            bullets = post_to_make(txt)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))

# --------------------------------------------

def live_ui():
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    rtc_cfg: RTCConfiguration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": "turn:global.relay.metered.ca:80", "username": "global", "credential": "global"},
        ]
    }

    ctx = webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration=rtc_cfg,
        async_processing=True,
    )

    level_ph = st.empty()
    bullet_ph = st.container()

    # start background worker once the stream is up
    if ctx.state.playing and ctx.audio_receiver:
        start_worker(ctx, (level_ph, bullet_ph))

    # render level bar (updates in place – no rerun needed)
    rms = st.session_state.get("_audio_rms", 0.0)
    bar_len = int(rms * 20)
    bar = "🟩" * bar_len + "▫️" * (20 - bar_len)
    level_ph.markdown(f"**Úroveň audia:** {bar}")

    # render bullets (after possible rerun)
    bullets: List[str] = st.session_state.get("bullets", [])
    bullet_ph.subheader("Živé bullet‑points")
    if bullets:
        bullet_ph.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        bullet_ph.info("Čekám na první segment…")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
