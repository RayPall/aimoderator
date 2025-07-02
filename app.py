# ai_moderator_live.py
"""
AI Moderátor – audio ➜ bullet‑points
===================================

* **Upload** (režim 1)   – statický soubor → Whisper → Make → bullet‑points.
* **Live**  (režim 2)     – stream z mikrofonu / virtuálního kabelu přes WebRTC.
  * ŽÁDNÉ autorefreshy, žádné `st.rerun()` v smyčce.
  * Audio se zpracovává **v AudioProcessoru** (API `streamlit_webrtc ≥ 0.46`).
  * Každých `SEGMENT_SEC` → WAV → Whisper → Make → bullet‑points ⮕ zapisujeme do
    `st.session_state['bullets']` → UI se překreslí samo při dalším script runu.
  * RMS úroveň se zapisuje do `st.session_state['_rms']` a vykresluje v
    indikátoru.

Testováno lokálně i na Streamlit Cloud (vyžaduje TURN 443/tcp).
"""
from __future__ import annotations

import asyncio, io, time, wave, logging
from typing import List

import av, numpy as np, streamlit as st, openai, requests
from streamlit_webrtc import (
    webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEGMENT_SEC      = 60
WHISPER_MODEL    = "whisper-1"
MAKE_WEBHOOK_URL = st.secrets.get("MAKE_WEBHOOK_URL", "")
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY", "")

client  = openai.OpenAI(api_key=OPENAI_API_KEY)
logger  = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pcm_to_wav(raw: bytes, sr: int) -> bytes:
    """16‑bit mono PCM → WAV (bytes)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(raw)
    return buf.getvalue()


def whisper_transcribe(wav: bytes) -> str:
    r = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("live.wav", io.BytesIO(wav), "audio/wav"),  # type: ignore[arg-type]
    )
    return r.text  # type: ignore[attr-defined]


def post_to_make(txt: str) -> List[str]:
    if not MAKE_WEBHOOK_URL:
        return ["⚠️ MAKE_WEBHOOK_URL chybí v Secrets"]
    r = requests.post(MAKE_WEBHOOK_URL, json={"transcript": txt}, timeout=30)
    r.raise_for_status()
    return r.json().get("bullets", [])

# ---------------------------------------------------------------------------
# Audio processor – běží v samostatném vlákně uvnitř streamlit‑webrtc
# ---------------------------------------------------------------------------

class LiveAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buf: bytearray = bytearray()
        self.last           = time.time()
        self.sr             = 48000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:  # noqa: N802 – API
        pcm = frame.to_ndarray().tobytes()
        self.buf.extend(pcm)
        self.sr = frame.sample_rate or self.sr

        # RMS → indikátor
        rms = float(np.sqrt(np.mean(np.square(np.frombuffer(pcm, np.int16)))) / 32768)
        st.session_state["_rms"] = rms

        # každých SEGMENT_SEC → zpracování segmentu
        if time.time() - self.last >= SEGMENT_SEC and self.buf:
            wav = pcm_to_wav(bytes(self.buf), self.sr)
            self.buf.clear(); self.last = time.time()
            asyncio.get_event_loop().create_task(self.process_segment(wav))
        return frame

    async def process_segment(self, wav: bytes):
        try:
            txt     = await asyncio.to_thread(whisper_transcribe, wav)
            bullets = await asyncio.to_thread(post_to_make, txt)
        except Exception as e:  # network/api
            logger.exception("segment error")
            bullets = [f"❌ {e}"]

        st.session_state.setdefault("bullets", []).extend(bullets)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config("AI Moderátor", "📝", layout="centered")
st.title("📝 AI Moderátor – audio ➜ bullet‑points")

st.sidebar.header("Nastavení")
mode = st.sidebar.radio("Režim", ["Upload", "Live"], index=1)
SEGMENT_SEC = st.sidebar.slider("Interval segmentu (s)", 15, 180, SEGMENT_SEC, 5)

# ---------------------------------------------------------------------------
# upload mód – statický soubor
# ---------------------------------------------------------------------------

if mode == "Upload":
    up = st.file_uploader("Nahraj audio", type=["wav", "mp3", "m4a"])
    if up and st.button("Zpracovat"):
        with st.spinner("Whisper → Make…"):
            txt     = whisper_transcribe(up.read())
            bullets = post_to_make(txt)
        st.subheader("Bullet‑points")
        st.markdown("\n".join(f"• {b}" for b in bullets))

# ---------------------------------------------------------------------------
# live mód – WebRTC
# ---------------------------------------------------------------------------

else:
    st.markdown("Klikni **Allow** pro mikrofon / virtuální kabel.")

    rtc_cfg: RTCConfiguration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:global.relay.metered.ca:443?transport=tcp",
                    "turn:global.relay.metered.ca:80?transport=tcp",
                ],
                "username": "global",
                "credential": "global",
            },
        ]
    }

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=LiveAudioProcessor,
    )

    # --------------------------- vizuální indikátory ------------------------

    rms  = st.session_state.get("_rms", 0.0)
    bar  = "🟩" * int(rms * 20) + "▫️" * (20 - int(rms * 20))
    st.markdown(f"**Úroveň audia:** {bar}")

    st.subheader("Živé bullet‑points")
    bullets: List[str] = st.session_state.get("bullets", [])
    if bullets:
        st.markdown("\n".join(f"• {b}" for b in bullets))
    else:
        st.info("Čekám na první segment…")
