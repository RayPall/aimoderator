import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import torch, whisper, queue, threading, numpy as np, time, os
from openai import OpenAI

MODEL_NAME   = st.sidebar.selectbox(
    "Whisper model", ["tiny", "base", "small", "medium", "turbo"], index=4)
SUMMARIZE    = st.sidebar.checkbox("Generovat odrážky GPT-4o", True)
openai_key   = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key) if SUMMARIZE and openai_key else None

@st.cache_resource  # načíst model jednou
def load_model(name): return whisper.load_model(name)
model = load_model(MODEL_NAME)

st.title("🗒️ Whisper Flipchart Live")
status = st.empty()
transcript_area = st.text_area("📜 Přepis (scrolluje se)", height=250)
bullets_area    = st.empty()

audio_q = queue.Queue()

def audio_callback(frame):
    audio = np.frombuffer(frame.to_ndarray(), np.int16).flatten().astype(np.float32) / 32768.0
    audio_q.put(audio)
    return frame

webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    audio_receiver_size=25600,  # ~0.5 s @ 48 kHz mono int16
    client_settings={"rtcv_audio": True},
    audio_frame_callback=audio_callback,
)

def transcribe_loop():
    buffer = []
    last_summary = 0
    while True:
        buffer.append(audio_q.get())
        if len(buffer) >= 96:  # ≈ 48 kHz * 2 s / 25600 ≈ 96 chunků 0.5 s
            segment = np.concatenate(buffer[-320:])  # posledních 5 s
            whisper.audio.save_audio(segment, "buf.wav", 48000)
            result = model.transcribe("buf.wav", language="cs", fp16=torch.cuda.is_available())
            text = result["text"].strip()
            if text:
                transcript_area.write(text + "\n", unsafe_allow_html=True)
            # Co 30 s pošleme summary
            if SUMMARIZE and time.time() - last_summary > 30 and openai_client:
                last_1min = "\n".join(transcript_area.value.splitlines()[-60:])
                prompt = f"Shrň následující český text do 3–5 bodů:\n\n{last_1min}"
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
                )
                bullets_area.markdown(resp.choices[0].message.content)
                last_summary = time.time()

threading.Thread(target=transcribe_loop, daemon=True).start()
