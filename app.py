# demo_audio_upload_whisper.py
"""
Upload MP3/WAV → Whisper → Make (automaticky) → Bullet-pointy
=============================================================
1. Uživatel nahraje audio soubor (.mp3 / .wav / .m4a)
2. Přepis proběhne přes OpenAI Whisper
3. Přepis se **okamžitě** odešle na Make webhook
4. Vrácené bullet-pointy se zobrazí na stránce

→ Jednoduchá demonstrace integrace Streamlit + Whisper + Make
"""

from __future__ import annotations
import io, re, logging, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ─── API klíče & URL ────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]          # povinné
MAKE_URL       = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN     = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

client = OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)

# ─── Funkce: Whisper & Make ─────────────────────────────────────────────
def whisper_transcribe(file_obj: io.BufferedReader | io.BytesIO) -> str:
    """Vrátí text (vyvolá Streamlit error při chybě)"""
    try:
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=file_obj, language="cs"
        )
        return resp.text
    except OpenAIError as e:
        st.error(f"❌ Whisper API: {e}")
        raise

def send_to_make(transcript: str) -> list[str]:
    try:
        r = requests.post(
            MAKE_URL,
            json={"token": MAKE_TOKEN, "transcript": transcript, "existing": []},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"❌ Make webhook: {e}")
        return []

# ─── Helper: formátování bullet-pointů ──────────────────────────────────
DASH = re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(raw: str) -> str:
    parts = ([ln.strip() for ln in raw.splitlines() if ln.strip()]
             if "\n" in raw else
             [p if i==0 else f"- {p}" for i,p in enumerate(DASH.split(raw.strip()))])
    head,*det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det: return head_html
    items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{items}</ul>"

# ─── UI ─────────────────────────────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make")
st.title("🎙️ Nahrát audio a získat bullet-pointy")

uploaded = st.file_uploader("➡️ Přetáhni MP3/WAV/M4A", type=["mp3", "wav", "m4a"])

if uploaded:
    with st.spinner("⏳ Přepisuji přes Whisper…"):
        transcript = whisper_transcribe(uploaded)

    st.success("✅ Přepis hotov")
    st.text_area("📄 Přepis", transcript, height=250)

    with st.spinner("📤 Odesílám přepis na Make…"):
        bullets = send_to_make(transcript)

    if bullets:
        st.markdown("---")
        st.subheader("📌 Body z Make")
        st.markdown("<ul>"+ "".join(f"<li>{fmt(b)}</li>" for b in bullets)+"</ul>",
                    unsafe_allow_html=True)
    else:
        st.info("Make nevrátil žádné body.")
