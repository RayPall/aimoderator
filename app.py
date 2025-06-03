# demo_audio_upload_whisper.py
"""
Upload MP3/WAV → OpenAI Whisper → (volitelně) Make
=================================================
* Uživatel nahraje krátký záznam (.mp3 / .wav / .m4a)  
* Aplikace použije Whisper k přepisu a ukáže text  
* Zaškrtnutím „Odeslat na Make“ se přepis pošle na webhook;  
  Make vrátí pole bullet-pointů, které hned zobrazíme

▶ Stačí jediný soubor – ideální showcase Streamlit + GitHub + Make
------------------------------------------------------------------
requirements.txt
----------------
streamlit  
openai  
requests
"""

from __future__ import annotations
import io, logging, re, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ───── Nastavení (secrets) ──────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"  # změň dle sebe
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

logging.basicConfig(level=logging.INFO)

# ───── Funkce Whisper & Make ────────────────────────────────────────────
def whisper_transcribe(file: io.BufferedReader | io.BytesIO) -> str | None:
    """Vrátí text, nebo None při chybě"""
    try:
        out = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            language="cs"
        )
        return out.text
    except OpenAIError as e:
        st.error(f"❌ Whisper API: {e}")
        return None

def send_to_make(transcript: str) -> list[str]:
    try:
        r = requests.post(MAKE_URL, json={
            "token": MAKE_TOKEN,
            "transcript": transcript,
            "existing": []          # nic zatím nemáme
        }, timeout=90)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"❌ Make webhook: {e}")
        return []

# ───── Flip-helper (nadpis tučně, podbody s puntíky) ────────────────────
DASH = re.compile(r"\s+-\s+"); STRIP="-–—• "
def fmt(pt: str) -> str:
    parts = [ln.strip() for ln in pt.splitlines() if ln.strip()] if "\n" in pt \
            else [x if i==0 else f"- {x}" for i,x in enumerate(DASH.split(pt.strip()))]
    head,*det = parts
    head = f"<strong>{head.upper()}</strong>"
    if not det: return head
    items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head}<ul>{items}</ul>"

# ───── UI ───────────────────────────────────────────────────────────────
st.set_page_config("Audio → Whisper demo")

st.title("🎙️ Nahrát audio & získat přepis")

uploaded = st.file_uploader("➡️ Přetáhni MP3/WAV/M4A", type=["mp3","wav","m4a"])
if uploaded:
    with st.spinner("⏳ Přepisuji přes Whisper…"):
        text = whisper_transcribe(uploaded)          # uploaded je již file-like
    if text:
        st.success("✅ Přepis hotov")
        st.text_area("📄 Přepis", text, height=250)

        if st.checkbox("Odeslat přepis na Make a zobrazit body"):
            with st.spinner("⏳ Odesílám na Make…"):
                bullets = send_to_make(text)
            if bullets:
                st.markdown("---")
                st.subheader("📌 Body od Make")
                st.markdown("<ul>"+ "".join(f"<li>{fmt(b)}</li>" for b in bullets)+"</ul>",
                            unsafe_allow_html=True)
            else:
                st.info("Make nevrátil žádné body.")
