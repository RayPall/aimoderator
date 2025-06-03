# audio_upload_whisper_sliced.py
"""
Upload → Whisper (sliced if > 25 MiB) → Make → Bullet-points
===========================================================

• user uploads MP3/WAV/M4A  
• if the file is ≤ 25 MiB → one Whisper call  
• if it is larger  → file is sliced into 24 MiB chunks, each chunk is
  sent to Whisper and transcripts are concatenated  
• final transcript is POST-ed to a Make webhook; Make returns JSON array
  of bullet-points, which we render as a simple flipchart

requirements.txt
----------------
streamlit
openai
requests
"""

from __future__ import annotations
import io, re, logging, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ────────── CONFIG ──────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]          # povinné
MAKE_URL       = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN     = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

WHISPER_LIMIT  = 25 * 1024 * 1024          # 25 MiB
CHUNK_SIZE     = 24 * 1024 * 1024          # safety margin

client = OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)

# ────────── Whisper wrapper ─────────────────────────────────────────────
def whisper_bytes(b: bytes, fname: str = "chunk") -> str:
    """Send raw bytes to Whisper; return text or raise."""
    file_like = io.BytesIO(b); file_like.name = fname
    resp = client.audio.transcriptions.create(
        model="whisper-1", file=file_like, language="cs"
    )
    return resp.text

# ────────── Make webhook ────────────────────────────────────────────────
def post_to_make(text: str) -> list[str]:
    try:
        r = requests.post(MAKE_URL, json={
            "token": MAKE_TOKEN,
            "transcript": text,
            "existing": [],
        }, timeout=90)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Make error: {e}")
        return []

# ────────── Flipchart formatting ────────────────────────────────────────
DASH = re.compile(r"\s+-\s+"); STRIP = "-–—• "
def fmt(pt: str) -> str:
    parts = ([ln.strip() for ln in pt.splitlines() if ln.strip()]
             if "\n" in pt else
             [p if i==0 else f"- {p}" for i, p in enumerate(DASH.split(pt.strip()))])
    head, *det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det: return head_html
    items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{items}</ul>"

# ────────── UI ──────────────────────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make demo")
st.title("🎙️ Přepis audia a bullet-pointy z Make")

uploaded = st.file_uploader("➕ Nahraj MP3 / WAV / M4A (max. 200 MB)", type=["mp3","wav","m4a"])

if uploaded:
    raw = uploaded.read()
    st.write(f"Velikost souboru: **{len(raw)/1_048_576:.1f} MB**")

    # ── slicing podle velikosti ────────────────────────────────────────
    chunks: list[bytes]
    if len(raw) <= WHISPER_LIMIT:
        chunks = [raw]
    else:
        st.info("Soubor > 25 MB → dělím na části")
        chunks = [raw[i:i+CHUNK_SIZE] for i in range(0, len(raw), CHUNK_SIZE)]

    # ── Whisper každého kusu ───────────────────────────────────────────
    full_text = ""
    for i, ch in enumerate(chunks, 1):
        with st.spinner(f"Whisper {i}/{len(chunks)}…"):
            try:
                txt = whisper_bytes(ch, fname=f"part{i}.{uploaded.type}")
                full_text += " " + txt
            except OpenAIError as e:
                st.error(f"Whisper chyba u části {i}: {e}")
                st.stop()
        st.success(f"Část {i} hotová")

    st.subheader("📄 Kompletní přepis")
    st.text_area(" ", full_text.strip(), height=250)

    # ── Odešli na Make a zobraz bullet-pointy ───────────────────────────
    with st.spinner("📤 Odesílám přepis na Make…"):
        bullets = post_to_make(full_text)

    if bullets:
        st.markdown("---")
        st.subheader("📌 Body z Make")
        st.markdown("<ul>"+"".join(f"<li>{fmt(b)}</li>" for b in bullets)+"</ul>",
                    unsafe_allow_html=True)
    else:
        st.info("Make nevrátil žádné body.")
