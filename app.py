# audio_auto_compress_whisper_make.py
"""
Streamlit: Upload → Auto-compress (pokud > 25 MiB) → Whisper → Make → Bullet-points
==================================================================================

Jak to funguje:
1. Uživatel nahraje audio soubor (MP3, WAV nebo M4A), max. např. 200 MB.
2. Když je jeho velikost nad 25 MiB, Pydub + FFmpeg jej převede na:
   • mono  
   • 16 kHz  
   • MP3 @ 64 kbps  
   Čímž se typicky zmenší na < 25 MiB.
3. (Zkomprimovaný) soubor se pošle v jednom požadavku na Whisper.
4. Získaný přepis se okamžitě pošle na Make webhook.
5. Make vrátí pole bullet-pointů, která aplikace vykreslí.

=== Pip dependencies ===
streamlit
openai
requests
pydub

=== V files: ===
– requirements.txt obsahuje výše uvedené balíčky  
– packages.txt (na Streamlit Cloud) obsahuje řádek:
    ffmpeg
"""

from __future__ import annotations
import io, re, logging, os, tempfile, streamlit as st, requests
from openai import OpenAI, OpenAIError
from pydub import AudioSegment

# ─────────── Konfigurace ─────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

WHISPER_LIMIT = 25 * 1024 * 1024    # 25 MiB
TARGET_BR_KBPS = 64                 # 64 kbps pro MP3 výsledné komprese

logging.basicConfig(level=logging.INFO)

# ─────────── Funkce auto-komprese ────────────────────────────────────────
def compress_if_needed(raw: bytes, ext: str) -> bytes:
    """
    Pokud raw > 25 MiB, zkomprimuje pomocí pydub/ffmpeg:
    převede na MP3, mono, 16 kHz, 64 kbps.
    Vrátí (může být zmenšené) byty MP3.
    """
    if len(raw) <= WHISPER_LIMIT:
        return raw

    # uložíme surový soubor do temp, aby ho pydub načetl
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as src:
        src.write(raw)
        src.flush()
        src_name = src.name

    # načteme audio, upravíme parametry
    audio = AudioSegment.from_file(src_name)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # exportujeme zpět do byte streamu
    mp3_buf = io.BytesIO()
    audio.export(mp3_buf, format="mp3", bitrate=f"{TARGET_BR_KBPS}k")
    os.unlink(src_name)

    mp3_bytes = mp3_buf.getvalue()
    return mp3_bytes

# ─────────── Funkce Whisper & Make ──────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty do Whisper (model whisper-1). Vrátí přepis.
    Zvedne OpenAIError, pokud selže.
    """
    try:
        file_like = io.BytesIO(b)
        file_like.name = filename  # Whisper API akceptuje i tento attribut
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=file_like,
            language="cs"
        )
        return resp.text
    except OpenAIError as e:
        st.error(f"❌ Whisper API Error: {e}")
        raise

def post_to_make(text: str) -> list[str]:
    """
    Pošle přepis na Make webhook. Vrací pole textových bullet-pointů
    (nebo prázdný list, pokud Make nevrátil array).
    """
    try:
        r = requests.post(
            MAKE_URL,
            json={
                "token": MAKE_TOKEN,
                "transcript": text,
                "existing": []
            },
            timeout=90
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"❌ Make webhook Error: {e}")
        return []

# ─────────── Formátování bullet-pointů ───────────────────────────────────
DASH = re.compile(r"\s+-\s+")
STRIP = "-–—• "

def fmt_bullet(raw: str) -> str:
    """
    Nadpis (první řádek) bude TUČNÝ, následující podbody budou
    jako <ul><li>…</li></ul>. Vrací HTML string.
    """
    if "\n" in raw:
        parts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    else:
        items = DASH.split(raw.strip())
        parts = [items[0]] + [f"- {p}" for p in items[1:]]
    if not parts:
        return ""

    head, *det = parts
    head_html = f"<strong>{head.upper()}</strong>"
    if not det:
        return head_html

    list_items = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{list_items}</ul>"

# ─────────── Layout Streamlit ───────────────────────────────────────────
st.set_page_config(page_title="Audio → Whisper → Make", layout="centered")
st.title("🎙️ Přepis audia + bullet-pointy z Make")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (ideálně do 200 MB)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost souboru: **{size_mb:.1f} MB**")

    # ─ 1) Komprese, pokud nad 25 MiB
    extension = uploaded.type.split("/")[-1]  # "mp3", "wav", "m4a", ...
    compressed = compress_if_needed(raw_bytes, ext=extension)
    comp_mb = len(compressed) / (1024 * 1024)
    if comp_mb < size_mb:
        st.info(f"Soubor byl > 25 MB → po kompresi nyní: **{comp_mb:.1f} MB**")
    else:
        st.info("Soubor ≤ 25 MB → posílám originál")

    # ─ 2) Přepis přes Whisper
    with st.spinner("⏳ Přepisuji audio přes Whisper…"):
        try:
            transcript = whisper_transcribe(compressed, filename=uploaded.name)
        except OpenAIError:
            st.stop()  # chybová hláška již vypdána v funkcí
    st.success("✅ Přepis dokončen")
    st.text_area("📄 Přepis", transcript, height=250)

    # ─ 3) Odeslání přepisu na Make
    with st.spinner("📤 Posílám přepis na Make…"):
        bullets = post_to_make(transcript)

    # ─ 4) Zobrazení bullet-pointů
    if bullets:
        st.markdown("---")
        st.subheader("📌 Bullet-pointy z Make")
        st.markdown(
            "<ul>" + "".join(f"<li>{fmt_bullet(b)}</li>" for b in bullets) + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Make nevrátil žádné bullet-pointy.")
