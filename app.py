# audio_upload_whisper_autocompress_cli.py
"""
Streamlit: Upload → Auto-compress (CLI-FFmpeg) → Whisper → Make → Bullet-points
===============================================================================
Místo pydub používáme FFmpeg přímo z příkazové řádky, takže odpadá závislost
na `pydub` a případném `audioop` modulu v Pythonu.

1. Uživatel nahraje audio (MP3, WAV nebo M4A), max. např. 200 MB.
2. Když je soubor > 25 MiB, spustíme FFmpeg:
   • převedeme na MP3
   • mono
   • vzorkovací kmitočet 16 kHz
   • bitrate 64 kbps  
   → výsledný datový proud čteme ze stdout FFmpeg, typicky < 25 MiB.
3. (Zkomprimovaný) MP3 binárně pošleme do Whisper v jednom requestu.
4. Vrácený přepis pošleme do Make webhook.
5. Make vrátí JSON array bullet-pointů, ta vykreslíme jako flipchart.

Pip requirements:
-----------------
streamlit
openai
requests

Na Streamlit Cloud přidej do `packages.txt`:
    ffmpeg
"""

from __future__ import annotations
import io, re, logging, subprocess, tempfile, os, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ────────── Konfigurace ───────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

WHISPER_LIMIT = 25 * 1024 * 1024    # 25 MiB
# Parametry pro kompresi FFmpeg:
TARGET_SR      = 16_000             # 16 kHz
TARGET_CHANNELS= 1                  # mono
TARGET_BITRATE = "64k"              # 64 kbps

logging.basicConfig(level=logging.INFO)


# ────────── Funkce pro kompresi přes FFmpeg CLI ──────────────────────────
def compress_if_needed(raw: bytes, ext: str) -> bytes:
    """
    • Pokud raw ≤ 25 MiB → vrátí raw beze změny.
    • Pokud raw > 25 MiB → zapíše raw do dočasného souboru s příponou ext,
      spustí FFmpeg pro převod na: MP3, 16 kHz, mono, 64 kbps → čte stdout
      z FFmpeg (pipe:1) a vrátí bytes mp3.
    """
    if len(raw) <= WHISPER_LIMIT:
        return raw

    # Uložíme původní audiobytový řetězec do TMP arquivo:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        in_path = tmp_in.name

    # Vytvoříme dočasný výstupní soubor, kam FFmpeg zapíše
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_path = tmp_out.name
    tmp_out.close()

    # Sestavíme FFmpeg příkaz:
    # ffmpeg -i <in_path> -ar 16000 -ac 1 -b:a 64k -f mp3 <out_path>
    cmd = [
        "ffmpeg",
        "-y",                  # přepsat out_path, pokud existuje
        "-i", in_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-b:a", TARGET_BITRATE,
        "-f", "mp3",
        out_path
    ]

    # Spustíme FFmpeg (sestavíme čekání na dokončení):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.unlink(in_path)
        os.unlink(out_path)
        st.error(f"❌ Chyba při kompresi FFmpeg: {e.stderr.decode(errors='ignore')}")
        raise

    # Přečteme výsledek (MP3 byty) a smažeme tempky:
    with open(out_path, "rb") as f:
        compressed = f.read()
    os.unlink(in_path)
    os.unlink(out_path)
    return compressed


# ────────── Funkce Whisper & Make ────────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty do Whisper (model whisper-1). Vrátí přepis.
    Zvedne OpenAIError, pokud dojde k chybě.
    """
    try:
        file_like = io.BytesIO(b)
        file_like.name = filename  # GPT-4 / Whisper knihovna akceptuje i tento atribut
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=file_like,
            language="cs"
        )
        return resp.text
    except OpenAIError as e:
        st.error(f"❌ Whisper API chyba: {e}")
        raise


def post_to_make(text: str) -> list[str]:
    """
    Pošle přepis na Make webhook → vrátí pole stringů.
    Pokud Make vrátí cokoliv jiného nebo selže, vrátí [].
    """
    try:
        r = requests.post(
            MAKE_URL,
            json={"token": MAKE_TOKEN, "transcript": text, "existing": []},
            timeout=90
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"❌ Make webhook chyba: {e}")
        return []


# ────────── Helper pro formátování bullet-pointů ─────────────────────────
DASH = re.compile(r"\s+-\s+")
STRIP = "-–—• "

def fmt_bullet(raw: str) -> str:
    """
    Převede string raw na HTML:
      * první řádek → nadpis (tučně, UPPERCASE)
      * další řádky („- detail“) → <ul><li>…</li></ul>
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
    items_html = "".join(f"<li>{d.lstrip(STRIP)}</li>" for d in det)
    return f"{head_html}<ul>{items_html}</ul>"


# ────────── UI – Streamlit layout ────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make Demo", layout="centered")
st.title("🎙️ Audio Upload → Auto-compress → Whisper → Make → Bullet-points")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (max. ~200 MB)", 
    type=["mp3","wav","m4a"]
)

if uploaded:
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost souboru: **{size_mb:.1f} MB**")

    # 1) Auto-komprese přes FFmpeg, pokud nad 25 MiB
    ext = uploaded.name.split('.')[-1]  # přípona: mp3, wav, m4a
    try:
        compressed_bytes = compress_if_needed(raw_bytes, ext=ext)
    except Exception:
        st.stop()  # chyba byla vypisována v compress_if_needed

    comp_mb = len(compressed_bytes) / (1024 * 1024)
    if comp_mb < size_mb:
        st.info(f"Soubor byl > 25 MB → po kompresi nyní: **{comp_mb:.1f} MB**")
    else:
        st.info("Soubor ≤ 25 MiB → posílám originál (bez komprese)")

    # 2) Přepis přes Whisper
    with st.spinner("⏳ Přepisuji přes Whisper…"):
        try:
            transcript = whisper_transcribe(compressed_bytes, filename=uploaded.name)
        except OpenAIError:
            st.stop()  # chybovou hlášku již vypsal whisper_transcribe
    st.success("✅ Přepis dokončen")
    st.text_area("📄 Přepis", transcript, height=250)

    # 3) Odeslání přepisu na Make a zobrazení bullet-pointů
    with st.spinner("📤 Odesílám přepis na Make…"):
        bullets = post_to_make(transcript)

    if bullets:
        st.markdown("---")
        st.subheader("📌 Bullet-points z Make")
        st.markdown(
            "<ul>" + "".join(f"<li>{fmt_bullet(b)}</li>" for b in bullets) + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Make nevrátil žádné bullet-pointy.")
