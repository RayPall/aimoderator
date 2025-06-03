# audio_upload_whisper_segmenter.py
"""
Streamlit: Upload → FFmpeg segmentace (30 s / MP3 mono 16 kHz 32 kb/s) → Whisper → Make → Bullet-points

1) Uživatel nahraje audio (MP3/WAV/M4A), až třeba 200 MB.
2) Vložíme upload do dočasného souboru (tmp_in.ext).
3) Spustíme FFmpeg segmentaci:
     ffmpeg -i tmp_in.ext
            -ar 16000 -ac 1 -b:a 32k
            -f segment -segment_time 30 -reset_timestamps 1
            tmp_dir/chunk%03d.mp3
   → FFmpeg vytvoří v tmp_dir MP3 soubory chunk000.mp3, chunk001.mp3, …,
     z nichž každý je mono, 16 kHz, 32 kb/s a dlouhý ~30 s (nebo méně v posledním).
4) Načteme každý MP3 z tmp_dir, pošleme je do OpenAI Whisper 
   (model="whisper-1") a spojíme výsledné texty do full_transcript.
5) Počkáme, až budou všechny části zpracované.  
6) Celý full_transcript pošleme do Make webhook.  
7) Make vrátí JSON pole bullet-pointů; ty vykreslíme jako flipchart.

Požadavky:
-----------
pip install streamlit openai requests

Ve Streamlit Cloudu do packages.txt:
    ffmpeg
"""

from __future__ import annotations
import io
import os
import re
import uuid
import shutil
import logging
import tempfile
import requests
import streamlit as st

from openai import OpenAI, OpenAIError
from subprocess import run, PIPE, CalledProcessError

# ─────────── Konfigurace ───────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

TARGET_SR       = 16_000   # vzorkování 16 kHz
TARGET_CHANNELS = 1        # mono
TARGET_BR       = "32k"    # 32 kb/s
SEGMENT_SEC     = 30       # každý segment ~30 s

logging.basicConfig(level=logging.INFO)

# ─────────── Whisper a Make ───────────────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty (MP3) do Whisper API (model="whisper-1").
    Vrátí přepis (text) nebo vyhodí OpenAIError.
    """
    try:
        file_like = io.BytesIO(b)
        file_like.name = filename
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=file_like, language="cs"
        )
        return resp.text
    except OpenAIError as e:
        st.error(f"❌ Whisper API chyba: {e}")
        raise

def post_to_make(text: str) -> list[str]:
    """
    Pošle celý přepis jako JSON na Make webhook.
    Vrátí seznam stringů (bullet-pointů) nebo [] při chybě.
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


# ─────────── Formátování bullet-pointů ─────────────────────────────────────
DASH  = re.compile(r"\s+-\s+")
STRIP = "-–—• "

def fmt_bullet(raw: str) -> str:
    """
    Nadpis (první řádek) tučně (uppercase), pod-body jako <li>…</li>.
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


# ─────────── Segmentace přes FFmpeg CLI ─────────────────────────────────────
def ffmpeg_segment_to_mp3s(input_path: str, out_dir: str) -> list[str]:
    """
    Ze souboru input_path vytvoří v adresáři out_dir MP3 segmenty:
      - mono, 16 kHz, 32 kb/s
      - formát segmentů po SEGMENT_SEC sekundách
      - každý segment: chunkNNN.mp3
    Vrací seznam absolutních cest k postupně vytvořeným chunk*.mp3.
    """

    # Vytvoříme adresář out_dir (pokud neexistuje):
    os.makedirs(out_dir, exist_ok=True)

    # Složíme FFmpeg příkaz:
    #
    # ffmpeg -i input_path
    #        -ar 16000 -ac 1 -b:a 32k
    #        -f segment -segment_time 30 -reset_timestamps 1
    #        out_dir/chunk%03d.mp3
    cmd = [
        "ffmpeg",
        "-y",                       # přepiš existující výstupy
        "-i", input_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-b:a", TARGET_BR,
        "-f", "segment",
        "-segment_time", str(SEGMENT_SEC),
        "-reset_timestamps", "1",
        os.path.join(out_dir, "chunk%03d.mp3")
    ]

    try:
        proc = run(cmd, check=True, stdout=PIPE, stderr=PIPE)
    except CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")
        st.error(f"❌ Chyba FFmpeg segmentace:\n{err}")
        raise

    # Nyní v out_dir máme např. chunk000.mp3, chunk001.mp3, …
    # Seřadíme je lexikograficky:
    files = sorted(
        fname for fname in os.listdir(out_dir) 
        if fname.startswith("chunk") and fname.endswith(".mp3")
    )
    return [os.path.join(out_dir, fname) for fname in files]


# ─────────── UI – Streamlit layout ────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make Demo", layout="centered")
st.title("🎙️ Note-bot")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (až ~200 MB)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    # 1) Uložíme upload do temp souboru
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost nahrávky: **{size_mb:.1f} MB**")

    # Do dočasného souboru:
    ext = uploaded.name.split('.')[-1].lower()
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    tmp_input.write(raw_bytes)
    tmp_input.flush()
    tmp_input_path = tmp_input.name
    tmp_input.close()

    # 2) Vytvoříme dočasný adresář pro segmenty:
    tmp_dir = tempfile.mkdtemp(prefix="audio_chunks_")

    # 3) Spustíme segmentaci (každý chunk je MP3)
    with st.spinner("🔀 Spouštím FFmpeg segmentaci na 30 s MP3 kousky…"):
        try:
            chunk_paths = ffmpeg_segment_to_mp3s(tmp_input_path, tmp_dir)
        except Exception:
            # pokud FFmpeg padnul, ukončíme
            os.unlink(tmp_input_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    num_chunks = len(chunk_paths)
    if num_chunks == 0:
        st.error("❌ Nebyl vytvořen žádný chunk. Zkontrolujte vstupní soubor.")
        os.unlink(tmp_input_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        st.stop()

    st.info(f"Audio rozděleno do **{num_chunks}** kousků (~{SEGMENT_SEC}s každý).")

    # 4) Pro každý chunk: načteme ho, pošleme do Whisperu
    full_transcript = ""
    progress_bar = st.progress(0)
    status_txt = st.empty()

    for i, path in enumerate(chunk_paths, start=1):
        fname = os.path.basename(path)
        status_txt.markdown(f"⏳ Zpracovávám chunk **{i}/{num_chunks}**: {fname}")

        # Načteme byty chunku:
        try:
            with open(path, "rb") as f:
                mp3_bytes = f.read()
        except Exception as e:
            st.error(f"❌ Chyba při čtení chunku: {e}")
            continue

        # Pošleme do Whisperu:
        with st.spinner(f"⏳ Whisper chunk {i}/{num_chunks}…"):
            try:
                txt = whisper_transcribe(mp3_bytes, filename=fname)
            except OpenAIError:
                # Ponecháme to, abychom viděli chybu a pokračovali dalším chunkem
                continue

        # Přidáme do výsledného přepisu
        full_transcript += " " + txt
        progress_bar.progress(i / num_chunks)

    status_txt.markdown("✅ Hotovo: všechny chunk-díly odeslány do Whisperu.")

    # 5) Zobrazíme kompletní přepis:
    st.subheader("📄 Kompletní přepis")
    st.text_area(" ", full_transcript.strip(), height=300)

    # 6) Odešleme full_transcript na Make:
    with st.spinner("📤 Odesílám přepis na Make…"):
        bullets = post_to_make(full_transcript)

    # 7) Vykreslíme bullet-pointy (flipchart-styl):
    if bullets:
        st.markdown("---")
        st.subheader("📌 Bullet-pointy z Make")
        st.markdown(
            "<ul>" + "".join(f"<li>{fmt_bullet(b)}</li>" for b in bullets) + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Make nevrátil žádné bullet-pointy.")

    # 8) Úklid dočasných souborů:
    os.unlink(tmp_input_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
