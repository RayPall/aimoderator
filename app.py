# audio_upload_whisper_chunking_progress.py
"""
Streamlit: Upload → Auto-compress (32 kb/s) → Pokud >25 MiB, rozděl do WAV chunků → Whisper (s indikací chunků) → Make → Bullet-points

1. Uživatel nahraje audio (MP3/WAV/M4A), max. např. 200 MB.
2. Pokud je soubor > 25 MiB, spustíme FFmpeg:
   • MP3, mono, 16 kHz, 32 kbit/s  
   → typicky výsledná MP3 < 25 MiB.  
   Pokud ale výsledná MP3 stále > 25 MiB, musíme dělit podle času.
3. Pokud je výsledná MP3 > 25 MiB:
   a) Dekódujeme celou MP3 do WAV (pipe:1).  
   b) Rozdělíme PCM na bloky po CHUNK_SEC (např. 120 s), každý blok zabalíme  
      jako samostatný WAV.  
   c) Vykreslíme progress bar + text “Chunk i / N” a pošleme každý WAV chunk  
      do Whisperu.  
   d) Spojíme všechny dílčí transkripty do jednoho řetězce.
4. Výsledný `full_transcript` pošleme do Make jedním požadavkem.
5. Make vrátí JSON pole bullet-pointů; ty vykreslíme jako jednoduchý flipchart.

=== Požadavky ===
pip install streamlit openai requests
Streamlit Cloud: v packages.txt přidej řádek `ffmpeg`
"""

import io, re, logging, os, subprocess, tempfile, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ────────────────── Konfigurace ───────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

WHISPER_LIMIT   = 25 * 1024 * 1024    # 25 MiB
TARGET_SR       = 16_000              # 16 kHz
TARGET_CHANNELS = 1                   # mono
TARGET_BR       = "32k"               # 32 kbit/s (komprese)
CHUNK_SEC       = 120                 # délka jednoho kusu v sekundách

logging.basicConfig(level=logging.INFO)

# ─────────── Funkce pro kompresi přes FFmpeg CLI ──────────────────────────
def compress_if_needed(raw: bytes, ext: str) -> bytes:
    """
    • Pokud raw ≤ 25 MiB → vrátí raw beze změny.  
    • Pokud raw > 25 MiB → uloží raw do tmp, spustí FFmpeg:
      MP3, 16 kHz, mono, 32 kbit/s → vrátí zkomprimované MP3 byty.
    """
    if len(raw) <= WHISPER_LIMIT:
        return raw

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        in_path = tmp_in.name

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_path = tmp_out.name
    tmp_out.close()

    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-b:a", TARGET_BR,
        "-f", "mp3",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.unlink(in_path)
        os.unlink(out_path)
        st.error(f"❌ Chyba při kompresi (FFmpeg): {e.stderr.decode(errors='ignore')}")
        raise

    with open(out_path, "rb") as f:
        compressed = f.read()
    os.unlink(in_path)
    os.unlink(out_path)
    return compressed

# ─────────── Funkce pro chunking a převod na PCM WAV ───────────────────────
def split_to_wav_chunks(mp3_bytes: bytes) -> list[bytes]:
    """
    1) Zapíše mp3_bytes do dočasného souboru (tmp.mp3).  
    2) Pomocí FFmpeg dekóduje MP3 na WAV (PCM) v paměti.  
    3) Rozdělí PCM na bloky po CHUNK_SEC sekundách a každému bloku
       přidá WAV hlavičku.  
    4) Vrátí seznam samostatných WAV byte-blocků.
    """
    # a) uložíme mp3_bytes do tmp souboru
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_in:
        tmp_in.write(mp3_bytes)
        tmp_in.flush()
        in_path = tmp_in.name

    # b) dekódujeme do raw WAV formátu
    cmd = [
        "ffmpeg", "-i", in_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_data, stderr = proc.communicate()
    if proc.returncode != 0:
        os.unlink(in_path)
        st.error(f"❌ Chyba při dekódování WAV (FFmpeg): {stderr.decode(errors='ignore')}")
        raise RuntimeError("FFmpeg decode failed")

    os.unlink(in_path)

    # c) oddělíme hlavičku (44 bajtů) a PCM data
    header = wav_data[:44]
    pcm = wav_data[44:]

    # d) vypočteme, kolik bajtů PCM na chunk
    bytes_per_chunk = TARGET_SR * 2 * CHUNK_SEC

    chunks: list[bytes] = []
    for i in range(0, len(pcm), bytes_per_chunk):
        block = pcm[i : i + bytes_per_chunk]
        wav_block = header[:4]  # "RIFF"
        size_chunk = 36 + len(block)
        wav_block += (size_chunk).to_bytes(4, "little")
        wav_block += header[8:44]
        wav_block += block
        chunks.append(wav_block)

    return chunks

# ─────────── Funkce Whisper & Make ────────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty (MP3 nebo WAV) do Whisper (model whisper-1). Vrátí přepis.
    """
    try:
        file_like = io.BytesIO(b)
        file_like.name = filename
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
    Pošle přepis na Make webhook → vrátí seznam stringů (bullet-pointy).
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

# ─────────── Helper pro formátování bulletu ─────────────────────────────────
DASH  = re.compile(r"\s+-\s+")
STRIP = "-–—• "
def fmt_bullet(raw: str) -> str:
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

# ─────────── UI – Streamlit layout ────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make Demo", layout="centered")
st.title("🎙️ Audio Upload → Auto-compress → Chunking → Whisper → Make → Bullet-points")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (do ~200 MB)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost souboru: **{size_mb:.1f} MB**")

    # 1) Komprimace (32 kb/s), pokud > 25 MiB
    ext = uploaded.name.split('.')[-1].lower()  # mp3, wav, m4a
    try:
        compressed = compress_if_needed(raw_bytes, ext=ext)
    except Exception:
        st.stop()  # chybová hláška se již zobrazila

    comp_mb = len(compressed) / (1024 * 1024)
    if comp_mb < size_mb:
        st.info(f"Soubor byl > 25 MiB → po kompresi nyní: **{comp_mb:.1f} MB**")
    else:
        st.info("Soubor ≤ 25 MiB → posílám originál (bez komprese)")

    # 2) Pokud je komprimované MP3 > 25 MiB, rozděl na WAV kousky
    if len(compressed) > WHISPER_LIMIT:
        st.warning("I po kompresi MP3 > 25 MiB – rozděluji na WAV kousky…")
        with st.spinner("🔪 Rozděluji na WAV chunks…"):
            wav_chunks = split_to_wav_chunks(compressed)
        num_chunks = len(wav_chunks)
    else:
        wav_chunks = []
        num_chunks = 1  # budeme mít jediný kus: 'compressed'

    # 3) Připrav seznam k zpracování (bytový blok + jméno)
    tasks: list[tuple[bytes,str]] = []
    if wav_chunks:
        for idx, w in enumerate(wav_chunks, start=1):
            tasks.append((w, f"chunk_{idx}.wav"))
    else:
        tasks.append((compressed, uploaded.name))

    # 4) Zpracovávej chunky postupně s indikací progresu
    full_transcript = ""
    progress_bar = st.progress(0)
    status_txt = st.empty()

    for i, (chunk_bytes, fname) in enumerate(tasks, start=1):
        status_txt.markdown(f"⏳ Zpracovávám chunk **{i}/{len(tasks)}**")
        with st.spinner(f"⏳ Whisper část {i}/{len(tasks)}…"):
            try:
                txt = whisper_transcribe(chunk_bytes, filename=fname)
            except OpenAIError:
                st.stop()
        full_transcript += " " + txt
        progress_bar.progress(i / len(tasks))

    status_txt.markdown("✅ Přepis dokončen (všechny chunky).")

    st.subheader("📄 Kompletní přepis")
    st.text_area(" ", full_transcript.strip(), height=250)

    # 5) Odešli celý přepis na Make a zobraz bullet-pointy
    with st.spinner("📤 Odesílám přepis na Make…"):
        bullets = post_to_make(full_transcript)

    if bullets:
        st.markdown("---")
        st.subheader("📌 Bullet-points z Make")
        st.markdown(
            "<ul>" + "".join(f"<li>{fmt_bullet(b)}</li>" for b in bullets) + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ Make nevrátil žádné bullet-pointy.")
