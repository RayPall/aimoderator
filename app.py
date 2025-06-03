# audio_upload_whisper_chunking.py
"""
Streamlit: Upload → Auto-compress (32 kb/s) → Pokud zůstane >25 MiB, rozděl na časové kusy → Whisper → Make → Bullet-points
=========================================================================================================================
1) Uživatel nahraje audio (MP3/WAV/M4A), max. např. 200 MB.
2) Když je soubor > 25 MiB, spustíme FFmpeg:
   • MP3, mono, 16 kHz, 32 kb/s  
   → typicky výstup < 25 MiB.  
   Pokud ale stále > 25 MiB, musíme dělit podle času.
3) Pokud výsledná MP3 > 25 MiB:
   a) Použijeme FFmpeg pro dekódování té MP3 na WAV / PCM v paměti  
   b) „Kus po kusu“ rozdělíme PCM na danou délku (např. 2–3 min bloky),  
      převedeme každý blok buď přímo do WAV bufferu nebo do MP3 znova  
      (WAV je v pořádku, protože Whisper přijímá WAV nativně).  
   c) Posíláme každý kousek zvlášť do Whisper API → spojíme texty.
4) Vzniklý `full_transcript` se pošle do Make jednokusově.
5) Make vrátí pole bullet-pointů, která vykreslíme.

Požadavky:
----------
streamlit
openai
requests

Do `packages.txt` (Streamlit Cloud):
-------------------------------------
ffmpeg
"""

from __future__ import annotations
import io, re, logging, os, subprocess, tempfile, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ─────────── Konfigurace ─────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

WHISPER_LIMIT   = 25 * 1024 * 1024    # 25 MiB
TARGET_SR       = 16_000              # 16 kHz
TARGET_CHANNELS = 1                   # mono
TARGET_BR       = "32k"               # 32 kb/s (komprese)

# Délka jednoho chunku pro případ, že MP3 > 25 MiB (v sekundách)
# Např. 120 s = 2 minuty; 120 s PCM @ 16 kHz/16bit/mono je ~4 MB,  
# takže se to vejde vč. overheadu. Upravit podle potřeby.
CHUNK_SEC       = 120                 # délka jednoho kusu pro řezání

logging.basicConfig(level=logging.INFO)

# ─────────── Funkce pro „hrubou“ kompresi MP3 CLI (mono, 16 kHz, 32 kb/s) ─────────────
def compress_if_needed(raw: bytes, ext: str) -> bytes:
    """
    Pokud raw ≤ 25 MiB → vrátí raw beze změny.  
    Pokud raw > 25 MiB → zapíše raw do tmp, spustí FFmpeg: MP3, 16 kHz, mono, 32 kb/s → výsledné MP3 vrátí.
    """
    if len(raw) <= WHISPER_LIMIT:
        return raw

    # 1) Uložíme původní audio do tmp souboru
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        in_path = tmp_in.name

    # 2) Komprimujeme pomocí FFmpeg
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

    # 3) Přečteme výsledek a smažeme temp soubory
    with open(out_path, "rb") as f:
        compressed = f.read()
    os.unlink(in_path)
    os.unlink(out_path)
    return compressed

# ─────────── Funkce pro chunking a převod na PCM WAV ───────────────────────────────────
def split_to_wav_chunks(mp3_bytes: bytes) -> list[bytes]:
    """
    1) Zapíše mp3_bytes do dočasného souboru (tmp.mp3).  
    2) Pomocí FFmpeg dekóduje celou MP3 do WAV (pipe:1).  
    3) Ve WAV binárním headeru si vyčteme hlavičku a chunkujeme PCM po CHUNK_SEC.  
       Každý samostatný WAV block vrátíme jako samostatné WAV bytes.
    """
    # a) uložíme mp3_bytes do tmp souboru
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_in:
        tmp_in.write(mp3_bytes)
        tmp_in.flush()
        in_path = tmp_in.name

    # b) dekódujeme do raw WAV do binárního bufferu
    #    ffmpeg -i tmp.mp3 -ar 16000 -ac 1 -f wav pipe:1
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

    # c) teď máme celé WAV v `wav_data`, což vypadá takto:
    #    b'RIFF....WAVEfmt ... data....' (hlavička + PCM)
    #    WAV header = 44 bajtů (obvykle), poté PCM vzorky 16bit int16
    # d) Rozdělíme to na hlavičku (header) a PCM data:
    header = wav_data[:44]         # standardní WAV hlavička 44 bajtů
    pcm = wav_data[44:]            # zbytek jsou 16bit-int16 vzorky

    # e) Kolik PCM bajtů = (SR * 2 B/sample * CHUNK_SEC)
    bytes_per_chunk = TARGET_SR * 2 * CHUNK_SEC

    chunks: list[bytes] = []
    for i in range(0, len(pcm), bytes_per_chunk):
        block = pcm[i : i + bytes_per_chunk]
        # Zabalíme každý PCM blok znova do validní WAV hlavičky + data:
        wav_block = header[:4]                # "RIFF"
        size_chunk = 36 + len(block)          # riff chunk size = 36 + data_len
        wav_block += (size_chunk).to_bytes(4, "little")
        wav_block += header[8:44]             # zbytek hlavičky od "WAVE" po "data"
        wav_block += block                     # PCM data
        chunks.append(wav_block)

    return chunks

# ─────────── Funkce Whisper & Make ─────────────────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty (MP3 nebo WAV) do Whisper (model whisper-1). Vrátí text.
    Zvedne OpenAIError, pokud dojde k chybě.
    """
    try:
        file_like = io.BytesIO(b)
        file_like.name = filename  # OpenAI Whisper lib akceptuje tento attribut
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
    Pokud Make selže → vrátí prázdný list.
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

# ─────────── Helper pro formátování bullet-pointů ───────────────────────────────────
DASH  = re.compile(r"\s+-\s+")
STRIP = "-–—• "
def fmt_bullet(raw: str) -> str:
    """Prvního řádku každého bulletu udělá <strong>...</strong>, další <li>…</li>."""
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

# ─────────── UI – Streamlit layout ─────────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make Demo", layout="centered")
st.title("🎙️ Audio Upload → Auto-Compress → Chunk → Whisper → Make → Bullet-points")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (do ~200 MB)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost souboru: **{size_mb:.1f} MB**")

    # 1) Zkomprimuj (32 kbit/s), pokud >25 MiB
    ext = uploaded.name.split('.')[-1].lower()  # přípona, např. "mp3", "wav", "m4a"
    try:
        compressed = compress_if_needed(raw_bytes, ext=ext)
    except Exception:
        st.stop()
    comp_mb = len(compressed) / (1024 * 1024)
    if comp_mb < size_mb:
        st.info(f"Složil jsem >25 MB → po kompresi: **{comp_mb:.1f} MB**")
    else:
        st.info("Soubor ≤ 25 MiB → posílám originál (bez komprese)")

    # 2) Pokud je komprimované MP3 > 25 MiB, rozděl ho na WAV chunk
    if len(compressed) > WHISPER_LIMIT:
        st.warning("I po kompresi MP3 > 25 MiB → provádím chunking na WAV kousky…")
        with st.spinner("🔪 Rozděluji do WAV chunků…"):
            wav_chunks = split_to_wav_chunks(compressed)
    else:
        wav_chunks = []  # žádné chunky, protože původní MIME je OK (MP3 < limit)

    # 3) Připrav finální seznam „k převodu do Whisperu“:
    #    • pokud máme WAV_chunks, budou je testovat jako jednotlivé WAV bloky
    #    • jinak pošleme originál (nebo komprimované MP3) jako jeden kousek
    tasks: list[tuple[bytes,str]] = []
    if wav_chunks:
        for idx, w in enumerate(wav_chunks, start=1):
            tasks.append((w, f"chunk_{idx}.wav"))
    else:
        # pošli MP3 přímo
        tasks.append((compressed, uploaded.name))

    # 4) Pošli každou část do Whisper API (sčítejme texty dohromady)
    full_transcript = ""
    for i, (chunk_bytes, fname) in enumerate(tasks, start=1):
        with st.spinner(f"⏳ Whisper část {i}/{len(tasks)}…"):
            try:
                txt = whisper_transcribe(chunk_bytes, filename=fname)
            except OpenAIError:
                st.stop()
        full_transcript += " " + txt
        st.success(f"Část {i} hotová")

    st.subheader("📄 Kompletní přepis")
    st.text_area(" ", full_transcript.strip(), height=250)

    # 5) Odešli na Make a vykresli bullet-pointy
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
