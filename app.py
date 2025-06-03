# audio_upload_whisper_chunking_shorter.py
"""
Streamlit: Upload → Rozdělení na WAV chunks po 30 s → Komprese každého kousku → Whisper → Make → Bullet-points
===========================================================================================================

1) Uživatel nahraje audio (MP3/WAV/M4A), třeba i 1h dlouhé (až ~200 MB).
2) Celé audio se nejprve rozdělí na WAV kousky po 30 sekundách:
   • Každý kousek je validní WAV s PCM 16 kHz/mono.
3) Každý wav-kousek se uloží do dočasného souboru a pomocí FFmpeg skrze  
   `ffmpeg -y -i tmp.wav -ar 16000 -ac 1 -b:a 32k -f mp3 pipe:1`  
   zkonvertuje na MP3 (mono, 16 kHz, 32 kb/s). Obvykle ~1 MB na 30 s audio.
4) Každý MP3 kousek se pošle do Whisper API → vrátí fragment textu.  
   Všechny fragmenty se spojí do `full_transcript`.
5) Jednou zaviňkem se pošle `full_transcript` do Make webhook → Make vrátí  
   JSON pole bullet-pointů. Ty vykreslíme jako jednoduchý flipchart.

Požadavky:
-----------
pip install streamlit openai requests

Ve Streamlit Cloudu přidej do `packages.txt`:
    ffmpeg
"""

from __future__ import annotations
import io, re, logging, os, subprocess, tempfile, requests, streamlit as st
from openai import OpenAI, OpenAIError

# ───────── Konfigurace ───────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

MAKE_URL   = "https://hook.eu2.make.com/k08ew9w6ozdfougyjg917nzkypgq24f7"
MAKE_TOKEN = st.secrets.get("WEBHOOK_OUT_TOKEN", "demo-token")

TARGET_SR       = 16_000   # výsledné vzorkování 16 kHz
TARGET_CHANNELS = 1        # mono
TARGET_BR       = "32k"    # 32 kb/s MP3
CHUNK_SEC       = 30       # délka jednoho audio-kusu (ve vteřinách)

logging.basicConfig(level=logging.INFO)

# ───────── Funkce: Rozdělení původního audio souboru na WAV chunks ───────
def split_input_to_wav_chunks(raw: bytes, ext: str) -> list[bytes]:
    """
    1) Zapíše raw audio (MP3/WAV/M4A) do dočasného souboru (tmp_in.ext).
    2) Pomocí FFmpeg dekóduje celý soubor do WAV (PCM16, 16 kHz, mono) do paměti 
       (pipe:1).  
    3) Z WAV binárních dat oddělí hlavičku (44 B) a PCM data a rozdělí PCM na bloky 
       po CHUNK_SEC sekundách (bytes_per_chunk = TARGET_SR * 2 B * CHUNK_SEC).
    4) Každý PCM blok zabalí zpět s drobnou WAV hlavičkou (RIFF, velikost, atp.) 
       a vrátí seznam byte‐řetězců, kde každý je validní WAV (PCM).
    """
    # a) Uložíme raw data do dočasného souboru s příponou ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        in_path = tmp_in.name

    # b) Dekodujeme full audio na WAV PCM (16 kHz, mono)
    cmd = [
        "ffmpeg",
        "-i", in_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-f", "wav",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_data, stderr = proc.communicate()
    if proc.returncode != 0:
        os.unlink(in_path)
        st.error(
            "❌ Chyba při dekódování na WAV (FFmpeg):\n"
            + stderr.decode(errors="ignore")
        )
        raise RuntimeError("FFmpeg decode failed")
    os.unlink(in_path)

    # c) Oddělíme hlavičku (44 B) a PCM data (int16)
    header = wav_data[:44]
    pcm = wav_data[44:]

    # d) Vypočteme, kolik bajtů PCM na CHUNK_SEC
    bytes_per_chunk = TARGET_SR * 2 * CHUNK_SEC
    wav_chunks: list[bytes] = []

    # e) Rozdělíme PCM na bloky po bytes_per_chunk
    for i in range(0, len(pcm), bytes_per_chunk):
        block = pcm[i : i + bytes_per_chunk]
        # Vytvoříme novou WAV hlavičku pro tento PCM blok:
        riff = header[:4]  # "RIFF"
        size_chunk = 36 + len(block)  # 36 = header bez prvních 8 B, + data_len
        riff += size_chunk.to_bytes(4, "little")
        rest_header = header[8:44]  # zbývající část “WAVEfmt … data …” hlavičky
        wav_block = riff + rest_header + block
        wav_chunks.append(wav_block)

    return wav_chunks

# ───────── Funkce: Komprese WAV → MP3 (mono, 16 kHz, 32 kb/s) ─────────────
def compress_wav_to_mp3(wav_bytes: bytes) -> bytes:
    """
    1) WAV uložíme do tmp.wav
    2) Spustíme FFmpeg: mp3 (mono, 16 kHz, 32 kb/s), výstup do stdout (pipe:1)
    3) Vrátíme byty MP3 a smažeme dočasný soubor
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(wav_bytes)
        tmp_wav.flush()
        wav_path = tmp_wav.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-b:a", TARGET_BR,
        "-f", "mp3",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mp3_data, stderr = proc.communicate()
    if proc.returncode != 0:
        os.unlink(wav_path)
        st.error(
            "❌ Chyba při kompresi WAV→MP3 (FFmpeg):\n"
            + stderr.decode(errors="ignore")
        )
        raise RuntimeError("FFmpeg compress failed")
    os.unlink(wav_path)
    return mp3_data

# ───────── Funkce Whisper & Make ────────────────────────────────────────────
def whisper_transcribe(b: bytes, filename: str) -> str:
    """
    Pošle byty (MP3 nebo WAV) do Whisper (model whisper-1). Vrátí přepis.
    Zvedne OpenAIError při chybě a vypíše ji v UI.
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
    Pošle kompletní přepis na Make webhook, vrátí seznam stringů (bullet-pointů).
    Při chybě vrátí prázdný list.
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

# ───────── Helper pro formátování bulletů ─────────────────────────────────
DASH  = re.compile(r"\s+-\s+")
STRIP = "-–—• "
def fmt_bullet(raw: str) -> str:
    """
    * První řádek nadpis (TUČNĚ, uppercase).
    * Další řádky (popisné `- detail`) → <li>detail</li>.
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

# ───────── UI – Streamlit layout ──────────────────────────────────────────
st.set_page_config("Audio → Whisper → Make Demo", layout="centered")
st.title("🎙️ Audio Upload → Chunk(30 s) → Komprese → Whisper → Make → Bullet-points")

uploaded = st.file_uploader(
    "➕ Nahraj MP3 / WAV / M4A (až ~200 MB)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    raw_bytes = uploaded.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    st.write(f"Velikost nahrávky: **{size_mb:.1f} MB**")

    # 1) Rozděl celý upload na WAV chunks po 30 s
    ext = uploaded.name.split('.')[-1].lower()  # mp3, wav, m4a
    with st.spinner("🔀 Rozděluji audio na WAV kousky po 30 s…"):
        try:
            wav_chunks = split_input_to_wav_chunks(raw_bytes, ext=ext)
        except Exception:
            st.stop()
    num_chunks = len(wav_chunks)
    st.info(f"Audio rozdělěno do **{num_chunks}** kousků (každý ≈ {CHUNK_SEC} s).")

    # 2) Pro každý WAV chunk:
    #    a) zkomprimuj na MP3 (mono, 16 kHz, 32 kbit/s)
    #    b) pošli do Whisperu
    full_transcript = ""
    progress_bar = st.progress(0)
    status_txt = st.empty()

    for i, wav_byt in enumerate(wav_chunks, start=1):
        # Zobrazíme, jak dlouhý je tento chunk (v sekundách), pro debug
        data_len = len(wav_byt) - 44  # bez hlavičky
        sec_len = data_len / (TARGET_SR * 2)
        status_txt.markdown(
            f"🔄 Chunk **{i}/{num_chunks}** (délka ~{sec_len:.1f} s): komprimace → Whisper…"
        )

        # a) Komprese WAV → MP3
        with st.spinner(f"🔂 Komprese chunk {i}/{num_chunks}…"):
            try:
                mp3_byt = compress_wav_to_mp3(wav_byt)
            except Exception:
                st.stop()

        # b) Přepis do Whisperu
        with st.spinner(f"⏳ Whisper chunk {i}/{num_chunks}…"):
            try:
                txt = whisper_transcribe(mp3_byt, filename=f"chunk_{i}.mp3")
            except OpenAIError:
                st.stop()
        full_transcript += " " + txt

        # Aktualizujeme progress bar
        progress_bar.progress(i / num_chunks)

    status_txt.markdown("✅ Přepsáno všech **chucků**.")

    # 3) Zobrazíme kompletní přepis
    st.subheader("📄 Kompletní přepis")
    st.text_area(" ", full_transcript.strip(), height=300)

    # 4) Odešli jednorázově celý přepis na Make a zobraz output
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
