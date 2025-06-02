# ai_flipchart_streamlit_whisper_api.py  (jen rozdíly proti předchozí verzi)

AUDIO_BATCH_SECONDS    = 5          # kratší blok
RECEIVER_SIZE_FRAMES   = 1024       # velká fronta (≈ 20 s audia při 48 kHz)

# … (vše až k blokům s webrtc_streamer zůstává beze změny)

    webrtc_ctx = webrtc_streamer(
        key="workshop-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=RECEIVER_SIZE_FRAMES,   # 🆕 1024
        rtc_configuration={...},
        media_stream_constraints={"audio": True, "video": False},
    )

    # ------------------ DVOJVLÁKNOVÁ PIPELINE ---------------------------
    stop_evt = threading.Event(); st.session_state.audio_stop_event = stop_evt
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)   # 8 bloků puffer

    async def reader(ctx):
        SR = 48000; tgt = AUDIO_BATCH_SECONDS*SR*2
        while not stop_evt.is_set():
            if not ctx.audio_receiver:
                await asyncio.sleep(0.05); continue
            frames = await ctx.audio_receiver.get_frames(timeout=1)
            st.session_state.audio_buffer.extend(f.to_ndarray().tobytes() for f in frames)
            if sum(map(len, st.session_state.audio_buffer)) >= tgt:
                wav = pcm_to_wav(st.session_state.audio_buffer)
                st.session_state.audio_buffer.clear()
                try:
                    queue.put_nowait(wav)
                except asyncio.QueueFull:
                    # zahodím nejstarší, aby se fronta hýbala
                    _ = queue.get_nowait()
                    queue.put_nowait(wav)

    async def worker():
        while not stop_evt.is_set():
            wav = await queue.get()
            set_status("🟣 Whisper (worker)…")
            txt = await asyncio.to_thread(whisper_safe, io.BytesIO(wav), "mic")
            if not txt:
                continue
            st.session_state.transcript_buffer += " "+txt
            if len(st.session_state.transcript_buffer.split()) >= 325:
                set_status("📤 Make…")
                new = await asyncio.to_thread(
                    call_make, st.session_state.transcript_buffer, st.session_state.flip_points
                )
                st.session_state.flip_points.extend(
                    [p for p in new if p not in st.session_state.flip_points]
                )
                st.session_state.transcript_buffer = ""
            queue.task_done()

    async def main_pipeline():
        await asyncio.gather(reader(webrtc_ctx), worker())

    t = threading.Thread(target=lambda: asyncio.run(main_pipeline()), daemon=True)
    add_script_run_ctx(t); t.start(); st.session_state.runner_thread = t
