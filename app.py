from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from video_analyzer.models import format_timestamp
from video_analyzer.providers import create_refiner
from video_analyzer.service import VideoAnalysisService

st.set_page_config(page_title="Framewise · Video Intelligence", page_icon="◉", layout="wide")
st.markdown("""<style>
@keyframes scan{from{transform:translateX(-120%)}to{transform:translateX(600%)}}.stApp{background:#080a10;color:#f5f7ff}
.hero{position:relative;overflow:hidden;padding:2.2rem;border-radius:24px;background:linear-gradient(120deg,#151a2d,#241443);border:1px solid #8b5cf644}.hero:after{content:'';position:absolute;inset:0;width:18%;background:linear-gradient(90deg,transparent,#a78bfa20,transparent);animation:scan 6s infinite}
.kicker{color:#a78bfa;font-weight:800;letter-spacing:.2em}.chapter{padding:1rem 1.2rem;border-radius:16px;background:#111521;border:1px solid #ffffff16;margin:.6rem 0}.time{font-family:monospace;color:#22d3ee}
[data-testid="stSidebar"]{background:#0d1018}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style><div class="hero"><div class="kicker">FRAMEWISE</div><h1>See the structure inside every video.</h1><p>Timestamped chapters, searchable ideas, visual context, and exportable intelligence.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analyze")
    mode = st.radio("Input", ["Transcript", "Video (optional models)"])
    if mode == "Transcript":
        upload = st.file_uploader("JSON, SRT, VTT, or TXT", ["json", "srt", "vtt", "txt"])
    else:
        upload = st.file_uploader("MP4, MOV, MP3, WAV", ["mp4", "mov", "mp3", "wav", "m4a"])
        with_visuals = st.checkbox("Caption sampled frames with BLIP", value=False)
    run = st.button("Analyze content", type="primary", use_container_width=True)
    refiner_options = ["local", "openai", "gemini"]
    configured_refiner = os.getenv("ANALYSIS_REFINER", "local").lower()
    refiner_name = st.selectbox("Chapter segmentation", refiner_options, index=refiner_options.index(configured_refiner) if configured_refiner in refiner_options else 0)
    st.caption("Transcript mode is fully offline and lightweight. Video mode downloads/runs configured local models.")

if run and upload:
    suffix = Path(upload.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getbuffer())
        temp_path = Path(handle.name)
    try:
        with st.status("Building the timeline…", expanded=True) as status:
            service = VideoAnalysisService(refiner=create_refiner(refiner_name))
            if mode == "Transcript":
                result = service.from_transcript(temp_path)
            else:
                from video_analyzer.providers import BlipSceneCaptioner, WhisperTranscriber
                transcriber = WhisperTranscriber(
                    model_size=os.getenv("WHISPER_MODEL", "base"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                )
                captioner = BlipSceneCaptioner(os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")) if with_visuals else None
                interval = float(os.getenv("FRAME_INTERVAL_SECONDS", "15"))
                result = service.from_video(temp_path, transcriber=transcriber, captioner=captioner, frame_interval=interval)
            status.update(label="Analysis ready", state="complete")
        st.session_state.analysis = result
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)
elif run:
    st.warning("Choose a file first.")

result = st.session_state.get("analysis")
if result:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", format_timestamp(result.duration))
    c2.metric("Chapters", len(result.chapters))
    c3.metric("Transcript cues", result.transcript_segments)
    c4.metric("Visual scenes", result.scene_count)
    st.markdown("### Executive brief")
    st.write(result.executive_summary)
    st.markdown("### Chapter navigator")
    for chapter in result.chapters:
        with st.expander(f"{format_timestamp(chapter.start)} · {chapter.title}", expanded=len(result.chapters) <= 4):
            st.write(chapter.summary)
            st.markdown("**Key moments**")
            for point in chapter.key_points:
                st.markdown(f"- {point}")
            st.caption(f"Keywords: {', '.join(chapter.keywords)} · Tone: {chapter.sentiment}")
            if chapter.scenes:
                st.markdown("**Scene captions**")
                for scene in chapter.scenes:
                    st.write(f"`{format_timestamp(scene.timestamp)}` {scene.caption}")
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    st.download_button("Download structured JSON", payload, "video-analysis.json", "application/json")
else:
    st.info("Try the included sample: upload `data/sample_transcript.json`.")
