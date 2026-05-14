"""Streamlit UI — Demo Walkthrough Summarizer v3 (dark theme, parallel summaries)."""

from __future__ import annotations

import html
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
os.chdir(_APP_DIR)

from utils.aligner import align
from utils import exporter
from utils.scene_detector import detect_scenes
from utils.summarizer import (
    check_ollama_running,
    get_available_models,
    pick_summary_model,
    summarize_all,
)
from utils.transcriber import extract_audio, transcribe

st.set_page_config(
    layout="wide",
    page_title="Demo Walkthrough Summarizer",
    page_icon="🎬",
)

V3_CSS = """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  .stApp {
    background:
      radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 28%),
      radial-gradient(circle at top right, rgba(168,85,247,0.10), transparent 25%),
      #080808;
  }

  .block-container {
    max-width: 1400px;
    padding-top: 3rem;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    animation: pageFade 0.45s ease-in-out;
  }

  /* Hero title */
  h1 {
    font-size: 3.2rem !important;
    letter-spacing: -0.04em;
    font-weight: 800 !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(15, 15, 15, 0.92);
    border-right: 1px solid #242424;
  }

  section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
  }

  /* Upload box */
  [data-testid="stFileUploader"] section {
    background: #111827;
    border: 1px dashed #3b82f6;
    border-radius: 14px;
    padding: 18px;
  }

  /* Info box */
  [data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(96,165,250,0.25);
    background: rgba(30,64,175,0.22);
  }

  /* Buttons */
  .stButton > button,
  .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid #333;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 700;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(59,130,246,0.25);
  }

  /* Sliders */
  .stSlider [data-baseweb="slider"] {
    padding-top: 10px;
  }

  /* Cards — no border, hover handled on Streamlit row wrapper */
  .segment-card {
    background: rgba(17, 24, 39, 0.82);
    border: none;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
  }

  /* Hover: target the Streamlit horizontal row that contains a segment card */
  [data-testid="stHorizontalBlock"]:has(.segment-card) {
    border-radius: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
  }

  [data-testid="stHorizontalBlock"]:has(.segment-card):hover {
    transform: translateY(-4px);
    box-shadow: 0 24px 50px rgba(0,0,0,0.35), 0 0 24px rgba(59,130,246,0.2);
  }

  /* Text areas */
  textarea {
    border-radius: 12px !important;
    background: #0f172a !important;
    border: 1px solid #334155 !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 10px;
  }

  .stTabs [data-baseweb="tab"] {
    background: #111827;
    border-radius: 999px;
    padding: 8px 18px;
    font-size: 13px;
    color: #666666;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white !important;
    border-bottom: none;
  }

  /* Timestamp badge */
  .timestamp-badge {
    font-family: monospace;
    font-size: 11px;
    color: #888888;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 10px;
  }

  /* Segment title */
  .segment-title {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.3;
  }

  /* Segment note / why it matters */
  .segment-note {
    font-size: 13px;
    color: #cccccc;
    line-height: 1.6;
    margin-bottom: 12px;
  }

  /* Transcript timestamps */
  .ts { color: #666666; font-family: monospace; font-size: 11px; }

  /* Muted meta text */
  .muted-meta { color: #888888; font-size: 13px; }

  hr { border-color: #1a1a1a; margin: 24px 0; }

  /* Hero text */
  .hero-subtitle {
    font-size: 1.05rem;
    line-height: 1.65;
    color: #a1a1aa;
    margin: 0 0 1.5rem 0;
    max-width: 36rem;
  }

  .hero-meta {
    font-size: 0.85rem;
    color: #71717a;
    margin-top: 0.25rem;
  }

  .hero-fade {
    animation: heroFadeIn 0.55s ease-out forwards;
    opacity: 0;
    margin-bottom: 0.5rem;
  }

  /* Sidebar session label */
  .sidebar-session-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #71717a;
    margin-bottom: 0.35rem;
  }

  /* Step progress label */
  .step-label {
    font-family: monospace;
    font-size: 12px;
    color: #888888;
    margin-bottom: 4px;
  }

  /* Timeline thumbnails */
  .timeline-thumb {
    cursor: pointer;
    border-radius: 6px;
    border: 1px solid #222;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }

  .timeline-thumb:hover {
    transform: translateY(-3px);
    border-color: #555;
  }

  /* Animations */
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes pageFade {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Landing — single panel (no Streamlit border); lift + glow on hover */
  .hero-card-panel {
    background: linear-gradient(165deg, rgba(24,24,27,0.96), rgba(12,12,14,0.99));
    border: 1px solid rgba(63,63,70,0.5);
    border-radius: 18px;
    padding: 1.75rem 2rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
  }
  .hero-card-panel:hover {
    transform: translateY(-4px);
    box-shadow: 0 22px 56px rgba(0,0,0,0.5), 0 0 40px rgba(99,102,241,0.14);
    border-color: rgba(129,140,248,0.35);
  }
  .hero-card-panel .hero-title-html {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #fafafa;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
  }

  /* Results dashboard — tighter chrome */
  .results-dedicated .block-container {
    padding-top: 0.75rem !important;
    max-width: 1320px !important;
  }

  @keyframes resultsEnter {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Summaries — documentation layout */
  .segment-doc-block {
    margin-bottom: 3.5rem;
    padding-bottom: 2.75rem;
    border-bottom: 1px solid #27272a;
  }
  .segment-doc-block:last-child {
    border-bottom: none;
    margin-bottom: 1rem;
  }
  .segment-doc-title {
    font-size: 1.45rem;
    font-weight: 650;
    color: #fafafa;
    letter-spacing: -0.025em;
    margin: 0 0 0.4rem 0;
    line-height: 1.22;
  }
  .segment-doc-summary {
    font-size: 1.08rem;
    line-height: 1.75;
    color: #d4d4d8;
    margin: 0.85rem 0 1.35rem 0;
  }
  .segment-steps-wrap ol.segment-steps-list {
    margin: 0.35rem 0 0 0;
    padding-left: 1.4rem;
    font-size: 1.08rem;
    line-height: 1.8;
    color: #e4e4e7;
  }
  .segment-steps-wrap ol.segment-steps-list li {
    margin-bottom: 0.65rem;
  }
  .segment-wim {
    font-size: 1rem;
    color: #a1a1aa;
    margin-top: 1.25rem;
    line-height: 1.6;
  }
</style>
"""

st.markdown(V3_CSS, unsafe_allow_html=True)

MAX_DURATION_SEC = 3600.0
MAX_UPLOAD_BYTES = 2000 * 1024 * 1024


def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps > 0 and frame_count > 0:
        d = float(frame_count) / float(fps)
        cap.release()
        return d
    cap.set(cv2.CAP_PROP_POS_MSEC, 1e9)
    cap.read()
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
    cap.release()
    return float(pos_ms) / 1000.0 if pos_ms > 0 else 0.0


def _ensure_output_dirs() -> None:
    for sub in ("screenshots", "transcripts", "summaries"):
        Path("outputs", sub).mkdir(parents=True, exist_ok=True)


def _clear_outputs_for_new_run() -> None:
    for sub in ("screenshots", "transcripts", "summaries"):
        d = Path("outputs") / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _startup_checks() -> None:
    if not shutil.which("ffmpeg"):
        st.error(
            "❌ ffmpeg is not installed.\n\n"
            "**Fix:** `brew install ffmpeg`"
        )
        st.stop()

    if not check_ollama_running():
        st.error(
            "❌ Ollama is not running.\n\n"
            "**Fix:** open a terminal and run → `ollama serve`"
        )
        st.stop()

    models = get_available_models()
    chosen = pick_summary_model(models)
    if not chosen:
        st.error(
            "❌ No compatible model found. You need llava (recommended) or llama3.\n\n"
            "**Fix:** `ollama pull llava`  \n or: `ollama pull llama3`"
        )
        st.stop()

    st.session_state["ollama_model"] = chosen


def _init_session_state() -> None:
    defaults = {
        "transcripts": [],
        "scenes": [],
        "aligned": [],
        "summaries": [],
        "selected_segment": None,
        "video_filename": "",
        "video_duration": 0.0,
        "analysis_complete": False,
        "probe_signature": None,
        "probe_duration_sec": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _progress_bar_ascii(pct: int, width: int = 10) -> str:
    return ""  # no longer used

def show_step(step: int, total: int, label: str, pct: int) -> None:
    base = int((step - 1) / total * 100)
    span = int(100 / total)
    overall = min(100, base + int(pct * span / 100))
    progress_slot.progress(overall / 100.0)
    status_slot.markdown(
        f"""
        <div style="font-family:monospace; font-size:13px; color:#ffffff; margin-top:6px;">
            <span style="color:#71717a;">Step {step}/{total}</span>
            &nbsp;·&nbsp;
            <span>{html.escape(label)}</span>
            &nbsp;·&nbsp;
            <span style="color:#71717a;">{pct}%</span>
        </div>
        <div style="margin-top:8px; height:3px; border-radius:999px; background:#1e293b; overflow:hidden;">
            <div style="height:100%; width:{overall}%; background:#ffffff; border-radius:999px; transition:width 0.3s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _format_ts(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _clear_edit_session_keys() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (
            k.startswith("edit_summary_") or k.startswith("edit_step_")
        ):
            del st.session_state[k]


def _reset_to_landing() -> None:
    st.session_state["analysis_complete"] = False
    st.session_state["transcripts"] = []
    st.session_state["scenes"] = []
    st.session_state["aligned"] = []
    st.session_state["summaries"] = []
    st.session_state["selected_segment"] = None
    st.session_state["video_filename"] = ""
    st.session_state["video_duration"] = 0.0
    st.session_state["probe_signature"] = None
    st.session_state["probe_duration_sec"] = None
    _clear_edit_session_keys()
    st.rerun()


def _ensure_edit_fields(summaries: list) -> None:
    for i, s in enumerate(summaries):
        sk = f"edit_summary_{i}"
        if sk not in st.session_state:
            raw = s.get("summary", s.get("note", ""))
            st.session_state[sk] = str(raw) if raw is not None else ""
        steps = s.get("steps") if isinstance(s.get("steps"), list) else []
        for j, step in enumerate(steps):
            tk = f"edit_step_{i}_{j}"
            if tk not in st.session_state:
                st.session_state[tk] = str(step)


def _build_summary_edited_md(summaries: list) -> str:
    parts: list[str] = ["# Demo Walkthrough Summary (edited)\n", "\n"]
    for i, s in enumerate(summaries):
        label = s.get("label", "")
        title = s.get("title", "")
        summary = st.session_state.get(
            f"edit_summary_{i}", s.get("summary", s.get("note", ""))
        )
        steps_src = s.get("steps") if isinstance(s.get("steps"), list) else []
        parts.append(f"## [{label}] {title}\n\n")
        parts.append(f"**Summary:** {summary}\n\n")
        parts.append("**Steps:**\n")
        if not steps_src:
            parts.append("_No steps._\n\n")
        else:
            for j in range(len(steps_src)):
                line = st.session_state.get(
                    f"edit_step_{i}_{j}", str(steps_src[j])
                )
                parts.append(f"{j + 1}. {line}\n")
            parts.append("\n")
        wim = s.get("why_it_matters")
        if wim:
            parts.append(f"**Why it matters:** {wim}\n\n")
        parts.append("---\n\n")
    return "".join(parts)


def _probe_upload_duration(uploaded_file) -> float | None:
    if uploaded_file is None:
        st.session_state["probe_signature"] = None
        st.session_state["probe_duration_sec"] = None
        return None
    sig = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("probe_signature") == sig and st.session_state.get(
        "probe_duration_sec"
    ) is not None:
        return float(st.session_state["probe_duration_sec"])

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        duration = get_video_duration(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    st.session_state["probe_signature"] = sig
    st.session_state["probe_duration_sec"] = duration
    return duration


_ensure_output_dirs()
_init_session_state()
_startup_checks()

ollama_model = st.session_state.get("ollama_model", "llava")

_analysis_done = bool(st.session_state.get("analysis_complete"))

# Full-width app: no sidebar (controls live in hero or results top bar)
st.markdown(
    "<style>"
    'section[data-testid="stSidebar"]{display:none!important;}'
    '[data-testid="collapsedControl"]{display:none!important;}'
    "</style>",
    unsafe_allow_html=True,
)

# Defaults for `if run` (both flows assign before use)
uploaded = None
probe_dur = None
over_limit = False
over_size = False
whisper_size = "base"
threshold = 25
min_gap = 8
run = False
progress_slot = None
status_slot = None


def _render_analysis_controls(*, uploader_key: str = "upload_recording") -> tuple:
    """Upload + sliders + duration checks + Run + progress placeholders. Returns control tuple."""
    ol = False
    osz = False
    up = st.file_uploader(
        "Upload recording (MP4, max ~2GB / 60 min)",
        type=["mp4"],
        key=uploader_key,
    )
    pd: float | None = None
    if up is not None:
        if up.size > MAX_UPLOAD_BYTES:
            st.error("❌ File exceeds ~2GB. Compress the video and try again.")
            osz = True
        else:
            pd = _probe_upload_duration(up)

    if up is not None and not osz and pd is not None:
        mins = int(pd // 60)
        secs = int(pd % 60)
        if pd > MAX_DURATION_SEC:
            st.error(f"❌ {mins}m {secs}s — exceeds 60 min limit")
            ol = True
        else:
            st.success(f"✅ {mins}m {secs}s — within limits")

    st.markdown("**Transcription**")
    ws = st.select_slider(
        "Whisper model",
        options=["tiny", "base", "small"],
        value="base",
    )
    st.markdown("**Scene detection**")
    th = st.slider("Sensitivity", 10, 50, 25)
    mg = st.slider("Min gap (seconds)", 5, 20, 8)

    rn = st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=up is None or ol or osz,
    )
    ps = st.empty()
    ss = st.empty()
    return up, pd, ol, osz, ws, th, mg, rn, ps, ss


if not _analysis_done:
    st.markdown('<div class="landing-main">', unsafe_allow_html=True)
    _, hero_col, _ = st.columns([1, 2.2, 1])
    with hero_col:
        st.markdown('<div class="hero-fade">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-card-panel">'
            '<h1 class="hero-title-html">Demo Walkthrough Summarizer</h1>'
            '<p class="hero-subtitle">Turn walkthrough recordings into transcripts, scene '
            "screenshots, structured summaries, and exportable notes — all processed locally "
            "on your machine.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        (
            uploaded,
            probe_dur,
            over_limit,
            over_size,
            whisper_size,
            threshold,
            min_gap,
            run,
            progress_slot,
            status_slot,
        ) = _render_analysis_controls(uploader_key="hero_upload")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    vf = st.session_state.get("video_filename") or "—"
    vd = float(st.session_state.get("video_duration") or 0.0)
    vm = int(vd // 60)
    vs = int(vd % 60)
    tb_l, tb_r = st.columns([1, 4])
    with tb_l:
        if st.button("← New analysis", key="results_new_analysis"):
            _reset_to_landing()
    with tb_r:
        st.markdown(
            f'<p class="muted-meta" style="margin-top:0.35rem;">'
            f"<b>File:</b> {html.escape(str(vf))} &nbsp;·&nbsp; "
            f"<b>Duration:</b> {vm}m {vs}s &nbsp;·&nbsp; "
            f"<b>Model:</b> <code>{html.escape(str(ollama_model))}</code></p>",
            unsafe_allow_html=True,
        )


def _run_analysis(
    video_path: str,
    filename: str,
    video_duration_sec: float,
    whisper_size: str,
    threshold: int,
    min_gap: int,
    ollama_model: str,
    progress_slot,
    status_slot,
) -> None:
    if video_duration_sec > MAX_DURATION_SEC:
        st.error(
            f"❌ Video is {video_duration_sec / 60:.0f} minutes long. Maximum allowed is 60 minutes. "
            "Please trim the video and re-upload."
        )
        return

    _clear_outputs_for_new_run()

    with st.status("Running pipeline…", expanded=True) as status:
        try:
            show_step(1, 4, "Extracting audio…", 5)
            wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(wav_fd)
            extract_audio(video_path, wav_path)
            show_step(1, 4, "Extracting audio…", 100)

            def on_transcribe(step: str, pct: int) -> None:
                status.update(label=f"Transcription: {step}")
                show_step(2, 4, "Transcribing with Whisper", pct)

            segments = transcribe(
                wav_path, model_size=whisper_size, on_progress=on_transcribe
            )
            try:
                os.unlink(wav_path)
            except OSError:
                pass

            def on_scenes(step: str, pct: int) -> None:
                status.update(label=f"Scenes: {step}")
                show_step(3, 4, "Detecting scene changes", pct)

            scenes = detect_scenes(
                video_path,
                threshold=float(threshold),
                min_gap=float(min_gap),
                max_scenes=35,
                on_progress=on_scenes,
            )

            aligned = align(scenes, segments)

            def on_summary(step: str, pct: int) -> None:
                status.update(label=f"Summaries: {step}")
                show_step(4, 4, "Generating AI summaries", pct)

            summaries = summarize_all(
                aligned,
                model=ollama_model,
                on_progress=on_summary,
            )

            exporter.write_markdown_exports(summaries, filename, video_duration_sec)

            st.session_state["transcripts"] = segments
            st.session_state["scenes"] = scenes
            st.session_state["aligned"] = aligned
            st.session_state["summaries"] = summaries
            _clear_edit_session_keys()
            st.session_state["video_filename"] = filename
            st.session_state["video_duration"] = video_duration_sec
            st.session_state["analysis_complete"] = True
            st.session_state["_results_page_enter"] = True
            st.session_state["selected_segment"] = None
            status.update(label="Complete", state="complete")
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Analysis failed: {exc}")
            return
    st.rerun()


if run:
    if not uploaded:
        st.error("Please upload an MP4 file first.")
    elif over_limit:
        st.error("Video exceeds the 60-minute limit.")
    elif over_size:
        st.error("File exceeds the upload size limit.")
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds ~2GB.")
            else:
                dur = get_video_duration(tmp_path)
                if dur > MAX_DURATION_SEC:
                    st.error(
                        f"❌ Video is {dur / 60:.0f} minutes long. Maximum allowed is 60 minutes. "
                        "Please trim the video and re-upload."
                    )
                else:
                    _run_analysis(
                        tmp_path,
                        uploaded.name,
                        dur,
                        whisper_size,
                        threshold,
                        min_gap,
                        ollama_model,
                        progress_slot,
                        status_slot,
                    )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

if not st.session_state.get("analysis_complete"):
    st.stop()

_enter = st.session_state.pop("_results_page_enter", False)
_anim = (
    "animation: resultsEnter 0.72s ease-out forwards !important;"
    if _enter
    else ""
)
st.markdown(
    f"<style>.main .block-container {{ padding-top: 0.85rem !important; {_anim} }}</style>",
    unsafe_allow_html=True,
)

tab_timeline, tab_transcript, tab_summaries, tab_export = st.tabs(
    ["🗂 Timeline", "📝 Transcript", "🧠 Summaries", "📦 Export"]
)

summaries = st.session_state["summaries"]
segments = st.session_state["transcripts"]

with tab_timeline:
    st.subheader("Scene thumbnails")
    scenes = st.session_state["scenes"]
    if not scenes:
        st.write("No scenes detected.")
    else:
        st.markdown(
            '<div style="overflow-x:auto;padding-bottom:8px;">',
            unsafe_allow_html=True,
        )
        n = len(scenes)
        batch = 6
        for start in range(0, n, batch):
            cols = st.columns(batch)
            for j, col in enumerate(cols):
                idx = start + j
                if idx >= n:
                    break
                sc = scenes[idx]
                with col:
                    if Path(sc["path"]).is_file():
                        st.image(sc["path"], use_container_width=True)
                    st.caption(sc["label"])
                    if st.button("Open summary", key=f"tl_btn_{idx}"):
                        st.session_state["selected_segment"] = idx
                        st.toast("Open the **Summaries** tab for this segment.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_transcript:
    st.subheader("Full transcript")
    words = []
    parts = []
    for seg in segments:
        t = _format_ts(float(seg["start"]))
        txt = (seg.get("text") or "").strip()
        words.extend(txt.split())
        parts.append(
            f'<span class="ts">[{html.escape(t)}]</span> {html.escape(txt)}'
        )
    wc = len(words)
    minutes = wc / 200.0
    est = "under 1 min" if wc == 0 or minutes < 1 else f"~{max(1, int(round(minutes)))} min"
    st.markdown(
        f'<p class="muted-meta"><b>Words:</b> {wc} · <b>Est. reading time:</b> {html.escape(est)}</p>',
        unsafe_allow_html=True,
    )
    body = "\n\n".join(parts)
    st.markdown(
        f'<div style="max-height:480px;overflow-y:auto;">{body}</div>',
        unsafe_allow_html=True,
    )

with tab_summaries:
    _ensure_edit_fields(summaries)
    selected = st.session_state.get("selected_segment")
    for i, s in enumerate(summaries):
        is_sel = selected is not None and i == int(selected)
        label_e = html.escape(str(s.get("label", "")))
        title_e = html.escape(str(s.get("title", "")))
        summary_text = st.session_state.get(
            f"edit_summary_{i}", str(s.get("summary", s.get("note", "")) or "")
        )
        steps_src = s.get("steps") if isinstance(s.get("steps"), list) else []
        step_items: list[str] = []
        for j in range(len(steps_src)):
            step_items.append(
                st.session_state.get(
                    f"edit_step_{i}_{j}", str(steps_src[j] if steps_src[j] is not None else "")
                )
            )
        ol_parts = []
        for j, line in enumerate(step_items):
            ol_parts.append(f"<li>{html.escape(line)}</li>")
        ol_html = (
            f'<ol class="segment-steps-list">{"".join(ol_parts)}</ol>'
            if ol_parts
            else '<p class="muted-meta" style="margin:0;">No steps.</p>'
        )

        col_thumb, col_doc = st.columns([1, 2])
        with col_thumb:
            p = s.get("screenshot_path") or ""
            if p and Path(p).is_file():
                st.image(p, use_container_width=True)
        with col_doc:
            sel_note = (
                '<p class="muted-meta" style="margin:0 0 0.75rem 0;">Selected segment</p>'
                if is_sel
                else ""
            )
            wim = s.get("why_it_matters")
            wim_html = (
                f'<div class="segment-wim"><b>Why it matters:</b> '
                f"{html.escape(str(wim))}</div>"
                if wim
                else ""
            )
            sum_esc = html.escape(summary_text)
            st.markdown(
                f'<div class="segment-doc-block">{sel_note}'
                f'<p class="timestamp-badge">{label_e}</p>'
                f'<h3 class="segment-doc-title">{title_e}</h3>'
                f'<div class="segment-doc-summary" style="white-space:pre-wrap;">{sum_esc}</div>'
                f'<div class="segment-steps-wrap">{ol_html}</div>'
                f"{wim_html}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Edit summary & steps"):
                st.text_area(
                    "Summary",
                    key=f"edit_summary_{i}",
                    height=120,
                    label_visibility="collapsed",
                )
                st.markdown("**Steps** (one field per step)")
                for j in range(len(steps_src)):
                    st.text_area(
                        f"Step {j + 1}",
                        key=f"edit_step_{i}_{j}",
                        height=72,
                        label_visibility="collapsed",
                    )

with tab_export:
    st.subheader("Downloads")
    sm_path = exporter.SUMMARY_MD
    edited_md = _build_summary_edited_md(summaries)
    st.download_button(
        "📝 Download summary with my edits",
        data=edited_md.encode("utf-8"),
        file_name="summary_edited.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if sm_path.is_file():
        st.download_button(
            "Download summary.md",
            data=sm_path.read_bytes(),
            file_name="summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
    zip_bytes = exporter.create_zip("outputs")
    st.download_button(
        "Download outputs ZIP",
        data=zip_bytes,
        file_name="outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown("#### Preview: summary.md")
    if sm_path.is_file():
        st.code(sm_path.read_text(encoding="utf-8"), language="markdown")
