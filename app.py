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
  .stApp { background-color: #0a0a0a; }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  .segment-card {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: fadeSlideUp 0.4s ease forwards;
    opacity: 0;
  }
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .segment-card:nth-child(1) { animation-delay: 0.05s; }
  .segment-card:nth-child(2) { animation-delay: 0.10s; }
  .segment-card:nth-child(3) { animation-delay: 0.15s; }
  .segment-card:nth-child(4) { animation-delay: 0.20s; }
  .segment-card:nth-child(5) { animation-delay: 0.25s; }

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
  .segment-title {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.3;
  }
  .segment-note {
    font-size: 13px;
    color: #cccccc;
    line-height: 1.6;
    margin-bottom: 12px;
  }
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
  .step-label {
    font-family: monospace;
    font-size: 12px;
    color: #888888;
    margin-bottom: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 13px;
    color: #666666;
  }
  .stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff;
  }
  .ts { color: #666666; font-family: monospace; font-size: 11px; }
  hr { border-color: #1a1a1a; margin: 24px 0; }
  .muted-meta { color: #888888; font-size: 13px; }
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
    filled = int(round(width * pct / 100.0))
    filled = min(width, max(0, filled))
    return "█" * filled + "░" * (width - filled)


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

st.title("Demo Walkthrough Summarizer")

with st.sidebar:
    st.markdown("### 🎬 Demo Summarizer")
    st.markdown("---")

    over_limit = False
    over_size = False
    uploaded = st.file_uploader(
        "Upload recording (MP4, max ~2GB / 60 min)",
        type=["mp4"],
    )

    probe_dur: float | None = None
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("❌ File exceeds ~2GB. Compress the video and try again.")
            over_size = True
        else:
            probe_dur = _probe_upload_duration(uploaded)

    if uploaded is not None and not over_size and probe_dur is not None:
        mins = int(probe_dur // 60)
        secs = int(probe_dur % 60)
        if probe_dur > MAX_DURATION_SEC:
            st.error(f"❌ {mins}m {secs}s — exceeds 60 min limit")
            over_limit = True
        else:
            st.success(f"✅ {mins}m {secs}s — within limits")

    st.markdown("---")
    st.markdown("**Transcription**")
    whisper_size = st.select_slider(
        "Whisper model",
        options=["tiny", "base", "small"],
        value="base",
    )

    st.markdown("**Scene detection**")
    threshold = st.slider("Sensitivity", 10, 50, 25)
    min_gap = st.slider("Min gap (seconds)", 5, 20, 8)

    st.markdown("---")
    st.markdown(f"**Model:** `{ollama_model}`")

    run = st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None or over_limit or over_size,
    )

    progress_slot = st.empty()
    status_slot = st.empty()


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
    def show_step(step: int, total: int, label: str, pct: int) -> None:
        bar = _progress_bar_ascii(pct)
        base = int((step - 1) / total * 100)
        span = int(100 / total)
        overall = min(100, base + int(pct * span / 100))
        progress_slot.progress(overall / 100.0)
        status_slot.markdown(
            f'<p class="step-label">Step {step}/{total}: {html.escape(label)}  '
            f"<code>{bar}</code>  <b>{pct}%</b></p>",
            unsafe_allow_html=True,
        )

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
            st.session_state["selected_segment"] = None
            status.update(label="Complete", state="complete")
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Analysis failed: {exc}")
            return


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
    st.info(
        "Upload an MP4 (≤ 60 min, ≤ ~2 GB). Configure options in the sidebar, then run analysis."
    )
    st.stop()

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
    st.subheader("Summaries by segment")
    _ensure_edit_fields(summaries)
    selected = st.session_state.get("selected_segment")
    for i, s in enumerate(summaries):
        is_sel = selected is not None and i == int(selected)
        if is_sel:
            st.success(f"Selected — {s.get('label', '')}")
        col1, col2 = st.columns([1, 2])
        with col1:
            p = s.get("screenshot_path") or ""
            if p and Path(p).is_file():
                st.image(p, use_container_width=True)
        with col2:
            label_e = html.escape(str(s.get("label", "")))
            title_e = html.escape(str(s.get("title", "")))
            st.markdown(
                f'<div class="segment-card">'
                f'<div class="timestamp-badge">{label_e}</div>'
                f'<div class="segment-title">{title_e}</div></div>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "Summary",
                key=f"edit_summary_{i}",
                height=90,
                label_visibility="collapsed",
            )
            st.markdown("**Steps**")
            steps_list = s.get("steps") if isinstance(s.get("steps"), list) else []
            for j in range(len(steps_list)):
                st.text_area(
                    f"Step {j + 1}",
                    key=f"edit_step_{i}_{j}",
                    height=68,
                    label_visibility="collapsed",
                )
            wim = s.get("why_it_matters")
            if wim:
                st.markdown(
                    f'<div class="segment-note"><b>Why it matters:</b> '
                    f"{html.escape(str(wim))}</div>",
                    unsafe_allow_html=True,
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
