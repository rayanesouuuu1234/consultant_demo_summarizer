# Demo Walkthrough Summarizer (v3)

Fully local Streamlit app: **faster-whisper**, **OpenCV** scene detection, **Ollama** (llava preferred). Summaries are **consultant-style briefing notes** — direct, skimmable, no passive “the speaker says…” framing.

**v3 highlights:** dark UI (theme + injected CSS), **parallel** Ollama calls (`max_workers=3`), faster Whisper (`beam_size=1` + VAD), **adaptive frame sampling** for scene detection, **2GB** upload limit via `.streamlit/config.toml`.

## Prerequisites

```bash
brew install ffmpeg
brew install ollama
ollama pull llava
ollama pull llama3
ollama serve
```

## Compress your video first (recommended)

```bash
ffmpeg -i your_recording.mp4 -vcodec libx264 -crf 28 -preset fast compressed.mp4
```

## Setup

```bash
cd demo-summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repo includes **`.streamlit/config.toml`** (dark theme + `maxUploadSize = 2000` MB). Run from the app directory:

```bash
streamlit run app.py
```

## Limits

| Limit | Value |
|--------|--------|
| Max duration | 60 minutes |
| Max upload (Streamlit) | ~2 GB (config); app also blocks oversized uploads early |

## How it works

1. Upload MP4 → duration and size checks in the sidebar.  
2. **Run analysis** → ffmpeg WAV → Whisper → adaptive-interval scene detection → align transcript windows → **up to 3 parallel** Ollama segment summaries.  
3. Browse **Timeline**, **Transcript**, **Summaries**, **Export**.  
4. Download `summary.md`, `notable_details.md`, or a ZIP of `outputs/`.

## Tuning

- Raise **Sensitivity** if too many scene screenshots; lower if transitions are missed.  
- **tiny** / **small** Whisper for speed vs accuracy.  
- **llava** uses screenshot + transcript; other models are transcript-only with the same JSON shape.
