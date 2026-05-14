"""Audio extraction and local transcription with faster-whisper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

OUTPUT_TRANSCRIPT = Path("outputs/transcripts/transcript.json")


def extract_audio(video_path: str, output_wav: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        output_wav,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def transcribe(
    wav_path: str,
    model_size: str = "base",
    on_progress=None,
) -> list[dict]:
    if on_progress:
        on_progress("Loading Whisper model…", 0)

    whisper_model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    if on_progress:
        on_progress("Transcribing audio…", 10)

    segments_iter, _info = whisper_model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments: list[dict] = []

    for seg in segments_iter:
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )
        if on_progress and len(segments) % 5 == 0:
            on_progress("Transcribing audio…", min(95, 10 + len(segments) // 2))

    if on_progress:
        on_progress("Saving transcript…", 98)

    OUTPUT_TRANSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRANSCRIPT, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    if on_progress:
        on_progress("Transcription complete", 100)

    return segments
