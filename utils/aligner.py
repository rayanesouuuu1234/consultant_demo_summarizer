"""Align scene timestamps to transcript windows (v3)."""

from __future__ import annotations


def align(scenes: list[dict], segments: list[dict]) -> list[dict]:
    aligned: list[dict] = []

    for scene in scenes:
        T = float(scene["timestamp"])
        lo = T - 30.0
        hi = T + 60.0

        window = [
            s["text"].strip()
            for s in segments
            if float(s["start"]) >= lo and float(s["end"]) <= hi and s.get("text")
        ]

        if not window and segments:
            nearest = min(segments, key=lambda s: abs(float(s["start"]) - T))
            if abs(float(nearest["start"]) - T) <= 120.0 and nearest.get("text"):
                window = [nearest["text"].strip()]

        transcript_window = (
            " ".join(window) if window else "[No speech detected near this segment]"
        )

        aligned.append(
            {
                "timestamp": T,
                "label": scene["label"],
                "screenshot_path": scene["path"],
                "transcript_window": transcript_window,
            }
        )

    return aligned
