"""Scene change detection with adaptive sampling interval (v3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

SCREENSHOT_DIR = Path("outputs/screenshots")


def get_sample_interval(duration_seconds: float) -> int:
    if duration_seconds <= 600:
        return 1
    if duration_seconds <= 1800:
        return 2
    return 3


def _format_label(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_filename(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}_{m:02d}_{s:02d}.png"


def _video_duration_sec(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps > 0 and frame_count > 0:
        return float(frame_count) / float(fps)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1e9)
    cap.read()
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
    if pos_ms > 0:
        return float(pos_ms) / 1000.0
    return 60.0


def detect_scenes(
    video_path: str,
    threshold: float = 25,
    min_gap: float = 8,
    max_scenes: int = 35,
    on_progress=None,
) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    duration = _video_duration_sec(cap)
    cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
    interval = float(get_sample_interval(duration))

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    candidates: list[dict] = []
    t = 0.0
    last_scene = -float(min_gap)

    while t < duration + 0.001:
        if on_progress and duration > 0:
            on_progress("Detecting scenes", int(min(100, (t / duration) * 100)))

        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(diff.mean())
            if score > threshold and (t - last_scene) >= min_gap:
                candidates.append(
                    {
                        "timestamp": t,
                        "diff_score": score,
                        "frame": frame.copy(),
                    }
                )
                last_scene = t
        prev_gray = gray
        t += interval

    cap.release()

    candidates.sort(key=lambda x: x["diff_score"], reverse=True)
    kept = candidates[:max_scenes]
    kept.sort(key=lambda x: x["timestamp"])

    if on_progress:
        on_progress("Saving screenshots…", 95)

    results: list[dict] = []
    for i, c in enumerate(kept):
        ts = float(c["timestamp"])
        fname = _format_filename(ts)
        path = SCREENSHOT_DIR / fname
        cv2.imwrite(str(path), c["frame"])
        label = _format_label(ts)
        results.append(
            {
                "timestamp": ts,
                "label": label,
                "path": str(path),
                "diff_score": c["diff_score"],
            }
        )
        if on_progress:
            on_progress(
                "Saving screenshots…",
                95 + int((i + 1) / max(len(kept), 1) * 5),
            )

    return results


def clear_screenshots_dir() -> None:
    if SCREENSHOT_DIR.exists():
        shutil.rmtree(SCREENSHOT_DIR)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
