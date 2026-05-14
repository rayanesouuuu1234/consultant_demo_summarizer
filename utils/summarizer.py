"""Ollama-backed summarization — v3 consultant notes, parallel workers."""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OLLAMA_BASE = "http://localhost:11434"
SUMMARY_JSON = Path("outputs/summaries/summary.json")

_JSON_SCHEMA_VISION = """{
  "title": "Action-oriented title — what was done here (5-8 words, e.g. 'Configured reorder levels for seasonal SKUs')",
  "note": "2-4 sentences. Direct consultant briefing. What happened, what was shown, what matters. No passive openers.",
  "key_concepts": ["only terms explicitly visible or spoken — empty array if none"],
  "notable_details": ["specific values, rules, decisions, names mentioned — empty array if none"],
  "why_it_matters": "One sentence on why this moment is significant in the overall flow — only if clearly inferable from the inputs, otherwise omit this field"
}"""


def check_ollama_running() -> bool:
    try:
        r = requests.get(OLLAMA_BASE, timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def get_available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = data.get("models") or []
        names: list[str] = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
        return names
    except (requests.RequestException, KeyError, ValueError):
        return []


def pick_summary_model(available: list[str]) -> str | None:

    def first_match(prefix: str) -> str | None:
        for orig in available:
            if orig.lower().startswith(prefix):
                return orig
        return None

    for pref in ("llava", "llama3", "mistral"):
        m = first_match(pref)
        if m:
            return m
    return None


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _fallback_ai_fields(label: str, transcript_window: str) -> dict:
    tw = transcript_window or ""
    note = tw[:300] if tw else "[No transcript]"
    if len(tw) > 300:
        note += "…"
    return {
        "title": f"Segment at {label}",
        "note": note,
        "key_concepts": [],
        "notable_details": [],
    }


def _fallback_from_exception(segment: dict, err: str) -> dict:
    label = segment.get("label", "")
    base = _fallback_ai_fields(label, segment.get("transcript_window") or "")
    base["note"] = f"[Summary error] {base['note']}"[:500]
    return {**segment, **base}


def _normalize_ai_v3(data: dict, label: str, transcript_window: str) -> dict:
    if not isinstance(data, dict):
        return _fallback_ai_fields(label, transcript_window)

    if "note" not in data and "summary" in data:
        data["note"] = data.get("summary", "")

    out: dict = {
        "title": str(data.get("title") or f"Segment at {label}"),
        "note": str(data.get("note") or ""),
        "key_concepts": data.get("key_concepts") if isinstance(data.get("key_concepts"), list) else [],
        "notable_details": data.get("notable_details")
        if isinstance(data.get("notable_details"), list)
        else [],
    }
    wim = data.get("why_it_matters")
    if isinstance(wim, str) and wim.strip():
        out["why_it_matters"] = wim.strip()
    return out


def summarize_segment(segment: dict, model: str) -> dict:
    label = segment["label"]
    transcript_window = segment.get("transcript_window") or ""
    screenshot_path = segment.get("screenshot_path") or ""
    model_lower = model.lower()
    use_vision = model_lower.startswith("llava") and bool(
        screenshot_path and Path(screenshot_path).is_file()
    )

    if use_vision:
        with open(screenshot_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        prompt = f"""You are reviewing a screen recording on behalf of a consultant. Your job is to write a brief, sharp note that helps the consultant instantly recall what happened at this moment — without rewatching the video.

TIMESTAMP: {label}

TRANSCRIPT (what was said near this moment):
\"\"\"{transcript_window}\"\"\"

The screenshot shows what was on screen at this moment.

Write your note as if briefing a colleague. Be direct and specific. Use active language. Do not start with "The speaker", "In this segment", "This section", or any passive opener.

Focus on:
- What was being done or decided
- What the screen was showing
- Any specific values, steps, rules, or configurations mentioned
- Why this moment matters in the overall flow

If the transcript is empty, base your note only on what you see on screen.
If the screen is unclear, base your note only on what was said.
Only include information actually present in the screenshot or transcript — do not invent context.

Respond ONLY in valid JSON, no markdown, no explanation outside the JSON:
{_JSON_SCHEMA_VISION}"""
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    else:
        prompt = f"""You are reviewing a screen recording on behalf of a consultant. Your job is to write a brief, sharp note that helps the consultant instantly recall what happened at this moment — without rewatching the video. There is no screenshot — transcript only.

TIMESTAMP: {label}

TRANSCRIPT (what was said near this moment):
\"\"\"{transcript_window}\"\"\"

Write your note as if briefing a colleague. Be direct and specific. Use active language. Do not start with "The speaker", "In this segment", "This section", or any passive opener.

If the transcript is sparse or empty, say so briefly and only include what can be grounded in the words above. You may add "(Note: summary based on transcript only — no screenshot available)" once inside the "note" field if the transcript is sparse.

Focus on:
- What was being done or decided
- Any specific values, steps, rules, or configurations mentioned
- Why this moment matters in the overall flow (only if inferable from the transcript)

Only include information actually present in the transcript — do not invent context.

Respond ONLY in valid JSON, no markdown, no explanation outside the JSON:
{_JSON_SCHEMA_VISION}"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
        body = r.json()
        raw = body.get("response") or ""
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)
        ai = _normalize_ai_v3(data, label, transcript_window)
        return {**segment, **ai}
    except (requests.RequestException, json.JSONDecodeError, TypeError, ValueError):
        return {**segment, **_fallback_ai_fields(label, transcript_window)}


def summarize_all(
    aligned_segments: list[dict],
    model: str,
    on_progress=None,
) -> list[dict]:
    n = len(aligned_segments)
    if n == 0:
        return []

    results: list[dict | None] = [None] * n

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(summarize_segment, dict(seg), model): i
            for i, seg in enumerate(aligned_segments)
        }
        completed = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = _fallback_from_exception(dict(aligned_segments[i]), str(exc))
            completed += 1
            if on_progress:
                on_progress(
                    "Generating summaries",
                    int((completed / n) * 100),
                )

    out = [r for r in results if r is not None]

    if on_progress:
        on_progress("Saving summaries…", 100)

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out
