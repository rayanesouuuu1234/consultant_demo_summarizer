"""Ollama-backed summarization — summary + numbered steps, parallel workers."""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OLLAMA_BASE = "http://localhost:11434"
SUMMARY_JSON = Path("outputs/summaries/summary.json")

_JSON_SCHEMA = """{
  "title": "Short label for the scene (5-10 words, e.g. 'Filing Setup — Entity Selection')",
  "summary": "2-3 sentences describing what was on screen and what was being discussed at this moment",
  "steps": [
    "Action-oriented step: what was clicked, navigated, or shown, combined with what was said about it",
    "Next concrete step..."
  ],
  "why_it_matters": "One sentence on why this moment matters in the flow — only if clearly inferable from inputs, else omit"
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


def _coerce_steps(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            t = item.get("text") or item.get("step")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
    return out


def _fallback_ai_fields(label: str, transcript_window: str) -> dict:
    tw = transcript_window or ""
    summary = tw[:400] if tw else "[No transcript]"
    if len(tw) > 400:
        summary += "…"
    return {
        "title": f"Segment at {label}",
        "summary": summary,
        "steps": [],
    }


def _fallback_from_exception(segment: dict, err: str) -> dict:
    label = segment.get("label", "")
    base = _fallback_ai_fields(label, segment.get("transcript_window") or "")
    base["summary"] = f"[Summary error: {err}] {base['summary']}"[:800]
    return {**segment, **base}


def _normalize_ai(data: dict, label: str, transcript_window: str) -> dict:
    if not isinstance(data, dict):
        return _fallback_ai_fields(label, transcript_window)

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        legacy = data.get("note")
        if isinstance(legacy, str) and legacy.strip():
            summary = legacy.strip()
        else:
            summary = _fallback_ai_fields(label, transcript_window)["summary"]

    steps = _coerce_steps(data.get("steps"))

    out: dict = {
        "title": str(data.get("title") or f"Segment at {label}"),
        "summary": str(summary).strip(),
        "steps": steps,
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
        prompt = f"""You are helping a consultant document a screen recording of a software demo. For this moment in the video, produce structured notes they can skim later.

TIMESTAMP: {label}

TRANSCRIPT (what was said near this moment):
\"\"\"{transcript_window}\"\"\"

A screenshot of the UI at this moment is attached.

Instructions:
1. Look carefully at the screenshot — read visible UI: buttons, labels, field names, table headers, and data values where legible.
2. Cross-reference with the transcript — capture what the presenter explained about what is visible.
3. Produce a `steps` array of short, numbered-style action lines (as separate strings) that combine what was done or shown on screen with what was said about it. Each step should be concrete enough that someone who missed the call could replay this part of the demo.
4. Write `summary` as 2-3 sentences describing what was on screen and what was being discussed (no passive "the speaker says" openers — stay direct).
5. Only include information grounded in the screenshot and/or transcript. Do not invent screens, clicks, or business facts not supported by the inputs.

Respond ONLY in valid JSON, no markdown outside the JSON:
{_JSON_SCHEMA}"""
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    else:
        prompt = f"""You are helping a consultant document a screen recording of a software demo. There is no screenshot — use only the transcript for this moment.

TIMESTAMP: {label}

TRANSCRIPT (what was said near this moment):
\"\"\"{transcript_window}\"\"\"

Instructions:
1. Infer what was likely happening on screen only when strongly implied by the transcript; otherwise stay conservative.
2. Produce a `steps` array of short action-oriented lines (as strings) capturing what was done or decided according to what was said.
3. Write `summary` as 2-3 direct sentences about what was discussed. If the transcript is sparse, say so briefly. You may mention once in `summary` that no screenshot was available.
4. Only include information grounded in the transcript.

Respond ONLY in valid JSON, no markdown outside the JSON:
{_JSON_SCHEMA}"""
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
        ai = _normalize_ai(data, label, transcript_window)
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
