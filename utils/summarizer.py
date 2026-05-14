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
  "title": "Short neutral label for the screen state (5-10 words, e.g. 'Entity type selection — new filing')",
  "summary": "2-3 sentences of neutral product documentation: what the UI shows and what the narration covers. Do not name roles (consultant, client, presenter). Use passive or imperative voice ('The form displays…', 'Tax ID is entered…').",
  "steps": [
    "Imperative or passive step grounded in UI + narration, e.g. 'Select LLC in the entity dropdown'",
    "Next concrete step — no 'they' or 'someone'"
  ],
  "why_it_matters": "One sentence on workflow significance — neutral, no roles — only if inferable from inputs, else omit"
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
        prompt = f"""Produce neutral, agentless documentation for this moment in a screen recording (as if writing internal product notes, not meeting minutes).

TIMESTAMP: {label}

NARRATION (audio near this moment, may describe UI actions):
\"\"\"{transcript_window}\"\"\"

A screenshot of the UI at this moment is attached.

Rules:
1. Examine the screenshot: buttons, labels, fields, tables, and legible values.
2. Cross-check the narration for factual tie-ins to what is visible.
3. Write `summary` in calm documentation style: what the screen shows and what the narration establishes. Never attribute actions to a person or role (no consultant, client, presenter, user names). Prefer passive voice or imperative ("The dropdown lists…", "Navigate to…").
4. `steps`: short strings, each one concrete UI/navigation or data fact tied to narration when relevant. No "they explained" or "someone clicked" — write as procedure ("Open X", "Field Y shows Z", "Value entered in …").
5. Only facts supported by screenshot and/or transcript.

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
        prompt = f"""Produce neutral, agentless documentation for this moment. No screenshot is available — use only the narration text.

TIMESTAMP: {label}

NARRATION:
\"\"\"{transcript_window}\"\"\"

Rules:
1. Stay conservative about unseen UI; only state what the narration reasonably implies.
2. `summary`: 2-3 sentences, neutral documentation tone. No roles (consultant, client, etc.). If narration is sparse, say so briefly and note that visual context was unavailable.
3. `steps`: short procedural lines grounded in what was said — no "they" or "someone".
4. No facts not supported by the transcript.

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
