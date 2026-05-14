"""Markdown exports and ZIP packaging (v3)."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SUMMARY_MD = Path("outputs/summaries/summary.md")
NOTABLE_DETAILS_MD = Path("outputs/summaries/notable_details.md")
OPEN_QUESTIONS_MD = Path("outputs/summaries/open_questions.md")


def _fmt_list(items: list | None, sep: str = " · ") -> str:
    if not items:
        return ""
    return sep.join(str(x) for x in items if x)


def _md_cell(val: str) -> str:
    return str(val).replace("|", "·").replace("\n", " ")


def _format_duration_hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_markdown_exports(
    summaries: list[dict],
    video_filename: str,
    video_duration_sec: float,
) -> None:
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    n = len(summaries)
    dur_label = _format_duration_hhmmss(video_duration_sec)

    lines = [
        "# Demo Walkthrough Summary",
        f"Generated: {now}",
        f"Video: {video_filename}",
        f"Duration: {dur_label}",
        f"Total Segments: {n}",
        "",
        "---",
        "",
    ]

    for s in summaries:
        title = s.get("title") or f"Segment {s.get('label', '')}"
        label = s.get("label", "")
        lines.append(f"## [{label}] {title}")
        lines.append("")
        lines.append(f"**Note:** {s.get('note', '')}")
        lines.append("")
        wim = s.get("why_it_matters")
        if wim:
            lines.append(f"**Why it matters:** {wim}")
            lines.append("")
        kc = s.get("key_concepts") or []
        lines.append(f"**Key concepts:** {_fmt_list(kc)}")
        lines.append("")
        nd = s.get("notable_details") or []
        lines.append("**Notable details:**")
        for d in nd:
            lines.append(f"- {d}")
        if not nd:
            lines.append("- _None listed_")
        lines.append("")
        lines.append("---")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    nd_lines = [
        "# Notable Details & Decisions",
        "",
        "Extracted across all segments — based only on transcript and screenshots.",
        "",
    ]
    for s in summaries:
        label = s.get("label", "")
        for detail in s.get("notable_details") or []:
            nd_lines.append(f"- [{label}] {detail}")
    if len(nd_lines) <= 4:
        nd_lines.append("- _No notable details extracted._")
    nd_lines.append("")
    NOTABLE_DETAILS_MD.write_text("\n".join(nd_lines), encoding="utf-8")

    oq_lines = [
        "# Open Questions Log",
        "",
        "| Timestamp | Topic | Question | Owner | Status |",
        "|-----------|-------|----------|-------|--------|",
    ]
    for s in summaries:
        label = s.get("label", "")
        title = s.get("title") or ""
        oq_lines.append(
            f"| {_md_cell(label)}  | {_md_cell(title)} |  |  | Open   |"
        )
    oq_lines.append("")
    OPEN_QUESTIONS_MD.write_text("\n".join(oq_lines), encoding="utf-8")


def create_zip(output_dir: str | Path = "outputs") -> bytes:
    root = Path(output_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if not root.exists():
            zf.writestr("outputs/README.txt", "No outputs directory yet.\n")
        else:
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path))
    return buf.getvalue()
