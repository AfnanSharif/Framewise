from __future__ import annotations

import json
import re
from pathlib import Path

from .models import TranscriptSegment


TIME_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{3})")


def parse_timestamp(value: str) -> float:
    match = TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid subtitle timestamp: {value!r}")
    hours = int(match.group(1) or 0)
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000


def parse_subtitles(text: str) -> list[TranscriptSegment]:
    normalized = text.replace("\r\n", "\n").strip()
    blocks = re.split(r"\n\s*\n", normalized)
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper() == "WEBVTT":
            continue
        time_index = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if time_index is None:
            continue
        start_raw, end_raw = lines[time_index].split(" --> ", 1)
        end_raw = end_raw.split()[0]
        caption = " ".join(re.sub(r"<[^>]+>", "", line) for line in lines[time_index + 1 :]).strip()
        if caption:
            segments.append(TranscriptSegment(parse_timestamp(start_raw), parse_timestamp(end_raw), caption))
    return segments


def parse_json(text: str) -> list[TranscriptSegment]:
    payload = json.loads(text)
    rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("JSON transcript must be a list or contain a 'segments' list")
    return [
        TranscriptSegment(float(row["start"]), float(row["end"]), str(row["text"]), row.get("speaker"))
        for row in rows
    ]


def parse_plain_text(text: str, words_per_minute: int = 145) -> list[TranscriptSegment]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]
    cursor = 0.0
    segments = []
    for paragraph in paragraphs:
        duration = max(2.0, len(paragraph.split()) / words_per_minute * 60)
        segments.append(TranscriptSegment(cursor, cursor + duration, paragraph))
        cursor += duration
    return segments


def load_transcript(path: str | Path) -> list[TranscriptSegment]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        result = parse_json(text)
    elif file_path.suffix.lower() in {".srt", ".vtt"}:
        result = parse_subtitles(text)
    else:
        result = parse_plain_text(text)
    if not result:
        raise ValueError("the transcript contains no usable segments")
    return sorted(result, key=lambda item: item.start)
