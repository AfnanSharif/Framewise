from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid transcript timestamps")
        if not self.text.strip():
            raise ValueError("transcript text cannot be empty")


@dataclass(frozen=True)
class Scene:
    timestamp: float
    caption: str
    confidence: float | None = None


@dataclass(frozen=True)
class Chapter:
    title: str
    start: float
    end: float
    summary: str
    key_points: tuple[str, ...]
    keywords: tuple[str, ...]
    sentiment: str
    transcript: tuple[TranscriptSegment, ...] = field(repr=False)
    scenes: tuple[Scene, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Analysis:
    duration: float
    executive_summary: str
    chapters: tuple[Chapter, ...]
    top_keywords: tuple[str, ...]
    transcript_segments: int
    scene_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_label"] = format_timestamp(self.duration)
        for chapter in data["chapters"]:
            chapter["start_label"] = format_timestamp(chapter["start"])
            chapter["end_label"] = format_timestamp(chapter["end"])
        return data
