from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .analyzer import ContentAnalyzer
from .models import Analysis, Chapter, Scene, TranscriptSegment
from .analyzer import NEGATIVE, POSITIVE, keywords, words
from .transcripts import load_transcript


class AnalysisRefiner(Protocol):
    name: str

    def refine(self, analysis: dict[str, Any]) -> dict[str, Any]: ...


class VideoAnalysisService:
    def __init__(self, analyzer: ContentAnalyzer | None = None, refiner: AnalysisRefiner | None = None) -> None:
        self.analyzer = analyzer or ContentAnalyzer()
        self.refiner = refiner

    def from_transcript(self, path: str | Path, scenes: list[Scene] | None = None) -> Analysis:
        return self._analyze(load_transcript(path), scenes, {"source": Path(path).name})

    def from_segments(self, transcript: list[TranscriptSegment], scenes: list[Scene] | None = None, source: str = "memory") -> Analysis:
        return self._analyze(transcript, scenes, {"source": source})

    def from_video(self, path: str | Path, *, transcriber=None, captioner=None, frame_interval: float = 15, language: str | None = None) -> Analysis:
        from .providers.multimodal import WhisperTranscriber, sample_video_frames

        transcriber = transcriber or WhisperTranscriber()
        transcript = transcriber.transcribe(path, language=language)
        scenes = captioner.caption(sample_video_frames(path, frame_interval)) if captioner else []
        return self._analyze(transcript, scenes, {"source": Path(path).name, "transcribed": True})

    def _analyze(self, transcript: list[TranscriptSegment], scenes: list[Scene] | None, metadata: dict[str, Any]) -> Analysis:
        local = self.analyzer.analyze(transcript, scenes, metadata)
        if self.refiner is None:
            return local
        refined = self.refiner.refine(local.to_dict())
        return self._apply_refinement(local, refined, getattr(self.refiner, "name", type(self.refiner).__name__))

    @staticmethod
    def _apply_refinement(local: Analysis, refined: dict[str, Any], provider_name: str) -> Analysis:
        if not isinstance(refined, dict) or not isinstance(refined.get("executive_summary"), str) or not refined["executive_summary"].strip():
            raise ValueError("refiner must return a non-empty executive_summary")
        rows = refined.get("chapters")
        if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 50:
            raise ValueError("refiner must return between one and fifty chapters")
        transcript = sorted(
            [segment for chapter in local.chapters for segment in chapter.transcript],
            key=lambda segment: (segment.start, segment.end, segment.text),
        )
        scenes = sorted(
            {scene for chapter in local.chapters for scene in chapter.scenes},
            key=lambda scene: (scene.timestamp, scene.caption),
        )
        chapters: list[Chapter] = []
        cursor = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("refined chapters must be objects")
            if cursor >= len(transcript):
                raise ValueError("refiner returned more chapter ranges than transcript cues")
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("refined chapters require numeric start and end timestamps") from exc
            if abs(start - transcript[cursor].start) > 1e-6:
                raise ValueError("refined chapter start must align to the next transcript cue")
            end_index = next((index for index in range(cursor, len(transcript)) if abs(transcript[index].end - end) <= 1e-6), None)
            if end_index is None:
                raise ValueError("refined chapter end must align to a transcript cue")
            group = transcript[cursor : end_index + 1]
            title, summary, points = row.get("title"), row.get("summary"), row.get("key_points")
            if not isinstance(title, str) or not title.strip() or not isinstance(summary, str) or not summary.strip():
                raise ValueError("refined chapter title and summary must be non-empty strings")
            if not isinstance(points, (list, tuple)) or not points or any(not isinstance(point, str) or not point.strip() for point in points):
                raise ValueError("refined key_points must be a list of non-empty strings")
            text = " ".join(segment.text for segment in group)
            chapter_keywords = row.get("keywords")
            if not isinstance(chapter_keywords, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in chapter_keywords):
                chapter_keywords = keywords(text, 6)
            score = sum(word in POSITIVE for word in words(text)) - sum(word in NEGATIVE for word in words(text))
            local_sentiment = "positive" if score > 1 else "cautionary" if score < -1 else "neutral"
            sentiment = row.get("sentiment", local_sentiment)
            if sentiment not in {"positive", "neutral", "cautionary"}:
                sentiment = local_sentiment
            chapters.append(Chapter(
                title=title.strip(),
                start=start,
                end=end,
                summary=summary.strip(),
                key_points=tuple(point.strip() for point in points),
                keywords=tuple(dict.fromkeys(item.strip().lower() for item in chapter_keywords)),
                sentiment=sentiment,
                transcript=tuple(group),
                scenes=tuple(scene for scene in scenes if start <= scene.timestamp <= end),
            ))
            cursor = end_index + 1
        if cursor != len(transcript):
            raise ValueError("refined chapter ranges must cover every transcript cue exactly once")
        return replace(
            local,
            executive_summary=refined["executive_summary"].strip(),
            chapters=tuple(chapters),
            metadata={**local.metadata, "refiner": provider_name, "segmentation_mode": "llm-validated"},
        )
