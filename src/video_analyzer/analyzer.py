from __future__ import annotations

import math
import re
from collections import Counter

from .models import Analysis, Chapter, Scene, TranscriptSegment

STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before", "being", "but", "can", "could", "did", "does", "each", "for", "from", "have", "here", "into", "just", "more", "most", "not", "now", "only", "other", "our", "out", "over", "should", "some", "such", "than", "that", "the", "their", "then", "there", "these", "they", "this", "those", "through", "today", "use", "using", "very", "was", "we", "were", "what", "when", "where", "which", "while", "will", "with", "would", "you", "your"
}
POSITIVE = {"accurate", "benefit", "clear", "effective", "efficient", "improve", "insight", "success", "useful", "valuable"}
NEGATIVE = {"bad", "challenge", "error", "fail", "failure", "issue", "problem", "risk", "slow", "wrong"}


def words(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text) if word.lower() not in STOPWORDS]


def similarity(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    return len(a & b) / len(a | b) if a and b else 0.0


def keywords(text: str, limit: int = 8) -> tuple[str, ...]:
    counts = Counter(words(text))
    return tuple(word for word, _ in counts.most_common(limit))


def summarize_segments(segments: list[TranscriptSegment], limit: int = 2) -> tuple[str, ...]:
    if not segments:
        return ()
    corpus_counts = Counter(words(" ".join(segment.text for segment in segments)))
    scored = []
    for index, segment in enumerate(segments):
        tokens = words(segment.text)
        score = sum(corpus_counts[token] for token in set(tokens)) / math.sqrt(max(1, len(tokens)))
        scored.append((score, index, segment.text.strip()))
    chosen = sorted(sorted(scored, reverse=True)[:limit], key=lambda row: row[1])
    return tuple(row[2] for row in chosen)


class ContentAnalyzer:
    def __init__(self, min_chapter_seconds: float = 45, max_chapter_seconds: float = 240, topic_threshold: float = 0.06) -> None:
        self.min_chapter_seconds = min_chapter_seconds
        self.max_chapter_seconds = max_chapter_seconds
        self.topic_threshold = topic_threshold

    def analyze(self, transcript: list[TranscriptSegment], scenes: list[Scene] | None = None, metadata: dict | None = None) -> Analysis:
        if not transcript:
            raise ValueError("at least one transcript segment is required")
        transcript = sorted(transcript, key=lambda segment: segment.start)
        scenes = sorted(scenes or [], key=lambda scene: scene.timestamp)
        groups = self._chapter_groups(transcript)
        chapters = tuple(self._build_chapter(group, scenes, number) for number, group in enumerate(groups, 1))
        executive_points = summarize_segments(transcript, 3)
        entire_text = " ".join(segment.text for segment in transcript)
        return Analysis(
            duration=max(segment.end for segment in transcript),
            executive_summary=" ".join(executive_points),
            chapters=chapters,
            top_keywords=keywords(entire_text, 12),
            transcript_segments=len(transcript),
            scene_count=len(scenes),
            metadata={"analysis_mode": "local-extractive", **(metadata or {})},
        )

    def _chapter_groups(self, transcript: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
        groups: list[list[TranscriptSegment]] = []
        current = [transcript[0]]
        for segment in transcript[1:]:
            elapsed = current[-1].end - current[0].start
            topic_shift = similarity(current[-1].text, segment.text) < self.topic_threshold
            gap = segment.start - current[-1].end
            should_break = elapsed >= self.max_chapter_seconds or (elapsed >= self.min_chapter_seconds and (topic_shift or gap > 8))
            if should_break:
                groups.append(current)
                current = [segment]
            else:
                current.append(segment)
        groups.append(current)
        return groups

    def _build_chapter(self, group: list[TranscriptSegment], scenes: list[Scene], number: int) -> Chapter:
        text = " ".join(segment.text for segment in group)
        key = keywords(text, 6)
        title_terms = " · ".join(term.title() for term in key[:3]) or f"Section {number}"
        points = summarize_segments(group, 3)
        score = sum(word in POSITIVE for word in words(text)) - sum(word in NEGATIVE for word in words(text))
        sentiment = "positive" if score > 1 else "cautionary" if score < -1 else "neutral"
        chapter_scenes = tuple(scene for scene in scenes if group[0].start <= scene.timestamp <= group[-1].end)
        return Chapter(
            title=title_terms,
            start=group[0].start,
            end=group[-1].end,
            summary=" ".join(points[:2]),
            key_points=points,
            keywords=key,
            sentiment=sentiment,
            transcript=tuple(group),
            scenes=chapter_scenes,
        )
