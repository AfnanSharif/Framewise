import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_analyzer.analyzer import ContentAnalyzer
from video_analyzer.models import Scene, TranscriptSegment
from video_analyzer.service import VideoAnalysisService
from video_analyzer.transcripts import parse_subtitles


class AnalyzerTests(unittest.TestCase):
    def test_srt_parser(self):
        segments = parse_subtitles("1\n00:00:01,000 --> 00:00:03,500\nHello world.\n\n2\n00:00:04,000 --> 00:00:06,000\nNext idea.\n")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].start, 1.0)
        self.assertEqual(segments[0].end, 3.5)

    def test_analysis_chapters_and_scene_alignment(self):
        transcript = [
            TranscriptSegment(0, 12, "Vectors represent words and documents for semantic retrieval."),
            TranscriptSegment(12, 25, "Embeddings make similar documents close in vector space."),
            TranscriptSegment(25, 40, "Evaluation reveals a problem and a retrieval failure risk."),
            TranscriptSegment(40, 55, "Monitoring measures latency and error rates in production."),
        ]
        analyzer = ContentAnalyzer(min_chapter_seconds=20, max_chapter_seconds=30)
        result = analyzer.analyze(transcript, [Scene(10, "vector diagram"), Scene(45, "monitoring dashboard")])
        self.assertGreaterEqual(len(result.chapters), 2)
        self.assertEqual(sum(len(ch.scenes) for ch in result.chapters), 2)
        self.assertEqual(result.scene_count, 2)
        json.dumps(result.to_dict())

    def test_sample_file_end_to_end(self):
        path = Path(__file__).resolve().parents[1] / "data" / "sample_transcript.json"
        result = VideoAnalysisService().from_transcript(path)
        self.assertEqual(result.transcript_segments, 9)
        self.assertIn("retrieval", result.top_keywords)
        self.assertTrue(result.executive_summary)

    def test_empty_transcript_rejected(self):
        with self.assertRaises(ValueError):
            ContentAnalyzer().analyze([])

    def test_refiner_is_invoked_and_cannot_change_timestamps(self):
        class FakeRefiner:
            name = "fake-refiner"

            def __init__(self):
                self.calls = 0

            def refine(self, analysis):
                self.calls += 1
                analysis["executive_summary"] = "Refined executive brief"
                for chapter in analysis["chapters"]:
                    chapter["title"] = "Refined chapter"
                    chapter["summary"] = "Refined summary"
                    chapter["key_points"] = ["Refined point"]
                return analysis

        refiner = FakeRefiner()
        segments = [
            TranscriptSegment(0, 10, "Vector search connects related documents for retrieval systems."),
            TranscriptSegment(10, 20, "Evaluation measures whether retrieved evidence answers user questions."),
        ]
        result = VideoAnalysisService(refiner=refiner).from_segments(segments)
        self.assertEqual(refiner.calls, 1)
        self.assertEqual(result.executive_summary, "Refined executive brief")
        self.assertEqual(result.chapters[0].start, 0)
        self.assertEqual(result.metadata["refiner"], "fake-refiner")

    def test_llm_can_propose_new_validated_chapter_boundaries(self):
        class SegmentingRefiner:
            name = "fake-segmenter"

            def refine(self, analysis):
                return {
                    "executive_summary": "Two grounded phases",
                    "chapters": [
                        {"title": "Foundations", "start": 0, "end": 20, "summary": "Vector foundations.", "key_points": ["Vectors and embeddings"], "keywords": ["vectors"], "sentiment": "neutral"},
                        {"title": "Operations", "start": 20, "end": 40, "summary": "Evaluation and monitoring.", "key_points": ["Measure production quality"], "keywords": ["monitoring"], "sentiment": "cautionary"},
                    ],
                }

        segments = [
            TranscriptSegment(0, 10, "Vectors encode semantic meaning."),
            TranscriptSegment(10, 20, "Embeddings support retrieval."),
            TranscriptSegment(20, 30, "Evaluation reveals retrieval failures."),
            TranscriptSegment(30, 40, "Monitoring tracks production errors."),
        ]
        analyzer = ContentAnalyzer(min_chapter_seconds=5, max_chapter_seconds=12)
        result = VideoAnalysisService(analyzer, SegmentingRefiner()).from_segments(segments)
        self.assertEqual(len(result.chapters), 2)
        self.assertEqual(sum(len(chapter.transcript) for chapter in result.chapters), 4)
        self.assertEqual(result.metadata["segmentation_mode"], "llm-validated")

    def test_llm_segmentation_cannot_skip_cues(self):
        class InvalidRefiner:
            name = "invalid"

            def refine(self, analysis):
                return {"executive_summary": "Bad", "chapters": [{"title": "Bad", "start": 0, "end": 10, "summary": "Bad", "key_points": ["Bad"]}]}

        segments = [TranscriptSegment(0, 10, "First grounded segment."), TranscriptSegment(10, 20, "Second grounded segment.")]
        with self.assertRaisesRegex(ValueError, "cover every"):
            VideoAnalysisService(refiner=InvalidRefiner()).from_segments(segments)


if __name__ == "__main__":
    unittest.main()
