from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..models import Scene, TranscriptSegment


class WhisperTranscriber:
    """Lazy local Whisper adapter. `faster-whisper` is never needed for transcript mode."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_size, self.device, self.compute_type = model_size, device, compute_type

    def transcribe(self, media_path: str | Path, language: str | None = None) -> list[TranscriptSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper to transcribe media files") from exc
        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        segments, _ = model.transcribe(str(media_path), language=language, vad_filter=True)
        return [TranscriptSegment(float(item.start), float(item.end), item.text.strip()) for item in segments if item.text.strip()]


def sample_video_frames(video_path: str | Path, interval_seconds: float = 15) -> list[tuple[float, Any]]:
    if interval_seconds <= 0:
        raise ValueError("frame interval must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to sample video frames") from exc
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps
    frames: list[tuple[float, Any]] = []
    timestamp = 0.0
    while timestamp <= duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if ok:
            frames.append((timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        timestamp += interval_seconds
    capture.release()
    return frames


class BlipSceneCaptioner:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base") -> None:
        self.model_id = model_id
        self._processor: Any = None
        self._model: Any = None

    def caption(self, frames: Iterable[tuple[float, Any]]) -> list[Scene]:
        if self._model is None:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError as exc:
                raise RuntimeError("Install transformers and torch for BLIP captions") from exc
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        results = []
        for timestamp, image in frames:
            inputs = self._processor(images=image, return_tensors="pt")
            output = self._model.generate(**inputs, max_new_tokens=40)
            results.append(Scene(timestamp, self._processor.decode(output[0], skip_special_tokens=True)))
        return results
