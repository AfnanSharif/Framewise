import os

from .gemini_refiner import GeminiRefiner
from .multimodal import BlipSceneCaptioner, WhisperTranscriber, sample_video_frames
from .openai_refiner import OpenAIRefiner

__all__ = ["BlipSceneCaptioner", "WhisperTranscriber", "sample_video_frames", "OpenAIRefiner", "GeminiRefiner"]


def create_refiner(name: str | None = None):
    selected = (name or os.getenv("ANALYSIS_REFINER", "local")).strip().lower()
    if selected in {"", "local", "none"}:
        return None
    if selected == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI refinement")
        return OpenAIRefiner(api_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if selected == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini refinement")
        return GeminiRefiner(api_key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    raise ValueError("ANALYSIS_REFINER must be local, openai, or gemini")


__all__.append("create_refiner")
