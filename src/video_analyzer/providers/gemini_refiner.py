from __future__ import annotations

import json
from typing import Any


class GeminiRefiner:
    """Opt-in Gemini chapter refinement with a JSON-only response contract."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to enable Gemini refinement") from exc
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def refine(self, analysis: dict[str, Any]) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Segment the timestamped transcript into coherent chapters. Every start/end must exactly match cue boundaries; "
                "cover every cue once in order; include executive_summary, title, summary, key_points, keywords, and sentiment; "
                "do not add facts; return JSON only.\n\n"
                + json.dumps(analysis, ensure_ascii=False)
            ),
            config={"response_mime_type": "application/json"},
        )
        refined = json.loads(response.text or "{}")
        if not isinstance(refined, dict) or "chapters" not in refined:
            raise ValueError("Gemini returned an invalid analysis object")
        return refined
