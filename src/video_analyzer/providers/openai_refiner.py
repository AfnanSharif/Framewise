from __future__ import annotations

import json
from typing import Any


class OpenAIRefiner:
    """Optional structured-output refinement; local analysis remains authoritative fallback."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to enable LLM refinement") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def refine(self, analysis: dict[str, Any]) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Segment the supplied timestamped transcript into coherent chapters, then write grounded titles, summaries, key points, keywords, and sentiment. "
                "Every chapter start/end must exactly match transcript cue boundaries; cover every cue exactly once in order; do not invent facts. "
                "Return executive_summary and chapters as strict JSON."
            ),
            input=json.dumps(analysis, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": "video_chapters", "strict": True, "schema": {
                "type": "object",
                "properties": {
                    "executive_summary": {"type": "string"},
                    "chapters": {"type": "array", "minItems": 1, "maxItems": 50, "items": {"type": "object", "properties": {
                        "title": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                        "summary": {"type": "string"}, "key_points": {"type": "array", "items": {"type": "string"}},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "sentiment": {"type": "string", "enum": ["positive", "neutral", "cautionary"]}
                    }, "required": ["title", "start", "end", "summary", "key_points", "keywords", "sentiment"], "additionalProperties": False}}
                },
                "required": ["executive_summary", "chapters"], "additionalProperties": False
            }}},
        )
        refined = json.loads(response.output_text)
        if not isinstance(refined, dict) or "chapters" not in refined:
            raise ValueError("provider returned an invalid analysis object")
        return refined
