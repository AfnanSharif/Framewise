from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .providers import create_refiner
from .service import VideoAnalysisService


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze a timestamped transcript or video")
    parser.add_argument("input", type=Path)
    parser.add_argument("--video", action="store_true", help="Transcribe media with optional faster-whisper")
    parser.add_argument("--language", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refiner", choices=["local", "openai", "gemini"], default=os.getenv("ANALYSIS_REFINER", "local"))
    args = parser.parse_args(argv)
    if args.video:
        from .providers import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_size=os.getenv("WHISPER_MODEL", "base"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        )
        result = VideoAnalysisService(refiner=create_refiner(args.refiner)).from_video(args.input, transcriber=transcriber, language=args.language)
    else:
        result = VideoAnalysisService(refiner=create_refiner(args.refiner)).from_transcript(args.input)
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
