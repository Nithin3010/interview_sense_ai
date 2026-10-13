import logging
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from backend.app.config import settings
from backend.app.core.exceptions import (
    InterviewSenseException,
    ModelNotLoadedError,
)

logger = logging.getLogger("interviewsense.transcription")


@dataclass
class WordRecord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptionResult:
    text: str
    words: List[WordRecord]
    language: str
    model_used: str


def transcribe_audio(
    samples: np.ndarray,
    model: Any,
    language: str = "en",
    model_name: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribes normalized 16kHz mono float32 audio using faster-whisper.
    Extracts complete transcript and chronologically ordered word-level timestamps.
    
    NOTE: Word probabilities represent acoustic-language model confidence
    and MUST NOT be interpreted as pronunciation accuracy scores.
    """
    if model is None:
        raise ModelNotLoadedError("faster-whisper")

    if not isinstance(samples, np.ndarray) or samples.ndim != 1:
        raise InterviewSenseException(
            error_code="INVALID_AUDIO_ARRAY",
            message="Transcription requires a 1D NumPy array of audio samples.",
            status_code=400,
        )

    resolved_model_name = model_name or getattr(model, "model_size_or_path", settings.WHISPER_MODEL_SIZE)

    try:
        segments_gen, info = model.transcribe(
            samples,
            word_timestamps=True,
            language=language,
        )

        detected_language = getattr(info, "language", language) or language
        words: List[WordRecord] = []
        segment_texts: List[str] = []

        for segment in segments_gen:
            seg_text = segment.text.strip() if hasattr(segment, "text") and segment.text else ""
            if seg_text:
                segment_texts.append(seg_text)

            seg_words = getattr(segment, "words", None)
            if seg_words:
                for w in seg_words:
                    word_str = w.word.strip()
                    if not word_str:
                        continue
                    start_sec = max(0.0, round(float(w.start), 3))
                    end_sec = max(start_sec, round(float(w.end), 3))
                    prob = max(0.0, min(1.0, round(float(getattr(w, "probability", 1.0)), 4)))

                    words.append(
                        WordRecord(
                            word=word_str,
                            start=start_sec,
                            end=end_sec,
                            probability=prob,
                        )
                    )

        # Sort words chronologically by start timestamp
        words.sort(key=lambda w: (w.start, w.end))

        full_text = " ".join(segment_texts).strip()
        if not full_text and words:
            full_text = " ".join(w.word for w in words).strip()

        return TranscriptionResult(
            text=full_text,
            words=words,
            language=detected_language,
            model_used=resolved_model_name,
        )

    except Exception as exc:
        logger.error(f"Transcription error: {exc}", exc_info=True)
        raise InterviewSenseException(
            error_code="TRANSCRIPTION_FAILED",
            message=f"Speech recognition execution failed: {exc}",
            status_code=500,
        )
