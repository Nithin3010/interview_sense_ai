import logging
import re
from dataclasses import dataclass
from typing import Any, List, Sequence, Set

from backend.app.utils.constants import (
    VOCAL_FILLERS,
    POTENTIAL_FILLERS,
    MULTIWORD_FILLERS,
)
from backend.app.services.pause_detector import PauseStatistics

logger = logging.getLogger("interviewsense.fluency_analyzer")


@dataclass
class FillerWord:
    word: str
    start: float
    end: float
    category: str


@dataclass
class RepetitionRecord:
    repeated_text: str
    start: float
    end: float
    count: int
    repetition_type: str  # "word_repetition" or "phrase_repetition"


@dataclass
class FluencyMetrics:
    word_count: int
    gross_wpm: float
    articulation_rate_wpm: float
    speaking_duration_seconds: float
    total_pause_duration_seconds: float
    pause_to_speech_ratio: float
    filler_count: int
    filler_percentage: float
    filler_words: List[FillerWord]
    repetitions: List[RepetitionRecord]


def clean_token(token: str) -> str:
    """Strips leading/trailing punctuation and lowercases the token for matching."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()


def detect_fillers(words: Sequence[Any]) -> List[FillerWord]:
    """
    Token-aware detection of vocal disfluencies, discourse markers, and multiword fillers.
    Matches discrete word tokens only, preventing substring false positives.
    """
    if not words:
        return []

    fillers: List[FillerWord] = []
    n = len(words)
    consumed_indices: Set[int] = set()

    # 1. Multiword filler detection (e.g. "you know", "i mean", "sort of")
    for phrase_tuple, category in MULTIWORD_FILLERS:
        phrase_len = len(phrase_tuple)
        for i in range(n - phrase_len + 1):
            if any((i + k) in consumed_indices for k in range(phrase_len)):
                continue

            window_tokens = tuple(clean_token(getattr(words[i + k], "word", "")) for k in range(phrase_len))
            if window_tokens == phrase_tuple:
                phrase_text = " ".join(getattr(words[i + k], "word", "") for k in range(phrase_len))
                start_time = float(getattr(words[i], "start", 0.0))
                end_time = float(getattr(words[i + phrase_len - 1], "end", 0.0))
                fillers.append(
                    FillerWord(
                        word=phrase_text,
                        start=round(start_time, 3),
                        end=round(end_time, 3),
                        category=category,
                    )
                )
                for k in range(phrase_len):
                    consumed_indices.add(i + k)

    # 2. Single-token filler detection
    for i, w in enumerate(words):
        if i in consumed_indices:
            continue

        raw_word = getattr(w, "word", "")
        norm_token = clean_token(raw_word)

        if norm_token in VOCAL_FILLERS:
            fillers.append(
                FillerWord(
                    word=raw_word,
                    start=round(float(getattr(w, "start", 0.0)), 3),
                    end=round(float(getattr(w, "end", 0.0)), 3),
                    category=VOCAL_FILLERS[norm_token],
                )
            )
        elif norm_token in POTENTIAL_FILLERS:
            fillers.append(
                FillerWord(
                    word=raw_word,
                    start=round(float(getattr(w, "start", 0.0)), 3),
                    end=round(float(getattr(w, "end", 0.0)), 3),
                    category=POTENTIAL_FILLERS[norm_token],
                )
            )

    fillers.sort(key=lambda f: f.start)
    return fillers


def detect_repetitions(words: Sequence[Any]) -> List[RepetitionRecord]:
    """
    Detects immediate word and phrase repetitions (2 to 4 words).
    Avoids double-counting sub-elements of an already recognized phrase repetition.
    """
    if not words or len(words) < 2:
        return []

    tokens = [clean_token(getattr(w, "word", "")) for w in words]
    n = len(words)
    covered_indices: Set[int] = set()
    repetitions: List[RepetitionRecord] = []

    # 1. Immediate phrase repetitions of length 4, 3, 2
    for k in (4, 3, 2):
        i = 0
        while i <= n - 2 * k:
            if any((i + offset) in covered_indices for offset in range(2 * k)):
                i += 1
                continue

            sub1 = tokens[i : i + k]
            sub2 = tokens[i + k : i + 2 * k]

            if all(sub1) and sub1 == sub2:
                phrase_str = " ".join(getattr(words[j], "word", "") for j in range(i, i + k))
                start_time = float(getattr(words[i], "start", 0.0))
                end_time = float(getattr(words[i + 2 * k - 1], "end", 0.0))
                repetitions.append(
                    RepetitionRecord(
                        repeated_text=phrase_str,
                        start=round(start_time, 3),
                        end=round(end_time, 3),
                        count=2,
                        repetition_type="phrase_repetition",
                    )
                )
                for offset in range(2 * k):
                    covered_indices.add(i + offset)
                i += 2 * k
            else:
                i += 1

    # 2. Immediate single-word repetitions
    i = 0
    while i < n - 1:
        if i in covered_indices or (i + 1) in covered_indices:
            i += 1
            continue

        if tokens[i] and tokens[i] == tokens[i + 1]:
            word_str = getattr(words[i], "word", "")
            start_time = float(getattr(words[i], "start", 0.0))
            end_time = float(getattr(words[i + 1], "end", 0.0))
            repetitions.append(
                RepetitionRecord(
                    repeated_text=word_str,
                    start=round(start_time, 3),
                    end=round(end_time, 3),
                    count=2,
                    repetition_type="word_repetition",
                )
            )
            covered_indices.add(i)
            covered_indices.add(i + 1)
            i += 2
        else:
            i += 1

    repetitions.sort(key=lambda r: r.start)
    return repetitions


def analyze_fluency(
    words: Sequence[Any],
    total_duration_seconds: float,
    pause_stats: PauseStatistics,
) -> FluencyMetrics:
    """
    Calculates pacing (Gross WPM), articulation rate (Net WPM), filler occurrences,
    and repeated-word/phrase markers.
    """
    word_count = len(words)
    total_dur = max(0.0, float(total_duration_seconds))
    total_pause_dur = max(0.0, float(pause_stats.total_pause_duration_seconds))

    # Gross Speaking Rate: (word_count / total_duration) * 60
    if total_dur > 0.0 and word_count > 0:
        gross_wpm = (word_count / total_dur) * 60.0
    else:
        gross_wpm = 0.0

    # Active Phonation Duration = total_duration - total_pause_duration
    active_speaking_duration = max(0.0, total_dur - total_pause_dur)

    # Articulation Rate: (word_count / active_speaking_duration) * 60
    if active_speaking_duration > 0.1 and word_count > 0:
        articulation_wpm = (word_count / active_speaking_duration) * 60.0
    else:
        articulation_wpm = gross_wpm

    # Fillers and Repetitions
    fillers = detect_fillers(words)
    repetitions = detect_repetitions(words)

    filler_count = len(fillers)
    filler_percentage = (filler_count / word_count * 100.0) if word_count > 0 else 0.0
    pause_ratio = pause_stats.pause_to_total_ratio

    return FluencyMetrics(
        word_count=word_count,
        gross_wpm=round(gross_wpm, 1),
        articulation_rate_wpm=round(articulation_wpm, 1),
        speaking_duration_seconds=round(active_speaking_duration, 3),
        total_pause_duration_seconds=round(total_pause_dur, 3),
        pause_to_speech_ratio=round(pause_ratio, 4),
        filler_count=filler_count,
        filler_percentage=round(filler_percentage, 1),
        filler_words=fillers,
        repetitions=repetitions,
    )
