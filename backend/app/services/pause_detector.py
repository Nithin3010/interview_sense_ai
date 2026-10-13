import logging
from dataclasses import dataclass
from typing import Any, List, Sequence

from backend.app.utils.constants import (
    MIN_PAUSE_THRESHOLD_SECONDS,
    SIGNIFICANT_PAUSE_SECONDS,
    EXTENDED_PAUSE_SECONDS,
)

logger = logging.getLogger("interviewsense.pause_detector")


@dataclass
class PauseInterval:
    start: float
    end: float
    duration: float


@dataclass
class PauseStatistics:
    total_pauses: int
    total_pause_duration_seconds: float
    pauses_over_500ms: int
    pauses_over_2s: int
    longest_pause_seconds: float
    average_pause_seconds: float
    pause_to_total_ratio: float
    pause_intervals: List[PauseInterval]
    note: str = (
        "Pause measurements are alignment-derived estimates based on word timestamp gaps. "
        "They approximate conversational pacing and hesitations, but are not guaranteed acoustic silence measurements."
    )


def detect_pauses(
    words: Sequence[Any],
    total_duration_seconds: float,
    min_pause_seconds: float = MIN_PAUSE_THRESHOLD_SECONDS,
) -> PauseStatistics:
    """
    Computes pause statistics and silence intervals derived from gaps between
    recognized word timestamps.
    
    Safely handles overlapping, out-of-order, or negative timestamps, clamping
    all intervals to the valid audio duration.
    """
    total_dur = max(0.0, float(total_duration_seconds))

    if total_dur == 0.0 or not words:
        if total_dur > 0.0 and total_dur >= min_pause_seconds:
            single_pause = PauseInterval(start=0.0, end=total_dur, duration=total_dur)
            return PauseStatistics(
                total_pauses=1,
                total_pause_duration_seconds=total_dur,
                pauses_over_500ms=1 if total_dur >= SIGNIFICANT_PAUSE_SECONDS else 0,
                pauses_over_2s=1 if total_dur >= EXTENDED_PAUSE_SECONDS else 0,
                longest_pause_seconds=total_dur,
                average_pause_seconds=total_dur,
                pause_to_total_ratio=1.0,
                pause_intervals=[single_pause],
            )
        return PauseStatistics(
            total_pauses=0,
            total_pause_duration_seconds=0.0,
            pauses_over_500ms=0,
            pauses_over_2s=0,
            longest_pause_seconds=0.0,
            average_pause_seconds=0.0,
            pause_to_total_ratio=0.0,
            pause_intervals=[],
        )

    # 1. Extract valid speech intervals from words, clamped to [0, total_dur]
    raw_intervals = []
    for w in words:
        try:
            start = max(0.0, min(float(w.start), total_dur))
            end = max(start, min(float(w.end), total_dur))
            if end > start:
                raw_intervals.append((start, end))
        except (AttributeError, TypeError, ValueError):
            continue

    if not raw_intervals:
        if total_dur >= min_pause_seconds:
            p = PauseInterval(start=0.0, end=total_dur, duration=total_dur)
            return PauseStatistics(
                total_pauses=1,
                total_pause_duration_seconds=total_dur,
                pauses_over_500ms=1 if total_dur >= SIGNIFICANT_PAUSE_SECONDS else 0,
                pauses_over_2s=1 if total_dur >= EXTENDED_PAUSE_SECONDS else 0,
                longest_pause_seconds=total_dur,
                average_pause_seconds=total_dur,
                pause_to_total_ratio=1.0,
                pause_intervals=[p],
            )
        return PauseStatistics(
            total_pauses=0,
            total_pause_duration_seconds=0.0,
            pauses_over_500ms=0,
            pauses_over_2s=0,
            longest_pause_seconds=0.0,
            average_pause_seconds=0.0,
            pause_to_total_ratio=0.0,
            pause_intervals=[],
        )

    # 2. Sort and merge overlapping or adjacent speech intervals
    raw_intervals.sort(key=lambda iv: iv[0])
    merged_speech: List[List[float]] = []
    for iv in raw_intervals:
        if not merged_speech or iv[0] > merged_speech[-1][1]:
            merged_speech.append([iv[0], iv[1]])
        else:
            merged_speech[-1][1] = max(merged_speech[-1][1], iv[1])

    # 3. Identify pause intervals (leading gap, inter-word gaps, trailing gap)
    pauses: List[PauseInterval] = []

    # Leading pause
    if merged_speech[0][0] >= min_pause_seconds:
        dur = merged_speech[0][0]
        pauses.append(PauseInterval(start=0.0, end=round(dur, 3), duration=round(dur, 3)))

    # Inter-speech pauses
    for i in range(len(merged_speech) - 1):
        gap_start = merged_speech[i][1]
        gap_end = merged_speech[i + 1][0]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_pause_seconds:
            pauses.append(
                PauseInterval(
                    start=round(gap_start, 3),
                    end=round(gap_end, 3),
                    duration=round(gap_dur, 3),
                )
            )

    # Trailing pause
    if total_dur - merged_speech[-1][1] >= min_pause_seconds:
        dur = total_dur - merged_speech[-1][1]
        pauses.append(
            PauseInterval(
                start=round(merged_speech[-1][1], 3),
                end=round(total_dur, 3),
                duration=round(dur, 3),
            )
        )

    total_pause_duration = sum(p.duration for p in pauses)
    total_pauses_count = len(pauses)
    pauses_500ms = sum(1 for p in pauses if p.duration >= SIGNIFICANT_PAUSE_SECONDS)
    pauses_2s = sum(1 for p in pauses if p.duration >= EXTENDED_PAUSE_SECONDS)
    longest_pause = max((p.duration for p in pauses), default=0.0)
    avg_pause = total_pause_duration / total_pauses_count if total_pauses_count > 0 else 0.0
    pause_ratio = total_pause_duration / total_dur if total_dur > 0.0 else 0.0

    return PauseStatistics(
        total_pauses=total_pauses_count,
        total_pause_duration_seconds=round(total_pause_duration, 3),
        pauses_over_500ms=pauses_500ms,
        pauses_over_2s=pauses_2s,
        longest_pause_seconds=round(longest_pause, 3),
        average_pause_seconds=round(avg_pause, 3),
        pause_to_total_ratio=round(min(1.0, max(0.0, pause_ratio)), 4),
        pause_intervals=pauses,
    )
