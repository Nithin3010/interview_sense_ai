from backend.app.services.pause_detector import (
    detect_pauses,
    PauseStatistics,
    PauseInterval,
)
from backend.app.services.transcription import WordRecord


def test_pause_detector_gaps_and_thresholds():
    words = [
        WordRecord(word="First", start=0.5, end=1.0, probability=0.9),
        WordRecord(word="second", start=2.0, end=3.0, probability=0.9),
        WordRecord(word="third", start=3.15, end=3.8, probability=0.9),  # 0.15s gap (< 0.25 threshold)
    ]
    total_duration = 5.0

    stats = detect_pauses(words, total_duration_seconds=total_duration)

    assert isinstance(stats, PauseStatistics)
    # Expected pauses:
    # 1. Leading: 0.0 -> 0.5 (0.5s)
    # 2. Inter-word: 1.0 -> 2.0 (1.0s)
    # (Gap 3.0 -> 3.15 is 0.15s, below 0.25s threshold, omitted)
    # 3. Trailing: 3.8 -> 5.0 (1.2s)
    assert stats.total_pauses == 3
    assert stats.total_pause_duration_seconds == 2.7
    assert stats.pauses_over_500ms == 3
    assert stats.pauses_over_2s == 0
    assert stats.longest_pause_seconds == 1.2
    assert round(stats.average_pause_seconds, 2) == 0.9
    assert stats.pause_to_total_ratio == 0.54
    assert len(stats.pause_intervals) == 3
    assert "timestamp" in stats.note.lower()


def test_pause_detector_over_2s_pause():
    words = [
        WordRecord(word="Start", start=0.0, end=1.0, probability=0.9),
        WordRecord(word="Later", start=3.5, end=4.5, probability=0.9),  # 2.5s gap
    ]
    stats = detect_pauses(words, total_duration_seconds=4.5)

    assert stats.total_pauses == 1
    assert stats.pauses_over_2s == 1
    assert stats.longest_pause_seconds == 2.5


def test_pause_detector_overlapping_and_out_of_order_words():
    words = [
        WordRecord(word="Word2", start=1.2, end=2.0, probability=0.9),  # Out of order
        WordRecord(word="Word1", start=0.0, end=1.5, probability=0.9),  # Overlaps Word2
        WordRecord(word="Word3", start=3.0, end=4.0, probability=0.9),  # Gap from 2.0 to 3.0
    ]
    stats = detect_pauses(words, total_duration_seconds=4.0)

    # Speech intervals merge to [0.0, 2.0] and [3.0, 4.0]
    # One inter-word pause: 2.0 -> 3.0 (1.0s)
    assert stats.total_pauses == 1
    assert stats.pause_intervals[0].start == 2.0
    assert stats.pause_intervals[0].end == 3.0
    assert stats.pause_intervals[0].duration == 1.0


def test_pause_detector_empty_words():
    stats = detect_pauses([], total_duration_seconds=3.0)
    assert stats.total_pauses == 1
    assert stats.total_pause_duration_seconds == 3.0
    assert stats.pause_to_total_ratio == 1.0


def test_pause_detector_zero_duration():
    stats = detect_pauses([], total_duration_seconds=0.0)
    assert stats.total_pauses == 0
    assert stats.total_pause_duration_seconds == 0.0
    assert stats.pause_to_total_ratio == 0.0
