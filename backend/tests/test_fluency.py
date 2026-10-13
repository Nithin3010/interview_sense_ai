from backend.app.services.transcription import WordRecord
from backend.app.services.pause_detector import PauseStatistics, PauseInterval
from backend.app.services.fluency_analyzer import (
    analyze_fluency,
    detect_fillers,
    detect_repetitions,
    clean_token,
)


def test_clean_token():
    assert clean_token("Hello,") == "hello"
    assert clean_token("...world!") == "world"
    assert clean_token('"Um"') == "um"


def test_wpm_and_articulation_rate_calculation():
    # 10 words over 4.0 seconds total duration, with 1.0s of pauses
    words = [
        WordRecord(word=f"word{i}", start=i * 0.3, end=i * 0.3 + 0.2, probability=0.9)
        for i in range(10)
    ]
    pause_stats = PauseStatistics(
        total_pauses=1,
        total_pause_duration_seconds=1.0,
        pauses_over_500ms=1,
        pauses_over_2s=0,
        longest_pause_seconds=1.0,
        average_pause_seconds=1.0,
        pause_to_total_ratio=0.25,
        pause_intervals=[PauseInterval(start=1.0, end=2.0, duration=1.0)],
    )

    metrics = analyze_fluency(words, total_duration_seconds=4.0, pause_stats=pause_stats)

    assert metrics.word_count == 10
    # Gross WPM: (10 words / 4.0s) * 60 = 150.0
    assert metrics.gross_wpm == 150.0
    # Articulation Rate: (10 words / 3.0s active) * 60 = 200.0
    assert metrics.articulation_rate_wpm == 200.0
    assert metrics.speaking_duration_seconds == 3.0
    assert metrics.total_pause_duration_seconds == 1.0


def test_fluency_empty_words():
    pause_stats = PauseStatistics(
        total_pauses=0,
        total_pause_duration_seconds=0.0,
        pauses_over_500ms=0,
        pauses_over_2s=0,
        longest_pause_seconds=0.0,
        average_pause_seconds=0.0,
        pause_to_total_ratio=0.0,
        pause_intervals=[],
    )
    metrics = analyze_fluency([], total_duration_seconds=5.0, pause_stats=pause_stats)

    assert metrics.word_count == 0
    assert metrics.gross_wpm == 0.0
    assert metrics.articulation_rate_wpm == 0.0
    assert metrics.filler_count == 0
    assert metrics.filler_percentage == 0.0


def test_detect_vocal_and_potential_fillers():
    words = [
        WordRecord(word="Well,", start=0.0, end=0.3, probability=0.9),
        WordRecord(word="um", start=0.4, end=0.7, probability=0.9),
        WordRecord(word="I", start=0.8, end=1.0, probability=0.9),
        WordRecord(word="like", start=1.1, end=1.4, probability=0.9),
        WordRecord(word="basically", start=1.5, end=1.9, probability=0.9),
    ]

    fillers = detect_fillers(words)

    assert len(fillers) == 3
    # "um" -> vocal_disfluency
    assert fillers[0].word == "um"
    assert fillers[0].category == "vocal_disfluency"
    assert fillers[0].start == 0.4

    # "like" -> potential_filler
    assert fillers[1].word == "like"
    assert fillers[1].category == "potential_filler"

    # "basically" -> potential_filler
    assert fillers[2].word == "basically"
    assert fillers[2].category == "potential_filler"


def test_detect_multiword_fillers():
    words = [
        WordRecord(word="And", start=0.0, end=0.2, probability=0.9),
        WordRecord(word="you", start=0.3, end=0.5, probability=0.9),
        WordRecord(word="know,", start=0.6, end=0.9, probability=0.9),
        WordRecord(word="it", start=1.0, end=1.2, probability=0.9),
        WordRecord(word="was", start=1.3, end=1.5, probability=0.9),
        WordRecord(word="sort", start=1.6, end=1.8, probability=0.9),
        WordRecord(word="of", start=1.9, end=2.1, probability=0.9),
        WordRecord(word="cool", start=2.2, end=2.5, probability=0.9),
    ]

    fillers = detect_fillers(words)

    assert len(fillers) == 2
    assert fillers[0].word == "you know,"
    assert fillers[0].start == 0.3
    assert fillers[0].end == 0.9
    assert fillers[0].category == "conversational_filler"

    assert fillers[1].word == "sort of"
    assert fillers[1].start == 1.6
    assert fillers[1].end == 2.1
    assert fillers[1].category == "conversational_filler"


def test_avoid_substring_false_positives():
    words = [
        WordRecord(word="Somewhere", start=0.0, end=0.4, probability=0.9),  # Should not match 'so'
        WordRecord(word="likeable", start=0.5, end=0.9, probability=0.9),   # Should not match 'like'
        WordRecord(word="human", start=1.0, end=1.4, probability=0.9),      # Should not match 'um'
    ]

    fillers = detect_fillers(words)
    assert len(fillers) == 0


def test_detect_immediate_word_repetitions():
    words = [
        WordRecord(word="I,", start=0.0, end=0.2, probability=0.9),
        WordRecord(word="I", start=0.3, end=0.5, probability=0.9),
        WordRecord(word="think", start=0.6, end=0.8, probability=0.9),
    ]

    reps = detect_repetitions(words)

    assert len(reps) == 1
    assert reps[0].repeated_text == "I,"
    assert reps[0].repetition_type == "word_repetition"
    assert reps[0].start == 0.0
    assert reps[0].end == 0.5


def test_detect_immediate_phrase_repetitions():
    words = [
        WordRecord(word="we", start=0.0, end=0.2, probability=0.9),
        WordRecord(word="have", start=0.3, end=0.5, probability=0.9),
        WordRecord(word="to,", start=0.6, end=0.8, probability=0.9),
        WordRecord(word="we", start=0.9, end=1.1, probability=0.9),
        WordRecord(word="have", start=1.2, end=1.4, probability=0.9),
        WordRecord(word="to", start=1.5, end=1.7, probability=0.9),
        WordRecord(word="go", start=1.8, end=2.0, probability=0.9),
    ]

    reps = detect_repetitions(words)

    assert len(reps) == 1
    assert reps[0].repeated_text == "we have to,"
    assert reps[0].repetition_type == "phrase_repetition"
    assert reps[0].start == 0.0
    assert reps[0].end == 1.7
