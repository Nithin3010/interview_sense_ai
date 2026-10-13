"""
Milestone 8 Reliability and Edge-Case Regression Tests
-------------------------------------------------------
Verifies calculations, boundaries, non-finite rejection, and error responses.
"""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.core.exceptions import InterviewSenseException
from backend.app.services.acoustic_analyzer import analyze_pitch, analyze_energy
from backend.app.services.fluency_analyzer import analyze_fluency
from backend.app.services.pause_detector import detect_pauses, PauseStatistics, PauseInterval
from backend.tests.conftest import MockWord, make_synthetic_wav, make_silent_wav


def test_fluency_edge_cases_zero_and_exceeded_pauses():
    """Verifies gross WPM and articulation rate calculations with zero duration or exceeded pauses."""
    # 1. Zero words and zero duration
    empty_stats = PauseStatistics(
        total_pauses=0,
        total_pause_duration_seconds=0.0,
        pauses_over_500ms=0,
        pauses_over_2s=0,
        longest_pause_seconds=0.0,
        average_pause_seconds=0.0,
        pause_to_total_ratio=0.0,
        pause_intervals=[],
    )
    res = analyze_fluency(words=[], total_duration_seconds=0.0, pause_stats=empty_stats)
    assert res.gross_wpm == 0.0
    assert res.articulation_rate_wpm == 0.0
    assert res.speaking_duration_seconds == 0.0
    assert res.filler_percentage == 0.0

    # 2. Total pause duration reported larger than total audio duration
    exceeded_pause_stats = PauseStatistics(
        total_pauses=1,
        total_pause_duration_seconds=15.0,
        pauses_over_500ms=1,
        pauses_over_2s=1,
        longest_pause_seconds=15.0,
        average_pause_seconds=15.0,
        pause_to_total_ratio=1.0,
        pause_intervals=[PauseInterval(start=0.0, end=15.0, duration=15.0)],
    )
    words = [MockWord("test", 0.0, 0.5)]
    res_exceeded = analyze_fluency(words=words, total_duration_seconds=10.0, pause_stats=exceeded_pause_stats)
    assert res_exceeded.speaking_duration_seconds == 0.0
    assert res_exceeded.articulation_rate_wpm == res_exceeded.gross_wpm
    assert not np.isnan(res_exceeded.gross_wpm)
    assert not np.isnan(res_exceeded.articulation_rate_wpm)


def test_pause_detector_ratio_and_duration_bounds():
    """Verifies that pause-to-total ratio is clamped to [0.0, 1.0]."""
    # Audio with out-of-bounds timestamps
    words = [
        MockWord("first", -2.0, 5.0),
        MockWord("second", 6.0, 20.0),  # extends past 10s audio duration
    ]
    stats = detect_pauses(words=words, total_duration_seconds=10.0)
    assert 0.0 <= stats.pause_to_total_ratio <= 1.0
    assert stats.total_pause_duration_seconds <= 10.0
    for p in stats.pause_intervals:
        assert 0.0 <= p.start <= 10.0
        assert 0.0 <= p.end <= 10.0
        assert p.duration >= 0.0


def test_acoustics_non_finite_sample_rejection():
    """Verifies rejection of NaN and Inf sample arrays."""
    nan_samples = np.array([0.0, np.nan, 0.5], dtype=np.float32)
    with pytest.raises(InterviewSenseException) as exc_info:
        analyze_pitch(nan_samples)
    assert exc_info.value.error_code == "NON_FINITE_SAMPLES"

    inf_samples = np.array([0.0, np.inf, 0.5], dtype=np.float32)
    with pytest.raises(InterviewSenseException) as exc_info:
        analyze_energy(inf_samples)
    assert exc_info.value.error_code == "NON_FINITE_SAMPLES"


def test_acoustics_ultra_short_signal():
    """Verifies that signals shorter than 50ms return unavailable pitch safely."""
    short_samples = np.zeros(200, dtype=np.float32)  # 200 samples at 16kHz = 12.5ms
    pitch_res = analyze_pitch(short_samples)
    assert pitch_res.status == "unavailable"
    assert pitch_res.median_f0_hz is None
    assert pitch_res.range_semitones is None
    assert pitch_res.voiced_fraction == 0.0


def test_api_upload_limits_and_validation(client: TestClient):
    """Verifies API responses for too-short, silent, and oversized audio."""
    # 1. Too short (< 2.0s) -> 422 Unprocessable Entity
    short_wav = make_synthetic_wav(duration=1.2)
    res_short = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("short.wav", io.BytesIO(short_wav), "audio/wav")},
        data={"language": "en"},
    )
    assert res_short.status_code == 422
    assert res_short.json()["detail"]["error_code"] == "AUDIO_TOO_SHORT"

    # 2. Silent audio -> 422 Unprocessable Entity
    silent_wav = make_silent_wav(duration=3.0)
    res_silent = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("silent.wav", io.BytesIO(silent_wav), "audio/wav")},
        data={"language": "en"},
    )
    assert res_silent.status_code == 422
    assert res_silent.json()["detail"]["error_code"] == "SILENT_AUDIO"

    # 3. Corrupt/malformed audio -> 400 Bad Request
    res_corrupt = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("corrupt.wav", io.BytesIO(b"NOT_A_VALID_AUDIO_FILE"), "audio/wav")},
        data={"language": "en"},
    )
    assert res_corrupt.status_code == 400
    assert res_corrupt.json()["detail"]["error_code"] == "INVALID_AUDIO_FORMAT"


def test_api_consecutive_requests_model_reuse(client: TestClient):
    """Verifies that consecutive requests succeed and reuse the model manager."""
    valid_wav = make_synthetic_wav(duration=3.0)
    for i in range(3):
        res = client.post(
            "/api/v1/analyze",
            files={"audio_file": (f"test_{i}.wav", io.BytesIO(valid_wav), "audio/wav")},
            data={"language": "en"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "validated"
        assert data["metadata"]["duration_seconds"] == 3.0
        assert data["pronunciation"]["status"] == "unavailable"
