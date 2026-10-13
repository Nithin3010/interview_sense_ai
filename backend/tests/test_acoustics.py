import numpy as np
import pytest

from backend.app.services.acoustic_analyzer import (
    analyze_pitch,
    analyze_energy,
    analyze_acoustics,
    PitchAnalysisResult,
    EnergyAnalysisResult,
)
from backend.app.core.exceptions import InterviewSenseException


def test_pitch_voiced_sine_200hz():
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 200 Hz pure tone (within normal pitch bounds 65 - 400 Hz)
    samples = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

    res = analyze_pitch(samples, sample_rate=sample_rate)

    assert isinstance(res, PitchAnalysisResult)
    assert res.status == "available"
    assert res.median_f0_hz is not None
    assert 198.0 <= res.median_f0_hz <= 202.0
    assert 198.0 <= res.mean_f0_hz <= 202.0
    assert res.std_f0_hz <= 5.0
    assert res.voiced_fraction > 0.8
    assert res.pitch_floor_hz == 65.0
    assert res.pitch_ceiling_hz == 400.0


def test_pitch_voiced_sine_150hz():
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (0.6 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

    res = analyze_pitch(samples, sample_rate=sample_rate)

    assert res.status == "available"
    assert res.median_f0_hz is not None
    assert 148.0 <= res.median_f0_hz <= 152.0


def test_pitch_silence_returns_unavailable():
    # Pure zero array produces zero voiced frames
    samples = np.zeros(16000 * 2, dtype=np.float32)

    res = analyze_pitch(samples, sample_rate=16000)

    assert res.status == "unavailable"
    assert res.median_f0_hz is None
    assert res.mean_f0_hz is None
    assert res.std_f0_hz is None
    assert res.p05_f0_hz is None
    assert res.p95_f0_hz is None
    assert res.range_semitones is None
    assert res.voiced_fraction == 0.0
    assert "No voiced frames detected" in res.note


def test_pitch_short_signal_returns_unavailable():
    # Signal shorter than 50ms (< 800 samples at 16kHz)
    short_samples = np.zeros(400, dtype=np.float32)

    res = analyze_pitch(short_samples, sample_rate=16000)

    assert res.status == "unavailable"
    assert res.median_f0_hz is None
    assert "too short" in res.note


def test_pitch_invalid_input_type_raises_error():
    with pytest.raises(InterviewSenseException) as exc_info:
        analyze_pitch("not-an-array")  # type: ignore
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "INVALID_AUDIO_ARRAY"


def test_pitch_non_finite_samples_raises_error():
    samples = np.full(16000, np.nan, dtype=np.float32)
    with pytest.raises(InterviewSenseException) as exc_info:
        analyze_pitch(samples)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "NON_FINITE_SAMPLES"


def test_energy_pure_sine_wave():
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave with peak amplitude A = 1.0. Theoretical continuous RMS is 1.0 / sqrt(2) ≈ 0.7071
    samples = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    res = analyze_energy(samples, sample_rate=sample_rate)

    assert isinstance(res, EnergyAnalysisResult)
    assert 0.70 <= res.mean_rms <= 0.72
    # Theoretical dBFS = 20 * log10(0.7071) ≈ -3.01 dBFS
    assert -3.10 <= res.mean_rms_dbfs <= -2.90
    assert res.frame_length_ms == 25.0
    assert res.hop_length_ms == 10.0


def test_energy_silence_safe_dbfs():
    samples = np.zeros(16000 * 2, dtype=np.float32)

    res = analyze_energy(samples, sample_rate=16000)

    assert res.mean_rms == 0.0
    assert res.std_rms == 0.0
    # Uses epsilon (1e-12) -> does not crash or return -inf/NaN
    assert np.isfinite(res.mean_rms_dbfs)
    assert res.mean_rms_dbfs <= -200.0
    assert res.dynamic_range_db == 0.0


def test_energy_dynamic_range_variation():
    sample_rate = 16000
    # First second: quiet tone (amplitude 0.01)
    # Second second: loud tone (amplitude 0.8)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    quiet_part = (0.01 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    loud_part = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    combined = np.concatenate([quiet_part, loud_part])

    res = analyze_energy(combined, sample_rate=sample_rate)

    # Dynamic range should clearly capture the amplitude difference (roughly 38 dB)
    assert res.dynamic_range_db >= 25.0


def test_analyze_acoustics_integration():
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    acoustics = analyze_acoustics(samples, sample_rate=sample_rate)

    assert acoustics.pitch.status == "available"
    assert 178.0 <= acoustics.pitch.median_f0_hz <= 182.0
    assert acoustics.energy.mean_rms > 0.0
