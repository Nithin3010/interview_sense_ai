import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import parselmouth

from backend.app.config import settings
from backend.app.core.exceptions import InterviewSenseException

logger = logging.getLogger("interviewsense.acoustic_analyzer")


@dataclass
class PitchAnalysisResult:
    status: str  # "available" or "unavailable"
    median_f0_hz: Optional[float]
    mean_f0_hz: Optional[float]
    std_f0_hz: Optional[float]
    p05_f0_hz: Optional[float]
    p95_f0_hz: Optional[float]
    range_semitones: Optional[float]
    voiced_fraction: Optional[float]
    pitch_floor_hz: float
    pitch_ceiling_hz: float
    note: str


@dataclass
class EnergyAnalysisResult:
    mean_rms: float
    std_rms: float
    mean_rms_dbfs: float
    dynamic_range_db: float
    frame_length_ms: float
    hop_length_ms: float
    note: str


@dataclass
class AcousticAnalysisResult:
    pitch: PitchAnalysisResult
    energy: EnergyAnalysisResult


def analyze_pitch(
    samples: np.ndarray,
    sample_rate: int = settings.SAMPLE_RATE,
    pitch_floor: float = settings.PITCH_FLOOR_HZ,
    pitch_ceiling: float = settings.PITCH_CEILING_HZ,
    time_step: Optional[float] = settings.PITCH_TIME_STEP_SECONDS,
) -> PitchAnalysisResult:
    """
    Extracts fundamental frequency (F0) estimates using Praat's peer-reviewed
    acoustic autocorrelation engine via parselmouth.
    
    Excludes unvoiced frames, zeros, NaNs, and infinities. Returns an explicit
    'unavailable' status if no voiced pitch candidates exist within configured bounds.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 1:
        raise InterviewSenseException(
            error_code="INVALID_AUDIO_ARRAY",
            message="Pitch analysis requires a 1D NumPy array of audio samples.",
            status_code=400,
        )

    if not np.isfinite(samples).all():
        raise InterviewSenseException(
            error_code="NON_FINITE_SAMPLES",
            message="Audio samples contain non-finite (NaN or Inf) values.",
            status_code=400,
        )

    if len(samples) < int(sample_rate * 0.05):  # Shorter than 50ms
        return PitchAnalysisResult(
            status="unavailable",
            median_f0_hz=None,
            mean_f0_hz=None,
            std_f0_hz=None,
            p05_f0_hz=None,
            p95_f0_hz=None,
            range_semitones=None,
            voiced_fraction=0.0,
            pitch_floor_hz=pitch_floor,
            pitch_ceiling_hz=pitch_ceiling,
            note="Audio signal too short (< 50ms) for reliable pitch candidate extraction.",
        )

    try:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(sample_rate))
        pitch_obj = sound.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        freq_array = pitch_obj.selected_array["frequency"]
        total_frames = max(1, len(freq_array))

        # Filter strictly for valid, positive, finite voiced frames
        voiced_f0 = freq_array[(freq_array > 0.0) & np.isfinite(freq_array)]

        if len(voiced_f0) == 0:
            return PitchAnalysisResult(
                status="unavailable",
                median_f0_hz=None,
                mean_f0_hz=None,
                std_f0_hz=None,
                p05_f0_hz=None,
                p95_f0_hz=None,
                range_semitones=None,
                voiced_fraction=0.0,
                pitch_floor_hz=pitch_floor,
                pitch_ceiling_hz=pitch_ceiling,
                note=(
                    f"No voiced frames detected within configured boundaries ({pitch_floor:.0f}-{pitch_ceiling:.0f} Hz). "
                    "Pitch metrics are unavailable."
                ),
            )

        median_f0 = float(np.median(voiced_f0))
        mean_f0 = float(np.mean(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        p05 = float(np.percentile(voiced_f0, 5))
        p95 = float(np.percentile(voiced_f0, 95))

        if p05 > 0.0 and p95 > 0.0 and p95 >= p05:
            range_st = float(12.0 * np.log2(p95 / p05))
        else:
            range_st = 0.0

        voiced_fraction = float(len(voiced_f0) / total_frames)

        return PitchAnalysisResult(
            status="available",
            median_f0_hz=round(median_f0, 2),
            mean_f0_hz=round(mean_f0, 2),
            std_f0_hz=round(std_f0, 2),
            p05_f0_hz=round(p05, 2),
            p95_f0_hz=round(p95, 2),
            range_semitones=round(range_st, 2),
            voiced_fraction=round(min(1.0, max(0.0, voiced_fraction)), 4),
            pitch_floor_hz=pitch_floor,
            pitch_ceiling_hz=pitch_ceiling,
            note=(
                "Fundamental frequency (F0) extracted using Praat autocorrelation. "
                "Pitch boundaries (65-400 Hz) constrain estimation. Pitch variation reflects intonational modulation, "
                "not personality traits or speaking competence."
            ),
        )

    except Exception as exc:
        logger.warning(f"Pitch extraction encountered an unexpected error: {exc}")
        return PitchAnalysisResult(
            status="unavailable",
            median_f0_hz=None,
            mean_f0_hz=None,
            std_f0_hz=None,
            p05_f0_hz=None,
            p95_f0_hz=None,
            range_semitones=None,
            voiced_fraction=0.0,
            pitch_floor_hz=pitch_floor,
            pitch_ceiling_hz=pitch_ceiling,
            note=f"Pitch extraction failed: {exc}",
        )


def analyze_energy(
    samples: np.ndarray,
    sample_rate: int = settings.SAMPLE_RATE,
    frame_length_ms: float = settings.RMS_FRAME_LENGTH_MS,
    hop_length_ms: float = settings.RMS_HOP_LENGTH_MS,
) -> EnergyAnalysisResult:
    """
    Calculates frame-level RMS energy and digital dynamic range metrics.
    
    NOTE: dBFS measures digital full-scale amplitude relative to digital ceiling (0 dBFS).
    It does not measure acoustic sound pressure level (SPL) or perceived loudness.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 1:
        raise InterviewSenseException(
            error_code="INVALID_AUDIO_ARRAY",
            message="Energy analysis requires a 1D NumPy array of audio samples.",
            status_code=400,
        )

    if not np.isfinite(samples).all():
        raise InterviewSenseException(
            error_code="NON_FINITE_SAMPLES",
            message="Audio samples contain non-finite (NaN or Inf) values.",
            status_code=400,
        )

    frame_len = max(1, int(sample_rate * (frame_length_ms / 1000.0)))
    hop_len = max(1, int(sample_rate * (hop_length_ms / 1000.0)))

    n = len(samples)
    frame_rms_list = []

    if n < frame_len:
        rms_val = float(np.sqrt(np.mean(samples ** 2))) if n > 0 else 0.0
        frame_rms_list.append(rms_val)
    else:
        for start_idx in range(0, n - frame_len + 1, hop_len):
            frame = samples[start_idx : start_idx + frame_len]
            rms_val = float(np.sqrt(np.mean(frame ** 2)))
            frame_rms_list.append(rms_val)

    frame_rms = np.array(frame_rms_list, dtype=np.float64)
    epsilon = 1e-12

    mean_rms = float(np.mean(frame_rms))
    std_rms = float(np.std(frame_rms))
    mean_rms_dbfs = float(20.0 * np.log10(max(mean_rms, epsilon)))

    p95_rms = float(np.percentile(frame_rms, 95))
    p05_rms = float(np.percentile(frame_rms, 5))
    p95_dbfs = float(20.0 * np.log10(max(p95_rms, epsilon)))
    p05_dbfs = float(20.0 * np.log10(max(p05_rms, epsilon)))
    dynamic_range_db = float(max(0.0, p95_dbfs - p05_dbfs))

    return EnergyAnalysisResult(
        mean_rms=round(mean_rms, 6),
        std_rms=round(std_rms, 6),
        mean_rms_dbfs=round(mean_rms_dbfs, 2),
        dynamic_range_db=round(dynamic_range_db, 2),
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        note=(
            "Energy computed as frame-level Root Mean Square (RMS). dBFS measures uncalibrated digital signal amplitude "
            "relative to full scale (0 dBFS), not acoustic sound pressure level (SPL) or vocal assertiveness."
        ),
    )


def analyze_acoustics(
    samples: np.ndarray,
    sample_rate: int = settings.SAMPLE_RATE,
) -> AcousticAnalysisResult:
    """Executes integrated pitch and energy acoustic feature extraction."""
    pitch_res = analyze_pitch(samples, sample_rate=sample_rate)
    energy_res = analyze_energy(samples, sample_rate=sample_rate)
    return AcousticAnalysisResult(pitch=pitch_res, energy=energy_res)
