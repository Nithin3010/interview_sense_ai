import io
import logging
from dataclasses import dataclass
from typing import Optional

import av
from fastapi import UploadFile
import numpy as np

from backend.app.config import settings
from backend.app.core.exceptions import (
    AudioFormatError,
    AudioTooLongError,
    AudioTooShortError,
    FileTooLargeError,
    SilentAudioError,
)

logger = logging.getLogger("interviewsense.audio_processor")


@dataclass
class ProcessedAudio:
    """Canonical audio representation for downstream speech and acoustic analysis."""
    samples: np.ndarray          # 1D float32 array normalized to [-1.0, 1.0]
    sample_rate: int             # Target sample rate (16000 Hz)
    duration_seconds: float      # Duration in seconds
    channels: int                # 1 (Mono)
    rms_energy: float            # RMS energy of the signal
    peak_amplitude: float        # Peak absolute amplitude
    original_format: Optional[str] = None


async def read_upload_with_limit(
    upload_file: UploadFile,
    max_bytes: int = settings.MAX_FILE_SIZE_BYTES,
    chunk_size: int = 64 * 1024,
) -> bytes:
    """
    Reads an uploaded file in chunks, enforcing the payload limit strictly
    during streaming to avoid unbounded memory allocation.
    """
    buffer = bytearray()
    total_bytes = 0

    try:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise FileTooLargeError(size_bytes=total_bytes, max_bytes=max_bytes)
            buffer.extend(chunk)
    finally:
        await upload_file.close()

    if total_bytes == 0:
        raise AudioFormatError("Uploaded file is empty (0 bytes).")

    return bytes(buffer)


def decode_and_normalize_audio(audio_bytes: bytes) -> ProcessedAudio:
    """
    Decodes an in-memory audio byte stream into 16kHz mono float32 PCM samples
    using PyAV without requiring external system FFmpeg binaries.
    
    Validates decodability, duration, finite sample values, and silence.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise AudioFormatError("Uploaded audio buffer is empty.")

    bio = io.BytesIO(audio_bytes)
    container = None

    try:
        container = av.open(bio)
    except Exception as exc:
        logger.warning(f"Audio container open failed: {exc}")
        raise AudioFormatError(f"Unable to decode audio container: {exc}")

    try:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise AudioFormatError("No audio stream found in the uploaded file.")

        stream = audio_streams[0]
        detected_format = getattr(container.format, "name", None)

        # PyAV AudioResampler for conversion to 16000 Hz, mono, float planar
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=settings.SAMPLE_RATE
        )

        decoded_chunks = []
        try:
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for rf in resampled_frames:
                        decoded_chunks.append(rf.to_ndarray())

            # Flush any trailing resampler frames
            flushed_frames = resampler.resample(None)
            if flushed_frames:
                for rf in flushed_frames:
                    decoded_chunks.append(rf.to_ndarray())
        except Exception as exc:
            logger.warning(f"Audio frame decoding failed: {exc}")
            raise AudioFormatError(f"Error while decoding audio frames: {exc}")

        if not decoded_chunks:
            raise AudioFormatError("Audio stream contains no decodable audio frames.")

        # Concatenate into 1D mono float32 array
        samples = np.concatenate(decoded_chunks, axis=1).squeeze()
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0)
        samples = samples.astype(np.float32)

        # Validate finite values (reject NaN or Inf)
        if not np.isfinite(samples).all():
            raise AudioFormatError("Audio stream contains non-finite (NaN or Inf) sample values.")

        # Check duration
        duration_seconds = float(len(samples) / settings.SAMPLE_RATE)
        if duration_seconds < settings.MIN_DURATION_SECONDS:
            raise AudioTooShortError(
                duration=duration_seconds,
                min_duration=settings.MIN_DURATION_SECONDS
            )
        if duration_seconds > settings.MAX_DURATION_SECONDS:
            raise AudioTooLongError(
                duration=duration_seconds,
                max_duration=settings.MAX_DURATION_SECONDS
            )

        # Silence Validation (MUST happen before normalization to avoid amplifying digital silence)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < settings.SILENCE_RMS_THRESHOLD:
            raise SilentAudioError("Audio is silent or below the audible speech threshold.")

        # Safe Normalization:
        # If signal peaks near or above clipping (> 0.98), rescale smoothly to 0.95 peak
        peak = float(np.max(np.abs(samples)))
        if peak > 0.98:
            samples = samples * (0.95 / peak)
        elif peak > 1.0:
            samples = np.clip(samples, -1.0, 1.0)

        # Recalculate definitive metrics
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2)))

        return ProcessedAudio(
            samples=samples,
            sample_rate=settings.SAMPLE_RATE,
            duration_seconds=round(duration_seconds, 3),
            channels=1,
            rms_energy=round(rms, 6),
            peak_amplitude=round(peak, 6),
            original_format=detected_format,
        )

    finally:
        if container is not None:
            container.close()
