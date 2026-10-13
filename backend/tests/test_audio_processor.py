import io
import pytest
import soundfile as sf
import numpy as np
from fastapi import UploadFile

from backend.app.services.audio_processor import (
    decode_and_normalize_audio,
    read_upload_with_limit,
)
from backend.app.core.exceptions import (
    AudioFormatError,
    AudioTooLongError,
    AudioTooShortError,
    FileTooLargeError,
    SilentAudioError,
)
from backend.tests.conftest import make_synthetic_wav, make_silent_wav


def test_decode_valid_mono_wav(valid_mono_wav_bytes):
    processed = decode_and_normalize_audio(valid_mono_wav_bytes)

    assert processed.sample_rate == 16000
    assert processed.channels == 1
    assert 2.95 <= processed.duration_seconds <= 3.05
    assert processed.rms_energy > 0.01
    assert processed.peak_amplitude <= 0.95
    assert processed.samples.dtype == np.float32
    assert processed.samples.ndim == 1
    assert np.isfinite(processed.samples).all()


def test_decode_stereo_44khz_resampled(valid_stereo_wav_bytes):
    processed = decode_and_normalize_audio(valid_stereo_wav_bytes)

    assert processed.sample_rate == 16000
    assert processed.channels == 1
    assert 2.45 <= processed.duration_seconds <= 2.55
    assert np.isfinite(processed.samples).all()


def test_decode_resampling_48khz():
    wav_48k = make_synthetic_wav(duration=2.5, sample_rate=48000, frequency=220.0)
    processed = decode_and_normalize_audio(wav_48k)

    assert processed.sample_rate == 16000
    assert processed.channels == 1
    assert 2.45 <= processed.duration_seconds <= 2.55


def test_decode_ogg_format():
    buf = io.BytesIO()
    data = 0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.5, int(2.5 * 22050), endpoint=False)).astype(np.float32)
    sf.write(buf, data, 22050, format="OGG")
    processed = decode_and_normalize_audio(buf.getvalue())

    assert processed.sample_rate == 16000
    assert processed.channels == 1
    assert 2.4 <= processed.duration_seconds <= 2.6
    assert processed.original_format == "ogg"


def test_decode_flac_format():
    buf = io.BytesIO()
    data = 0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.5, int(2.5 * 44100), endpoint=False)).astype(np.float32)
    sf.write(buf, data, 44100, format="FLAC")
    processed = decode_and_normalize_audio(buf.getvalue())

    assert processed.sample_rate == 16000
    assert processed.channels == 1
    assert 2.45 <= processed.duration_seconds <= 2.55
    assert processed.original_format == "flac"


def test_decode_short_audio_raises_error(short_wav_bytes):
    with pytest.raises(AudioTooShortError) as exc_info:
        decode_and_normalize_audio(short_wav_bytes)
    assert exc_info.value.status_code == 422
    assert "below the minimum required" in exc_info.value.message


def test_decode_long_audio_raises_error():
    # Test duration exceeding max allowed (300 seconds)
    # We can test by mocking or constructing a buffer with metadata or checking boundary
    wav_305s = make_synthetic_wav(duration=301.0, sample_rate=8000)
    with pytest.raises(AudioTooLongError) as exc_info:
        decode_and_normalize_audio(wav_305s)
    assert exc_info.value.status_code == 422
    assert "exceeds the maximum allowed" in exc_info.value.message


def test_decode_silent_audio_raises_error(silent_wav_bytes):
    with pytest.raises(SilentAudioError) as exc_info:
        decode_and_normalize_audio(silent_wav_bytes)
    assert exc_info.value.status_code == 422
    assert "below the audible speech threshold" in exc_info.value.message


def test_decode_empty_bytes_raises_error():
    with pytest.raises(AudioFormatError) as exc_info:
        decode_and_normalize_audio(b"")
    assert exc_info.value.status_code == 400


def test_decode_garbage_bytes_raises_error():
    with pytest.raises(AudioFormatError) as exc_info:
        decode_and_normalize_audio(b"random-non-audio-binary-payload")
    assert exc_info.value.status_code == 400


def test_decode_nan_values_raises_error():
    buf = io.BytesIO()
    data = np.full(16000 * 3, np.nan, dtype=np.float32)
    sf.write(buf, data, 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(AudioFormatError) as exc_info:
        decode_and_normalize_audio(buf.getvalue())
    assert exc_info.value.status_code == 400
    assert "non-finite" in exc_info.value.message


def test_decode_clipping_normalization():
    # Audio with deliberate peak exceeding 1.0 (amplitude = 2.5)
    loud_wav = make_synthetic_wav(duration=2.5, sample_rate=16000, amplitude=2.5)
    processed = decode_and_normalize_audio(loud_wav)

    # Safe normalization scales down signal so peak does not exceed 0.95
    assert processed.peak_amplitude <= 0.950001
    assert np.all(np.abs(processed.samples) <= 0.950001)
    assert np.isfinite(processed.samples).all()


@pytest.mark.anyio
async def test_read_upload_with_limit_enforced():
    large_payload = b"x" * 1024  # 1 KB
    small_limit = 512            # Limit to 512 bytes

    upload_file = UploadFile(
        file=io.BytesIO(large_payload),
        filename="test.wav",
    )

    with pytest.raises(FileTooLargeError) as exc_info:
        await read_upload_with_limit(upload_file, max_bytes=small_limit, chunk_size=256)

    assert exc_info.value.status_code == 413


@pytest.mark.anyio
async def test_read_upload_empty_file_rejected():
    upload_file = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.wav",
    )

    with pytest.raises(AudioFormatError) as exc_info:
        await read_upload_with_limit(upload_file)

    assert exc_info.value.status_code == 400
