import numpy as np
import pytest

from backend.app.services.transcription import (
    transcribe_audio,
    TranscriptionResult,
    WordRecord,
)
from backend.app.core.exceptions import (
    InterviewSenseException,
    ModelNotLoadedError,
)
from backend.app.core.model_manager import ModelManager
from backend.tests.conftest import MockWhisperModel, MockSegment, MockWord, MockInfo


def test_transcribe_audio_structure_and_timestamp_ordering():
    samples = np.zeros(16000 * 3, dtype=np.float32)
    mock_model = MockWhisperModel()

    result = transcribe_audio(samples, model=mock_model, language="en")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world this is a test"
    assert result.language == "en"
    assert len(result.words) == 6

    # Verify chronological ordering and timestamp bounds
    prev_start = 0.0
    for w in result.words:
        assert isinstance(w, WordRecord)
        assert w.start >= prev_start
        assert w.end >= w.start
        assert 0.0 <= w.probability <= 1.0
        prev_start = w.start


def test_transcribe_audio_empty_segments():
    samples = np.zeros(16000 * 2, dtype=np.float32)

    class EmptyWhisperModel:
        model_size_or_path = "mock-empty"
        def transcribe(self, samples, word_timestamps=True, language="en"):
            return [], MockInfo(language=language)

    result = transcribe_audio(samples, model=EmptyWhisperModel(), language="en")

    assert result.text == ""
    assert result.words == []
    assert result.language == "en"
    assert result.model_used == "mock-empty"


def test_transcribe_audio_model_not_loaded_raises_error():
    samples = np.zeros(16000 * 2, dtype=np.float32)

    with pytest.raises(ModelNotLoadedError) as exc_info:
        transcribe_audio(samples, model=None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "MODEL_NOT_LOADED"


def test_transcribe_audio_invalid_input_type():
    mock_model = MockWhisperModel()

    with pytest.raises(InterviewSenseException) as exc_info:
        transcribe_audio("not-a-numpy-array", model=mock_model)  # type: ignore

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "INVALID_AUDIO_ARRAY"


def test_transcribe_audio_runtime_exception_handled():
    samples = np.zeros(16000 * 2, dtype=np.float32)

    class FailingWhisperModel:
        def transcribe(self, samples, word_timestamps=True, language="en"):
            raise RuntimeError("CTranslate2 internal runtime error")

    with pytest.raises(InterviewSenseException) as exc_info:
        transcribe_audio(samples, model=FailingWhisperModel())

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "TRANSCRIPTION_FAILED"


def test_model_manager_status_and_lifecycle():
    mm = ModelManager()
    assert not mm.is_asr_ready
    status = mm.get_status()
    assert "uninitialized" in status["asr"]

    # Inject mock model
    mock = MockWhisperModel("test-small.en")
    mm.set_asr_model(mock, "test-small.en")
    assert mm.is_asr_ready
    assert mm.asr_model is mock

    status = mm.get_status()
    assert "test-small.en (loaded)" in status["asr"]

    # Unload
    mm.unload_models()
    assert not mm.is_asr_ready
    status = mm.get_status()
    assert "unloaded" in status["asr"]
