import io
import os
import sys
from typing import Any, List
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

# Add workspace root to sys.path so 'backend.app...' imports work seamlessly
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.app.main import app
from backend.app.core.model_manager import model_manager


class MockWord:
    def __init__(self, word: str, start: float, end: float, probability: float = 0.95):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class MockSegment:
    def __init__(self, text: str, words: List[MockWord]):
        self.text = text
        self.words = words


class MockInfo:
    def __init__(self, language: str = "en"):
        self.language = language


class MockWhisperModel:
    """Mock ASR model avoiding network downloads in unit and integration tests."""
    def __init__(self, model_size_or_path: str = "mock-small.en"):
        self.model_size_or_path = model_size_or_path

    def transcribe(self, samples: Any, word_timestamps: bool = True, language: str = "en"):
        words = [
            MockWord("Hello", 0.12, 0.45, 0.98),
            MockWord("world", 0.48, 0.92, 0.96),
            MockWord("this", 1.05, 1.25, 0.99),
            MockWord("is", 1.28, 1.42, 0.97),
            MockWord("a", 1.45, 1.55, 0.95),
            MockWord("test", 1.58, 2.05, 0.99),
        ]
        segment = MockSegment("Hello world this is a test", words)
        return [segment], MockInfo(language=language)


@pytest.fixture(autouse=True, scope="session")
def setup_test_model():
    """Inject mock ASR model to guarantee no external weights download during test suite."""
    mock_model = MockWhisperModel()
    model_manager.set_asr_model(mock_model, "mock-small.en")
    original_load = model_manager.load_models
    original_unload = model_manager.unload_models

    model_manager.load_models = lambda: model_manager.set_asr_model(mock_model, "mock-small.en")
    model_manager.unload_models = lambda: None

    yield mock_model

    model_manager.load_models = original_load
    model_manager.unload_models = original_unload


@pytest.fixture(scope="session")
def client():
    """Test client fixture running lifespan handlers."""
    with TestClient(app) as test_client:
        yield test_client


def make_synthetic_wav(
    duration: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    channels: int = 1,
    amplitude: float = 0.5,
) -> bytes:
    """Generates an in-memory WAV byte stream with a pure sine wave."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine = amplitude * np.sin(2 * np.pi * frequency * t)

    if channels == 1:
        data = sine.astype(np.float32)
    else:
        data = np.stack([sine] * channels, axis=-1).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def make_silent_wav(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generates an in-memory WAV byte stream containing pure digital silence."""
    num_samples = int(duration * sample_rate)
    data = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture
def valid_mono_wav_bytes():
    return make_synthetic_wav(duration=3.0, sample_rate=16000, channels=1)


@pytest.fixture
def valid_stereo_wav_bytes():
    return make_synthetic_wav(duration=2.5, sample_rate=44100, channels=2)


@pytest.fixture
def short_wav_bytes():
    return make_synthetic_wav(duration=1.2, sample_rate=16000, channels=1)


@pytest.fixture
def silent_wav_bytes():
    return make_silent_wav(duration=3.0, sample_rate=16000)
