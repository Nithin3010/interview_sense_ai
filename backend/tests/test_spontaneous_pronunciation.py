import io
import threading
from typing import List
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.exceptions import InterviewSenseException
from backend.app.core.model_manager import model_manager
from backend.app.services.pronunciation.base import (
    PronunciationAssessmentResult,
    WordPronunciationDiagnostic,
)
from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner
from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel
from backend.app.services.transcription import WordRecord


def _make_sine_wave(duration: float = 2.0, sample_rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return (0.35 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_spontaneous_mode_basic_ctc_alignment():
    # Tokens for "hello" (4=h, 5=ʌ, 6=l, 7=oʊ) and "world" (8=w, 9=ɜː, 6=l, 10=d)
    mock_model = MockPhonemeModel(
        confidence_level=0.89,
        simulate_aligned_tokens=[4, 5, 6, 7, 8, 9, 6, 10],
    )
    aligner = CTCPhonemeAligner(model=mock_model, model_name="mock-spontaneous-wav2vec2")
    samples = _make_sine_wave(duration=2.5)

    words = [
        WordRecord(word="hello", start=0.1, end=0.8, probability=0.98),
        WordRecord(word="world", start=0.9, end=1.7, probability=0.95),
    ]

    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello world",
        reference_text=None,
    )

    assert isinstance(res, PronunciationAssessmentResult)
    assert res.status == "ready"
    assert res.reference_text is None
    assert res.transcript_source == "ASR-generated (faster-whisper)"
    assert res.transcript_evaluated == "hello world"
    assert res.alignment_coverage_ratio == 1.0
    assert res.mean_acoustic_confidence is not None
    assert res.mean_acoustic_confidence > 0.70
    assert len(res.word_diagnostics) == 2

    # Check words
    w0 = res.word_diagnostics[0]
    assert w0.word == "hello"
    assert w0.start >= 0.0
    assert w0.end <= 2.5
    assert len(w0.phonemes) == 4
    for p in w0.phonemes:
        assert p.is_supported is True
        assert p.acoustic_match_confidence is not None
        assert p.start <= p.end
        assert p.duration > 0.0


def test_spontaneous_mode_with_unsupported_phoneme():
    # Vocab missing token 'h' (id 4)
    limited_vocab = {
        "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        "ʌ": 5, "l": 6, "oʊ": 7,
    }
    mock_model = MockPhonemeModel(vocab=limited_vocab, confidence_level=0.85)
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = _make_sine_wave(duration=1.5)

    words = [WordRecord(word="hello", start=0.1, end=0.8, probability=0.95)]
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello",
        reference_text=None,
    )

    assert len(res.word_diagnostics) == 1
    w = res.word_diagnostics[0]
    unsupported_p = [p for p in w.phonemes if not p.is_supported]
    assert len(unsupported_p) >= 1
    assert unsupported_p[0].phoneme == "HH"
    assert unsupported_p[0].acoustic_match_confidence is None
    assert "unsupported" in unsupported_p[0].diagnostic_note.lower()


def test_spontaneous_mode_low_whisper_word_probability_flagged():
    mock_model = MockPhonemeModel(confidence_level=0.90, simulate_aligned_tokens=[4, 5, 6, 7])
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = _make_sine_wave(duration=1.5)

    # Word with very low ASR recognition probability (uncertain hypothesis)
    words = [WordRecord(word="hello", start=0.1, end=0.8, probability=0.25)]
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello",
        reference_text=None,
    )

    assert len(res.word_diagnostics) == 1
    w = res.word_diagnostics[0]
    assert w.flagged is True
    assert w.diagnostic_note is not None
    assert "low asr hypothesis probability" in w.diagnostic_note.lower()


def test_spontaneous_mode_empty_transcript_and_words():
    mock_model = MockPhonemeModel()
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = _make_sine_wave(duration=1.0)

    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=[],
        transcript_text="   ",
        reference_text=None,
    )

    assert res.status == "experimental"
    assert res.reference_text is None
    assert res.word_diagnostics == []
    assert res.alignment_coverage_ratio == 0.0
    assert "empty" in res.methodology_note.lower()


def test_spontaneous_mode_silence_guard():
    mock_model = MockPhonemeModel()
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = np.zeros(16000 * 2, dtype=np.float32)

    words = [WordRecord(word="hello", start=0.1, end=0.8, probability=0.9)]
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello",
        reference_text=None,
    )

    assert res.status == "experimental"
    assert res.reference_text is None
    assert res.word_diagnostics == []
    assert res.alignment_coverage_ratio == 0.0
    assert "silent" in res.methodology_note.lower() or "discernible" in res.methodology_note.lower()


def test_spontaneous_mode_model_unavailable():
    aligner = CTCPhonemeAligner(model=None, model_name="uninitialized-model")
    assert not aligner.is_available
    samples = _make_sine_wave(duration=1.5)

    words = [WordRecord(word="hello", start=0.1, end=0.8, probability=0.95)]
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello",
        reference_text=None,
    )

    assert res.status == "unavailable"
    assert res.reference_text is None
    assert res.word_diagnostics == []
    assert "unavailable" in res.methodology_note.lower()


def test_spontaneous_mode_non_finite_samples_rejection():
    mock_model = MockPhonemeModel()
    aligner = CTCPhonemeAligner(model=mock_model)
    bad_samples = np.array([0.0, 0.5, np.nan, 0.2], dtype=np.float32)

    with pytest.raises(InterviewSenseException) as exc_info:
        aligner.assess(
            samples=bad_samples,
            sample_rate=16000,
            words=[],
            transcript_text="hello",
            reference_text=None,
        )
    assert exc_info.value.error_code == "NON_FINITE_SAMPLES"


def test_spontaneous_mode_boundaries_within_audio_duration():
    duration_s = 2.4
    mock_model = MockPhonemeModel(
        confidence_level=0.91,
        simulate_aligned_tokens=[4, 5, 6, 7, 8, 9, 6, 10],
    )
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = _make_sine_wave(duration=duration_s)

    words = [
        WordRecord(word="hello", start=0.0, end=1.0, probability=0.99),
        WordRecord(word="world", start=1.1, end=2.4, probability=0.99),
    ]

    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=words,
        transcript_text="hello world",
        reference_text=None,
    )

    for w in res.word_diagnostics:
        assert 0.0 <= w.start <= duration_s
        assert 0.0 <= w.end <= duration_s
        assert w.start <= w.end
        for p in w.phonemes:
            if p.acoustic_match_confidence is not None:
                assert 0.0 <= p.start <= duration_s
                assert 0.0 <= p.end <= duration_s
                assert p.start <= p.end


def test_switching_modes_isolation_via_api(client: TestClient, valid_mono_wav_bytes):
    mock_model = MockPhonemeModel(confidence_level=0.92, simulate_aligned_tokens=[4, 5, 6, 7])
    original_model = model_manager._pronunciation_model
    original_scorer = model_manager.pronunciation_scorer
    model_manager.set_pronunciation_model(mock_model, "mock-wav2vec2-api")

    try:
        # Request 1: Reference-based mode
        resp_ref = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
            data={"reference_text": "hello"},
        )
        assert resp_ref.status_code == 200
        data_ref = resp_ref.json()
        assert data_ref["pronunciation"]["reference_text"] == "hello"
        assert data_ref["pronunciation"]["transcript_source"] == "User-provided reference"

        # Request 2: Spontaneous mode (no reference_text)
        resp_spont = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        )
        assert resp_spont.status_code == 200
        data_spont = resp_spont.json()
        # Verify reference_text is strictly null and not leaked from previous request
        assert data_spont["pronunciation"]["reference_text"] is None
        assert data_spont["pronunciation"]["transcript_source"] == "ASR-generated (faster-whisper)"
        assert len(data_spont["pronunciation"]["word_diagnostics"]) > 0

    finally:
        model_manager._pronunciation_model = original_model
        model_manager.pronunciation_scorer = original_scorer


def test_model_reuse_concurrency_spontaneous():
    mock_model = MockPhonemeModel(confidence_level=0.88, simulate_aligned_tokens=[4, 5, 6, 7])
    aligner = CTCPhonemeAligner(model=mock_model)
    samples = _make_sine_wave(duration=2.0)
    words = [WordRecord(word="hello", start=0.1, end=0.7, probability=0.95)]

    results: List[PronunciationAssessmentResult] = []
    errors: List[Exception] = []

    def worker():
        try:
            r = aligner.assess(
                samples=samples,
                sample_rate=16000,
                words=words,
                transcript_text="hello",
                reference_text=None,
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 8
    for r in results:
        assert r.status == "ready"
        assert r.reference_text is None
        assert r.alignment_coverage_ratio == 1.0
