import numpy as np
import pytest

from backend.app.services.pronunciation.base import (
    PronunciationAssessmentResult,
    PronunciationScorer,
)
from backend.app.services.pronunciation.g2p import word_to_phonemes
from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner
from backend.app.services.transcription import WordRecord
from backend.app.core.exceptions import InterviewSenseException


def test_g2p_word_to_phonemes_known_and_fallback():
    # Known lexicon words
    assert word_to_phonemes("hello") == ["HH", "AH0", "L", "OW1"]
    assert word_to_phonemes("world") == ["W", "ER1", "L", "D"]
    assert word_to_phonemes("test") == ["T", "EH1", "S", "T"]

    # Digraph / fallback word
    phonemes = word_to_phonemes("speech")
    assert "CH" in phonemes
    assert len(phonemes) >= 3

    # Empty string
    assert word_to_phonemes("") == []


def test_ctc_aligner_unavailable_when_uninitialized():
    aligner = CTCPhonemeAligner(model=None)
    assert not aligner.is_available

    samples = np.zeros(16000 * 2, dtype=np.float32)
    words = [WordRecord(word="hello", start=0.0, end=0.5, probability=0.98)]

    res = aligner.assess(samples, 16000, words, "hello")

    assert isinstance(res, PronunciationAssessmentResult)
    assert res.status == "unavailable"
    assert res.word_diagnostics == []
    assert "unavailable" in res.methodology_note.lower()
    assert res.transcript_source == "ASR-generated (faster-whisper)"


def test_ctc_aligner_experimental_with_mock_model():
    class MockCTCModel:
        def score_phoneme(self, phoneme: str, start: float, end: float) -> float:
            return 0.88

    aligner = CTCPhonemeAligner(model=MockCTCModel(), model_name="mock-ctc-phoneme")
    assert aligner.is_available

    samples = np.zeros(16000 * 2, dtype=np.float32)
    words = [
        WordRecord(word="hello", start=0.1, end=0.6, probability=0.95),
        WordRecord(word="world", start=0.7, end=1.2, probability=0.99),
    ]

    res = aligner.assess(samples, 16000, words, "hello world")

    assert res.status == "experimental"
    assert res.model_identifier == "mock-ctc-phoneme"
    assert res.phoneme_inventory == "ARPAbet-39"
    assert res.transcript_evaluated == "hello world"
    assert res.transcript_source == "ASR-generated (faster-whisper)"
    assert len(res.word_diagnostics) == 2

    hello_diag = res.word_diagnostics[0]
    assert hello_diag.word == "hello"
    assert hello_diag.start == 0.1
    assert hello_diag.end == 0.6
    assert len(hello_diag.phonemes) == 4
    assert hello_diag.phonemes[0].phoneme == "HH"
    assert hello_diag.phonemes[0].acoustic_match_confidence == 0.88

    # Ensure timestamps are ordered within word bounds
    prev_start = 0.1
    for ph in hello_diag.phonemes:
        assert ph.start >= prev_start
        assert ph.end >= ph.start
        prev_start = ph.start


def test_ctc_aligner_empty_transcript():
    class MockCTCModel:
        pass

    aligner = CTCPhonemeAligner(model=MockCTCModel())
    samples = np.zeros(16000, dtype=np.float32)

    res = aligner.assess(samples, 16000, [], "")
    assert res.status == "experimental"
    assert res.word_diagnostics == []


def test_ctc_aligner_invalid_samples():
    aligner = CTCPhonemeAligner(model=object())
    with pytest.raises(InterviewSenseException):
        aligner.assess("not-an-array", 16000, [], "text")  # type: ignore

    with pytest.raises(InterviewSenseException):
        aligner.assess(np.array([np.nan]), 16000, [], "text")


def test_ctc_aligner_runtime_failure_reports_error_status():
    class FailingCTCModel:
        def score_phoneme(self, phoneme, start, end):
            raise RuntimeError("Model execution failed")

    aligner = CTCPhonemeAligner(model=FailingCTCModel())
    samples = np.zeros(16000, dtype=np.float32)
    words = [WordRecord(word="test", start=0.0, end=0.4, probability=0.9)]

    res = aligner.assess(samples, 16000, words, "test")
    assert res.status == "error"
    assert res.error_message is not None
    assert "Model execution failed" in res.error_message


def test_reference_text_validation_success_and_errors():
    from backend.app.services.pronunciation.g2p import validate_reference_text
    from backend.app.core.exceptions import InvalidReferenceTextError

    # Valid sentences
    valid1 = "I have worked on distributed systems for three years."
    assert validate_reference_text(valid1) == valid1

    valid2 = "  Thank you   for the opportunity   today!  "
    assert validate_reference_text(valid2) == "Thank you for the opportunity today!"

    # None
    with pytest.raises(InvalidReferenceTextError, match="cannot be null"):
        validate_reference_text(None)

    # Empty string
    with pytest.raises(InvalidReferenceTextError, match="cannot be empty"):
        validate_reference_text("")

    # Whitespace only
    with pytest.raises(InvalidReferenceTextError, match="cannot be empty"):
        validate_reference_text("   \n\t  ")

    # Excessively long text (>80 words)
    long_words = " ".join(["word"] * 85)
    with pytest.raises(InvalidReferenceTextError, match="exceeds maximum supported practice length"):
        validate_reference_text(long_words)

    # Excessively long text (>500 characters)
    long_chars = "a" * 505
    with pytest.raises(InvalidReferenceTextError, match="exceeds maximum allowed length"):
        validate_reference_text(long_chars)

    # Unsupported characters / excessive symbols
    bad_symbols = "Testing $$$$$$$$ %%%%%%%% ^^^^^^^^ &&&&&&&& ******* !!!"
    with pytest.raises(InvalidReferenceTextError, match="unsupported symbols"):
        validate_reference_text(bad_symbols)


def test_text_to_reference_phonemes_mapping_and_inventory():
    from backend.app.services.pronunciation.g2p import text_to_reference_phonemes

    words = text_to_reference_phonemes("hello world")
    assert len(words) == 2
    assert words[0].word == "hello"
    assert words[1].word == "world"

    hello_phones = words[0].phonemes
    assert len(hello_phones) == 4
    assert hello_phones[0].phoneme_base == "HH"
    assert hello_phones[0].model_token == "h"
    assert hello_phones[0].is_supported is True

    # Check ARPAbet 39 to eSpeak inventory coverage
    from backend.app.services.pronunciation.g2p import ARPABET_TO_ESPEAK
    assert len(ARPABET_TO_ESPEAK) == 39
    assert ARPABET_TO_ESPEAK["TH"] == "θ"
    assert ARPABET_TO_ESPEAK["SH"] == "ʃ"
    assert ARPABET_TO_ESPEAK["AE"] == "æ"


def test_ctc_viterbi_alignment_algorithm_deterministic():
    from backend.app.services.pronunciation.ctc_aligner import ctc_viterbi_alignment

    # 10 frames, vocab size 5
    # Token 0 is blank
    # Target tokens: [2, 3]
    T, V = 10, 5
    emissions = np.full((T, V), -5.0, dtype=np.float32)

    # High blank probability in frame 0 and frame 9
    emissions[0, 0] = -0.1
    emissions[9, 0] = -0.1

    # High token 2 probability in frames 1-4
    emissions[1:5, 2] = -0.05

    # High token 3 probability in frames 5-8
    emissions[5:9, 3] = -0.05

    spans = ctc_viterbi_alignment(emissions, [2, 3], blank_id=0)
    assert len(spans) == 2
    assert spans[0] is not None
    assert spans[1] is not None

    # Check that boundaries are ordered
    st0, en0 = spans[0]
    st1, en1 = spans[1]
    assert st0 <= en0
    assert en0 <= st1
    assert st1 <= en1


def test_reference_based_assessment_with_mock_model():
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    mock_model = MockPhonemeModel(
        confidence_level=0.92,
        # Tokens: "h"=4, "ʌ"=5, "l"=6, "oʊ"=7
        simulate_aligned_tokens=[4, 5, 6, 7],
    )
    aligner = CTCPhonemeAligner(model=mock_model, model_name="mock-wav2vec2-espeak")
    assert aligner.is_available

    # 2 seconds of 16kHz audio with speech-level energy
    t = np.linspace(0, 2.0, 16000 * 2, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=[],
        transcript_text="",
        reference_text="hello",
    )

    assert isinstance(res, PronunciationAssessmentResult)
    assert res.status == "ready"
    assert res.transcript_source == "User-provided reference"
    assert res.reference_text == "hello"
    assert res.alignment_coverage_ratio == 1.0
    assert res.mean_acoustic_confidence is not None
    assert res.mean_acoustic_confidence > 0.80
    assert len(res.word_diagnostics) == 1

    hello_diag = res.word_diagnostics[0]
    assert hello_diag.word == "hello"
    assert len(hello_diag.phonemes) == 4
    for ph in hello_diag.phonemes:
        assert ph.is_supported is True
        assert ph.acoustic_match_confidence is not None
        assert ph.end >= ph.start


def test_reference_based_assessment_silence_guard():
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    mock_model = MockPhonemeModel()
    aligner = CTCPhonemeAligner(model=mock_model)

    # Pure silence (all zeros)
    samples = np.zeros(16000 * 2, dtype=np.float32)
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=[],
        transcript_text="",
        reference_text="hello world",
    )

    assert res.status in ("experimental", "unavailable")
    assert res.alignment_coverage_ratio == 0.0
    assert "silent" in res.methodology_note.lower()


def test_reference_based_assessment_unsupported_phoneme_explicit_note():
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    # Mock model with empty vocab (no phonemes supported)
    mock_model = MockPhonemeModel(vocab={"<pad>": 0})
    aligner = CTCPhonemeAligner(model=mock_model)

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=[],
        transcript_text="",
        reference_text="hello",
    )

    assert res.status == "experimental"
    assert res.alignment_coverage_ratio == 0.0
    assert len(res.word_diagnostics) == 1
    for ph in res.word_diagnostics[0].phonemes:
        assert ph.acoustic_match_confidence is None
        assert "unsupported" in (ph.diagnostic_note or "").lower()


def test_reference_based_assessment_unavailable_without_model():
    aligner = CTCPhonemeAligner(model=None)
    assert not aligner.is_available

    samples = np.ones(16000, dtype=np.float32) * 0.1
    res = aligner.assess(
        samples=samples,
        sample_rate=16000,
        words=[],
        transcript_text="",
        reference_text="hello world",
    )

    assert res.status == "unavailable"
    assert res.word_diagnostics == []
    assert res.transcript_source == "User-provided reference"
    assert "unavailable" in res.methodology_note.lower()


def test_model_manager_lazy_pronunciation_and_concurrency():
    import threading
    from backend.app.core.model_manager import ModelManager
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    manager = ModelManager()
    assert manager._pronunciation_status == "unavailable"
    assert not manager.is_pronunciation_ready

    # Inject mock pronunciation model
    mock = MockPhonemeModel()
    manager.set_pronunciation_model(mock, model_name="mock-test-ctc")
    assert manager.is_pronunciation_ready
    assert "mock-test-ctc (ready)" in manager.get_status()["pronunciation"]

    # Concurrency test: multiple threads calling load_pronunciation_model
    results = []

    def call_loader():
        results.append(manager.load_pronunciation_model())

    threads = [threading.Thread(target=call_loader) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert all(results)
    assert manager.is_pronunciation_ready


def test_g2p_audited_lexicon_and_fallback_defects():
    # Audit: verified canonical pronunciations for common reference words
    assert word_to_phonemes("worked") == ["W", "ER1", "K", "T"]
    assert word_to_phonemes("she") == ["SH", "IY1"]
    assert word_to_phonemes("sells") == ["S", "EH1", "L", "Z"]
    assert word_to_phonemes("seashells") == ["S", "IY1", "SH", "EH1", "L", "Z"]
    assert word_to_phonemes("by") == ["B", "AY1"]
    assert word_to_phonemes("seashore") == ["S", "IY1", "SH", "AO1", "R"]

    # Audit: fallback heuristics should collapse double consonants
    # e.g., 'banner' -> B AE N ER (not B AE N N ER)
    banner_phones = word_to_phonemes("banner")
    assert banner_phones.count("N") == 1

    # Audit: fallback heuristics should not emit trailing EH for silent final 'e'
    # e.g. 'cone' -> K AA N (not K AA N EH)
    cone_phones = word_to_phonemes("cone")
    assert "EH" not in cone_phones


def test_ctc_repeated_phonemes_require_intermediate_blank():
    from backend.app.services.pronunciation.ctc_aligner import ctc_viterbi_alignment

    # Sequence of two identical target tokens: [5, 5]
    # In CTC, consecutive identical labels cannot be decoded without an intervening blank
    T, V = 6, 10
    emissions = np.full((T, V), -5.0, dtype=np.float32)

    # Frame 0: token 5
    emissions[0, 5] = -0.05
    # Frame 1: blank (0)
    emissions[1, 0] = -0.05
    # Frame 2: token 5
    emissions[2, 5] = -0.05
    # Frames 3-5: blank
    emissions[3:, 0] = -0.05

    spans = ctc_viterbi_alignment(emissions, [5, 5], blank_id=0)
    assert len(spans) == 2
    assert spans[0] is not None
    assert spans[1] is not None
    # The first token is at frame 0, second token must be at frame >= 2 (after blank)
    assert spans[0][1] < spans[1][0]
    assert spans[0][0] == 0
    assert spans[1][0] == 2


def test_reference_assessment_boundaries_remain_inside_audio_duration():
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    # Audio duration exactly 1.0s (16000 samples)
    sample_rate = 16000
    total_dur = 1.0
    samples = (0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, total_dur, int(total_dur * sample_rate)))).astype(np.float32)

    mock_model = MockPhonemeModel(
        confidence_level=0.85,
        simulate_aligned_tokens=[4, 5, 6, 7],
    )
    aligner = CTCPhonemeAligner(model=mock_model)

    res = aligner.assess(
        samples=samples,
        sample_rate=sample_rate,
        words=[],
        transcript_text="",
        reference_text="hello",
    )

    assert res.status == "ready"
    assert len(res.word_diagnostics) == 1
    word_diag = res.word_diagnostics[0]
    assert word_diag.start >= 0.0
    assert word_diag.end <= total_dur + 1e-4

    for ph in word_diag.phonemes:
        assert ph.start >= 0.0
        assert ph.end <= total_dur + 1e-4
        assert ph.duration <= total_dur


