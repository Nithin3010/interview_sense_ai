import io
from fastapi.testclient import TestClient
from backend.app.core.model_manager import model_manager


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "health_check" in data


def test_health_check_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "InterviewSense AI"
    assert "models" in data
    assert "asr" in data["models"]
    assert "mock-small.en (loaded)" in data["models"]["asr"]
    assert "acoustic" in data["models"]
    assert "pronunciation" in data["models"]


def test_404_not_found(client: TestClient):
    response = client.get("/api/v1/non_existent_route")
    assert response.status_code == 404


def test_analyze_valid_audio(client: TestClient, valid_mono_wav_bytes):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "validated"
    assert "metadata" in data
    meta = data["metadata"]
    assert meta["sample_rate"] == 16000
    assert meta["channels"] == 1
    assert 2.95 <= meta["duration_seconds"] <= 3.05
    assert meta["peak_amplitude"] <= 0.95
    assert meta["rms_energy"] > 0.0

    # Milestone 3 Transcription Assertions
    assert "transcript" in data
    trans = data["transcript"]
    assert trans["text"] == "Hello world this is a test"
    assert trans["language"] == "en"
    assert trans["model_used"] == "mock-small.en"
    assert len(trans["words"]) == 6
    assert trans["words"][0]["word"] == "Hello"
    assert trans["words"][0]["start"] == 0.12
    assert trans["words"][0]["end"] == 0.45
    assert trans["words"][0]["probability"] == 0.98

    # Milestone 4 Fluency & Pause Assertions
    assert "fluency" in data
    fluency = data["fluency"]
    assert fluency["word_count"] == 6
    assert fluency["gross_wpm"] > 0.0
    assert fluency["articulation_rate_wpm"] >= fluency["gross_wpm"]
    assert "filler_count" in fluency
    assert "filler_percentage" in fluency
    assert "repetitions" in fluency

    assert "pauses" in data
    pauses = data["pauses"]
    assert "total_pauses" in pauses
    assert "total_pause_duration_seconds" in pauses
    assert "pause_intervals" in pauses
    assert "note" in pauses

    # Milestone 5 Acoustics Assertions
    assert "acoustics" in data
    acoustics = data["acoustics"]
    assert "pitch" in acoustics
    assert "energy" in acoustics
    pitch = acoustics["pitch"]
    assert pitch["status"] in ("available", "unavailable")
    assert "note" in pitch
    energy = acoustics["energy"]
    assert "mean_rms" in energy
    assert "mean_rms_dbfs" in energy
    assert "dynamic_range_db" in energy
    assert "note" in energy

    # Milestone 6 Pronunciation Assessment Assertions
    assert "pronunciation" in data
    pron = data["pronunciation"]
    assert pron["status"] in ("unavailable", "experimental", "error")
    assert "model_identifier" in pron
    assert "phoneme_inventory" in pron
    assert "transcript_source" in pron
    assert "ASR" in pron["transcript_source"]
    assert "methodology_note" in pron
    assert "word_diagnostics" in pron



def test_analyze_short_audio(client: TestClient, short_wav_bytes):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("short.wav", io.BytesIO(short_wav_bytes), "audio/wav")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "AUDIO_TOO_SHORT"


def test_analyze_silent_audio(client: TestClient, silent_wav_bytes):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("silent.wav", io.BytesIO(silent_wav_bytes), "audio/wav")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "SILENT_AUDIO"


def test_analyze_corrupted_file(client: TestClient):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("broken.wav", io.BytesIO(b"not-valid-audio-data"), "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "INVALID_AUDIO_FORMAT"


def test_analyze_empty_file(client: TestClient):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "INVALID_AUDIO_FORMAT"


def test_analyze_transcription_failure(client: TestClient, valid_mono_wav_bytes):
    class ExplodingModel:
        def transcribe(self, samples, word_timestamps=True, language="en"):
            raise RuntimeError("Engine failure")

    original_model = model_manager.asr_model
    model_manager.set_asr_model(ExplodingModel(), "exploding-model")

    try:
        response = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error_code"] == "TRANSCRIPTION_FAILED"
    finally:
        model_manager.set_asr_model(original_model, "mock-small.en")


def test_analyze_pronunciation_experimental_integration(client: TestClient, valid_mono_wav_bytes):
    from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner

    class DummyAcousticModel:
        def score_phoneme(self, phoneme: str, start: float, end: float) -> float:
            return 0.85

    custom_scorer = CTCPhonemeAligner(
        model=DummyAcousticModel(),
        model_name="mock-wav2vec2-phoneme",
    )
    original_scorer = model_manager.pronunciation_scorer
    model_manager.pronunciation_scorer = custom_scorer

    try:
        response = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "pronunciation" in data
        pron = data["pronunciation"]
        assert pron["status"] == "experimental"
        assert pron["model_identifier"] == "mock-wav2vec2-phoneme"
        assert len(pron["word_diagnostics"]) > 0
        first_diag = pron["word_diagnostics"][0]
        assert first_diag["word"] == "Hello"
        assert "phonemes" in first_diag
        assert len(first_diag["phonemes"]) > 0
        assert first_diag["phonemes"][0]["phoneme"] == "HH"
        assert first_diag["phonemes"][0]["acoustic_match_confidence"] == 0.85
    finally:
        model_manager.pronunciation_scorer = original_scorer


def test_analyze_reference_based_pronunciation_success(client: TestClient, valid_mono_wav_bytes):
    from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel

    mock_model = MockPhonemeModel(confidence_level=0.91, simulate_aligned_tokens=[4, 5, 6, 7])
    original_model = model_manager._pronunciation_model
    original_scorer = model_manager.pronunciation_scorer

    model_manager.set_pronunciation_model(mock_model, "mock-wav2vec2-api")

    try:
        ref_text = "hello"
        response = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
            data={"reference_text": ref_text},
        )
        assert response.status_code == 200
        data = response.json()
        assert "pronunciation" in data
        pron = data["pronunciation"]
        assert pron["status"] in ("ready", "experimental")
        assert pron["reference_text"] == "hello"
        assert pron["transcript_source"] == "User-provided reference"
        assert pron["alignment_coverage_ratio"] is not None
        assert pron["mean_acoustic_confidence"] is not None
        assert len(pron["word_diagnostics"]) == 1
        assert pron["word_diagnostics"][0]["word"] == "hello"
    finally:
        model_manager._pronunciation_model = original_model
        model_manager.pronunciation_scorer = original_scorer


def test_analyze_whitespace_reference_text_rejected(client: TestClient, valid_mono_wav_bytes):
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        data={"reference_text": "    \t\n   "},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "INVALID_REFERENCE_TEXT"


def test_analyze_excessive_reference_text_rejected(client: TestClient, valid_mono_wav_bytes):
    too_many_words = " ".join(["practice"] * 90)
    response = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
        data={"reference_text": too_many_words},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "INVALID_REFERENCE_TEXT"


def test_analyze_reference_text_when_model_unavailable(client: TestClient, valid_mono_wav_bytes):
    from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner

    original_scorer = model_manager.pronunciation_scorer
    model_manager.pronunciation_scorer = CTCPhonemeAligner(model=None)

    try:
        response = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("speech.wav", io.BytesIO(valid_mono_wav_bytes), "audio/wav")},
            data={"reference_text": "I have worked on distributed systems."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "validated"
        assert data["pronunciation"]["status"] == "unavailable"
        assert data["pronunciation"]["transcript_source"] == "User-provided reference"
        assert "unavailable" in data["pronunciation"]["methodology_note"].lower()
    finally:
        model_manager.pronunciation_scorer = original_scorer


