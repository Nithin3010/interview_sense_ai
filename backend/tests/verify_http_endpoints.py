"""
InterviewSense AI — End-to-End HTTP Verification Script
Verifies both reference-based and spontaneous speech analysis modes at HTTP level.
"""

import io
import sys
import os

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.model_manager import model_manager
from backend.app.services.pronunciation.wav2vec2_model import MockPhonemeModel
from backend.tests.conftest import make_synthetic_wav, MockWhisperModel


def run_http_verification():
    print("=" * 60)
    print("InterviewSense AI -- HTTP-Level Endpoint & Workflow Verification")
    print("=" * 60)

    # Set mock ASR model so standalone script does not need pytest fixtures
    model_manager.set_asr_model(MockWhisperModel(), "mock-small.en")

    client = TestClient(app)

    # 1. Verify health endpoint
    print("\n1. Testing GET /api/v1/health...")
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    health_data = health_resp.json()
    print(f"   Health status: {health_data['status']}")
    print(f"   ASR status: {health_data['models']['asr']}")
    print(f"   Acoustic status: {health_data['models']['acoustic']}")
    print(f"   Pronunciation status: {health_data['models']['pronunciation']}")
    print("   [OK] Health check passed!")

    # Generate synthetic audio for testing
    audio_bytes = make_synthetic_wav(duration=3.0, sample_rate=16000, frequency=220.0)

    # Set up mock pronunciation model for deterministic testing
    mock_model = MockPhonemeModel(confidence_level=0.91, simulate_aligned_tokens=[4, 5, 6, 7])
    model_manager.set_pronunciation_model(mock_model, "mock-wav2vec2-http")

    # 2. Test Reference-Based Analysis
    print("\n2. Testing POST /api/v1/analyze (Reference Mode)...")
    ref_resp = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("test_ref.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"reference_text": "hello", "language": "en"},
    )
    assert ref_resp.status_code == 200, f"Reference analysis failed: {ref_resp.status_code} {ref_resp.text}"
    ref_data = ref_resp.json()
    assert ref_data["status"] == "validated"
    assert ref_data["pronunciation"]["reference_text"] == "hello"
    assert ref_data["pronunciation"]["transcript_source"] == "User-provided reference"
    assert ref_data["pronunciation"]["status"] in ("ready", "experimental")
    print(f"   Reference evaluated: \"{ref_data['pronunciation']['reference_text']}\"")
    print(f"   Alignment coverage: {ref_data['pronunciation']['alignment_coverage_ratio']}")
    print(f"   Mean acoustic evidence: {ref_data['pronunciation']['mean_acoustic_confidence']}")
    print("   [OK] Reference-based analysis verified!")

    # 3. Test Spontaneous Speech Analysis
    print("\n3. Testing POST /api/v1/analyze (Spontaneous Mode -- no reference text)...")
    spont_resp = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("test_spont.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"language": "en"},
    )
    assert spont_resp.status_code == 200, f"Spontaneous analysis failed: {spont_resp.status_code} {spont_resp.text}"
    spont_data = spont_resp.json()
    assert spont_data["status"] == "validated"
    assert spont_data["pronunciation"]["reference_text"] is None
    assert spont_data["pronunciation"]["transcript_source"] == "ASR-generated (faster-whisper)"
    assert len(spont_data["transcript"]["words"]) > 0
    assert len(spont_data["pronunciation"]["word_diagnostics"]) > 0
    print(f"   ASR Transcript: \"{spont_data['transcript']['text']}\"")
    print(f"   Transcript source: {spont_data['pronunciation']['transcript_source']}")
    print(f"   Reference text: {spont_data['pronunciation']['reference_text']} (strictly null)")
    print(f"   Alignment coverage: {spont_data['pronunciation']['alignment_coverage_ratio']}")
    print(f"   Word diagnostics count: {len(spont_data['pronunciation']['word_diagnostics'])}")
    print("   [OK] Spontaneous analysis verified!")

    # 4. Test Invalid Reference Text
    print("\n4. Testing POST /api/v1/analyze with invalid whitespace reference...")
    inv_resp = client.post(
        "/api/v1/analyze",
        files={"audio_file": ("test_err.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"reference_text": "    \t   "},
    )
    assert inv_resp.status_code == 422
    assert inv_resp.json()["detail"]["error_code"] == "INVALID_REFERENCE_TEXT"
    print("   [OK] Invalid reference text properly rejected with 422!")

    # 5. Test Model-Unavailable Safety
    print("\n5. Testing model-unavailable graceful state...")
    from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner
    original_scorer = model_manager.pronunciation_scorer
    original_load = model_manager.load_pronunciation_model
    model_manager.load_pronunciation_model = lambda: False
    model_manager.pronunciation_scorer = CTCPhonemeAligner(model=None)
    try:
        unavail_resp = client.post(
            "/api/v1/analyze",
            files={"audio_file": ("test_unavail.wav", io.BytesIO(audio_bytes), "audio/wav")},
        )
        assert unavail_resp.status_code == 200
        unavail_data = unavail_resp.json()
        assert unavail_data["pronunciation"]["status"] == "unavailable"
        assert "unavailable" in unavail_data["pronunciation"]["methodology_note"].lower()
        # Verify core features still returned
        assert "transcript" in unavail_data
        assert "fluency" in unavail_data
        assert "acoustics" in unavail_data
        print("   [OK] Pronunciation-unavailable state safely handled without breaking core analysis!")
    finally:
        model_manager.pronunciation_scorer = original_scorer
        model_manager.load_pronunciation_model = original_load

    print("\n" + "=" * 60)
    print("ALL HTTP-LEVEL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_http_verification()
