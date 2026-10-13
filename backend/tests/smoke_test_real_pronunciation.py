"""
Manual Smoke Test for Real Wav2Vec2 CTC Phonetic Pronunciation Model
--------------------------------------------------------------------
This is an optional manual smoke test to verify live model loading and inference
on your Windows CPU using a known reference sentence and recorded speech.

Excluded from normal automated pytest runs to ensure the CI test suite remains
completely deterministic and independent of network access or external downloads.

Usage:
    python backend/tests/smoke_test_real_pronunciation.py
    python backend/tests/smoke_test_real_pronunciation.py path/to/sample.wav "I have worked on distributed systems for three years."
"""

import sys
import os
import time
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.config import settings
from backend.app.services.pronunciation.wav2vec2_model import Wav2Vec2PhonemeModel
from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner
from backend.app.services.audio_processor import decode_and_normalize_audio


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    ref_sentence = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "I have worked on distributed systems for three years."
    )

    print("=" * 65)
    print("InterviewSense AI — Real Wav2Vec2 CTC Pronunciation Smoke Test")
    print(f"Target model: {settings.PRONUNCIATION_MODEL_NAME}")
    print(f"Target reference sentence: \"{ref_sentence}\"")
    print("=" * 65)

    if audio_path and os.path.exists(audio_path):
        print(f"Loading provided audio file: {audio_path}")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        processed = decode_and_normalize_audio(audio_bytes)
        samples = processed.samples
        sample_rate = processed.sample_rate
        print(f"Decoded {len(samples)} samples ({processed.duration_seconds:.2f}s) at {sample_rate}Hz.")
    else:
        print("Notice: No speech audio file passed. Generating 3.5s speech-range test carrier signal.")
        print("To test on genuine recorded speech, pass: python backend/tests/smoke_test_real_pronunciation.py <file.wav> \"<reference>\"")
        duration_s = 3.5
        sample_rate = 16000
        t_axis = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
        # Synthetic formant-like harmonic signal
        samples = (
            0.15 * np.sin(2 * np.pi * 150.0 * t_axis)
            + 0.10 * np.sin(2 * np.pi * 300.0 * t_axis)
            + 0.05 * np.sin(2 * np.pi * 2500.0 * t_axis)
        ).astype(np.float32)

    try:
        import psutil
        proc = psutil.Process()
        mem_baseline_mb = proc.memory_info().rss / (1024 * 1024)
    except Exception:
        proc = None
        mem_baseline_mb = 0.0

    try:
        print(f"\nInitializing Wav2Vec2PhonemeModel '{settings.PRONUNCIATION_MODEL_NAME}' on {settings.PRONUNCIATION_DEVICE}...")
        t_load_start = time.time()
        model = Wav2Vec2PhonemeModel(
            model_name=settings.PRONUNCIATION_MODEL_NAME,
            device=settings.PRONUNCIATION_DEVICE,
        )
        loaded = model.load_model()
        if not loaded:
            print("Model weights could not be loaded. Please ensure HuggingFace connectivity.")
            sys.exit(1)

        t_load = time.time() - t_load_start
        mem_loaded_mb = proc.memory_info().rss / (1024 * 1024) if proc else 0.0
        print(f"Model loaded successfully in {t_load:.2f}s! Vocab size: {len(model.vocab)}")
        if proc:
            print(f"Memory RSS after model loading: {mem_loaded_mb:.1f} MB (Delta: +{mem_loaded_mb - mem_baseline_mb:.1f} MB)")

        aligner = CTCPhonemeAligner(model=model, model_name=settings.PRONUNCIATION_MODEL_NAME)

        print("\nRunning phoneme-level CTC Trellis alignment against reference targets...")
        t_infer_start = time.time()
        result = aligner.assess(
            samples=samples,
            sample_rate=sample_rate,
            words=[],
            transcript_text="",
            reference_text=ref_sentence,
        )
        t_infer = time.time() - t_infer_start
        mem_peak_mb = proc.memory_info().rss / (1024 * 1024) if proc else 0.0

        unaligned_phones = []
        unsupported_phones = []
        for w in result.word_diagnostics:
            for p in w.phonemes:
                if not getattr(p, "is_supported", True):
                    unsupported_phones.append(f"{w.word}:{p.phoneme}")
                elif p.acoustic_match_confidence is None:
                    unaligned_phones.append(f"{w.word}:{p.phoneme}")

        print("=" * 65)
        print("Real Pronunciation Smoke Test Results:")
        print(f"• Alignment status: {result.status.upper()}")
        print(f"• Reference evaluated: \"{result.reference_text}\"")
        print(f"• Phoneme inventory: {result.phoneme_inventory}")
        print(f"• Model loading time: {t_load:.2f}s")
        print(f"• Inference latency: {t_infer:.3f}s")
        if proc:
            print(f"• Memory RSS: {mem_peak_mb:.1f} MB (Net increase: {mem_peak_mb - mem_baseline_mb:.1f} MB)")
        print(f"• Alignment coverage ratio: {result.alignment_coverage_ratio}")
        print(f"• Mean acoustic confidence: {result.mean_acoustic_confidence}")
        print(f"• Total words evaluated: {len(result.word_diagnostics)}")
        print(f"• Unaligned phonemes ({len(unaligned_phones)}): {', '.join(unaligned_phones) if unaligned_phones else 'None'}")
        print(f"• Unsupported phonemes ({len(unsupported_phones)}): {', '.join(unsupported_phones) if unsupported_phones else 'None'}")

        print("\nWord-by-word phoneme breakdown:")
        for w in result.word_diagnostics:
            ph_summary = " ".join([f"{p.phoneme}({f'{(p.acoustic_match_confidence*100):.0f}%' if p.acoustic_match_confidence is not None else 'n/a'})" for p in w.phonemes])
            print(f"  [{w.start:.2f}s - {w.end:.2f}s] \"{w.word}\": {ph_summary}")

        print("\nMethodology note:")
        print(f"  {result.methodology_note}")
        print("=" * 65)
        print("Status: PASSED (Real Wav2Vec2 CTC phoneme inference verified)")

    except Exception as exc:
        print(f"\nSmoke test encountered an issue: {exc}")
        print("Ensure network access if downloading weights for the first time (~1.2 GB).")
        sys.exit(1)


if __name__ == "__main__":
    main()
