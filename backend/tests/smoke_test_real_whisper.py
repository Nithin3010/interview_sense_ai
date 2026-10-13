"""
Manual Smoke Test for Real faster-whisper Model
-----------------------------------------------
This is an optional manual smoke test to verify live model loading and inference
on your Windows CPU without affecting the deterministic automated test suite.

Run manually with:
    python backend/tests/smoke_test_real_whisper.py
    python backend/tests/smoke_test_real_whisper.py tiny.en
    python backend/tests/smoke_test_real_whisper.py small.en
"""

import sys
import os
import time
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.config import settings
from backend.app.services.transcription import transcribe_audio


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else settings.WHISPER_MODEL_SIZE

    print("=" * 60)
    print("InterviewSense AI — Real Whisper Model Smoke Test")
    print(f"Target model: {model_name}")
    print(f"Device: {settings.WHISPER_DEVICE}, Compute Type: {settings.WHISPER_COMPUTE_TYPE}")
    print(f"Configured CPU Threads: {settings.WHISPER_CPU_THREADS}")
    print("=" * 60)

    try:
        from faster_whisper import WhisperModel

        print(f"Initializing WhisperModel '{model_name}'...")
        t_load_start = time.time()
        try:
            model = WhisperModel(
                model_size_or_path=model_name,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                cpu_threads=settings.WHISPER_CPU_THREADS,
            )
        except Exception as load_err:
            if model_name != "tiny.en":
                print(f"Could not load '{model_name}' ({load_err}). Attempting cached fallback 'tiny.en'...")
                model_name = "tiny.en"
                model = WhisperModel(
                    model_size_or_path="tiny.en",
                    device=settings.WHISPER_DEVICE,
                    compute_type=settings.WHISPER_COMPUTE_TYPE,
                    cpu_threads=settings.WHISPER_CPU_THREADS,
                )
            else:
                raise load_err

        t_load = time.time() - t_load_start
        print(f"Model loaded successfully in {t_load:.2f}s!")

        # Synthesize a 3-second test carrier signal (440Hz sine wave)
        duration_s = 3.0
        sample_rate = 16000
        t_axis = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
        # Low-amplitude tone to verify signal ingestion
        samples = (0.2 * np.sin(2 * np.pi * 440.0 * t_axis)).astype(np.float32)

        print(f"Running inference on {duration_s:.1f}s test audio...")
        t_infer_start = time.time()
        result = transcribe_audio(samples, model=model, language="en")
        t_infer = time.time() - t_infer_start

        rtf = t_infer / duration_s

        print("=" * 60)
        print("Smoke Test Verification Results:")
        print(f"• Model verified: {model_name}")
        print(f"• Audio duration: {duration_s:.2f}s")
        print(f"• Inference latency: {t_infer:.3f}s (Real-Time Factor: {rtf:.3f}x)")
        print(f"• Transcript result: '{result.text}'")
        print(f"• Word tokens count: {len(result.words)}")
        print(f"• Model attribute: {result.model_used}")
        print("• Status: PASSED (Real faster-whisper CPU inference operational)")
        print("=" * 60)

    except Exception as exc:
        print(f"Smoke test encountered an issue: {exc}")
        print("Ensure internet connectivity if downloading weights for the first time.")
        sys.exit(1)


if __name__ == "__main__":
    main()
