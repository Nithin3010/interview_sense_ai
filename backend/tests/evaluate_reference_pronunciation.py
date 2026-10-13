"""
InterviewSense AI — Reference-Based Pronunciation Evaluation Protocol
---------------------------------------------------------------------
Reproducible local evaluation workflow for measuring:
- Reference sentence alignment coverage
- Per-phoneme and mean frame posterior acoustic evidence
- Unaligned and unsupported phoneme counts
- Repeatability / determinism across consecutive runs
- Latency and diagnostic bounds

Generates machine-readable JSON and CSV evaluation reports saved to
local directory 'evaluation_reports/' (ignored by Git).
"""

import os
import sys
import time
import json
import csv
from dataclasses import asdict
from typing import Dict, List, Optional, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from backend.app.config import settings
from backend.app.services.audio_processor import decode_and_normalize_audio
from backend.app.services.pronunciation.wav2vec2_model import Wav2Vec2PhonemeModel
from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner


def run_single_evaluation(
    aligner: CTCPhonemeAligner,
    audio_path: str,
    reference_text: str,
    case_id: str,
    description: str = "",
) -> Dict[str, Any]:
    """Evaluates a single audio recording against target reference text."""
    if not os.path.exists(audio_path):
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"Audio file not found: {audio_path}",
            "description": description,
        }

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    t_decode_start = time.time()
    processed = decode_and_normalize_audio(audio_bytes)
    t_decode = time.time() - t_decode_start

    t_infer_start = time.time()
    result = aligner.assess(
        samples=processed.samples,
        sample_rate=processed.sample_rate,
        words=[],
        transcript_text="",
        reference_text=reference_text,
    )
    t_infer = time.time() - t_infer_start

    # Phoneme breakdown counts
    unaligned_count = 0
    unsupported_count = 0
    total_phonemes = 0
    phoneme_details = []

    for w in result.word_diagnostics:
        for p in w.phonemes:
            total_phonemes += 1
            if not getattr(p, "is_supported", True):
                unsupported_count += 1
            elif p.acoustic_match_confidence is None:
                unaligned_count += 1

            phoneme_details.append({
                "word": w.word,
                "phoneme": p.phoneme,
                "start": p.start,
                "end": p.end,
                "duration": p.duration,
                "confidence": p.acoustic_match_confidence,
                "is_supported": getattr(p, "is_supported", True),
                "diagnostic_note": p.diagnostic_note,
            })

    return {
        "case_id": case_id,
        "description": description,
        "audio_path": audio_path,
        "audio_duration_sec": processed.duration_seconds,
        "reference_text": reference_text,
        "status": result.status,
        "alignment_coverage_ratio": result.alignment_coverage_ratio,
        "mean_acoustic_confidence": result.mean_acoustic_confidence,
        "total_phonemes": total_phonemes,
        "aligned_phonemes": total_phonemes - unaligned_count - unsupported_count,
        "unaligned_phonemes_count": unaligned_count,
        "unsupported_phonemes_count": unsupported_count,
        "decode_time_sec": round(t_decode, 4),
        "inference_time_sec": round(t_infer, 4),
        "methodology_note": result.methodology_note,
        "word_diagnostics_count": len(result.word_diagnostics),
        "phoneme_details": phoneme_details,
    }


def main():
    print("=" * 70)
    print("InterviewSense AI — Pronunciation Evaluation & Calibration Protocol")
    print(f"Model: {settings.PRONUNCIATION_MODEL_NAME} ({settings.PRONUNCIATION_DEVICE})")
    print("=" * 70)

    # Initialize model
    print("\n[1/4] Initializing Wav2Vec2 phonetic model...")
    t_load_start = time.time()
    model = Wav2Vec2PhonemeModel(
        model_name=settings.PRONUNCIATION_MODEL_NAME,
        device=settings.PRONUNCIATION_DEVICE,
    )
    loaded = model.load_model()
    if not loaded:
        print(f"Error loading model: {model._error}")
        sys.exit(1)
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s (Vocab: {len(model.vocab)} tokens).")

    aligner = CTCPhonemeAligner(model=model, model_name=settings.PRONUNCIATION_MODEL_NAME)

    # Output directory
    out_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "evaluation_reports")
    os.makedirs(out_dir, exist_ok=True)

    # Known evaluation audio files
    sample_wav = os.path.join(os.path.dirname(__file__), "genuine_speech_sample.wav")
    ref_sentence = "I have worked on distributed systems for three years."

    eval_cases = [
        {
            "case_id": "trial_01_canonical",
            "audio_path": sample_wav,
            "reference_text": ref_sentence,
            "description": "Baseline trial 1 of reference sentence on genuine recorded speech.",
        },
        {
            "case_id": "trial_02_repeatability",
            "audio_path": sample_wav,
            "reference_text": ref_sentence,
            "description": "Repeatability trial 2 (identical audio) to verify determinism.",
        },
        {
            "case_id": "trial_03_repeatability",
            "audio_path": sample_wav,
            "reference_text": ref_sentence,
            "description": "Repeatability trial 3 (identical audio) to verify determinism.",
        },
        {
            "case_id": "mismatched_reference",
            "audio_path": sample_wav,
            "reference_text": "Thank you for the opportunity to interview today.",
            "description": "Negative control: audio says 'distributed systems' but reference is different sentence.",
        },
        {
            "case_id": "partial_reference",
            "audio_path": sample_wav,
            "reference_text": "I have worked on distributed systems.",
            "description": "Partial reference: truncates 'for three years' to test alignment with trailing silence.",
        },
    ]

    print("\n[2/4] Executing evaluation test cases...")
    results = []
    for c in eval_cases:
        print(f"  • Running case: {c['case_id']}...")
        res = run_single_evaluation(
            aligner=aligner,
            audio_path=c["audio_path"],
            reference_text=c["reference_text"],
            case_id=c["case_id"],
            description=c["description"],
        )
        results.append(res)
        print(f"    -> Status: {res.get('status')}, Coverage: {res.get('alignment_coverage_ratio')}, "
              f"Mean Evidence: {res.get('mean_acoustic_confidence')}, Latency: {res.get('inference_time_sec')}s")

    # Repeatability Analysis (Trials 1, 2, 3)
    print("\n[3/4] Analyzing repeatability across trials 1, 2, and 3:")
    t1 = results[0]
    t2 = results[1]
    t3 = results[2]

    cov_identical = (t1["alignment_coverage_ratio"] == t2["alignment_coverage_ratio"] == t3["alignment_coverage_ratio"])
    conf_identical = (t1["mean_acoustic_confidence"] == t2["mean_acoustic_confidence"] == t3["mean_acoustic_confidence"])

    # Compare phoneme timestamps
    phones_identical = True
    for p1, p2, p3 in zip(t1["phoneme_details"], t2["phoneme_details"], t3["phoneme_details"]):
        if (p1["start"], p1["end"], p1["confidence"]) != (p2["start"], p2["end"], p2["confidence"]) or \
           (p1["start"], p1["end"], p1["confidence"]) != (p3["start"], p3["end"], p3["confidence"]):
            phones_identical = False
            break

    print(f"  • Coverage Determinism: {'IDENTICAL (Deterministic)' if cov_identical else 'VARIED'}")
    print(f"  • Mean Evidence Determinism: {'IDENTICAL (Deterministic)' if conf_identical else 'VARIED'}")
    print(f"  • Phoneme Boundaries Determinism: {'IDENTICAL (Deterministic)' if phones_identical else 'VARIED'}")
    latencies = [t1["inference_time_sec"], t2["inference_time_sec"], t3["inference_time_sec"]]
    print(f"  • Inference Latencies: {latencies} (Mean: {np.mean(latencies):.3f}s, Std: {np.std(latencies):.4f}s)")

    # Negative control analysis
    neg = results[3]
    print("\n  Negative Control (Acoustic vs Mismatched Text):")
    print(f"  • Reference: \"{neg['reference_text']}\"")
    print(f"  • Alignment coverage: {neg['alignment_coverage_ratio']}")
    print(f"  • Mean acoustic evidence: {neg['mean_acoustic_confidence']} (vs {t1['mean_acoustic_confidence']} for matched)")
    print(f"  • Evidence degradation: {float(t1['mean_acoustic_confidence']) - float(neg['mean_acoustic_confidence']):.3f}")

    # [4/4] Save Reports
    print("\n[4/4] Writing machine-readable evaluation reports...")
    json_path = os.path.join(out_dir, "pronunciation_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": settings.PRONUNCIATION_MODEL_NAME,
                "device": settings.PRONUNCIATION_DEVICE,
                "model_load_time_sec": t_load,
                "repeatability": {
                    "trials_evaluated": 3,
                    "coverage_deterministic": cov_identical,
                    "evidence_deterministic": conf_identical,
                    "boundaries_deterministic": phones_identical,
                    "latencies_sec": latencies,
                },
                "cases": results,
            },
            f,
            indent=2,
        )
    print(f"  Saved JSON report: {json_path}")

    csv_path = os.path.join(out_dir, "pronunciation_eval_report.csv")
    csv_fields = [
        "case_id",
        "status",
        "audio_duration_sec",
        "alignment_coverage_ratio",
        "mean_acoustic_confidence",
        "total_phonemes",
        "aligned_phonemes",
        "unaligned_phonemes_count",
        "unsupported_phonemes_count",
        "inference_time_sec",
        "reference_text",
        "description",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved CSV report: {csv_path}")

    print("\n" + "=" * 70)
    print("Pronunciation Evaluation Protocol Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
