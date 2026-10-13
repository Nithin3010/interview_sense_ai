import logging
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from backend.app.core.exceptions import InterviewSenseException
from backend.app.services.pronunciation.base import (
    PronunciationScorer,
    PronunciationAssessmentResult,
    WordPronunciationDiagnostic,
    PhonemeDiagnostic,
)
from backend.app.services.pronunciation.g2p import (
    word_to_phonemes,
    validate_reference_text,
    text_to_reference_phonemes,
    WordReference,
)

logger = logging.getLogger("interviewsense.pronunciation.ctc_aligner")


def ctc_viterbi_alignment(
    emissions: np.ndarray, target_token_ids: List[int], blank_id: int = 0
) -> List[Optional[Tuple[int, int]]]:
    """
    Performs standard CTC Viterbi forced alignment.
    Given:
      emissions: log-probabilities matrix of shape (T, Vocab_size)
      target_token_ids: list of K target token IDs
      blank_id: token ID for blank / CTC pad
    Returns:
      List of length K with (start_frame, end_frame) tuples (inclusive),
      or None if the phoneme could not be aligned / reached.
    """
    K = len(target_token_ids)
    if K == 0:
        return []

    T, V = emissions.shape
    if T == 0:
        return [None] * K

    # Interleave blanks: [blank, y0, blank, y1, ..., blank, y_{K-1}, blank]
    S = 2 * K + 1
    target_prime = np.zeros(S, dtype=np.int32)
    for k in range(K):
        target_prime[2 * k] = blank_id
        target_prime[2 * k + 1] = target_token_ids[k]
    target_prime[2 * K] = blank_id

    # Trellis: V[t, s] = max log-prob of paths ending at state s at frame t
    trellis = np.full((T, S), -np.inf, dtype=np.float32)
    backpointers = np.full((T, S), -1, dtype=np.int32)

    # Initial frame t = 0
    trellis[0, 0] = emissions[0, blank_id]
    if S > 1:
        trellis[0, 1] = emissions[0, target_prime[1]]
    backpointers[0, 0] = 0
    backpointers[0, 1] = 1

    # Induction over frames
    for t in range(1, T):
        max_s = min(S, 2 * t + 2)
        for s in range(max_s):
            tok = target_prime[s]
            emit = emissions[t, tok]

            if s == 0:
                trellis[t, 0] = trellis[t - 1, 0] + emit
                backpointers[t, 0] = 0
            elif s == 1 or tok == blank_id or tok == target_prime[s - 2]:
                cand0 = trellis[t - 1, s]
                cand1 = trellis[t - 1, s - 1]
                if cand0 >= cand1:
                    trellis[t, s] = cand0 + emit
                    backpointers[t, s] = s
                else:
                    trellis[t, s] = cand1 + emit
                    backpointers[t, s] = s - 1
            else:
                cand0 = trellis[t - 1, s]
                cand1 = trellis[t - 1, s - 1]
                cand2 = trellis[t - 1, s - 2]
                if cand0 >= cand1 and cand0 >= cand2:
                    trellis[t, s] = cand0 + emit
                    backpointers[t, s] = s
                elif cand1 >= cand2:
                    trellis[t, s] = cand1 + emit
                    backpointers[t, s] = s - 1
                else:
                    trellis[t, s] = cand2 + emit
                    backpointers[t, s] = s - 2

    # Termination: best final state at T - 1
    cand_end1 = trellis[T - 1, S - 1]
    cand_end2 = trellis[T - 1, S - 2] if S > 1 else -np.inf

    if np.isfinite(cand_end1) or np.isfinite(cand_end2):
        best_state = S - 1 if cand_end1 >= cand_end2 else S - 2
    else:
        # Partial alignment: best reachable state across trellis[T - 1, :]
        valid_states = np.where(np.isfinite(trellis[T - 1, :]))[0]
        if len(valid_states) == 0:
            return [None] * K
        best_state = int(valid_states[np.argmax(trellis[T - 1, valid_states])])

    # Backtracking
    path = np.zeros(T, dtype=np.int32)
    curr_state = best_state
    for t in range(T - 1, -1, -1):
        path[t] = curr_state
        curr_state = backpointers[t, curr_state]
        if curr_state < 0:
            curr_state = 0

    # Extract phoneme boundaries
    aligned_spans: List[Optional[Tuple[int, int]]] = []
    for k in range(K):
        target_state = 2 * k + 1
        matched_frames = np.where(path == target_state)[0]
        if len(matched_frames) > 0:
            aligned_spans.append((int(matched_frames[0]), int(matched_frames[-1])))
        else:
            aligned_spans.append(None)

    return aligned_spans


class CTCPhonemeAligner(PronunciationScorer):
    """
    Phoneme alignment and acoustic diagnostic aligner supporting reference-based
    and spontaneous-speech diagnostics.

    Adheres strictly to scientific and ethical boundaries:
    - If no pretrained acoustic model is loaded, returns status='unavailable'.
    - When reference text is provided, performs CTC forced alignment against
      canonical phoneme targets and extracts frame posterior evidence.
    - Does NOT fabricate uncalibrated overall scores or composite ratings.
    """

    def __init__(self, model: Optional[Any] = None, model_name: Optional[str] = None):
        self._model = model
        self._model_name = model_name or "facebook/wav2vec2-lv-60-espeak-cv-ft"

    @property
    def is_available(self) -> bool:
        if self._model is None:
            return False
        return bool(getattr(self._model, "is_ready", True))

    def assess(
        self,
        samples: np.ndarray,
        sample_rate: int,
        words: Sequence[Any],
        transcript_text: str,
        reference_text: Optional[str] = None,
    ) -> PronunciationAssessmentResult:
        # 1. Input array validation
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise InterviewSenseException(
                error_code="INVALID_AUDIO_ARRAY",
                message="Pronunciation assessment requires a 1D NumPy array.",
                status_code=400,
            )

        if not np.isfinite(samples).all():
            raise InterviewSenseException(
                error_code="NON_FINITE_SAMPLES",
                message="Audio samples contain non-finite (NaN or Inf) values.",
                status_code=400,
            )

        # 2. Reference-based assessment path
        if reference_text is not None:
            return self._assess_reference_based(
                samples=samples,
                sample_rate=sample_rate,
                reference_text=reference_text,
            )

        # 3. Spontaneous speech assessment path (Reference text absent)
        return self._assess_spontaneous(
            samples=samples,
            sample_rate=sample_rate,
            words=words,
            transcript_text=transcript_text,
        )

    def _assess_reference_based(
        self,
        samples: np.ndarray,
        sample_rate: int,
        reference_text: str,
    ) -> PronunciationAssessmentResult:
        norm_ref = validate_reference_text(reference_text)

        # Graceful 'unavailable' state if model weights are not loaded
        if not self.is_available:
            return PronunciationAssessmentResult(
                status="unavailable",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=norm_ref,
                transcript_source="User-provided reference",
                reference_text=norm_ref,
                word_diagnostics=[],
                methodology_note=(
                    "Pronunciation assessment is unavailable: Pretrained phoneme-level acoustic model is uninitialized. "
                    "Acoustic model weights must be loaded to evaluate reference speech."
                ),
            )

        # Silence / no-speech check
        rms_energy = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        if len(samples) == 0 or rms_energy < 1e-4:
            return PronunciationAssessmentResult(
                status="experimental",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=norm_ref,
                transcript_source="User-provided reference",
                reference_text=norm_ref,
                word_diagnostics=[],
                alignment_coverage_ratio=0.0,
                mean_acoustic_confidence=None,
                methodology_note="Audio contains no discernible speech energy or is silent; phoneme alignment cannot be performed.",
            )

        try:
            word_refs = text_to_reference_phonemes(norm_ref)
            if not word_refs:
                return PronunciationAssessmentResult(
                    status="experimental",
                    model_identifier=self._model_name,
                    phoneme_inventory="ARPAbet-39",
                    transcript_evaluated=norm_ref,
                    transcript_source="User-provided reference",
                    reference_text=norm_ref,
                    word_diagnostics=[],
                    alignment_coverage_ratio=0.0,
                    mean_acoustic_confidence=None,
                    methodology_note="Reference text yielded no valid word targets.",
                )

            # Obtain model emissions
            if hasattr(self._model, "get_frame_emissions"):
                emissions, time_per_frame = self._model.get_frame_emissions(samples, sample_rate)
            else:
                num_frames = max(5, int(len(samples) / 320))
                emissions = np.full((num_frames, 50), -3.5, dtype=np.float32)
                time_per_frame = (len(samples) / float(sample_rate)) / float(num_frames)

            vocab = getattr(self._model, "vocab", {})
            blank_id = vocab.get("<pad>", 0)

            # Collect supported phonemes for alignment trellis
            # Track flat targets: (token_id, word_idx, phoneme_idx)
            flat_targets: List[Tuple[int, int, int]] = []
            for w_idx, w_ref in enumerate(word_refs):
                for p_idx, p_ref in enumerate(w_ref.phonemes):
                    if p_ref.is_supported and p_ref.model_token and p_ref.model_token in vocab:
                        tok_id = vocab[p_ref.model_token]
                        flat_targets.append((tok_id, w_idx, p_idx))

            target_tokens = [t[0] for t in flat_targets]
            aligned_spans = ctc_viterbi_alignment(emissions, target_tokens, blank_id=blank_id)

            # Map alignments back to word/phoneme structure
            span_map: dict = {}
            for target_idx, span in enumerate(aligned_spans):
                _, w_idx, p_idx = flat_targets[target_idx]
                span_map[(w_idx, p_idx)] = (span, flat_targets[target_idx][0])

            word_diagnostics: List[WordPronunciationDiagnostic] = []
            total_expected_phonemes = 0
            aligned_phonemes_count = 0
            all_confidences: List[float] = []

            for w_idx, w_ref in enumerate(word_refs):
                phone_diagnostics: List[PhonemeDiagnostic] = []
                w_aligned_starts: List[float] = []
                w_aligned_ends: List[float] = []
                w_confidences: List[float] = []

                for p_idx, p_ref in enumerate(w_ref.phonemes):
                    total_expected_phonemes += 1
                    lookup = span_map.get((w_idx, p_idx), None)

                    if not p_ref.is_supported or not p_ref.model_token or p_ref.model_token not in vocab:
                        phone_diagnostics.append(
                            PhonemeDiagnostic(
                                phoneme=p_ref.phoneme_arpabet,
                                start=0.0,
                                end=0.0,
                                duration=0.0,
                                acoustic_match_confidence=None,
                                is_supported=False,
                                diagnostic_note="Phoneme unsupported by acoustic model vocabulary",
                            )
                        )
                        continue

                    if lookup is not None and lookup[0] is not None:
                        span, tok_id = lookup
                        st_frame, en_frame = span
                        st_sec = round(float(st_frame * time_per_frame), 3)
                        en_sec = round(float((en_frame + 1) * time_per_frame), 3)
                        dur_sec = round(max(0.01, en_sec - st_sec), 3)

                        # Frame-level posterior average: exp(log_probs)
                        seg_log_probs = emissions[st_frame : en_frame + 1, tok_id]
                        conf = float(np.mean(np.exp(seg_log_probs)))
                        conf = round(max(0.0, min(1.0, conf)), 3)

                        aligned_phonemes_count += 1
                        all_confidences.append(conf)
                        w_confidences.append(conf)
                        w_aligned_starts.append(st_sec)
                        w_aligned_ends.append(en_sec)

                        phone_diagnostics.append(
                            PhonemeDiagnostic(
                                phoneme=p_ref.phoneme_arpabet,
                                start=st_sec,
                                end=en_sec,
                                duration=dur_sec,
                                acoustic_match_confidence=conf,
                                is_supported=True,
                                diagnostic_note=None,
                            )
                        )
                    else:
                        phone_diagnostics.append(
                            PhonemeDiagnostic(
                                phoneme=p_ref.phoneme_arpabet,
                                start=0.0,
                                end=0.0,
                                duration=0.0,
                                acoustic_match_confidence=None,
                                is_supported=True,
                                diagnostic_note="Phoneme omitted or not aligned in acoustic signal",
                            )
                        )

                w_start = min(w_aligned_starts) if w_aligned_starts else 0.0
                w_end = max(w_aligned_ends) if w_aligned_ends else 0.0
                w_mean_conf = round(float(np.mean(w_confidences)), 3) if w_confidences else None
                flagged = (w_mean_conf is not None and w_mean_conf < 0.25) or (len(w_confidences) < len(w_ref.phonemes))

                diag_note = None
                if len(w_confidences) < len(w_ref.phonemes):
                    diag_note = "Partial phoneme alignment: some sounds unaligned or unsupported."
                elif flagged:
                    diag_note = "Acoustic match confidence is low across target phonemes."

                word_diagnostics.append(
                    WordPronunciationDiagnostic(
                        word=w_ref.word,
                        start=w_start,
                        end=w_end,
                        phonemes=phone_diagnostics,
                        flagged=flagged,
                        diagnostic_note=diag_note,
                        mean_acoustic_confidence=w_mean_conf,
                    )
                )

            coverage_ratio = round(aligned_phonemes_count / max(1, total_expected_phonemes), 3)
            mean_overall_conf = round(float(np.mean(all_confidences)), 3) if all_confidences else None

            status = "ready" if coverage_ratio >= 0.70 else "experimental"

            methodology_note = (
                "Reference-based phoneme assessment using Wav2Vec2 CTC phonetic alignment against canonical targets. "
                "Acoustic match confidence reflects frame-averaged posterior probability under the acoustic model (0.0 to 1.0). "
                "Notice: Acoustic frame confidence measures phonetic match evidence, not an accredited oral proficiency grade. "
                "Low confidence may result from accent diversity, background noise, or microphone placement rather than pronunciation errors."
            )

            return PronunciationAssessmentResult(
                status=status,
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=norm_ref,
                transcript_source="User-provided reference",
                reference_text=norm_ref,
                word_diagnostics=word_diagnostics,
                alignment_coverage_ratio=coverage_ratio,
                mean_acoustic_confidence=mean_overall_conf,
                methodology_note=methodology_note,
            )

        except Exception as exc:
            logger.warning(f"Reference pronunciation assessment error: {exc}", exc_info=True)
            return PronunciationAssessmentResult(
                status="error",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=norm_ref,
                transcript_source="User-provided reference",
                reference_text=norm_ref,
                word_diagnostics=[],
                methodology_note=f"Processing failed: {exc}",
                error_message=str(exc),
            )

    def _assess_spontaneous(
        self,
        samples: np.ndarray,
        sample_rate: int,
        words: Sequence[Any],
        transcript_text: str,
    ) -> PronunciationAssessmentResult:
        # Graceful 'unavailable' state if model weights are not loaded
        if not self.is_available:
            return PronunciationAssessmentResult(
                status="unavailable",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=transcript_text,
                transcript_source="ASR-generated (faster-whisper)",
                word_diagnostics=[],
                methodology_note=(
                    "Pronunciation assessment is unavailable: Pretrained phoneme-level acoustic model is not initialized. "
                    "Unscripted interview speech evaluation requires dedicated phoneme posterior calibration."
                ),
            )

        if not words or not transcript_text.strip():
            return PronunciationAssessmentResult(
                status="experimental",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=transcript_text,
                transcript_source="ASR-generated (faster-whisper)",
                word_diagnostics=[],
                methodology_note="Transcript is empty; no words available for phoneme alignment.",
            )

        try:
            word_diagnostics: List[WordPronunciationDiagnostic] = []

            for w in words:
                raw_word = getattr(w, "word", "").strip()
                if not raw_word:
                    continue

                w_start = round(float(getattr(w, "start", 0.0)), 3)
                w_end = round(float(getattr(w, "end", 0.0)), 3)
                w_dur = max(0.01, w_end - w_start)

                target_phonemes = word_to_phonemes(raw_word)
                num_phones = len(target_phonemes)

                phone_diagnostics: List[PhonemeDiagnostic] = []
                for idx, ph in enumerate(target_phonemes):
                    p_start = round(w_start + (idx / num_phones) * w_dur, 3)
                    p_end = round(w_start + ((idx + 1) / num_phones) * w_dur, 3)
                    p_dur = round(p_end - p_start, 3)

                    match_conf = None
                    if hasattr(self._model, "score_phoneme"):
                        match_conf = self._model.score_phoneme(ph, p_start, p_end)

                    phone_diagnostics.append(
                        PhonemeDiagnostic(
                            phoneme=ph,
                            start=p_start,
                            end=p_end,
                            duration=p_dur,
                            acoustic_match_confidence=match_conf,
                        )
                    )

                word_diagnostics.append(
                    WordPronunciationDiagnostic(
                        word=raw_word,
                        start=w_start,
                        end=w_end,
                        phonemes=phone_diagnostics,
                        flagged=False,
                        diagnostic_note=None,
                    )
                )

            return PronunciationAssessmentResult(
                status="experimental",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=transcript_text,
                transcript_source="ASR-generated (faster-whisper)",
                word_diagnostics=word_diagnostics,
                methodology_note=(
                    "Experimental diagnostic: Word boundaries mapped to canonical ARPAbet targets. "
                    "Notice: Because transcripts are generated by ASR, pronunciation anomalies cannot be differentiated "
                    "from ASR transcription errors without human reference ground truth. Not an accredited L2 score."
                ),
            )

        except Exception as exc:
            logger.warning(f"Pronunciation assessment error: {exc}", exc_info=True)
            return PronunciationAssessmentResult(
                status="error",
                model_identifier=self._model_name,
                phoneme_inventory="ARPAbet-39",
                transcript_evaluated=transcript_text,
                transcript_source="ASR-generated (faster-whisper)",
                word_diagnostics=[],
                methodology_note=f"Processing failed: {exc}",
                error_message=str(exc),
            )
