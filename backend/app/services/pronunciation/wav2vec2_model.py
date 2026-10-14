import logging
import threading
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("interviewsense.pronunciation.wav2vec2")


class Wav2Vec2PhonemeModel:
    """
    Lazy-loading wrapper for a Wav2Vec2 CTC phoneme acoustic model.
    Extracts frame-level phoneme log-probability emissions from 16kHz mono audio.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._vocab: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._status: str = "unavailable"  # "unavailable", "loading", "ready", "failed"
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status == "ready" and self._model is not None

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def id2token(self) -> Dict[int, str]:
        return self._id2token

    def load_model(self) -> bool:
        """
        Thread-safe lazy initialization of the Wav2Vec2 phoneme CTC model.
        Returns True if loaded, False otherwise.
        """
        if self.is_ready:
            return True

        with self._lock:
            if self.is_ready:
                return True

            self._status = "loading"
            logger.info(f"Lazily loading Wav2Vec2 phoneme model '{self.model_name}' on {self.device}...")

            try:
                import torch
                from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, AutoModelForCTC

                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCTC.from_pretrained(self.model_name)
                model.to(self.device)
                model.eval()

                # Build vocabulary mappings
                if hasattr(tokenizer, "get_vocab"):
                    self._vocab = dict(tokenizer.get_vocab())
                else:
                    self._vocab = {}

                self._id2token = {v: k for k, v in self._vocab.items()}
                self._feature_extractor = feature_extractor
                self._tokenizer = tokenizer
                self._model = model
                self._status = "ready"
                self._error = None
                logger.info(f"Wav2Vec2 phoneme model '{self.model_name}' ready. Vocab size: {len(self._vocab)}.")
                return True

            except Exception as exc:
                self._status = "failed"
                self._error = str(exc)
                self._feature_extractor = None
                self._tokenizer = None
                self._model = None
                logger.warning(
                    f"Wav2Vec2 phoneme model '{self.model_name}' could not be initialized: {exc}. "
                    "Pronunciation assessment will report as unavailable."
                )
                return False

    def get_frame_emissions(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """
        Runs forward inference on 16kHz mono audio and returns:
        - log_probs: ndarray of shape (T, Vocab_size) containing log emission probabilities
        - time_per_frame: float, duration in seconds per frame (approx 0.020s)
        """
        if not self.is_ready:
            raise RuntimeError(f"Wav2Vec2 phoneme model is not ready (status: {self._status}).")

        import torch

        with torch.no_grad():
            inputs = self._feature_extractor(
                samples,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(self.device)
            logits = self._model(input_values).logits  # (1, T, Vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            emissions = log_probs.squeeze(0).cpu().numpy()  # (T, Vocab)

        num_frames = max(1, emissions.shape[0])
        total_dur = len(samples) / float(sample_rate)
        time_per_frame = total_dur / float(num_frames)

        return emissions, time_per_frame


class MockPhonemeModel:
    """
    Deterministic mock acoustic phoneme model for testing CTC Trellis alignment
    and diagnostics without downloading external weights.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        model_name: str = "mock-wav2vec2-phoneme",
        confidence_level: float = 0.88,
        custom_emissions: Optional[np.ndarray] = None,
        simulate_aligned_tokens: Optional[List[int]] = None,
    ):
        self.model_name = model_name
        self.confidence_level = confidence_level
        self.custom_emissions = custom_emissions
        self.simulate_aligned_tokens = simulate_aligned_tokens
        self._vocab = vocab or {
            "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
            "h": 4, "ʌ": 5, "l": 6, "oʊ": 7, "w": 8, "ɜː": 9, "d": 10,
            "ð": 11, "ɪ": 12, "s": 13, "z": 14, "t": 15, "ɛ": 16,
            "aɪ": 17, "æ": 18, "v": 19, "b": 20, "n": 21, "k": 22,
            "ŋ": 23, "ɑː": 24, "ɹ": 25, "f": 26, "ɔː": 27, "aʊ": 28,
            "θ": 29, "iː": 30, "j": 31, "p": 32, "tʃ": 33, "m": 34,
            "uː": 35, "dʒ": 36, "ɡ": 37, "ʃ": 38, "ʒ": 39, "ʊ": 40,
            "eɪ": 41, "ɔɪ": 42,
        }
        self._id2token = {v: k for k, v in self._vocab.items()}
        self._status = "ready"

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def id2token(self) -> Dict[int, str]:
        return self._id2token

    def score_phoneme(self, phoneme: str, start: float, end: float) -> float:
        """Backward compatibility for legacy mock tests."""
        return self.confidence_level

    def get_frame_emissions(
        self, samples: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """
        Generates synthetic frame emissions (T, V) with customizable peaks.
        """
        if self.custom_emissions is not None:
            time_per_frame = (len(samples) / float(sample_rate)) / float(max(1, len(self.custom_emissions)))
            return self.custom_emissions, time_per_frame

        num_frames = max(5, int(len(samples) / 320))
        max_id = max(self._vocab.values()) if self._vocab else 49
        vocab_size = max(len(self._vocab), max_id + 1)

        # Baseline background noise log-probs
        emissions = np.full((num_frames, vocab_size), -4.5, dtype=np.float32)

        # High blank log-prob initially
        emissions[:, 0] = -0.5

        if self.simulate_aligned_tokens:
            tokens = self.simulate_aligned_tokens
            k = len(tokens)
            frames_per_token = max(1, (num_frames - 2) // max(1, k))
            peak_log_prob = float(np.log(max(0.01, min(0.99, self.confidence_level))))
            other_log_prob = float(np.log((1.0 - np.exp(peak_log_prob)) / max(1, vocab_size - 1)))

            for idx, tok in enumerate(tokens):
                st = min(num_frames - 1, 1 + idx * frames_per_token)
                en = min(num_frames, st + frames_per_token)
                if tok < vocab_size:
                    emissions[st:en, :] = other_log_prob
                    emissions[st:en, tok] = peak_log_prob

        time_per_frame = (len(samples) / float(sample_rate)) / float(num_frames)
        return emissions, time_per_frame
