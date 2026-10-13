import logging
import threading
from typing import Optional, Dict, Any

from backend.app.config import settings
from backend.app.services.pronunciation.base import PronunciationScorer
from backend.app.services.pronunciation.ctc_aligner import CTCPhonemeAligner

logger = logging.getLogger("interviewsense.models")


class ModelManager:
    """
    Singleton lifecycle manager for acoustic and speech recognition models.
    Loads models once at startup and reuses them across requests.
    """

    def __init__(self):
        self._asr_model: Any = None
        self._asr_status: str = "uninitialized"
        self._asr_error: Optional[str] = None
        self._model_name: str = settings.WHISPER_MODEL_SIZE

        # Pronunciation acoustic model state & synchronization lock
        self._pronunciation_model: Optional[Any] = None
        self._pronunciation_model_name: str = settings.PRONUNCIATION_MODEL_NAME
        self._pronunciation_status: str = "unavailable"
        self._pronunciation_error: Optional[str] = None
        self._pronunciation_lock = threading.Lock()
        self._pronunciation_scorer: PronunciationScorer = CTCPhonemeAligner(model=None, model_name=self._pronunciation_model_name)

    @property
    def asr_model(self) -> Any:
        return self._asr_model

    @property
    def is_asr_ready(self) -> bool:
        return self._asr_model is not None and self._asr_status == "loaded"

    @property
    def is_pronunciation_ready(self) -> bool:
        return self._pronunciation_scorer is not None and self._pronunciation_scorer.is_available

    @property
    def pronunciation_scorer(self) -> PronunciationScorer:
        return self._pronunciation_scorer

    @pronunciation_scorer.setter
    def pronunciation_scorer(self, scorer: PronunciationScorer) -> None:
        self._pronunciation_scorer = scorer
        if scorer and getattr(scorer, "is_available", False):
            self._pronunciation_status = "ready"
        else:
            self._pronunciation_status = "unavailable"

    def set_asr_model(self, model: Any, model_name: str = "mock-small.en") -> None:
        """Test utility to inject a mock ASR model without triggering weights download."""
        self._asr_model = model
        self._model_name = model_name
        self._asr_status = "loaded"
        self._asr_error = None

    def set_pronunciation_scorer(self, scorer: PronunciationScorer) -> None:
        """Test utility to inject a mock pronunciation scorer."""
        self._pronunciation_scorer = scorer
        if scorer and getattr(scorer, "is_available", False):
            self._pronunciation_status = "ready"
        else:
            self._pronunciation_status = "unavailable"

    def set_pronunciation_model(self, model: Any, model_name: str = "mock-wav2vec2") -> None:
        """Test utility to inject a mock pronunciation acoustic model."""
        self._pronunciation_model = model
        self._pronunciation_model_name = model_name
        self._pronunciation_status = "ready"
        self._pronunciation_error = None
        self._pronunciation_scorer = CTCPhonemeAligner(model=model, model_name=model_name)

    def load_pronunciation_model(self) -> bool:
        """
        Thread-safe lazy initialization of the Wav2Vec2 phoneme acoustic model.
        Returns True if loaded, False otherwise.
        """
        if self.is_pronunciation_ready:
            return True

        with self._pronunciation_lock:
            if self.is_pronunciation_ready:
                return True

            self._pronunciation_status = "loading"
            logger.info(f"Lazily loading pronunciation model '{self._pronunciation_model_name}' on {settings.PRONUNCIATION_DEVICE}...")

            try:
                from backend.app.services.pronunciation.wav2vec2_model import Wav2Vec2PhonemeModel

                model = Wav2Vec2PhonemeModel(
                    model_name=self._pronunciation_model_name,
                    device=settings.PRONUNCIATION_DEVICE,
                )
                loaded = model.load_model()
                if loaded:
                    self._pronunciation_model = model
                    self._pronunciation_status = "ready"
                    self._pronunciation_error = None
                    self._pronunciation_scorer = CTCPhonemeAligner(
                        model=model, model_name=self._pronunciation_model_name
                    )
                    logger.info("Pronunciation acoustic model loaded successfully.")
                    return True
                else:
                    self._pronunciation_status = model.status
                    self._pronunciation_error = model._error
                    self._pronunciation_model = None
                    return False

            except Exception as exc:
                self._pronunciation_status = "failed"
                self._pronunciation_error = str(exc)
                self._pronunciation_model = None
                logger.warning(f"Could not initialize pronunciation acoustic model: {exc}")
                return False

    def get_status(self) -> Dict[str, str]:
        if self._asr_status == "loaded":
            asr_status_str = f"{self._model_name} (loaded)"
        elif self._asr_status == "failed":
            asr_status_str = f"{self._model_name} (failed: {self._asr_error})"
        else:
            asr_status_str = f"{self._model_name} ({self._asr_status})"

        if self._pronunciation_status == "ready" or (self._pronunciation_scorer and self._pronunciation_scorer.is_available):
            model_name = getattr(self._pronunciation_scorer, "_model_name", self._pronunciation_model_name)
            pron_status = f"{model_name} (ready)"
        elif self._pronunciation_status == "loading":
            pron_status = f"{self._pronunciation_model_name} (loading)"
        elif self._pronunciation_status == "failed":
            pron_status = f"{self._pronunciation_model_name} (failed: {self._pronunciation_error})"
        else:
            pron_status = f"{self._pronunciation_model_name} (unavailable)"

        return {
            "asr": asr_status_str,
            "acoustic": "praat-parselmouth (ready)",
            "pronunciation": pron_status,
        }

    def load_models(self) -> None:
        """
        Instantiates faster-whisper WhisperModel once at application startup.
        Uses configured model size, device, compute type, and CPU threads.
        """
        logger.info(
            f"Initializing faster-whisper model '{settings.WHISPER_MODEL_SIZE}' "
            f"(device={settings.WHISPER_DEVICE}, compute_type={settings.WHISPER_COMPUTE_TYPE}, "
            f"cpu_threads={settings.WHISPER_CPU_THREADS})..."
        )
        self._asr_status = "loading"
        try:
            from faster_whisper import WhisperModel
            self._asr_model = WhisperModel(
                model_size_or_path=settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                cpu_threads=settings.WHISPER_CPU_THREADS,
            )
            self._model_name = settings.WHISPER_MODEL_SIZE
            self._asr_status = "loaded"
            self._asr_error = None
            logger.info("faster-whisper model loaded successfully.")
        except Exception as exc:
            if settings.WHISPER_MODEL_SIZE != "tiny.en":
                logger.warning(
                    f"Could not initialize primary model '{settings.WHISPER_MODEL_SIZE}': {exc}. "
                    "Attempting fallback to local 'tiny.en' model..."
                )
                try:
                    self._asr_model = WhisperModel(
                        model_size_or_path="tiny.en",
                        device=settings.WHISPER_DEVICE,
                        compute_type=settings.WHISPER_COMPUTE_TYPE,
                        cpu_threads=settings.WHISPER_CPU_THREADS,
                    )
                    self._model_name = "tiny.en"
                    self._asr_status = "loaded"
                    self._asr_error = None
                    logger.info("Fallback 'tiny.en' faster-whisper model loaded successfully.")
                    return
                except Exception as fallback_exc:
                    logger.error(f"Fallback to 'tiny.en' also failed: {fallback_exc}")

            self._asr_status = "failed"
            self._asr_error = str(exc)
            self._asr_model = None
            logger.error(f"Failed to initialize faster-whisper model: {exc}")

    def unload_models(self) -> None:
        """Clean teardown of allocated model resources."""
        self._asr_model = None
        self._asr_status = "unloaded"
        self._asr_error = None
        self._pronunciation_model = None
        self._pronunciation_status = "unloaded"
        self._pronunciation_error = None
        self._pronunciation_scorer = CTCPhonemeAligner(model=None, model_name=self._pronunciation_model_name)
        logger.info("Model resources unloaded.")


model_manager = ModelManager()
