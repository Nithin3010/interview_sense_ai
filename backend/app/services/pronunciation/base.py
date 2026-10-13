from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
import numpy as np


@dataclass
class PhonemeDiagnostic:
    phoneme: str
    start: float
    end: float
    duration: float
    acoustic_match_confidence: Optional[float] = None  # Frame-averaged posterior (0.0 to 1.0) or None if uncalibrated/uncertain
    is_supported: bool = True
    diagnostic_note: Optional[str] = None


@dataclass
class WordPronunciationDiagnostic:
    word: str
    start: float
    end: float
    phonemes: List[PhonemeDiagnostic] = field(default_factory=list)
    flagged: bool = False
    diagnostic_note: Optional[str] = None
    mean_acoustic_confidence: Optional[float] = None


@dataclass
class PronunciationAssessmentResult:
    status: str  # "unavailable", "experimental", or "error"
    model_identifier: Optional[str] = None
    phoneme_inventory: Optional[str] = None  # e.g., "ARPAbet-39" or "eSpeak-IPA"
    transcript_evaluated: Optional[str] = None
    transcript_source: str = "ASR-generated (faster-whisper)"  # or "User-provided reference"
    word_diagnostics: List[WordPronunciationDiagnostic] = field(default_factory=list)
    methodology_note: str = ""
    error_message: Optional[str] = None
    reference_text: Optional[str] = None
    alignment_coverage_ratio: Optional[float] = None
    mean_acoustic_confidence: Optional[float] = None


class PronunciationScorer(ABC):
    """
    Abstract interface for phoneme-level pronunciation assessment models.
    Enforces clear separation between ASR recognition and acoustic phonetic scoring.
    """

    @abstractmethod
    def assess(
        self,
        samples: np.ndarray,
        sample_rate: int,
        words: Sequence[Any],
        transcript_text: str,
        reference_text: Optional[str] = None,
    ) -> PronunciationAssessmentResult:
        """
        Assesses pronunciation and outputs diagnostic phonetic boundaries.
        When reference_text is provided, aligns and scores expected target phonemes.
        Must return explicit 'unavailable' or 'experimental' status without fabricating uncalibrated scores.
        """
        pass
