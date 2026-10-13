from typing import List, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    error_code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable explanation of the error")


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class ModelStatus(BaseModel):
    asr: str
    acoustic: str
    pronunciation: str


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Overall backend health status")
    project: str = Field(..., description="Project name")
    version: str = Field(..., description="Application semantic version")
    models: ModelStatus = Field(..., description="Status of acoustic and ASR models")


class AudioMetadata(BaseModel):
    duration_seconds: float = Field(..., description="Duration of decoded audio in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz (normalized to 16000)")
    channels: int = Field(..., description="Channel count (normalized to 1 mono)")
    original_format: Optional[str] = Field(None, description="Original container format if detected")
    rms_energy: float = Field(..., description="Root Mean Square energy of normalized signal")
    peak_amplitude: float = Field(..., description="Peak absolute amplitude")


class WordRecordSchema(BaseModel):
    word: str = Field(..., description="Recognized word token text")
    start: float = Field(..., description="Word start timestamp in seconds")
    end: float = Field(..., description="Word end timestamp in seconds")
    probability: float = Field(..., description="Acoustic/language model token probability")


class TranscriptResponse(BaseModel):
    text: str = Field(..., description="Full recognized speech transcript")
    language: str = Field(..., description="Detected or configured language")
    model_used: str = Field(..., description="ASR model name")
    words: List[WordRecordSchema] = Field(
        default_factory=list,
        description="Chronologically ordered word timestamps"
    )


class FillerWordSchema(BaseModel):
    word: str = Field(..., description="Filler word or idiom text")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    category: str = Field(..., description="Disfluency category (e.g. vocal_disfluency, potential_filler)")


class RepetitionRecordSchema(BaseModel):
    repeated_text: str = Field(..., description="Repeated word or phrase text")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    count: int = Field(default=2, description="Number of immediate repetitions")
    repetition_type: str = Field(..., description="Type of repetition (word_repetition or phrase_repetition)")


class FluencyMetricsSchema(BaseModel):
    word_count: int = Field(..., description="Total recognized words")
    gross_wpm: float = Field(..., description="Gross speaking rate in words per minute")
    articulation_rate_wpm: float = Field(..., description="Articulation rate in words per minute (excluding pauses)")
    speaking_duration_seconds: float = Field(..., description="Active phonation time in seconds")
    total_pause_duration_seconds: float = Field(..., description="Total pause duration in seconds")
    pause_to_speech_ratio: float = Field(..., description="Ratio of pause duration to total audio duration")
    filler_count: int = Field(..., description="Total detected filler expressions")
    filler_percentage: float = Field(..., description="Percentage of words identified as fillers")
    filler_words: List[FillerWordSchema] = Field(default_factory=list, description="Detected filler words with timestamps")
    repetitions: List[RepetitionRecordSchema] = Field(default_factory=list, description="Detected immediate repetitions")


class PauseIntervalSchema(BaseModel):
    start: float = Field(..., description="Pause start timestamp in seconds")
    end: float = Field(..., description="Pause end timestamp in seconds")
    duration: float = Field(..., description="Pause duration in seconds")


class PauseStatisticsSchema(BaseModel):
    total_pauses: int = Field(..., description="Total detected pause intervals")
    total_pause_duration_seconds: float = Field(..., description="Total pause time in seconds")
    pauses_over_500ms: int = Field(..., description="Number of pauses >= 0.5s")
    pauses_over_2s: int = Field(..., description="Number of pauses >= 2.0s")
    longest_pause_seconds: float = Field(..., description="Duration of the longest pause in seconds")
    average_pause_seconds: float = Field(..., description="Average pause duration in seconds")
    pause_to_total_ratio: float = Field(..., description="Ratio of pause time to total audio duration")
    pause_intervals: List[PauseIntervalSchema] = Field(default_factory=list, description="List of pause intervals")
    note: str = Field(..., description="Methodological note regarding timestamp-derived pauses")


class PitchMetricsSchema(BaseModel):
    status: str = Field(..., description="'available' if voiced frames exist, else 'unavailable'")
    median_f0_hz: Optional[float] = Field(None, description="Median fundamental frequency (F0) in Hz")
    mean_f0_hz: Optional[float] = Field(None, description="Mean fundamental frequency (F0) in Hz")
    std_f0_hz: Optional[float] = Field(None, description="Standard deviation of fundamental frequency (F0) in Hz")
    p05_f0_hz: Optional[float] = Field(None, description="5th percentile of F0 in Hz")
    p95_f0_hz: Optional[float] = Field(None, description="95th percentile of F0 in Hz")
    range_semitones: Optional[float] = Field(None, description="Pitch range span in semitones (12 * log2(P95 / P05))")
    voiced_fraction: Optional[float] = Field(None, description="Proportion of frames classified as voiced")
    pitch_floor_hz: float = Field(..., description="Configured pitch floor boundary in Hz")
    pitch_ceiling_hz: float = Field(..., description="Configured pitch ceiling boundary in Hz")
    note: str = Field(..., description="Methodological note regarding pitch extraction")


class EnergyMetricsSchema(BaseModel):
    mean_rms: float = Field(..., description="Mean frame-level linear Root Mean Square energy")
    std_rms: float = Field(..., description="Standard deviation of frame-level RMS energy")
    mean_rms_dbfs: float = Field(..., description="Mean frame RMS in digital decibels relative to full scale (dBFS)")
    dynamic_range_db: float = Field(..., description="RMS energy dynamic range in dB (P95 dBFS - P05 dBFS)")
    frame_length_ms: float = Field(..., description="Frame window length in milliseconds")
    hop_length_ms: float = Field(..., description="Frame step/hop length in milliseconds")
    note: str = Field(..., description="Methodological note regarding digital dBFS vs sound pressure level")


class AcousticsSchema(BaseModel):
    pitch: PitchMetricsSchema = Field(..., description="Fundamental frequency and pitch modulation metrics")
    energy: EnergyMetricsSchema = Field(..., description="Frame RMS energy dynamics and digital volume range")


class PhonemeDiagnosticSchema(BaseModel):
    phoneme: str = Field(..., description="Target ARPAbet canonical phoneme label")
    start: float = Field(..., description="Phoneme estimated start timestamp in seconds")
    end: float = Field(..., description="Phoneme estimated end timestamp in seconds")
    duration: float = Field(..., description="Phoneme duration in seconds")
    acoustic_match_confidence: Optional[float] = Field(
        None, description="Phonetic acoustic match probability (null if uncalibrated)"
    )
    is_supported: bool = Field(default=True, description="Whether phoneme is supported in acoustic model inventory")
    diagnostic_note: Optional[str] = Field(None, description="Diagnostic observation note for this phoneme")


class WordPronunciationDiagnosticSchema(BaseModel):
    word: str = Field(..., description="Recognized or reference word text")
    start: float = Field(..., description="Word start timestamp in seconds")
    end: float = Field(..., description="Word end timestamp in seconds")
    phonemes: List[PhonemeDiagnosticSchema] = Field(default_factory=list, description="Phoneme segment diagnostics")
    flagged: bool = Field(default=False, description="Flag indicating potential phonetic discrepancy or alignment uncertainty")
    diagnostic_note: Optional[str] = Field(None, description="Diagnostic observation note")
    mean_acoustic_confidence: Optional[float] = Field(None, description="Mean posterior confidence across word phonemes")


class PronunciationAssessmentSchema(BaseModel):
    status: str = Field(..., description="Status: 'unavailable', 'experimental', 'ready', or 'error'")
    model_identifier: Optional[str] = Field(None, description="Acoustic alignment model identifier")
    phoneme_inventory: Optional[str] = Field(None, description="Phoneme standard used (e.g. ARPAbet-39)")
    transcript_evaluated: Optional[str] = Field(None, description="Transcript text evaluated")
    transcript_source: str = Field(
        default="ASR-generated (faster-whisper)",
        description="Source of the reference transcript evaluated"
    )
    word_diagnostics: List[WordPronunciationDiagnosticSchema] = Field(
        default_factory=list,
        description="Detailed phoneme-level diagnostic breakdowns"
    )
    methodology_note: str = Field(..., description="Transparency statement regarding unscripted L2 evaluation")
    error_message: Optional[str] = Field(None, description="Error explanation if status is 'error'")
    reference_text: Optional[str] = Field(None, description="User-provided reference text for assessment")
    alignment_coverage_ratio: Optional[float] = Field(
        None, description="Proportion of expected phonemes successfully aligned (0.0 to 1.0)"
    )
    mean_acoustic_confidence: Optional[float] = Field(
        None, description="Mean posterior confidence across scorable phonemes (0.0 to 1.0)"
    )


class AnalysisResponse(BaseModel):
    status: str = Field(default="validated", description="Analysis execution status")
    metadata: AudioMetadata = Field(..., description="Acoustic file and signal metadata")
    transcript: TranscriptResponse = Field(..., description="Speech transcription and word timestamps")
    fluency: FluencyMetricsSchema = Field(..., description="Pacing, speaking rate, fillers, and repetitions")
    pauses: PauseStatisticsSchema = Field(..., description="Pause intervals and timing distribution")
    acoustics: AcousticsSchema = Field(..., description="Acoustic pitch and RMS energy dynamics")
    pronunciation: PronunciationAssessmentSchema = Field(..., description="Experimental phoneme diagnostic assessment")


class AudioValidationResponse(BaseModel):
    status: str = Field(default="validated", description="Audio processing and validation status")
    message: str = Field(
        default="Audio successfully decoded, resampled to 16kHz mono, and validated.",
        description="Human-readable status summary"
    )
    metadata: AudioMetadata
