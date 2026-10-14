from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from backend.app.config import settings
from backend.app.api.schemas import (
    HealthResponse,
    ModelStatus,
    AnalysisResponse,
    AudioMetadata,
    TranscriptResponse,
    WordRecordSchema,
    FluencyMetricsSchema,
    FillerWordSchema,
    RepetitionRecordSchema,
    PauseStatisticsSchema,
    PauseIntervalSchema,
    AcousticsSchema,
    PitchMetricsSchema,
    EnergyMetricsSchema,
    PronunciationAssessmentSchema,
    WordPronunciationDiagnosticSchema,
    PhonemeDiagnosticSchema,
)
from backend.app.api.dependencies import get_model_manager
from backend.app.core.model_manager import ModelManager
from backend.app.services.audio_processor import (
    read_upload_with_limit,
    decode_and_normalize_audio,
)
from backend.app.services.transcription import transcribe_audio
from backend.app.services.pause_detector import detect_pauses
from backend.app.services.fluency_analyzer import analyze_fluency
from backend.app.services.acoustic_analyzer import analyze_acoustics

from backend.app.core.exceptions import InvalidReferenceTextError

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns backend service health and model initialization status.",
)
async def health_check(
    models: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    status_dict = models.get_status()
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        models=ModelStatus(
            asr=status_dict["asr"],
            acoustic=status_dict["acoustic"],
            pronunciation=status_dict["pronunciation"],
        ),
    )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Audio",
    description="Ingests audio and calculates transcription, fluency, fillers, repetitions, pauses, acoustic metrics, and pronunciation feasibility diagnostics.",
)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Multipart audio file (WebM, WAV, OGG, MP3, etc.)"),
    language: str = Form("en", description="Target spoken language code (default 'en')"),
    reference_text: Optional[str] = Form(None, description="Optional target reference sentence for pronunciation assessment"),
    models: ModelManager = Depends(get_model_manager),
) -> AnalysisResponse:
    # 1. Validate reference text if provided
    cleaned_reference: Optional[str] = None
    if reference_text is not None:
        trimmed = reference_text.strip()
        if not trimmed:
            raise InvalidReferenceTextError("Reference text cannot be empty or whitespace only.")
        cleaned_reference = trimmed

    # 2. Lazily initialize acoustic pronunciation model if needed
    if not models.is_pronunciation_ready:
        models.load_pronunciation_model()

    # 2. Read uploaded audio stream while strictly enforcing max file size
    audio_bytes = await read_upload_with_limit(audio_file)

    # 3. Decode in-memory via PyAV, resample to 16kHz mono, validate duration and silence
    processed = decode_and_normalize_audio(audio_bytes)

    # 4. Transcribe using singleton faster-whisper model
    transcript_res = transcribe_audio(
        samples=processed.samples,
        model=models.asr_model,
        language=language,
        model_name=getattr(models, "_model_name", None),
    )

    # 5. Compute pause statistics derived from word gaps
    pause_stats = detect_pauses(
        words=transcript_res.words,
        total_duration_seconds=processed.duration_seconds,
    )

    # 6. Compute fluency metrics (Gross WPM, Articulation WPM, fillers, repetitions)
    fluency_res = analyze_fluency(
        words=transcript_res.words,
        total_duration_seconds=processed.duration_seconds,
        pause_stats=pause_stats,
    )

    # 7. Compute acoustic delivery features (Praat F0 pitch and frame RMS energy dynamics)
    acoustics_res = analyze_acoustics(
        samples=processed.samples,
        sample_rate=processed.sample_rate,
    )

    # 8. Evaluate pronunciation assessment (reference-based if supplied, otherwise spontaneous)
    pronunciation_res = models.pronunciation_scorer.assess(
        samples=processed.samples,
        sample_rate=processed.sample_rate,
        words=transcript_res.words,
        transcript_text=transcript_res.text,
        reference_text=cleaned_reference,
    )

    # 8. Return aggregated analysis payload
    return AnalysisResponse(
        status="validated",
        metadata=AudioMetadata(
            duration_seconds=processed.duration_seconds,
            sample_rate=processed.sample_rate,
            channels=processed.channels,
            original_format=processed.original_format,
            rms_energy=processed.rms_energy,
            peak_amplitude=processed.peak_amplitude,
        ),
        transcript=TranscriptResponse(
            text=transcript_res.text,
            language=transcript_res.language,
            model_used=transcript_res.model_used,
            words=[
                WordRecordSchema(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in transcript_res.words
            ],
        ),
        fluency=FluencyMetricsSchema(
            word_count=fluency_res.word_count,
            gross_wpm=fluency_res.gross_wpm,
            articulation_rate_wpm=fluency_res.articulation_rate_wpm,
            speaking_duration_seconds=fluency_res.speaking_duration_seconds,
            total_pause_duration_seconds=fluency_res.total_pause_duration_seconds,
            pause_to_speech_ratio=fluency_res.pause_to_speech_ratio,
            filler_count=fluency_res.filler_count,
            filler_percentage=fluency_res.filler_percentage,
            filler_words=[
                FillerWordSchema(
                    word=f.word,
                    start=f.start,
                    end=f.end,
                    category=f.category,
                )
                for f in fluency_res.filler_words
            ],
            repetitions=[
                RepetitionRecordSchema(
                    repeated_text=r.repeated_text,
                    start=r.start,
                    end=r.end,
                    count=r.count,
                    repetition_type=r.repetition_type,
                )
                for r in fluency_res.repetitions
            ],
        ),
        pauses=PauseStatisticsSchema(
            total_pauses=pause_stats.total_pauses,
            total_pause_duration_seconds=pause_stats.total_pause_duration_seconds,
            pauses_over_500ms=pause_stats.pauses_over_500ms,
            pauses_over_2s=pause_stats.pauses_over_2s,
            longest_pause_seconds=pause_stats.longest_pause_seconds,
            average_pause_seconds=pause_stats.average_pause_seconds,
            pause_to_total_ratio=pause_stats.pause_to_total_ratio,
            pause_intervals=[
                PauseIntervalSchema(
                    start=p.start,
                    end=p.end,
                    duration=p.duration,
                )
                for p in pause_stats.pause_intervals
            ],
            note=pause_stats.note,
        ),
        acoustics=AcousticsSchema(
            pitch=PitchMetricsSchema(
                status=acoustics_res.pitch.status,
                median_f0_hz=acoustics_res.pitch.median_f0_hz,
                mean_f0_hz=acoustics_res.pitch.mean_f0_hz,
                std_f0_hz=acoustics_res.pitch.std_f0_hz,
                p05_f0_hz=acoustics_res.pitch.p05_f0_hz,
                p95_f0_hz=acoustics_res.pitch.p95_f0_hz,
                range_semitones=acoustics_res.pitch.range_semitones,
                voiced_fraction=acoustics_res.pitch.voiced_fraction,
                pitch_floor_hz=acoustics_res.pitch.pitch_floor_hz,
                pitch_ceiling_hz=acoustics_res.pitch.pitch_ceiling_hz,
                note=acoustics_res.pitch.note,
            ),
            energy=EnergyMetricsSchema(
                mean_rms=acoustics_res.energy.mean_rms,
                std_rms=acoustics_res.energy.std_rms,
                mean_rms_dbfs=acoustics_res.energy.mean_rms_dbfs,
                dynamic_range_db=acoustics_res.energy.dynamic_range_db,
                frame_length_ms=acoustics_res.energy.frame_length_ms,
                hop_length_ms=acoustics_res.energy.hop_length_ms,
                note=acoustics_res.energy.note,
            ),
        ),
        pronunciation=PronunciationAssessmentSchema(
            status=pronunciation_res.status,
            model_identifier=pronunciation_res.model_identifier,
            phoneme_inventory=pronunciation_res.phoneme_inventory,
            transcript_evaluated=pronunciation_res.transcript_evaluated,
            transcript_source=pronunciation_res.transcript_source,
            reference_text=pronunciation_res.reference_text,
            alignment_coverage_ratio=pronunciation_res.alignment_coverage_ratio,
            mean_acoustic_confidence=pronunciation_res.mean_acoustic_confidence,
            word_diagnostics=[
                WordPronunciationDiagnosticSchema(
                    word=wd.word,
                    start=wd.start,
                    end=wd.end,
                    phonemes=[
                        PhonemeDiagnosticSchema(
                            phoneme=ph.phoneme,
                            start=ph.start,
                            end=ph.end,
                            duration=ph.duration,
                            acoustic_match_confidence=ph.acoustic_match_confidence,
                            is_supported=getattr(ph, "is_supported", True),
                            diagnostic_note=getattr(ph, "diagnostic_note", None),
                        )
                        for ph in wd.phonemes
                    ],
                    flagged=wd.flagged,
                    diagnostic_note=wd.diagnostic_note,
                    mean_acoustic_confidence=getattr(wd, "mean_acoustic_confidence", None),
                )
                for wd in pronunciation_res.word_diagnostics
            ],
            methodology_note=pronunciation_res.methodology_note,
            error_message=pronunciation_res.error_message,
        ),
    )
