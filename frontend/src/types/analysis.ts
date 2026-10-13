export interface ModelStatus {
  asr: string;
  acoustic: string;
  pronunciation: string;
}

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
  models: ModelStatus;
}

export interface ApiErrorDetail {
  error_code: string;
  message: string;
}

export interface ApiErrorResponse {
  detail: ApiErrorDetail;
}

export interface AudioMetadata {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  original_format?: string | null;
  rms_energy: number;
  peak_amplitude: number;
}

export interface WordRecord {
  word: string;
  start: number;
  end: number;
  probability: number;
}

export interface TranscriptResponse {
  text: string;
  language: string;
  model_used: string;
  words: WordRecord[];
}

export interface FillerWord {
  word: string;
  start: number;
  end: number;
  category: string;
}

export interface RepetitionRecord {
  repeated_text: string;
  start: number;
  end: number;
  count: number;
  repetition_type: string;
}

export interface FluencyMetrics {
  word_count: number;
  gross_wpm: number;
  articulation_rate_wpm: number;
  speaking_duration_seconds: number;
  total_pause_duration_seconds: number;
  pause_to_speech_ratio: number;
  filler_count: number;
  filler_percentage: number;
  filler_words: FillerWord[];
  repetitions: RepetitionRecord[];
}

export interface PauseInterval {
  start: number;
  end: number;
  duration: number;
}

export interface PauseStatistics {
  total_pauses: number;
  total_pause_duration_seconds: number;
  pauses_over_500ms: number;
  pauses_over_2s: number;
  longest_pause_seconds: number;
  average_pause_seconds: number;
  pause_to_total_ratio: number;
  pause_intervals: PauseInterval[];
  note: string;
}

export interface PitchMetrics {
  status: 'available' | 'unavailable' | string;
  median_f0_hz?: number | null;
  mean_f0_hz?: number | null;
  std_f0_hz?: number | null;
  p05_f0_hz?: number | null;
  p95_f0_hz?: number | null;
  range_semitones?: number | null;
  voiced_fraction?: number | null;
  pitch_floor_hz: number;
  pitch_ceiling_hz: number;
  note: string;
}

export interface EnergyMetrics {
  mean_rms: number;
  std_rms: number;
  mean_rms_dbfs: number;
  dynamic_range_db: number;
  frame_length_ms: number;
  hop_length_ms: number;
  note: string;
}

export interface AcousticsMetrics {
  pitch: PitchMetrics;
  energy: EnergyMetrics;
}

export interface PhonemeDiagnostic {
  phoneme: string;
  start: number;
  end: number;
  duration: number;
  acoustic_match_confidence?: number | null;
  is_supported?: boolean;
  diagnostic_note?: string | null;
}

export interface WordPronunciationDiagnostic {
  word: string;
  start: number;
  end: number;
  phonemes: PhonemeDiagnostic[];
  flagged: boolean;
  diagnostic_note?: string | null;
  mean_acoustic_confidence?: number | null;
}

export interface PronunciationAssessment {
  status: 'unavailable' | 'experimental' | 'ready' | 'error' | string;
  model_identifier?: string | null;
  phoneme_inventory?: string | null;
  transcript_evaluated?: string | null;
  transcript_source: string;
  word_diagnostics: WordPronunciationDiagnostic[];
  methodology_note: string;
  error_message?: string | null;
  reference_text?: string | null;
  alignment_coverage_ratio?: number | null;
  mean_acoustic_confidence?: number | null;
}

export interface AnalysisResponse {
  status: string;
  metadata: AudioMetadata;
  transcript: TranscriptResponse;
  fluency: FluencyMetrics;
  pauses: PauseStatistics;
  acoustics: AcousticsMetrics;
  pronunciation: PronunciationAssessment;
}
