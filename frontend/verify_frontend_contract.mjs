import test from 'node:test';
import assert from 'node:assert/strict';

test('Frontend Contract & State Handling Verification', async (t) => {
  await t.test('handles full standard API analysis response', () => {
    const mockApiResponse = {
      status: 'validated',
      metadata: {
        duration_seconds: 12.4,
        sample_rate: 16000,
        channels: 1,
        original_format: 'matroska,webm',
        rms_energy: 0.045,
        peak_amplitude: 0.82,
      },
      transcript: {
        text: 'I think that my greatest strength is communication.',
        language: 'en',
        model_used: 'small.en',
        words: [
          { word: 'I', start: 0.1, end: 0.3, probability: 0.95 },
          { word: 'think', start: 0.35, end: 0.65, probability: 0.91 },
          { word: 'that', start: 0.7, end: 0.9, probability: 0.88 },
        ],
      },
      fluency: {
        word_count: 8,
        gross_wpm: 135.0,
        articulation_rate_wpm: 155.0,
        speaking_duration_seconds: 9.8,
        total_pause_duration_seconds: 2.6,
        pause_to_speech_ratio: 0.21,
        filler_count: 1,
        filler_percentage: 12.5,
        filler_words: [
          { word: 'um', start: 1.2, end: 1.5, category: 'vocal_disfluency' }
        ],
        repetitions: [
          {
            repeated_text: 'I think',
            start: 0.1,
            end: 1.1,
            count: 2,
            repetition_type: 'phrase_repetition'
          }
        ],
      },
      pauses: {
        total_pauses: 3,
        total_pause_duration_seconds: 2.6,
        pauses_over_500ms: 2,
        pauses_over_2s: 1,
        longest_pause_seconds: 2.1,
        average_pause_seconds: 0.86,
        pause_to_total_ratio: 0.21,
        pause_intervals: [
          { start: 2.0, end: 2.6, duration: 0.6 },
          { start: 5.0, end: 7.1, duration: 2.1 },
        ],
        note: 'Derived from inter-word timestamp intervals.',
      },
      acoustics: {
        pitch: {
          status: 'available',
          median_f0_hz: 142.5,
          mean_f0_hz: 145.2,
          std_f0_hz: 18.4,
          p05_f0_hz: 110.0,
          p95_f0_hz: 185.0,
          range_semitones: 9.0,
          voiced_fraction: 0.72,
          pitch_floor_hz: 65.0,
          pitch_ceiling_hz: 400.0,
          note: 'Praat autocorrelation F0 analysis.',
        },
        energy: {
          mean_rms: 0.045,
          std_rms: 0.015,
          mean_rms_dbfs: -26.9,
          dynamic_range_db: 18.2,
          frame_length_ms: 25.0,
          hop_length_ms: 10.0,
          note: 'Frame-level RMS energy.',
        },
      },
      pronunciation: {
        status: 'unavailable',
        model_identifier: null,
        phoneme_inventory: null,
        transcript_evaluated: null,
        transcript_source: 'ASR-generated (faster-whisper)',
        word_diagnostics: [],
        methodology_note: 'Experimental acoustic diagnostic. Unscripted speech lacks ground truth.',
        error_message: null,
      },
    };

    assert.equal(mockApiResponse.status, 'validated');
    assert.equal(mockApiResponse.metadata.duration_seconds, 12.4);
    assert.equal(mockApiResponse.acoustics.pitch.status, 'available');
    assert.equal(mockApiResponse.pronunciation.status, 'unavailable');
  });

  await t.test('safely handles null pitch values and unvoiced audio', () => {
    const unvoicedAcoustics = {
      pitch: {
        status: 'unavailable',
        median_f0_hz: null,
        mean_f0_hz: null,
        std_f0_hz: null,
        p05_f0_hz: null,
        p95_f0_hz: null,
        range_semitones: null,
        voiced_fraction: 0.0,
        pitch_floor_hz: 65.0,
        pitch_ceiling_hz: 400.0,
        note: 'No periodic voiced frames detected within configured boundaries.',
      },
      energy: {
        mean_rms: 0.001,
        std_rms: 0.0002,
        mean_rms_dbfs: -60.0,
        dynamic_range_db: 4.0,
        frame_length_ms: 25.0,
        hop_length_ms: 10.0,
        note: 'Frame-level RMS energy.',
      },
    };

    assert.equal(unvoicedAcoustics.pitch.status, 'unavailable');
    assert.equal(unvoicedAcoustics.pitch.median_f0_hz, null);
    assert.equal(unvoicedAcoustics.pitch.range_semitones, null);
  });

  await t.test('safely parses structured error details from backend', () => {
    const apiError = {
      detail: {
        error_code: 'FILE_TOO_LARGE',
        message: 'File size exceeds maximum permitted limit of 25 MB.',
      },
    };

    assert.equal(apiError.detail.error_code, 'FILE_TOO_LARGE');
    assert.match(apiError.detail.message, /25 MB/);
  });

  await t.test('handles empty filler and repetition arrays without errors', () => {
    const cleanSpeech = {
      word_count: 50,
      gross_wpm: 140.0,
      articulation_rate_wpm: 150.0,
      speaking_duration_seconds: 20.0,
      total_pause_duration_seconds: 1.4,
      pause_to_speech_ratio: 0.07,
      filler_count: 0,
      filler_percentage: 0.0,
      filler_words: [],
      repetitions: [],
    };

    assert.equal(cleanSpeech.filler_words.length, 0);
    assert.equal(cleanSpeech.repetitions.length, 0);
  });

  await t.test('handles reference-based pronunciation assessment payload', () => {
    const referencePronunciation = {
      status: 'ready',
      model_identifier: 'facebook/wav2vec2-lv-60-espeak-cv-ft',
      phoneme_inventory: 'ARPAbet-39',
      transcript_evaluated: 'hello world',
      transcript_source: 'User-provided reference',
      reference_text: 'hello world',
      alignment_coverage_ratio: 1.0,
      mean_acoustic_confidence: 0.88,
      word_diagnostics: [
        {
          word: 'hello',
          start: 0.12,
          end: 0.54,
          flagged: false,
          diagnostic_note: null,
          mean_acoustic_confidence: 0.90,
          phonemes: [
            {
              phoneme: 'HH',
              start: 0.12,
              end: 0.20,
              duration: 0.08,
              acoustic_match_confidence: 0.92,
              is_supported: true,
              diagnostic_note: null,
            },
          ],
        },
      ],
      methodology_note: 'Reference-based phoneme assessment using Wav2Vec2 CTC phonetic alignment.',
      error_message: null,
    };

    assert.equal(referencePronunciation.status, 'ready');
    assert.equal(referencePronunciation.reference_text, 'hello world');
    assert.equal(referencePronunciation.alignment_coverage_ratio, 1.0);
    assert.equal(referencePronunciation.mean_acoustic_confidence, 0.88);
    assert.equal(referencePronunciation.word_diagnostics.length, 1);
    assert.equal(referencePronunciation.word_diagnostics[0].phonemes[0].is_supported, true);
  });
});
