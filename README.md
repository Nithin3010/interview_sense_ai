# InterviewSense AI

An AI-powered local speech delivery and acoustic analysis platform for interview practice. InterviewSense AI evaluates temporal pacing, disfluencies, pauses, intonational pitch modulation, volume dynamics, and speech recognition without cloud dependencies or persistent recording storage.

---

## 1. System Architecture & Overview

InterviewSense AI runs entirely on local models and in-memory signal processing pipelines:

```
[ Browser UI / React 19 ]
       │
       ├── Live Microphone (MediaRecorder + Web Audio Analyser)
       └── File Upload (WAV, MP3, OGG, FLAC, WEBM, M4A <= 25MB)
       │
       ▼ (multipart/form-data via Vite proxy)
[ FastAPI Backend / Python 3.14+ ]
       │
       ├── 1. In-Memory Ingestion (PyAV)
       │      • Decodes directly from RAM byte stream (no disk writes)
       │      • Resamples to 16,000 Hz, mono float32 PCM
       │      • Enforces duration bounds (2.0s – 300.0s) & silence guard
       │
       ├── 2. Speech Recognition (faster-whisper)
       │      • In-memory transcription & word-level timestamps
       │      • Token-level recognition probabilities
       │
       ├── 3. Fluency & Pacing Analyzer
       │      • Gross WPM (total audio) vs. Articulation Rate (active phonation)
       │      • Discrete lexicon filler identification & repetition detection
       │
       ├── 4. Pause & Phonation Engine
       │      • Word gap duration modeling (standard >=0.5s, extended >=2.0s)
       │      • Pause-to-speech ratio & timeline distribution
       │
       ├── 5. Acoustic Delivery Analyzer (Praat-Parselmouth)
       │      • Fundamental frequency ($F_0$) autocorrelation (65–400 Hz)
       │      • Intonational span in semitones & voiced fraction
       │      • Digital frame RMS energy & dynamic range (dBFS)
       │
       └── 6. Pronunciation Assessment Engine (Dual Mode Wav2Vec2 CTC)
              • Reference Mode: CTC Trellis alignment against canonical user reference targets
              • Spontaneous Mode: CTC alignment against ASR word hypotheses with uncertainty flags
              • Frame-averaged posterior confidence & unsupported phoneme reporting
              • Explicit "unavailable" status when acoustic model weights are uninitialized
```

---

## 2. Features & Known Limitations

### Features
- **Zero Cloud Leakage**: All audio processing, speech recognition, and acoustic analysis run locally on Windows CPU.
- **In-Memory Audio Processing**: Uploaded audio is processed in RAM buffers using PyAV and discarded immediately after response synthesis.
- **Interactive Audio Waveform**: In-browser amplitude extraction and scrubbing playback via Web Audio API.
- **Dual Pronunciation Analysis Modes**:
  - **Target Sentence Practice (Reference-Based)**: Forced alignment against validated user-provided sentence targets with per-phoneme acoustic evidence and alignment coverage.
  - **Spontaneous Speech Response**: Automatic candidate phoneme generation from recognized ASR hypotheses with uncertainty indicators for unaligned, unsupported, or low-probability words.
- **Multi-Dimensional Fluency**: Gross WPM, net articulation rate, vocal fillers (`um`, `uh`), discourse markers (`like`, `you know`), and immediate repetitions.
- **Scientific Acoustic Profiling**: Fundamental frequency ($F_0$) median, standard deviation, P05–P95 bounds, and semitone pitch range calculated via Praat autocorrelation.
- **Pause Distribution Timeline**: Visual timeline mapping conversational pauses against active speech phonation.

### Known Limitations & Scientific Boundaries
- **ASR Confidence vs. Pronunciation**: Whisper word probabilities quantify acoustic and language model recognition likelihood, **not** speaker pronunciation accuracy or phonetic precision.
- **Word-Gap Pauses vs. Acoustic Silence**: Pause intervals are derived from inter-word timestamp gaps from faster-whisper. They approximate conversational pacing and hesitation, but do not guarantee physical acoustic silence.
- **Acoustic Modulation vs. Confidence/Personality**: Pitch semitone range and RMS energy dynamic range measure vocal modulation and recording levels; they do **not** reliably measure candidate confidence, emotional state, or job readiness.
- **Spontaneous ASR Hypothesis Limitation**: In spontaneous mode, Whisper transcripts are treated as candidate hypotheses, not ground truth. A complete CTC alignment does not guarantee the transcript or pronunciation is correct. Low acoustic posterior evidence may stem from ASR transcription errors, accent diversity, conversational coarticulation, or background noise rather than speaker mispronunciation.
- **No CEFR / IELTS Oral Proficiency Claims**: Raw frame posteriors reflect acoustic model likelihood under the training distribution ($[0.0, 1.0]$). They are uncalibrated and MUST NOT be presented as standardized educational, clinical, or hiring readiness scores.

---

## 3. Windows Setup & Environment Instructions

### Prerequisites
- **Operating System**: Windows 10/11 x64
- **Python**: 3.12, 3.13, or 3.14 x64
- **Node.js**: v20+ LTS (Node v24 tested)
- **Audio Device**: Microphone for live recording

### 1. Clone & Virtual Environment Setup
Open PowerShell in the project directory:

```powershell
cd d:\interview_sense_ai

# (Optional) Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Backend Dependency Installation
Install required Python packages:

```powershell
pip install -r backend/requirements.txt
```

Key dependencies:
- `fastapi` & `uvicorn`: API routing and ASGI server.
- `av` (PyAV): Standalone FFmpeg bindings for in-memory audio decoding.
- `faster-whisper`: CTranslate2-accelerated Whisper speech recognition.
- `praat-parselmouth`: Python bindings for Praat acoustic phonetics.
- `numpy` & `soundfile`: Signal processing and WAV manipulation.

### 3. Frontend Dependency Installation
In a separate terminal or the same directory:

```powershell
cd d:\interview_sense_ai\frontend
npm install
```

---

## 4. Configuration & Environment Variables

A `.env` file at the root of the workspace configures runtime parameters:

```ini
# Model Selection: 'tiny.en', 'base.en', or 'small.en'
WHISPER_MODEL_SIZE=tiny.en

# Execution Target & Quantization
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4

# Server Flags
DEBUG=False
```

### Hardware & Model Requirements
| Model | Disk Footprint | RAM Usage | Approximate CPU RTF (Real-Time Factor) |
|---|---|---|---|
| `tiny.en` | ~75 MB | ~200 MB | ~0.3x – 0.8x (Fastest) |
| `base.en` | ~140 MB | ~350 MB | ~0.8x – 1.5x (Balanced) |
| `small.en` | ~480 MB | ~900 MB | ~1.5x – 3.0x (Highest Accuracy) |

*Note: On first startup, `faster-whisper` checks the local HuggingFace cache (`~/.cache/huggingface/hub`). If the requested weights are absent, they are downloaded once. The engine automatically falls back to local `tiny.en` if the primary model cannot be downloaded.*

---

## 5. Startup Commands

### Start the Backend Server
```powershell
cd d:\interview_sense_ai
python backend/run_server.py
```
- Server URL: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive OpenAPI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Endpoint: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

### Start the Frontend Dev Server
```powershell
cd d:\interview_sense_ai\frontend
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173) (Vite proxies `/api` to `127.0.0.1:8000`)

---

## 6. API Reference & Analysis Modes

### `GET /api/v1/health`
Returns backend service health and model readiness status:
```json
{
  "status": "healthy",
  "project": "InterviewSense AI",
  "version": "1.0.0",
  "models": {
    "asr": "tiny.en (loaded)",
    "acoustic": "praat-parselmouth (ready)",
    "pronunciation": "facebook/wav2vec2-lv-60-espeak-cv-ft (ready)"
  }
}
```

### `POST /api/v1/analyze`
Accepts `multipart/form-data`:
- `audio_file`: Audio recording stream (WebM, WAV, OGG, MP3, FLAC, M4A; max 25 MB; 2.0s–300.0s duration).
- `language`: Spoken language code (default: `"en"`).
- `reference_text`: *(Optional)* Target practice sentence.
  - When provided: triggers **Reference-Based Pronunciation Mode** against canonical targets.
  - When omitted / null: triggers **Spontaneous Speech Pronunciation Mode** against recognized ASR hypotheses.

#### Mode 1: Spontaneous Speech Response (No Reference Text)
```json
{
  "status": "validated",
  "metadata": {
    "duration_seconds": 3.774,
    "sample_rate": 16000,
    "channels": 1,
    "original_format": "wav",
    "rms_energy": 0.0541,
    "peak_amplitude": 0.7241
  },
  "transcript": {
    "text": "I have worked on distributed systems for three years.",
    "language": "en",
    "model_used": "tiny.en",
    "words": [
      { "word": "I", "start": 0.14, "end": 0.22, "probability": 0.985 },
      { "word": "have", "start": 0.24, "end": 0.40, "probability": 0.962 }
    ]
  },
  "fluency": {
    "word_count": 9,
    "gross_wpm": 143.1,
    "articulation_rate_wpm": 162.4,
    "speaking_duration_seconds": 3.32,
    "total_pause_duration_seconds": 0.45,
    "pause_to_speech_ratio": 0.119,
    "filler_count": 0,
    "filler_percentage": 0.0,
    "filler_words": [],
    "repetitions": []
  },
  "pauses": {
    "total_pauses": 1,
    "total_pause_duration_seconds": 0.45,
    "pauses_over_500ms": 0,
    "pauses_over_2s": 0,
    "longest_pause_seconds": 0.45,
    "average_pause_seconds": 0.45,
    "pause_to_total_ratio": 0.119,
    "pause_intervals": []
  },
  "acoustics": {
    "pitch": {
      "status": "available",
      "median_f0_hz": 138.4,
      "mean_f0_hz": 141.2,
      "std_f0_hz": 16.8,
      "p05_f0_hz": 112.0,
      "p95_f0_hz": 178.5,
      "range_semitones": 8.04,
      "voiced_fraction": 0.684,
      "pitch_floor_hz": 65.0,
      "pitch_ceiling_hz": 400.0,
      "note": "Praat autocorrelation F0 analysis."
    },
    "energy": {
      "mean_rms": 0.0541,
      "std_rms": 0.0211,
      "mean_rms_dbfs": -25.33,
      "dynamic_range_db": 19.45,
      "frame_length_ms": 25.0,
      "hop_length_ms": 10.0,
      "note": "Frame-level RMS energy."
    }
  },
  "pronunciation": {
    "status": "ready",
    "model_identifier": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "phoneme_inventory": "ARPAbet-39",
    "transcript_evaluated": "I have worked on distributed systems for three years.",
    "transcript_source": "ASR-generated (faster-whisper)",
    "reference_text": null,
    "alignment_coverage_ratio": 1.0,
    "mean_acoustic_confidence": 0.723,
    "word_diagnostics": [
      {
        "word": "distributed",
        "start": 0.88,
        "end": 1.53,
        "flagged": false,
        "diagnostic_note": null,
        "mean_acoustic_confidence": 0.725,
        "phonemes": [
          { "phoneme": "D", "start": 0.88, "end": 0.94, "duration": 0.06, "acoustic_match_confidence": 1.0, "is_supported": true, "diagnostic_note": null }
        ]
      }
    ],
    "methodology_note": "Spontaneous speech phoneme assessment using Wav2Vec2 CTC phonetic alignment against ASR-recognized word hypotheses. Acoustic match confidence reflects frame-averaged posterior probability under the acoustic model (0.0 to 1.0). Notice: Transcripts are generated by ASR and represent hypotheses rather than verified ground truth."
  }
}
```

#### Mode 2: Target Sentence Practice (With Reference Text)
- Same payload structure as above, with:
  - `reference_text`: `"I have worked on distributed systems for three years."`
  - `transcript_source`: `"User-provided reference"`
  - `word_diagnostics`: Aligned against canonical target phonemes from the validated reference text.

---

## 7. Testing & Verification

### 1. Run Complete Backend Automated Test Suite
The automated backend test suite uses deterministic mocks and synthetic audio to run entirely offline without network access or downloading weights:
```powershell
python -m pytest backend/tests -v -k "not smoke"
```
- **Tests**: 90 unit, integration, and reliability regression tests.
- **Execution Time**: ~1.5 seconds.
- **Status**: 90 passed.

### 2. Run End-to-End HTTP Endpoint Verification
Tests both reference and spontaneous analysis modes, error validation, and model safety at the HTTP level:
```powershell
python backend/tests/verify_http_endpoints.py
```
- **Status**: 5/5 verification checks passed.

### 3. Run Real-Model Smoke Tests (Optional)
Verifies live model loading and inference on Windows CPU using actual weights:

#### ASR Smoke Test (faster-whisper)
```powershell
python backend/tests/smoke_test_real_whisper.py tiny.en
```

#### Pronunciation Smoke Test: Reference-Based Mode
```powershell
python backend/tests/smoke_test_real_pronunciation.py backend/tests/genuine_speech_sample.wav "I have worked on distributed systems for three years."
```

#### Pronunciation Smoke Test: Spontaneous Mode
```powershell
python backend/tests/smoke_test_real_pronunciation.py backend/tests/genuine_speech_sample.wav --spontaneous
```

#### Synthetic Carrier Smoke Test (Negative Control)
```powershell
python backend/tests/smoke_test_real_pronunciation.py
```

### 4. Run Frontend Verification Suite
```powershell
cd d:\interview_sense_ai\frontend

# Contract & state handling tests (7 tests passed)
npm test

# Linter checks (0 warnings, 0 errors across 19 files)
npm run lint

# Production bundle compilation
npm run build
```

### 5. Manual Browser Demonstration Checklist
If testing via browser interface ([http://localhost:5173](http://localhost:5173)):
1. **Health Badges**: Verify Header shows healthy status with ASR loaded, Praat ready, and Pronunciation model status.
2. **Mode Switching**:
   - Click *Target Sentence Practice*: verify textarea appears with quick prompts and char/word counter.
   - Click *Spontaneous Interview Response*: verify textarea disappears and spontaneous explanation appears.
3. **Reference Practice Workflow**:
   - Select a prompt (e.g. *"I have worked on distributed systems for three years."*).
   - Record via mic or upload `backend/tests/genuine_speech_sample.wav`.
   - Click *Analyze Speech*.
   - Verify: Target Reference Sentence banner appears, coverage shows ~100%, and scorable phoneme chips display posterior confidence.
4. **Spontaneous Speech Workflow**:
   - Select *Spontaneous Interview Response*.
   - Record or upload audio without providing a reference sentence.
   - Click *Analyze Speech*.
   - Verify: ASR Hypothesis banner appears, transcript card displays recognized text and word timestamps, and Pronunciation card displays aligned phoneme hypotheses and uncertainty indicators.
5. **Safety & Error Handling**:
   - In reference mode, clear the textarea and click *Analyze* -> verify explicit warning: *"Target Sentence Required"*.
   - Upload very short audio (<2s) -> verify `AUDIO_TOO_SHORT` error callout.
   - Click *New Analysis* -> verify all analysis cards reset and no stale references are reused.

---

## 8. Data Privacy & Memory Safety

InterviewSense AI is engineered with strict privacy defaults:
- **No Disk Logging of Audio**: Audio uploaded or recorded in the browser is sent over HTTPS/HTTP as multipart data directly into memory. PyAV decodes the audio from an in-memory RAM stream without temporary files on disk.
- **Immediate Buffer Cleanup**: Once acoustic and lexical metrics are returned in the response, Python garbage collection releases the decoded NumPy audio buffer.
- **Frontend Object URL Revocation**: Media object URLs (`blob:http://...`) created for in-browser playback are revoked immediately when a new recording is started, discarded, or unmounted.
- **Zero Cloud Transmission**: All inference (Whisper ASR, Praat F0, Wav2Vec2 CTC) runs locally on the host machine.

---

## 9. Pronunciation Assessment: Model Specifications & Diagnostic Method (Milestones 9 & 10)

### Acoustic Model Selection
- **Model**: `facebook/wav2vec2-lv-60-espeak-cv-ft`
- **Architecture**: `Wav2Vec2ForCTC` (large lv-60 fine-tuned on Common Voice using phonetic targets)
- **License**: Apache 2.0 (commercial and open source friendly)
- **Expected Sample Rate**: 16,000 Hz mono (native match for InterviewSense AI audio pipeline)
- **Output Vocabulary**: eSpeak-IPA phonetic characters + blank token (`<pad>`)
- **Inventory Mapping**: Canonical ARPAbet-39 phonemes mapped to eSpeak-IPA via `ARPABET_TO_ESPEAK` in `backend/app/services/pronunciation/g2p.py`
- **Hardware & Memory**: ~1.2 GB model weights, requires ~2 GB system RAM on CPU; compatible with Windows 10/11 x64.
- **Lazy Loading & Guardrails**: Model weights are loaded lazily on demand and reused across requests. If weights are uninitialized or unavailable, the application gracefully returns `status: "unavailable"` without impacting ASR, fluency, pauses, or acoustic metrics.

### Dual Evaluation Methodology

| Dimension | Mode 1: Target Sentence Practice (Reference-Based) | Mode 2: Spontaneous Speech Response |
|---|---|---|
| **Target Source** | User-provided practice sentence | ASR recognized transcript (faster-whisper) |
| **Phoneme Generation** | Canonical ARPAbet via G2P with fallback | Candidate ARPAbet from recognized word tokens |
| **Epistemic Status** | Ground-truth reference target | Candidate acoustic hypothesis |
| **Alignment Method** | CTC Viterbi Trellis forced alignment | CTC Viterbi Trellis alignment against hypothesis |
| **Confidence Metric** | Frame-averaged posterior probability | Frame-averaged posterior probability |
| **Uncertainty Flags** | Omitted / unaligned / unsupported phonemes | Low ASR probability, unaligned / unsupported phonemes |
| **Coverage Calculation** | Aligned phones / expected reference phones | Aligned phones / candidate hypothesis phones |
| **Methodology Caveat** | Acoustic evidence, not accredited oral grade | Conflates ASR errors with pronunciation; not an L2 grade |

### Scientific Boundaries & Scoring Disclaimers
- **Acoustic Match vs. Proficiency**: Acoustic match confidence quantifies acoustic frame posterior probability under the CTC model ($[0.0, 1.0]$). It does **not** represent an accredited CEFR/IELTS L2 oral proficiency grade or human-calibrated score.
- **Pronunciation Variance vs. Acoustic Conditions**: Lower posterior confidence can stem from regional accent diversity, non-native phonetic variants, microphone frequency response, room reverberation, or alignment uncertainty rather than speech production errors.
- **Spontaneous ASR Hypothesis Boundary**: Spontaneous speech lacks ground-truth target phonemes. Evaluating spontaneous speech using ASR output conflates ASR decoding errors with speaker pronunciation differences; the engine explicitly surfaces diagnostic warnings and never outputs an overall "pronunciation score" or "interview readiness score".

---

## 10. Implementation Milestones

- [x] **Milestone 1**: Project skeleton, modular architecture, FastAPI `/api/v1/health` endpoint, Vite React setup, and test suite.
- [x] **Milestone 2**: In-memory audio ingestion service (`PyAV` decoding, 16kHz mono resampling, silence guard).
- [x] **Milestone 3**: Speech recognition service (`faster-whisper` singleton with word-level timestamps).
- [x] **Milestone 4**: Fluency, filler-word classification, repetition detection, and pause statistics engines.
- [x] **Milestone 5**: Acoustic analysis engine (`Praat-Parselmouth` pitch extraction, semitone range, RMS energy dynamics).
- [x] **Milestone 6**: Isolated experimental pronunciation assessment module (Feasibility report, typed `PronunciationScorer`, G2P lexicon, CTC aligner with honest diagnostics).
- [x] **Milestone 7**: Frontend recording UI, `AudioWaveform` interactive player, live mic level meter, and delivery analysis dashboard.
- [x] **Milestone 8**: End-to-end verification, calculation reliability, regression test suite, and documentation.
- [x] **Milestone 9**: Reference-based pronunciation assessment (Wav2Vec2 CTC phonetic alignment, G2P reference text validation, coverage ratio, acoustic posterior evidence, and reference practice UI).
- [x] **Milestone 9.1**: Real-model pronunciation verification (live `facebook/wav2vec2-lv-60-espeak-cv-ft` weights on Windows CPU, smoke test suite).
- [x] **Milestone 9.2**: Pronunciation evaluation, repeatability protocol, G2P mapping audit, negative control validation, and calibration boundaries.
- [x] **Milestone 10 (FINAL)**: Spontaneous-speech pronunciation assessment (candidate phoneme generation from Whisper hypotheses, CTC Trellis alignment, uncertainty indicators, dual-mode API and UI, comprehensive automated regression tests, and full project completion).

