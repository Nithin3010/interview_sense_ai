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
       └── 6. Pronunciation Assessment Interface (Experimental)
              • Strict acoustic boundary alignment interface
              • Honest "unavailable" status when dedicated L2 models are unconfigured
```

---

## 2. Features & Known Limitations

### Features
- **Zero Cloud Leakage**: All audio processing, speech recognition, and acoustic analysis run locally.
- **In-Memory Audio Processing**: Uploaded audio is processed in RAM buffers using PyAV and discarded immediately after response synthesis.
- **Interactive Audio Waveform**: In-browser amplitude extraction and scrubbing playback via Web Audio API.
- **Multi-Dimensional Fluency**: Gross WPM, net articulation rate, vocal fillers (`um`, `uh`), discourse markers (`like`, `you know`), and immediate repetitions.
- **Scientific Acoustic Profiling**: Fundamental frequency ($F_0$) median, standard deviation, P05–P95 bounds, and semitone pitch range calculated via Praat autocorrelation.
- **Pause Distribution Timeline**: Visual timeline mapping conversational pauses against active speech phonation.

### Known Limitations & Scientific Boundaries
- **ASR Confidence vs. Pronunciation**: Whisper word probabilities quantify acoustic and language model recognition likelihood, **not** speaker pronunciation accuracy or phonetic precision.
- **Word-Gap Pauses vs. Acoustic Silence**: Pause intervals are derived from inter-word timestamp gaps from faster-whisper. They approximate conversational pacing and hesitation, but do not guarantee physical acoustic silence.
- **Acoustic Modulation vs. Confidence/Personality**: Pitch semitone range and RMS energy dynamic range measure vocal modulation and recording levels; they do **not** reliably measure candidate confidence, emotional state, or job readiness.
- **Pronunciation Assessment Feasibility**: Without ground-truth reference phonemes, unscripted spontaneous speech evaluation cannot reliably separate speaker pronunciation variance from ASR decoding errors. The platform explicitly returns `status: "unavailable"` unless a calibrated phoneme alignment model is provided.

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

## 6. API Reference & Response Contract

### `GET /api/v1/health`
Returns backend health and model readiness status:
```json
{
  "status": "healthy",
  "project": "InterviewSense AI",
  "version": "1.0.0",
  "models": {
    "asr": "tiny.en (loaded)",
    "acoustic": "praat-parselmouth (ready)",
    "pronunciation": "experimental-interface (ready)"
  }
}
```

### `POST /api/v1/analyze`
Accepts `multipart/form-data`:
- `audio_file`: Audio file (WebM, WAV, OGG, MP3, FLAC, M4A; max 25 MB; 2.0s–300.0s duration).
- `language`: Spoken language code (default: `"en"`).

#### Response Schema Structure:
```json
{
  "status": "validated",
  "metadata": {
    "duration_seconds": 14.52,
    "sample_rate": 16000,
    "channels": 1,
    "original_format": "wav",
    "rms_energy": 0.05412,
    "peak_amplitude": 0.7241
  },
  "transcript": {
    "text": "My greatest strength is solving complex engineering problems under tight deadlines.",
    "language": "en",
    "model_used": "tiny.en",
    "words": [
      {
        "word": "My",
        "start": 0.12,
        "end": 0.35,
        "probability": 0.962
      }
    ]
  },
  "fluency": {
    "word_count": 11,
    "gross_wpm": 138.2,
    "articulation_rate_wpm": 154.6,
    "speaking_duration_seconds": 11.2,
    "total_pause_duration_seconds": 3.32,
    "pause_to_speech_ratio": 0.2286,
    "filler_count": 0,
    "filler_percentage": 0.0,
    "filler_words": [],
    "repetitions": []
  },
  "pauses": {
    "total_pauses": 3,
    "total_pause_duration_seconds": 3.32,
    "pauses_over_500ms": 2,
    "pauses_over_2s": 0,
    "longest_pause_seconds": 1.45,
    "average_pause_seconds": 1.11,
    "pause_to_total_ratio": 0.2286,
    "pause_intervals": [
      {
        "start": 4.12,
        "end": 5.57,
        "duration": 1.45
      }
    ],
    "note": "Pause measurements are alignment-derived estimates based on word timestamp gaps."
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
      "note": "Fundamental frequency (F0) extracted using Praat autocorrelation."
    },
    "energy": {
      "mean_rms": 0.05412,
      "std_rms": 0.0211,
      "mean_rms_dbfs": -25.33,
      "dynamic_range_db": 19.45,
      "frame_length_ms": 25.0,
      "hop_length_ms": 10.0,
      "note": "Energy computed as frame-level Root Mean Square (RMS)."
    }
  },
  "pronunciation": {
    "status": "unavailable",
    "model_identifier": "experimental-wav2vec2-ctc",
    "phoneme_inventory": "ARPAbet-39",
    "transcript_evaluated": "My greatest strength...",
    "transcript_source": "ASR-generated (faster-whisper)",
    "word_diagnostics": [],
    "methodology_note": "Pronunciation assessment is unavailable: Pretrained phoneme-level acoustic model is not initialized.",
    "error_message": null
  }
}
```

---

## 7. Testing & Verification

### 1. Run Complete Backend Test Suite
The automated backend test suite uses deterministic mocks and synthetic audio to run entirely offline:
```powershell
python -m pytest backend/tests -v
```
- **Tests**: 77 unit, integration, and reliability regression tests.
- **Execution Time**: ~1.0 second.

### 2. Run Real Model Smoke Tests
Tests live model loading and inference on Windows CPU using actual weights:

#### ASR Smoke Test (faster-whisper)
```powershell
python backend/tests/smoke_test_real_whisper.py tiny.en
```
Outputs load time, inference latency, Real-Time Factor (RTF), and transcription text.

#### Pronunciation Smoke Test (Wav2Vec2 CTC Phonetic Model)
```powershell
# Synthetic carrier smoke test:
python backend/tests/smoke_test_real_pronunciation.py

# Live speech smoke test with reference sentence:
python backend/tests/smoke_test_real_pronunciation.py path/to/sample.wav "I have worked on distributed systems for three years."
```
Verifies Wav2Vec2 CTC phonetic inference, frame emission generation, and Viterbi Trellis alignment without modifying the automated test suite.

### 3. Run Frontend Verification Suite
```powershell
cd d:\interview_sense_ai\frontend

# Contract & state handling tests (6 tests passed)
npm test

# Linter checks (0 warnings, 0 errors)
npm run lint

# Production bundle compilation
npm run build
```

---

## 8. Data Privacy & Memory Safety

InterviewSense AI is engineered with strict privacy defaults:
- **No Disk Logging of Audio**: Audio uploaded or recorded in the browser is sent over HTTPS/HTTP as multipart data directly into memory. PyAV decodes the audio from an in-memory stream without temporary files on disk.
- **Immediate Cleanup**: Once the acoustic and lexical metrics are returned in the response, Python garbage collection releases the decoded NumPy audio buffer.
- **Frontend URL Revocation**: Media object URLs (`blob:http://...`) created for in-browser playback are revoked immediately when a new recording is started, discarded, or unmounted.

---

## 9. Pronunciation Assessment: Model Specifications & Evaluation Protocol (Milestone 9)

### Acoustic Model Selection
- **Model**: `facebook/wav2vec2-lv-60-espeak-cv-ft`
- **Architecture**: `Wav2Vec2ForCTC` (large lv-60 fine-tuned on Common Voice using phonetic targets)
- **License**: Apache 2.0 (commercial and open source friendly)
- **Expected Sample Rate**: 16,000 Hz mono (native match for InterviewSense AI audio pipeline)
- **Output Vocabulary**: eSpeak-IPA phonetic characters + blank token (`<pad>`)
- **Inventory Mapping**: Canonical ARPAbet-39 phonemes mapped to eSpeak-IPA via `ARPABET_TO_ESPEAK` in `backend/app/services/pronunciation/g2p.py`
- **Hardware & Memory**: ~1.2 GB model weights, requires ~2 GB system RAM on CPU; compatible with Windows 10/11 x64.
- **Lazy Loading & Guardrails**: Model weights are loaded lazily on demand and reused across requests. If weights are not downloaded or unavailable, the application gracefully returns `status: "unavailable"` without crashing.

### Metric Definitions
1. **Target Phoneme Sequence**: Expected canonical ARPAbet targets generated via G2P dictionary lookup with rule-based fallback.
2. **Aligned Intervals `[start, end]`**: Boundary timestamps derived via Viterbi CTC Trellis alignment against frame-level log-emissions.
3. **Phonetic Acoustic Evidence**: Frame-averaged posterior probability $P(y_k \mid x) \in [0.0, 1.0]$ over aligned frames.
4. **Alignment Coverage Ratio**: Proportion of expected reference phonemes successfully aligned ($N_{\text{aligned}} / N_{\text{expected}}$).
5. **Mean Acoustic Confidence**: Unweighted average posterior confidence across all scorable, aligned phonemes.

### Scientific Boundaries & Scoring Disclaimers
- **Acoustic Match vs. Proficiency**: Acoustic match confidence quantifies acoustic frame posterior probability under the CTC model. It does **not** represent an accredited CEFR/IELTS L2 oral proficiency grade or human-calibrated score.
- **Pronunciation Variance vs. Acoustic Conditions**: Lower posterior confidence can stem from regional accent diversity, non-native phonetic variants, microphone frequency response, room reverberation, or alignment uncertainty rather than speech production errors.
- **Spontaneous Speech Boundary**: Spontaneous speech lacks ground-truth target phonemes. Evaluating spontaneous speech using ASR output conflates ASR decoding errors with speaker pronunciation differences; the engine explicitly returns `status: "unavailable"` for spontaneous speech.

### Evaluation Protocol
Before making accuracy claims regarding pronunciation evaluation:
1. **Dataset**: Collect recordings of 5–10 reference sentences from consenting speakers with diverse accents (native and non-native L2).
2. **Conditions**: Test both canonical pronunciations and intentional phonetic substitutions/omissions.
3. **Manual Review**: Compare model-detected alignment boundaries and frame posterior confidence against expert phonetic transcripts.
4. **Calibration**: Validate that low confidence correlates with perceptible phonetic discrepancies rather than background noise before reporting proficiency diagnostics.

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
