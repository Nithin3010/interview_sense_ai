import React, { useState, useEffect, useCallback } from 'react';
import {
  Mic,
  UploadCloud,
  Loader2,
  AlertCircle,
  Sparkles,
  CheckCircle2,
  ShieldCheck,
  Target,
  MessageSquare,
} from 'lucide-react';
import type { HealthResponse } from './types/analysis';
import { Header } from './components/layout/Header';
import { AudioRecorder } from './components/audio/AudioRecorder';
import { AudioUploader } from './components/audio/AudioUploader';
import { Dashboard } from './components/dashboard/Dashboard';
import { useSpeechAnalysis } from './hooks/useSpeechAnalysis';

const SAMPLE_REFERENCE_PROMPTS = [
  "I have worked on distributed systems for three years.",
  "My greatest strength is clear communication and problem solving.",
  "Thank you for the opportunity to interview with your team today.",
  "In my previous role, I led a project to improve system reliability.",
];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('record');
  const [practiceMode, setPracticeMode] = useState<'reference' | 'spontaneous'>('reference');
  const [referenceText, setReferenceText] = useState<string>(SAMPLE_REFERENCE_PROMPTS[0]);

  const {
    isAnalyzing,
    analysisResult,
    error: analysisError,
    errorCode,
    analyzeAudio,
    resetAnalysis,
  } = useSpeechAnalysis();

  const fetchHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const response = await fetch('/api/v1/health');
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
      }
      const data: HealthResponse = await response.json();
      setHealth(data);
    } catch (err: any) {
      setHealthError(err.message || 'Failed to connect to backend server');
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const initialHealthCheck = async () => {
      try {
        const response = await fetch('/api/v1/health');
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }
        const data: HealthResponse = await response.json();
        if (!ignore) {
          setHealth(data);
          setHealthError(null);
        }
      } catch (err: any) {
        if (!ignore) {
          setHealthError(err.message || 'Failed to connect to backend server');
        }
      } finally {
        if (!ignore) {
          setHealthLoading(false);
        }
      }
    };

    initialHealthCheck();

    return () => {
      ignore = true;
    };
  }, []);

  const [referenceWarning, setReferenceWarning] = useState<string | null>(null);

  const handleNewAnalysis = () => {
    resetAnalysis();
    setReferenceWarning(null);
    // Explicitly reset reference sentence so old references are never silently reused for new audio
    setReferenceText('');
  };

  const handleAudioSubmit = async (audioSource: File | Blob, filename: string) => {
    if (practiceMode === 'reference' && referenceText.trim().length === 0) {
      setReferenceWarning(
        'Please enter a target reference sentence or select a quick prompt below before analyzing in reference mode.'
      );
      return;
    }
    setReferenceWarning(null);
    const effectiveRef = practiceMode === 'reference' ? referenceText.trim() : undefined;
    await analyzeAudio(audioSource, filename, effectiveRef);
  };

  const wordCount = referenceText.trim() ? referenceText.trim().split(/\s+/).length : 0;
  const charCount = referenceText.length;

  return (
    <div className="app-container">
      {/* 1. Header with Backend Health & Model Badges */}
      <Header
        health={health}
        loading={healthLoading}
        error={healthError}
        onRefresh={fetchHealth}
      />

      <main style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* If an analysis result is present, display the complete delivery analysis dashboard */}
        {analysisResult ? (
          <Dashboard data={analysisResult} onNewAnalysis={handleNewAnalysis} />
        ) : isAnalyzing ? (
          /* 2. Loading State with Multi-Step Progress Indicator */
          <div className="glass-card">
            <div className="analysis-loading-box">
              <div className="analysis-spinner-wrap">
                <Loader2 size={36} className="spin" />
              </div>
              <h2 className="loading-title">Analyzing Speech Delivery</h2>
              <div className="loading-steps-list">
                <span>1. Decoding in-memory 16kHz mono audio via PyAV...</span>
                <span>2. Transcribing with faster-whisper ASR & extracting word timestamps...</span>
                <span>3. Calculating speech pacing, filler words, and pause metrics...</span>
                <span>4. Extracting Praat-Parselmouth acoustic pitch (F0) & volume dynamics...</span>
                <span>5. Aligning target phoneme diagnostics via Wav2Vec2 CTC acoustic model...</span>
              </div>
            </div>
          </div>
        ) : (
          /* 3. Audio Input Area (Standby / Primary Action) */
          <>
            <div className="glass-card">
              <div className="card-header-row">
                <div>
                  <h2 className="card-title">
                    <Sparkles size={20} color="var(--accent-primary)" />
                    Speech Delivery & Pronunciation Practice
                  </h2>
                  <p className="card-desc">
                    Practice reading target interview sentences for phoneme-level pronunciation alignment, or record spontaneous speech.
                  </p>
                </div>

                {/* Tab Switcher: Mic vs Upload */}
                <div className="tabs-nav">
                  <button
                    type="button"
                    className={`tab-btn ${activeTab === 'record' ? 'active' : ''}`}
                    onClick={() => setActiveTab('record')}
                  >
                    <Mic size={16} />
                    <span>Microphone</span>
                  </button>
                  <button
                    type="button"
                    className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                    onClick={() => setActiveTab('upload')}
                  >
                    <UploadCloud size={16} />
                    <span>Upload File</span>
                  </button>
                </div>
              </div>

              {/* Practice Mode Selector */}
              <div className="practice-mode-selector">
                <button
                  type="button"
                  className={`mode-pill-btn ${practiceMode === 'reference' ? 'active' : ''}`}
                  onClick={() => setPracticeMode('reference')}
                >
                  <Target size={16} />
                  <span>Target Sentence Practice (Reference-Based Pronunciation)</span>
                </button>
                <button
                  type="button"
                  className={`mode-pill-btn ${practiceMode === 'spontaneous' ? 'active' : ''}`}
                  onClick={() => {
                    setPracticeMode('spontaneous');
                    setReferenceWarning(null);
                  }}
                >
                  <MessageSquare size={16} />
                  <span>Spontaneous Interview Response</span>
                </button>
              </div>

              {/* Reference Sentence Input Box (when in reference mode) */}
              {practiceMode === 'reference' && (
                <div className="reference-input-box">
                  <div className="reference-input-header">
                    <label htmlFor="reference-input" className="subcard-title" style={{ fontSize: '0.85rem' }}>
                      Target Reference Sentence:
                    </label>
                    <span className="char-word-counter">
                      {wordCount} words • {charCount}/500 chars
                    </span>
                  </div>

                  <textarea
                    id="reference-input"
                    className="reference-textarea"
                    placeholder="Enter or select a target sentence to practice reading aloud..."
                    value={referenceText}
                    onChange={(e) => {
                      setReferenceText(e.target.value);
                      setReferenceWarning(null);
                    }}
                    maxLength={500}
                  />

                  {/* Quick Preset Sentences */}
                  <div className="prompt-chips-row">
                    <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>Quick prompts:</span>
                    {SAMPLE_REFERENCE_PROMPTS.map((prompt, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className="prompt-chip"
                        onClick={() => {
                          setReferenceText(prompt);
                          setReferenceWarning(null);
                        }}
                        title={`Load prompt: "${prompt}"`}
                      >
                        {prompt.length > 42 ? `${prompt.slice(0, 40)}...` : prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Reference Mode Validation Alert */}
              {referenceWarning && (
                <div className="error-callout" style={{ marginTop: '0.75rem' }}>
                  <AlertCircle size={18} />
                  <div>
                    <strong>Target Sentence Required:</strong>
                    <p style={{ marginTop: '0.15rem' }}>{referenceWarning}</p>
                  </div>
                </div>
              )}

              {/* Active Input Component */}
              {activeTab === 'record' ? (
                <AudioRecorder
                  onAnalyze={handleAudioSubmit}
                  isAnalyzing={isAnalyzing}
                />
              ) : (
                <AudioUploader
                  onAnalyze={handleAudioSubmit}
                  isAnalyzing={isAnalyzing}
                />
              )}

              {/* Error Callout */}
              {analysisError && (
                <div className="error-callout" style={{ marginTop: '1.25rem' }}>
                  <AlertCircle size={20} />
                  <div>
                    <strong>Analysis Failed {errorCode ? `(${errorCode})` : ''}:</strong>
                    <p style={{ marginTop: '0.2rem' }}>{analysisError}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Local Engine Capabilities & Safeguards Overview */}
            <div className="glass-card">
              <h2 className="card-title">
                <CheckCircle2 size={20} color="var(--accent-emerald)" />
                Engine Capabilities & Evaluation Safeguards
              </h2>
              <p className="card-desc">
                InterviewSense AI processes speech entirely in-memory using local open models without cloud transmission or persistent recording storage.
              </p>

              <div className="grid-3" style={{ marginTop: '1rem' }}>
                <div className="milestone-item active">
                  <span className="milestone-badge badge-current">Pronunciation CTC</span>
                  <span className="milestone-name">Wav2Vec2 Phonetic Alignment</span>
                  <span className="milestone-desc">
                    Reference-based CTC Trellis alignment, ARPAbet-to-eSpeak phoneme mapping, and frame posterior confidence.
                  </span>
                </div>
                <div className="milestone-item active">
                  <span className="milestone-badge badge-current">Temporal & ASR</span>
                  <span className="milestone-name">Whisper Pacing & Lexicon</span>
                  <span className="milestone-desc">
                    Word-level timestamps, gross WPM, articulation rate, filler classification, and token probabilities.
                  </span>
                </div>
                <div className="milestone-item active">
                  <span className="milestone-badge badge-current">Acoustics & Ethics</span>
                  <span className="milestone-name">Praat F0 & Scientific Bounds</span>
                  <span className="milestone-desc">
                    Fundamental frequency autocorrelation, volume dynamics, and honest disclaimers with no fabricated L2 grades.
                  </span>
                </div>
              </div>

              <div className="methodology-note-banner" style={{ marginTop: '1.25rem' }}>
                <ShieldCheck size={18} color="var(--accent-emerald)" />
                <span>
                  <strong>Data Privacy Notice:</strong> Audio inputs are held transiently in memory for signal processing and discarded immediately. No audio files or transcripts are written to persistent storage.
                </span>
              </div>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          InterviewSense AI • Milestone 9 Reference-Based Pronunciation Assessment • FastAPI, Wav2Vec2 CTC, faster-whisper, Praat & React
        </p>
      </footer>
    </div>
  );
};

export default App;
