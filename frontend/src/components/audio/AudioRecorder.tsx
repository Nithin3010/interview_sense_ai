import React from 'react';
import { Mic, Square, Trash2, ArrowRight, AlertCircle, Loader2, Volume2 } from 'lucide-react';
import { useAudioRecorder } from '../../hooks/useAudioRecorder';
import { AudioWaveform } from './AudioWaveform';

interface AudioRecorderProps {
  onAnalyze: (blob: Blob, filename: string) => void;
  isAnalyzing: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAnalyze, isAnalyzing }) => {
  const {
    isRecording,
    recordingDuration,
    audioBlob,
    audioUrl,
    mimeType,
    audioLevel,
    error,
    startRecording,
    stopRecording,
    clearRecording,
  } = useAudioRecorder();

  const formatDuration = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  const handleAnalyzeClick = () => {
    if (!audioBlob || isAnalyzing) return;
    const ext = mimeType?.includes('ogg')
      ? 'ogg'
      : mimeType?.includes('mp4')
      ? 'm4a'
      : mimeType?.includes('wav')
      ? 'wav'
      : 'webm';
    onAnalyze(audioBlob, `interview_speech_sample.${ext}`);
  };

  const isDurationValid = recordingDuration >= 2.0 && recordingDuration <= 300.0;

  return (
    <div className="recorder-container">
      {/* 1. Active Recording View */}
      {isRecording ? (
        <div className="recorder-active-view">
          <div className="recording-pulse-wrapper">
            <span className="recording-pulse-ring" />
            <span
              className="recording-pulse-core"
              style={{
                transform: `scale(${1 + (audioLevel / 100) * 0.35})`,
              }}
            />
            <Mic size={32} color="#fff" />
          </div>

          <div className="recording-timer">{formatDuration(recordingDuration)}</div>

          {/* Live Audio Level Meter */}
          <div className="live-meter-container" title={`Microphone Input Level: ${audioLevel}%`}>
            <Volume2 size={14} color="var(--accent-rose)" />
            <div className="live-meter-track">
              <div
                className="live-meter-bar"
                style={{ width: `${Math.max(4, audioLevel)}%` }}
              />
            </div>
            <span className="live-meter-val">{audioLevel}%</span>
          </div>

          <p className="recording-guidance">
            {recordingDuration < 2.0
              ? 'Keep speaking (minimum 2.0 seconds required)...'
              : recordingDuration >= 290
              ? 'Approaching maximum duration limit of 5 minutes (300s)...'
              : 'Recording in progress. Click Stop when finished.'}
          </p>

          <button
            type="button"
            onClick={stopRecording}
            className="action-btn danger-btn"
            style={{ marginTop: '0.75rem' }}
          >
            <Square size={18} fill="currentColor" />
            <span>Stop Recording</span>
          </button>
        </div>
      ) : audioUrl ? (
        /* 2. Review & Playback View */
        <div className="recorder-review-view">
          <div className="review-header">
            <h4>Recording Completed ({recordingDuration.toFixed(1)}s)</h4>
            <span className="format-badge">{mimeType?.split(';')[0] || 'audio/webm'}</span>
          </div>

          {/* Waveform audio player */}
          <AudioWaveform
            src={audioUrl}
            title="Recorded Speech Sample"
            initialDuration={recordingDuration}
          />

          {recordingDuration < 2.0 && (
            <div className="warning-callout">
              <AlertCircle size={16} />
              <span>
                Recording is {recordingDuration.toFixed(1)}s (minimum 2.0s required). Please record a longer sample for acoustic analysis.
              </span>
            </div>
          )}

          {recordingDuration > 300.0 && (
            <div className="warning-callout">
              <AlertCircle size={16} />
              <span>
                Recording exceeds 300.0s (5 minutes). Please keep responses under 5 minutes.
              </span>
            </div>
          )}

          <div className="recorder-actions">
            <button
              type="button"
              onClick={clearRecording}
              disabled={isAnalyzing}
              className="action-btn secondary-btn"
            >
              <Trash2 size={16} />
              <span>Re-record</span>
            </button>

            <button
              type="button"
              onClick={handleAnalyzeClick}
              disabled={!isDurationValid || isAnalyzing}
              className="action-btn primary-btn"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={18} className="spin" />
                  <span>Analyzing Audio...</span>
                </>
              ) : (
                <>
                  <span>Analyze Delivery</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </div>
        </div>
      ) : (
        /* 3. Standby / Start View */
        <div className="recorder-standby-view">
          <div className="standby-icon-wrapper">
            <Mic size={36} color="var(--accent-primary)" />
          </div>
          <h3 className="standby-title">Live Microphone Recording</h3>
          <p className="standby-desc">
            Record an unscripted interview response directly in your browser. Microphone access will be requested when you click Start.
          </p>

          <button
            type="button"
            onClick={startRecording}
            disabled={isAnalyzing}
            className="action-btn primary-btn large-btn"
          >
            <Mic size={20} />
            <span>Start Recording</span>
          </button>

          <span className="privacy-note">
            Microphone access is requested strictly upon user action. Audio is processed entirely in memory by local models and never written to disk.
          </span>
        </div>
      )}

      {/* Structured Error Callout */}
      {error && (
        <div className="error-callout" style={{ marginTop: '1rem' }}>
          <AlertCircle size={18} />
          <div>
            <strong>Recording Error:</strong>
            <p>{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};
