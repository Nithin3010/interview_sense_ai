import React from 'react';
import { RotateCcw, CheckCircle2 } from 'lucide-react';
import type { AnalysisResponse } from '../../types/analysis';
import { SummaryMetrics } from './SummaryMetrics';
import { TranscriptCard } from './TranscriptCard';
import { FluencyCard } from './FluencyCard';
import { PauseCard } from './PauseCard';
import { AcousticCard } from './AcousticCard';
import { PronunciationCard } from './PronunciationCard';

interface DashboardProps {
  data: AnalysisResponse;
  onNewAnalysis: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ data, onNewAnalysis }) => {
  const { metadata, transcript, fluency, pauses, acoustics, pronunciation } = data;

  return (
    <div className="dashboard-container">
      {/* Overview Top Bar */}
      <div className="dashboard-top-bar">
        <div className="top-bar-left">
          <div className="success-badge">
            <CheckCircle2 size={16} />
            <span>Analysis Complete</span>
          </div>

          <div className="audio-meta-pills">
            <span className="audio-meta-pill">
              Duration: <strong>{metadata.duration_seconds.toFixed(2)}s</strong>
            </span>
            <span className="audio-meta-pill">
              Sample Rate: <strong>{metadata.sample_rate} Hz (Mono)</strong>
            </span>
            <span className="audio-meta-pill">
              Peak Amplitude: <strong>{metadata.peak_amplitude.toFixed(2)}</strong>
            </span>
            {metadata.original_format && (
              <span className="audio-meta-pill">
                Original Container: <strong>{metadata.original_format}</strong>
              </span>
            )}
          </div>
        </div>

        <button
          type="button"
          onClick={onNewAnalysis}
          className="action-btn secondary-btn"
          title="Analyze another speech sample"
        >
          <RotateCcw size={16} />
          <span>New Analysis</span>
        </button>
      </div>

      {/* 1. High-level metric summary cards */}
      <SummaryMetrics data={data} />

      {/* 2. Full Transcript & Token probabilities */}
      <TranscriptCard
        transcript={transcript}
        fillerWords={fluency.filler_words}
        repetitions={fluency.repetitions}
      />

      {/* 3. Fluency & Pacing Analysis */}
      <FluencyCard fluency={fluency} />

      {/* 4. Pause & Gap Breakdown */}
      <PauseCard pauses={pauses} totalDurationSeconds={metadata.duration_seconds} />

      {/* 5. Acoustic Pitch & Volume Dynamics */}
      <AcousticCard acoustics={acoustics} />

      {/* 6. Pronunciation Assessment & Methodology */}
      <PronunciationCard pronunciation={pronunciation} />
    </div>
  );
};
