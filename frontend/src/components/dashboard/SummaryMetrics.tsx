import React from 'react';
import { Gauge, Zap, AlertTriangle, Repeat, PauseCircle, Music, Volume2 } from 'lucide-react';
import type { AnalysisResponse } from '../../types/analysis';

interface SummaryMetricsProps {
  data: AnalysisResponse;
}

export const SummaryMetrics: React.FC<SummaryMetricsProps> = ({ data }) => {
  const { fluency, pauses, acoustics } = data;
  const pitch = acoustics.pitch;
  const energy = acoustics.energy;

  return (
    <div className="metrics-grid">
      {/* Gross Speaking Rate */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Speaking Rate (Gross)</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--accent-primary)' }}>
            <Gauge size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{fluency.gross_wpm.toFixed(0)}</span>
          <span className="metric-unit">WPM</span>
        </div>
        <p className="metric-footer">
          Total words recognized divided by full audio duration ({data.metadata.duration_seconds.toFixed(1)}s).
        </p>
      </div>

      {/* Articulation Rate */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Articulation Rate</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>
            <Zap size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{fluency.articulation_rate_wpm.toFixed(0)}</span>
          <span className="metric-unit">WPM</span>
        </div>
        <p className="metric-footer">
          Active speaking rate excluding detected pauses ({fluency.speaking_duration_seconds.toFixed(1)}s active).
        </p>
      </div>

      {/* Filler Words */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Filler Frequency</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(245, 158, 11, 0.15)', color: 'var(--accent-amber)' }}>
            <AlertTriangle size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{fluency.filler_percentage.toFixed(1)}%</span>
          <span className="metric-unit">({fluency.filler_count} fillers)</span>
        </div>
        <p className="metric-footer">
          {fluency.filler_count === 0 ? 'No filler words detected' : `${fluency.filler_count} fillers out of ${fluency.word_count} total words`}.
        </p>
      </div>

      {/* Immediate Repetitions */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Repetitions</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(244, 63, 94, 0.15)', color: 'var(--accent-rose)' }}>
            <Repeat size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{fluency.repetitions.length}</span>
          <span className="metric-unit">instances</span>
        </div>
        <p className="metric-footer">
          Immediate repeated words or short phrase sequences (lengths 2–4).
        </p>
      </div>

      {/* Pauses */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Long Pauses (&ge; 2.0s)</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-emerald)' }}>
            <PauseCircle size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{pauses.pauses_over_2s}</span>
          <span className="metric-unit">of {pauses.total_pauses} pauses</span>
        </div>
        <p className="metric-footer">
          {pauses.longest_pause_seconds.toFixed(1)}s longest pause • {(pauses.pause_to_total_ratio * 100).toFixed(0)}% pause time.
        </p>
      </div>

      {/* Pitch Range */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Pitch Range (F0)</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc' }}>
            <Music size={18} />
          </div>
        </div>
        <div className="metric-body">
          {pitch.status === 'available' && pitch.range_semitones != null ? (
            <>
              <span className="metric-value">{pitch.range_semitones.toFixed(1)}</span>
              <span className="metric-unit">semitones</span>
            </>
          ) : (
            <span className="metric-value muted-val">Unavailable</span>
          )}
        </div>
        <p className="metric-footer">
          {pitch.status === 'available' && pitch.median_f0_hz != null
            ? `Median F0: ${pitch.median_f0_hz.toFixed(0)} Hz • span between P05 & P95`
            : 'Voiced fundamental frequency could not be extracted.'}
        </p>
      </div>

      {/* Volume Dynamic Range */}
      <div className="metric-card">
        <div className="metric-header">
          <span className="metric-title">Volume Dynamics</span>
          <div className="metric-icon-wrap" style={{ background: 'rgba(14, 165, 233, 0.15)', color: '#38bdf8' }}>
            <Volume2 size={18} />
          </div>
        </div>
        <div className="metric-body">
          <span className="metric-value">{energy.dynamic_range_db.toFixed(1)}</span>
          <span className="metric-unit">dB dynamic range</span>
        </div>
        <p className="metric-footer">
          Mean RMS: {energy.mean_rms_dbfs.toFixed(1)} dBFS • digital decibels relative to full scale.
        </p>
      </div>
    </div>
  );
};
