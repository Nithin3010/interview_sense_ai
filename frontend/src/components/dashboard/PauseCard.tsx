import React from 'react';
import { PauseCircle, Info, Clock } from 'lucide-react';
import type { PauseStatistics } from '../../types/analysis';

interface PauseCardProps {
  pauses: PauseStatistics;
  totalDurationSeconds: number;
}

export const PauseCard: React.FC<PauseCardProps> = ({ pauses, totalDurationSeconds }) => {
  const safeTotalDuration = Math.max(0.1, totalDurationSeconds);

  return (
    <div className="glass-card">
      <div className="card-header-row">
        <div>
          <h2 className="card-title">
            <PauseCircle size={20} color="var(--accent-emerald)" />
            Pause & Phonation Statistics
          </h2>
          <p className="card-desc">
            Gap analysis derived between word timestamps with standard (&ge;0.5s) and extended (&ge;2.0s) thresholds
          </p>
        </div>
      </div>

      {/* Pause Metrics Grid */}
      <div className="pacing-stats-grid">
        <div className="pacing-stat-box">
          <span className="pacing-label">Total Pauses Detected</span>
          <span className="pacing-value">{pauses.total_pauses}</span>
          <span className="pacing-subtext">Gaps &ge; 0.50 seconds</span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Pauses &ge; 2.0s (Extended)</span>
          <span
            className="pacing-value"
            style={{
              color: pauses.pauses_over_2s > 0 ? 'var(--accent-amber)' : 'var(--accent-emerald)',
            }}
          >
            {pauses.pauses_over_2s}
          </span>
          <span className="pacing-subtext">Extended thinking or transition pauses</span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Longest Pause</span>
          <span className="pacing-value">
            {pauses.longest_pause_seconds.toFixed(2)} <small>s</small>
          </span>
          <span className="pacing-subtext">Maximum single gap duration</span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Average Pause Duration</span>
          <span className="pacing-value">
            {pauses.average_pause_seconds.toFixed(2)} <small>s</small>
          </span>
          <span className="pacing-subtext">Mean length of detected pauses</span>
        </div>
      </div>

      {/* Visual Pause Distribution Timeline across total duration */}
      <div className="pause-timeline-container" style={{ marginTop: '1.5rem' }}>
        <div className="subcard-header" style={{ marginBottom: '0.65rem' }}>
          <h4 className="subcard-title">
            Speech & Pause Timeline (0.0s – {safeTotalDuration.toFixed(1)}s)
          </h4>
          <span className="badge-emerald" style={{ fontSize: '0.75rem' }}>
            {(pauses.pause_to_total_ratio * 100).toFixed(0)}% pause time
          </span>
        </div>

        <div className="pause-timeline-track" title="Visual timeline of active speech vs detected pauses">
          {pauses.pause_intervals.map((p, idx) => {
            const leftPct = Math.max(0, Math.min(100, (p.start / safeTotalDuration) * 100));
            const widthPct = Math.max(
              0.8,
              Math.min(100 - leftPct, (p.duration / safeTotalDuration) * 100)
            );
            const isExtended = p.duration >= 2.0;

            return (
              <div
                key={`tl-pause-${idx}`}
                className={`timeline-pause-segment ${isExtended ? 'extended' : ''}`}
                style={{
                  left: `${leftPct}%`,
                  width: `${widthPct}%`,
                }}
                title={`Pause #${idx + 1}: ${p.start.toFixed(2)}s – ${p.end.toFixed(2)}s (+${p.duration.toFixed(2)}s)`}
              />
            );
          })}
        </div>

        <div className="timeline-labels-row">
          <span>0.0s</span>
          <span>{(safeTotalDuration * 0.5).toFixed(1)}s</span>
          <span>{safeTotalDuration.toFixed(1)}s</span>
        </div>
      </div>

      {/* Pause Distribution Timeline / List of intervals */}
      <div style={{ marginTop: '1.25rem' }}>
        <h4 className="subcard-title" style={{ marginBottom: '0.75rem' }}>
          Pause Intervals ({pauses.pause_intervals.length})
        </h4>

        {pauses.pause_intervals.length === 0 ? (
          <div className="empty-subcard">
            <Clock size={16} color="var(--accent-cyan)" />
            <span>No pauses exceeding 0.50 seconds were detected in this audio.</span>
          </div>
        ) : (
          <div className="pause-interval-chips">
            {pauses.pause_intervals.map((p, idx) => {
              const isLong = p.duration >= 2.0;
              return (
                <div
                  key={`pause-${p.start}-${idx}`}
                  className={`pause-chip ${isLong ? 'long-pause' : ''}`}
                  title={`Pause: ${p.start.toFixed(2)}s to ${p.end.toFixed(2)}s`}
                >
                  <span className="pause-chip-time">
                    {p.start.toFixed(1)}s – {p.end.toFixed(1)}s
                  </span>
                  <span className="pause-chip-dur">
                    +{p.duration.toFixed(2)}s {isLong && '⚠️ Extended'}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Methodological Note */}
      <div className="methodology-note-banner" style={{ marginTop: '1.25rem' }}>
        <Info size={16} />
        <span>
          <strong>Methodology Note:</strong> {pauses.note}
        </span>
      </div>
    </div>
  );
};
