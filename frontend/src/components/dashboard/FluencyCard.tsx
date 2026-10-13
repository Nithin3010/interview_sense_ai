import React from 'react';
import { Activity, AlertTriangle, Repeat, CheckCircle } from 'lucide-react';
import type { FluencyMetrics } from '../../types/analysis';

interface FluencyCardProps {
  fluency: FluencyMetrics;
}

export const FluencyCard: React.FC<FluencyCardProps> = ({ fluency }) => {
  return (
    <div className="glass-card">
      <div className="card-header-row">
        <div>
          <h2 className="card-title">
            <Activity size={20} color="var(--accent-cyan)" />
            Fluency & Pacing Analysis
          </h2>
          <p className="card-desc">
            Speaking rate metrics, lexical disfluencies, and repetition patterns
          </p>
        </div>
      </div>

      {/* Pacing Comparison Grid */}
      <div className="pacing-stats-grid">
        <div className="pacing-stat-box">
          <span className="pacing-label">Gross Speaking Rate</span>
          <span className="pacing-value">{fluency.gross_wpm.toFixed(1)} <small>WPM</small></span>
          <span className="pacing-subtext">Includes all pause durations</span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Articulation Rate</span>
          <span className="pacing-value" style={{ color: 'var(--accent-cyan)' }}>
            {fluency.articulation_rate_wpm.toFixed(1)} <small>WPM</small>
          </span>
          <span className="pacing-subtext">Calculated over active speech only</span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Active Phonation Time</span>
          <span className="pacing-value">{fluency.speaking_duration_seconds.toFixed(1)} <small>s</small></span>
          <span className="pacing-subtext">
            {((1 - fluency.pause_to_speech_ratio) * 100).toFixed(0)}% of total audio
          </span>
        </div>

        <div className="pacing-stat-box">
          <span className="pacing-label">Total Pause Duration</span>
          <span className="pacing-value">{fluency.total_pause_duration_seconds.toFixed(1)} <small>s</small></span>
          <span className="pacing-subtext">
            {(fluency.pause_to_speech_ratio * 100).toFixed(0)}% of total audio
          </span>
        </div>
      </div>

      {/* Fillers & Repetitions Split Grid */}
      <div className="disfluency-grid" style={{ marginTop: '1.5rem' }}>
        {/* Fillers Column */}
        <div className="subcard-box">
          <div className="subcard-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertTriangle size={16} color="var(--accent-amber)" />
              <h3 className="subcard-title">Detected Fillers ({fluency.filler_count})</h3>
            </div>
            <span className="badge-amber">{fluency.filler_percentage.toFixed(1)}% of speech</span>
          </div>

          {fluency.filler_words.length === 0 ? (
            <div className="empty-subcard">
              <CheckCircle size={18} color="var(--accent-emerald)" />
              <span>No filler words identified in speech.</span>
            </div>
          ) : (
            <div className="disfluency-list">
              {fluency.filler_words.map((f, i) => (
                <div key={`${f.word}-${f.start}-${i}`} className="disfluency-row">
                  <div className="disfluency-name">
                    <strong>&ldquo;{f.word}&rdquo;</strong>
                    <span className="category-chip">{f.category.replace('_', ' ')}</span>
                  </div>
                  <span className="disfluency-time">
                    {f.start.toFixed(2)}s – {f.end.toFixed(2)}s
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Repetitions Column */}
        <div className="subcard-box">
          <div className="subcard-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Repeat size={16} color="var(--accent-rose)" />
              <h3 className="subcard-title">Immediate Repetitions ({fluency.repetitions.length})</h3>
            </div>
            <span className="badge-rose">{fluency.repetitions.length} occurrences</span>
          </div>

          {fluency.repetitions.length === 0 ? (
            <div className="empty-subcard">
              <CheckCircle size={18} color="var(--accent-emerald)" />
              <span>No immediate word or phrase repetitions detected.</span>
            </div>
          ) : (
            <div className="disfluency-list">
              {fluency.repetitions.map((r, i) => (
                <div key={`${r.repeated_text}-${r.start}-${i}`} className="disfluency-row">
                  <div className="disfluency-name">
                    <strong>&ldquo;{r.repeated_text}&rdquo;</strong>
                    <span className="category-chip">{r.repetition_type.replace('_', ' ')} (x{r.count})</span>
                  </div>
                  <span className="disfluency-time">
                    {r.start.toFixed(2)}s – {r.end.toFixed(2)}s
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
