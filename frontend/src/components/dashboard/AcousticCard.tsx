import React from 'react';
import { Volume2, Music, Info, AlertCircle } from 'lucide-react';
import type { AcousticsMetrics } from '../../types/analysis';

interface AcousticCardProps {
  acoustics: AcousticsMetrics;
}

export const AcousticCard: React.FC<AcousticCardProps> = ({ acoustics }) => {
  const { pitch, energy } = acoustics;
  const isPitchAvailable = pitch.status === 'available';

  return (
    <div className="glass-card">
      <div className="card-header-row">
        <div>
          <h2 className="card-title">
            <Music size={20} color="#c084fc" />
            Acoustic Analysis: Pitch (F0) & RMS Energy
          </h2>
          <p className="card-desc">
            Acoustic signal properties calculated via Praat-Parselmouth autocorrelation and digital frame dynamics
          </p>
        </div>

        <span className={`status-tag ${isPitchAvailable ? 'status-online' : 'status-offline'}`}>
          Pitch: {pitch.status.toUpperCase()}
        </span>
      </div>

      <div className="acoustics-columns-grid">
        {/* Pitch Column */}
        <div className="subcard-box">
          <div className="subcard-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Music size={16} color="#c084fc" />
              <h3 className="subcard-title">Fundamental Frequency (F0)</h3>
            </div>
          </div>

          {isPitchAvailable ? (
            <div className="acoustic-metrics-table">
              <div className="acoustic-metric-row">
                <span className="acoustic-label">Median F0</span>
                <span className="acoustic-val">
                  {pitch.median_f0_hz != null ? `${pitch.median_f0_hz.toFixed(1)} Hz` : '—'}
                </span>
              </div>
              <div className="acoustic-metric-row">
                <span className="acoustic-label">Mean F0 (± Std Dev)</span>
                <span className="acoustic-val">
                  {pitch.mean_f0_hz != null && pitch.std_f0_hz != null
                    ? `${pitch.mean_f0_hz.toFixed(1)} ± ${pitch.std_f0_hz.toFixed(1)} Hz`
                    : '—'}
                </span>
              </div>
              <div className="acoustic-metric-row">
                <span className="acoustic-label">Pitch Range (Span)</span>
                <span className="acoustic-val highlight-val">
                  {pitch.range_semitones != null
                    ? `${pitch.range_semitones.toFixed(1)} semitones`
                    : '—'}
                </span>
              </div>
              <div className="acoustic-metric-row">
                <span className="acoustic-label">P05 &ndash; P95 Bounds</span>
                <span className="acoustic-val">
                  {pitch.p05_f0_hz != null && pitch.p95_f0_hz != null
                    ? `${pitch.p05_f0_hz.toFixed(0)} Hz &ndash; ${pitch.p95_f0_hz.toFixed(0)} Hz`
                    : '—'}
                </span>
              </div>
              <div className="acoustic-metric-row">
                <span className="acoustic-label">Voiced Fraction</span>
                <span className="acoustic-val">
                  {pitch.voiced_fraction != null
                    ? `${(pitch.voiced_fraction * 100).toFixed(0)}% voiced frames`
                    : '—'}
                </span>
              </div>
            </div>
          ) : (
            <div className="empty-subcard" style={{ padding: '1.5rem 1rem' }}>
              <AlertCircle size={20} color="var(--accent-amber)" />
              <div>
                <strong>Pitch extraction unavailable</strong>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  No periodic voiced frames detected within the configured bounds ({pitch.pitch_floor_hz}–{pitch.pitch_ceiling_hz} Hz).
                </p>
              </div>
            </div>
          )}

          <p className="subcard-footer-note">{pitch.note}</p>
        </div>

        {/* RMS Energy Column */}
        <div className="subcard-box">
          <div className="subcard-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Volume2 size={16} color="var(--accent-cyan)" />
              <h3 className="subcard-title">RMS Energy & Volume Dynamics</h3>
            </div>
          </div>

          <div className="acoustic-metrics-table">
            <div className="acoustic-metric-row">
              <span className="acoustic-label">Mean RMS Level</span>
              <span className="acoustic-val">
                {energy.mean_rms_dbfs.toFixed(1)} dBFS
              </span>
            </div>
            <div className="acoustic-metric-row">
              <span className="acoustic-label">Dynamic Range (P95 - P05)</span>
              <span className="acoustic-val highlight-val">
                {energy.dynamic_range_db.toFixed(1)} dB
              </span>
            </div>
            <div className="acoustic-metric-row">
              <span className="acoustic-label">Linear RMS Energy</span>
              <span className="acoustic-val">
                {energy.mean_rms.toFixed(4)} (std: {energy.std_rms.toFixed(4)})
              </span>
            </div>
            <div className="acoustic-metric-row">
              <span className="acoustic-label">Analysis Window</span>
              <span className="acoustic-val">
                {energy.frame_length_ms.toFixed(0)}ms window / {energy.hop_length_ms.toFixed(0)}ms hop
              </span>
            </div>
          </div>

          <p className="subcard-footer-note">{energy.note}</p>
        </div>
      </div>

      {/* Honest Acoustic Disclaimer */}
      <div className="methodology-note-banner" style={{ marginTop: '1.25rem' }}>
        <Info size={16} />
        <span>
          <strong>Acoustic Interpretation Boundary:</strong> Pitch variability and dynamic range quantify speech modulation and recording dynamics; they do <em>not</em> correlate reliably with candidate confidence, personality, or emotional state.
        </span>
      </div>
    </div>
  );
};
