import React from 'react';
import { SpellCheck, Info, AlertCircle, ShieldAlert, Target } from 'lucide-react';
import type { PronunciationAssessment } from '../../types/analysis';

interface PronunciationCardProps {
  pronunciation: PronunciationAssessment;
}

export const PronunciationCard: React.FC<PronunciationCardProps> = ({ pronunciation }) => {
  const {
    status,
    model_identifier,
    phoneme_inventory,
    transcript_source,
    reference_text,
    alignment_coverage_ratio,
    mean_acoustic_confidence,
    word_diagnostics,
    methodology_note,
    error_message,
  } = pronunciation;

  const isUnavailable = status === 'unavailable';
  const isReady = status === 'ready';
  const isExperimental = status === 'experimental';
  const isError = status === 'error';

  const getStatusBadge = () => {
    if (isReady) return <span className="status-tag status-emerald">Status: ALIGNED (UNCALIBRATED EVIDENCE)</span>;
    if (isExperimental) return <span className="status-tag status-amber">Status: EXPERIMENTAL / PARTIAL</span>;
    if (isUnavailable) return <span className="status-tag status-dim">Status: UNAVAILABLE</span>;
    return <span className="status-tag status-offline">Status: ERROR</span>;
  };

  const getConfidenceColorClass = (conf: number | null | undefined, isSupported: boolean = true) => {
    if (!isSupported) return 'chip-unsupported';
    if (conf == null) return 'chip-unaligned';
    if (conf >= 0.70) return 'chip-conf-high';
    if (conf >= 0.40) return 'chip-conf-mid';
    return 'chip-conf-low';
  };

  return (
    <div className="glass-card">
      <div className="card-header-row">
        <div>
          <h2 className="card-title">
            <SpellCheck size={20} color="var(--accent-amber)" />
            Pronunciation Assessment (Phonetic CTC Diagnostic)
          </h2>
          <p className="card-desc">
            Frame-level acoustic alignment and phonetic evidence using local Wav2Vec2 CTC models
          </p>
        </div>

        {getStatusBadge()}
      </div>

      {/* Model & Alignment Metadata Bar */}
      <div className="meta-pills-row">
        <span className="meta-pill">
          Model: <code>{model_identifier || 'facebook/wav2vec2-lv-60-espeak-cv-ft'}</code>
        </span>
        <span className="meta-pill">
          Inventory: <code>{phoneme_inventory || 'ARPAbet-39'}</code>
        </span>
        <span className="meta-pill">
          Source: <code>{transcript_source}</code>
        </span>
        {alignment_coverage_ratio != null && (
          <span className="meta-pill" title="Proportion of expected reference phonemes successfully aligned">
            Coverage: <strong>{(alignment_coverage_ratio * 100).toFixed(0)}%</strong>
          </span>
        )}
        {mean_acoustic_confidence != null && (
          <span className="meta-pill" title="Average frame-level posterior probability across scorable aligned phonemes">
            Mean Acoustic Evidence: <strong>{(mean_acoustic_confidence * 100).toFixed(0)}%</strong>
          </span>
        )}
      </div>

      {/* Target Reference Sentence Callout */}
      {reference_text && (
        <div className="reference-sentence-banner" style={{ marginTop: '1rem' }}>
          <Target size={18} color="var(--accent-primary)" />
          <div>
            <span className="reference-label">Target Reference Sentence:</span>
            <p className="reference-content">&ldquo;{reference_text}&rdquo;</p>
          </div>
        </div>
      )}

      {/* Explicit Unavailable State */}
      {isUnavailable && (
        <div className="empty-subcard" style={{ marginTop: '1rem', padding: '1.25rem' }}>
          <AlertCircle size={20} color="var(--accent-amber)" />
          <div>
            <strong>Pronunciation assessment currently unavailable</strong>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {methodology_note}
            </p>
          </div>
        </div>
      )}

      {/* Error State */}
      {isError && (
        <div className="error-callout" style={{ marginTop: '1rem' }}>
          <ShieldAlert size={20} />
          <div>
            <strong>Pronunciation Diagnostic Error:</strong>
            <p>{error_message || 'Diagnostic model failed safely without impacting core analysis.'}</p>
          </div>
        </div>
      )}

      {/* Aligned Phoneme Diagnostics Grid */}
      {(isReady || isExperimental) && word_diagnostics.length > 0 && (
        <div style={{ marginTop: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h4 className="subcard-title" style={{ margin: 0 }}>
              {reference_text ? 'Reference Phoneme Alignment & Evidence' : 'Experimental Phoneme Breakdown'}
            </h4>
            <div className="legend-pills-row">
              <span className="legend-indicator dot-high">High Evidence (&ge;70%)</span>
              <span className="legend-indicator dot-mid">Moderate (40–69%)</span>
              <span className="legend-indicator dot-low">Low / Uncertain (&lt;40%)</span>
              <span className="legend-indicator dot-dim">Unaligned / Unsupported</span>
            </div>
          </div>

          <div className="phoneme-diagnostics-grid">
            {word_diagnostics.map((w, wIdx) => (
              <div
                key={`${w.word}-${w.start}-${wIdx}`}
                className={`phoneme-word-card ${w.flagged ? 'word-card-flagged' : ''}`}
              >
                <div className="phoneme-word-header">
                  <strong>&ldquo;{w.word}&rdquo;</strong>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {w.mean_acoustic_confidence != null && (
                      <span className="word-mean-score" title="Word average posterior match">
                        {(w.mean_acoustic_confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className="phoneme-word-time">
                      {w.start > 0 || w.end > 0 ? `${w.start.toFixed(2)}s – ${w.end.toFixed(2)}s` : 'unaligned'}
                    </span>
                  </div>
                </div>

                {w.diagnostic_note && (
                  <p className="word-diagnostic-subnote">{w.diagnostic_note}</p>
                )}

                <div className="phoneme-chips-wrap">
                  {w.phonemes.map((ph, pIdx) => {
                    const isSupported = ph.is_supported ?? true;
                    const conf = ph.acoustic_match_confidence;
                    const colorClass = getConfidenceColorClass(conf, isSupported);

                    let tooltip = `Phoneme: ${ph.phoneme}`;
                    if (!isSupported) {
                      tooltip += ' • [Unsupported in model inventory]';
                    } else if (conf == null) {
                      tooltip += ` • [${ph.diagnostic_note || 'Omitted or unaligned'}]`;
                    } else {
                      tooltip += ` • ${ph.start.toFixed(2)}s - ${ph.end.toFixed(2)}s (${(ph.duration * 1000).toFixed(0)}ms) • Match: ${(conf * 100).toFixed(0)}%`;
                    }

                    return (
                      <span
                        key={`${ph.phoneme}-${pIdx}`}
                        className={`phoneme-chip ${colorClass}`}
                        title={tooltip}
                      >
                        <span className="phoneme-symbol">{ph.phoneme}</span>
                        {conf != null ? (
                          <small className="phoneme-conf">{(conf * 100).toFixed(0)}%</small>
                        ) : !isSupported ? (
                          <small className="phoneme-conf dim-tag">n/a</small>
                        ) : (
                          <small className="phoneme-conf dim-tag">skip</small>
                        )}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mandatory Scientific Methodology & Boundaries Banner */}
      <div className="methodology-note-banner" style={{ marginTop: '1.25rem' }}>
        <Info size={16} />
        <div>
          <strong>Scientific Methodology & Boundaries:</strong> {methodology_note}
        </div>
      </div>
    </div>
  );
};
