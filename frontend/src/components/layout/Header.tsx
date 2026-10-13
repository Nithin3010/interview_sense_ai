import React from 'react';
import { Mic, RefreshCw, Cpu, Activity, SpellCheck } from 'lucide-react';
import type { HealthResponse } from '../../types/analysis';

interface HeaderProps {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, loading, error, onRefresh }) => {
  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-icon">
          <Mic size={24} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <span className="brand-title">InterviewSense AI</span>
            <span className="brand-badge">Milestone 7 Engine</span>
          </div>
          <p className="brand-subtitle">
            Local speech delivery analysis • Acoustic, temporal, and lexical evaluation
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        {/* Backend Models Status Badges */}
        {health && !error && (
          <div className="engine-models-group" title="Active local analysis models">
            <span className="model-chip" title={`ASR Model: ${health.models.asr}`}>
              <Cpu size={12} color="var(--accent-primary)" />
              <span>ASR: {health.models.asr}</span>
            </span>
            <span className="model-chip" title={`Acoustic Model: ${health.models.acoustic}`}>
              <Activity size={12} color="var(--accent-cyan)" />
              <span>Praat: {health.models.acoustic}</span>
            </span>
            <span className="model-chip" title={`Pronunciation Engine: ${health.models.pronunciation}`}>
              <SpellCheck size={12} color="var(--accent-amber)" />
              <span>Pronunciation: {health.models.pronunciation}</span>
            </span>
          </div>
        )}

        {/* Backend Health Status Indicator */}
        <div className="status-pill">
          {loading ? (
            <>
              <span className="status-dot loading" />
              <span>Checking backend...</span>
            </>
          ) : error ? (
            <>
              <span className="status-dot offline" />
              <span>Backend Offline</span>
            </>
          ) : (
            <>
              <span className="status-dot online" />
              <span>
                Backend v{health?.version || '1.0.0'} Online
              </span>
            </>
          )}
        </div>

        <button
          onClick={onRefresh}
          title="Refresh backend status"
          className="refresh-btn"
          aria-label="Refresh backend status"
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
        </button>
      </div>
    </header>
  );
};
