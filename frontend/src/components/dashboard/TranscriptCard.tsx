import React, { useState } from 'react';
import { FileText, Info } from 'lucide-react';
import type { TranscriptResponse, FillerWord, RepetitionRecord } from '../../types/analysis';

interface TranscriptCardProps {
  transcript: TranscriptResponse;
  fillerWords: FillerWord[];
  repetitions: RepetitionRecord[];
}

export const TranscriptCard: React.FC<TranscriptCardProps> = ({
  transcript,
  fillerWords,
  repetitions,
}) => {
  const [selectedWordIndex, setSelectedWordIndex] = useState<number | null>(null);

  const isFiller = (start: number, end: number) => {
    return fillerWords.some(
      (f) => Math.abs(f.start - start) < 0.08 && Math.abs(f.end - end) < 0.08
    );
  };

  const getFillerCategory = (start: number, end: number) => {
    const f = fillerWords.find(
      (item) => Math.abs(item.start - start) < 0.08 && Math.abs(item.end - end) < 0.08
    );
    return f ? f.category : null;
  };

  const isRepetition = (start: number, end: number) => {
    return repetitions.some((r) => start >= r.start - 0.05 && end <= r.end + 0.05);
  };

  const selectedWord = selectedWordIndex !== null ? transcript.words[selectedWordIndex] : null;

  return (
    <div className="glass-card">
      <div className="card-header-row">
        <div>
          <h2 className="card-title">
            <FileText size={20} color="var(--accent-primary)" />
            Speech Recognition Transcript
          </h2>
          <p className="card-desc">
            Model: <code>{transcript.model_used}</code> • Language: <code>{transcript.language}</code> • {transcript.words.length} words recognized
          </p>
        </div>

        <div className="transcript-legend">
          <span className="legend-item">
            <span className="legend-dot filler-dot" /> Filler Token
          </span>
          <span className="legend-item">
            <span className="legend-dot repetition-dot" /> Repeated Expression
          </span>
        </div>
      </div>

      {/* Main Transcript Text with Word Highlighting */}
      <div className="transcript-box">
        {transcript.words.length === 0 ? (
          <p className="empty-text">No speech tokens recognized in audio.</p>
        ) : (
          <div className="words-stream">
            {transcript.words.map((item, idx) => {
              const filler = isFiller(item.start, item.end);
              const rep = isRepetition(item.start, item.end);
              const isSelected = selectedWordIndex === idx;

              let tokenClass = 'word-token';
              if (filler) tokenClass += ' is-filler';
              if (rep) tokenClass += ' is-repetition';
              if (isSelected) tokenClass += ' is-selected';

              return (
                <button
                  key={`${item.word}-${idx}`}
                  type="button"
                  className={tokenClass}
                  onClick={() => setSelectedWordIndex(isSelected ? null : idx)}
                  title={`[${item.start.toFixed(2)}s - ${item.end.toFixed(2)}s] ASR Confidence: ${(item.probability * 100).toFixed(0)}%`}
                >
                  {item.word}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected Token Inspector */}
      {selectedWord && (
        <div className="token-detail-card">
          <div className="token-detail-header">
            <strong>Token Inspector: &ldquo;{selectedWord.word}&rdquo;</strong>
            <button
              type="button"
              className="text-link-btn"
              onClick={() => setSelectedWordIndex(null)}
            >
              Close
            </button>
          </div>
          <div className="token-metrics-row">
            <div>
              <span className="token-label">Start Time:</span>
              <span className="token-val">{selectedWord.start.toFixed(2)}s</span>
            </div>
            <div>
              <span className="token-label">End Time:</span>
              <span className="token-val">{selectedWord.end.toFixed(2)}s</span>
            </div>
            <div>
              <span className="token-label">Duration:</span>
              <span className="token-val">{(selectedWord.end - selectedWord.start).toFixed(2)}s</span>
            </div>
            <div>
              <span className="token-label">ASR Recognition Probability:</span>
              <span className="token-val">{(selectedWord.probability * 100).toFixed(1)}%</span>
            </div>
            {getFillerCategory(selectedWord.start, selectedWord.end) && (
              <div>
                <span className="token-label">Filler Category:</span>
                <span className="token-val badge-amber">
                  {getFillerCategory(selectedWord.start, selectedWord.end)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Transparent Disclaimer */}
      <div className="methodology-note-banner">
        <Info size={16} />
        <span>
          <strong>Note on Word Probabilities:</strong> Token probabilities indicate faster-whisper acoustic/language model recognition confidence, <em>not</em> speaker pronunciation accuracy. Speech recognition and derived metrics may contain errors.
        </span>
      </div>
    </div>
  );
};
