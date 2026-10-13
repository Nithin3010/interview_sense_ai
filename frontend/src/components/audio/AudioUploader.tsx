import React, { useState, useRef, useCallback, useEffect } from 'react';
import { UploadCloud, FileAudio, Trash2, ArrowRight, AlertCircle, Loader2 } from 'lucide-react';
import { AudioWaveform } from './AudioWaveform';

interface AudioUploaderProps {
  onAnalyze: (file: File, filename: string) => void;
  isAnalyzing: boolean;
}

const SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac', '.webm', '.m4a'];
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25 MB

export const AudioUploader: React.FC<AudioUploaderProps> = ({ onAnalyze, isAnalyzing }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  useEffect(() => {
    previewUrlRef.current = previewUrl;
  }, [previewUrl]);

  const clearSelection = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleFile = (file: File) => {
    clearSelection();
    setValidationError(null);

    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setValidationError(
        `Unsupported audio container (${ext}). Supported formats: ${SUPPORTED_EXTENSIONS.join(', ')}`
      );
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      setValidationError(
        `File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed limit of 25 MB.`
      );
      return;
    }

    if (file.size === 0) {
      setValidationError('The uploaded file is empty (0 bytes).');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    previewUrlRef.current = url;
    setPreviewUrl(url);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile || isAnalyzing) return;
    onAnalyze(selectedFile, selectedFile.name);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  // Revoke preview URL on component unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  return (
    <div className="uploader-container">
      {selectedFile && previewUrl ? (
        <div className="uploader-selected-view">
          <div className="selected-file-card">
            <div className="file-info-header">
              <div className="file-icon-box">
                <FileAudio size={24} color="var(--accent-cyan)" />
              </div>
              <div className="file-details">
                <span className="file-name">{selectedFile.name}</span>
                <span className="file-meta">
                  {formatFileSize(selectedFile.size)} • {selectedFile.type || 'audio file'}
                </span>
              </div>
            </div>

            {/* Audio Waveform preview */}
            <AudioWaveform src={previewUrl} title={selectedFile.name} />

            <div className="uploader-actions">
              <button
                type="button"
                onClick={clearSelection}
                disabled={isAnalyzing}
                className="action-btn secondary-btn"
              >
                <Trash2 size={16} />
                <span>Remove File</span>
              </button>

              <button
                type="button"
                onClick={handleAnalyzeClick}
                disabled={isAnalyzing}
                className="action-btn primary-btn"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={18} className="spin" />
                    <span>Analyzing Audio...</span>
                  </>
                ) : (
                  <>
                    <span>Analyze Audio</span>
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div
          className={`dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.ogg,.flac,.webm,.m4a,audio/*"
            onChange={handleInputChange}
            style={{ display: 'none' }}
          />

          <div className="dropzone-icon-box">
            <UploadCloud size={40} color="var(--accent-primary)" />
          </div>

          <h3 className="dropzone-title">Upload Speech Recording</h3>
          <p className="dropzone-desc">
            Drag and drop an audio file here, or <span className="highlight-text">browse files</span>
          </p>

          <div className="dropzone-badges">
            <span>WAV</span>
            <span>MP3</span>
            <span>OGG</span>
            <span>FLAC</span>
            <span>WEBM</span>
            <span>M4A</span>
          </div>

          <span className="dropzone-limits">Max file size: 25 MB • Min duration: 2.0s</span>
        </div>
      )}

      {validationError && (
        <div className="error-callout" style={{ marginTop: '1rem' }}>
          <AlertCircle size={18} />
          <div>
            <strong>File Validation Error:</strong>
            <p>{validationError}</p>
          </div>
        </div>
      )}
    </div>
  );
};
