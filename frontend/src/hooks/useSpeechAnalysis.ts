import { useState, useCallback } from 'react';
import type { AnalysisResponse, ApiErrorResponse } from '../types/analysis';

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25 MB

export interface UseSpeechAnalysisReturn {
  isAnalyzing: boolean;
  analysisResult: AnalysisResponse | null;
  error: string | null;
  errorCode: string | null;
  analyzeAudio: (audioSource: File | Blob, filename?: string, referenceText?: string) => Promise<AnalysisResponse | null>;
  resetAnalysis: () => void;
}

export const useSpeechAnalysis = (): UseSpeechAnalysisReturn => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);

  const resetAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    setAnalysisResult(null);
    setError(null);
    setErrorCode(null);
  }, []);

  const analyzeAudio = useCallback(
    async (audioSource: File | Blob, filename?: string, referenceText?: string): Promise<AnalysisResponse | null> => {
      // 1. Prevent duplicate submissions while an analysis request is in flight
      if (isAnalyzing) {
        return null;
      }

      setError(null);
      setErrorCode(null);

      // 2. Client-side usability size guard matching backend 25 MB limit
      if (audioSource.size > MAX_UPLOAD_BYTES) {
        const sizeMb = (audioSource.size / (1024 * 1024)).toFixed(1);
        const errorMsg = `Audio file size (${sizeMb} MB) exceeds the maximum allowed limit of 25 MB.`;
        setError(errorMsg);
        setErrorCode('FILE_TOO_LARGE');
        return null;
      }

      // 3. Client-side non-empty check
      if (audioSource.size === 0) {
        const errorMsg = 'The provided audio file is empty (0 bytes).';
        setError(errorMsg);
        setErrorCode('EMPTY_FILE');
        return null;
      }

      setIsAnalyzing(true);

      try {
        const formData = new FormData();
        const effectiveName =
          filename ||
          (audioSource instanceof File ? audioSource.name : 'recorded_speech.webm');
        formData.append('audio_file', audioSource, effectiveName);
        formData.append('language', 'en');

        if (referenceText && referenceText.trim().length > 0) {
          formData.append('reference_text', referenceText.trim());
        }

        const response = await fetch('/api/v1/analyze', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          let errDetail: string | null = null;
          let code: string | null = null;

          try {
            const errJson = (await response.json()) as ApiErrorResponse | Record<string, any>;
            if (errJson && 'detail' in errJson) {
              const detail = errJson.detail;
              if (typeof detail === 'object' && detail !== null) {
                if ('message' in detail && 'error_code' in detail) {
                  errDetail = String(detail.message);
                  code = String(detail.error_code);
                } else if (Array.isArray(detail)) {
                  // FastAPI Pydantic 422 validation errors
                  errDetail = detail
                    .map((item: any) => item.msg || JSON.stringify(item))
                    .join('; ');
                  code = 'VALIDATION_ERROR';
                } else {
                  errDetail = JSON.stringify(detail);
                }
              } else if (typeof detail === 'string') {
                errDetail = detail;
              }
            }
          } catch {
            // Server returned non-JSON error response (e.g. 502/504 or raw text)
            errDetail = `HTTP ${response.status}: ${response.statusText}`;
          }

          const finalMsg =
            errDetail || `Audio analysis request failed with HTTP status ${response.status}`;
          setError(finalMsg);
          setErrorCode(code || `HTTP_${response.status}`);
          setIsAnalyzing(false);
          return null;
        }

        const data: AnalysisResponse = await response.json();
        setAnalysisResult(data);
        setIsAnalyzing(false);
        return data;
      } catch (err: any) {
        const msg =
          err.message || 'Network error occurred while contacting the analysis server.';
        setError(msg);
        setErrorCode('NETWORK_ERROR');
        setIsAnalyzing(false);
        return null;
      }
    },
    [isAnalyzing]
  );

  return {
    isAnalyzing,
    analysisResult,
    error,
    errorCode,
    analyzeAudio,
    resetAnalysis,
  };
};
