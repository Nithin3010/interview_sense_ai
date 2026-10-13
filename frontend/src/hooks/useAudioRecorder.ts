import { useState, useRef, useCallback, useEffect } from 'react';

export interface UseAudioRecorderReturn {
  isRecording: boolean;
  recordingDuration: number;
  audioBlob: Blob | null;
  audioUrl: string | null;
  mimeType: string | null;
  audioLevel: number; // 0 to 100 for live mic visualizer
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  clearRecording: () => void;
}

const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
  'audio/wav',
];

export const useAudioRecorder = (): UseAudioRecorderReturn => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [mimeType, setMimeType] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const currentAudioUrlRef = useRef<string | null>(null);

  // Web Audio API refs for live volume metering
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Keep track of currentAudioUrl in a ref for clean revocation
  useEffect(() => {
    currentAudioUrlRef.current = audioUrl;
  }, [audioUrl]);

  const getSupportedMimeType = (): string | undefined => {
    if (typeof window === 'undefined' || typeof MediaRecorder === 'undefined') {
      return undefined;
    }
    for (const type of PREFERRED_MIME_TYPES) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return undefined;
  };

  const stopAudioMeter = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
  }, []);

  const stopTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore error if track already stopped
        }
      });
      streamRef.current = null;
    }
    stopAudioMeter();
  }, [stopAudioMeter]);

  const clearRecording = useCallback(() => {
    if (currentAudioUrlRef.current) {
      URL.revokeObjectURL(currentAudioUrlRef.current);
      currentAudioUrlRef.current = null;
    }
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingDuration(0);
    setError(null);
    setAudioLevel(0);
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // mediaRecorder may already be stopping
      }
    }

    setIsRecording(false);
    stopTracks();
  }, [stopTracks]);

  const startRecording = useCallback(async () => {
    clearRecording();
    setError(null);

    // 1. Verify browser support
    if (
      typeof window === 'undefined' ||
      !navigator?.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === 'undefined'
    ) {
      setError(
        'Audio recording is not supported in this browser. Please use a modern browser such as Chrome, Firefox, or Edge.'
      );
      return;
    }

    try {
      // 2. Request microphone stream explicitly upon user action
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // 3. Setup live audio level metering (if AudioContext is available)
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          const audioCtx = new AudioCtx();
          audioContextRef.current = audioCtx;
          const source = audioCtx.createMediaStreamSource(stream);
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 256;
          analyser.smoothingTimeConstant = 0.5;
          source.connect(analyser);
          analyserRef.current = analyser;

          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          const updateLevel = () => {
            if (analyserRef.current) {
              analyserRef.current.getByteFrequencyData(dataArray);
              let sum = 0;
              for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
              }
              const avg = sum / dataArray.length;
              // Map 0-128 to 0-100 percentage
              const normalized = Math.min(100, Math.round((avg / 128) * 100));
              setAudioLevel(normalized);
              animFrameRef.current = requestAnimationFrame(updateLevel);
            }
          };
          updateLevel();
        }
      } catch {
        // Non-blocking: live meter is an optional visual enhancement
      }

      // 4. MIME-type selection with fallback
      const selectedMime = getSupportedMimeType();
      let mediaRecorder: MediaRecorder;
      try {
        const options: MediaRecorderOptions = selectedMime ? { mimeType: selectedMime } : {};
        mediaRecorder = new MediaRecorder(stream, options);
      } catch {
        // Fallback without explicit options if browser rejected mimeType
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      const effectiveMime = mediaRecorder.mimeType || selectedMime || 'audio/webm';
      setMimeType(effectiveMime);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const recordedType = mediaRecorder.mimeType || effectiveMime;
        const blob = new Blob(chunksRef.current, { type: recordedType });

        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
        }

        const url = URL.createObjectURL(blob);
        currentAudioUrlRef.current = url;
        setAudioBlob(blob);
        setAudioUrl(url);
        chunksRef.current = [];
      };

      mediaRecorder.onerror = (e: Event) => {
        const msg = (e as any).error?.message || 'MediaRecorder encountered a recording error.';
        setError(`Recording error: ${msg}`);
        stopRecording();
      };

      // 5. Start recording (chunks every 250ms)
      mediaRecorder.start(250);
      setIsRecording(true);
      startTimeRef.current = Date.now();
      setRecordingDuration(0);

      // 6. Accurate timer & auto-stop safeguard at 300 seconds (backend max limit)
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        setRecordingDuration(elapsed);

        if (elapsed >= 300) {
          // Exceeded maximum allowable speech duration (5 minutes)
          stopRecording();
        }
      }, 100);
    } catch (err: any) {
      stopTracks();
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError(
          'Microphone permission was denied. Please allow microphone access in your browser address bar or settings to record.'
        );
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No microphone found. Please connect an audio input device and retry.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError('Microphone is currently in use by another application.');
      } else {
        setError(err.message || 'Unable to access microphone for recording.');
      }
    }
  }, [clearRecording, stopRecording, stopTracks]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
      stopTracks();
      if (currentAudioUrlRef.current) {
        URL.revokeObjectURL(currentAudioUrlRef.current);
        currentAudioUrlRef.current = null;
      }
    };
  }, [stopTracks]);

  return {
    isRecording,
    recordingDuration,
    audioBlob,
    audioUrl,
    mimeType,
    audioLevel,
    error,
    startRecording,
    stopRecording,
    clearRecording,
  };
};
