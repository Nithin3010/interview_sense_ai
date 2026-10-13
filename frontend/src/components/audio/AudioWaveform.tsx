import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Loader2 } from 'lucide-react';

interface AudioWaveformProps {
  src: string;
  title?: string;
  initialDuration?: number;
}

const BAR_COUNT = 54;

export const AudioWaveform: React.FC<AudioWaveformProps> = ({
  src,
  title = 'Recorded Audio Preview',
  initialDuration,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(initialDuration || 0);
  const [isMuted, setIsMuted] = useState(false);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [isDecoding, setIsDecoding] = useState(false);

  // Generate fallback peaks in case audio context decoding is not supported
  const generateFallbackPeaks = useCallback((): number[] => {
    const fallback: number[] = [];
    for (let i = 0; i < BAR_COUNT; i++) {
      // Gentle pseudo-random sine wave
      const v = 0.25 + 0.45 * Math.abs(Math.sin((i / BAR_COUNT) * Math.PI * 3.5));
      fallback.push(Math.max(0.12, Math.min(0.95, v)));
    }
    return fallback;
  }, []);

  // Decode audio data into waveform peaks via Web Audio API
  useEffect(() => {
    let isCancelled = false;
    let audioCtx: AudioContext | null = null;

    const extractPeaks = async () => {
      setIsDecoding(true);
      try {
        const response = await fetch(src);
        const arrayBuffer = await response.arrayBuffer();

        if (isCancelled) return;

        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (!AudioCtx) {
          setPeaks(generateFallbackPeaks());
          setIsDecoding(false);
          return;
        }

        audioCtx = new AudioCtx();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        if (isCancelled) return;

        if (audioBuffer.duration && audioBuffer.duration > 0) {
          setDuration(audioBuffer.duration);
        }

        const channelData = audioBuffer.getChannelData(0);
        const step = Math.floor(channelData.length / BAR_COUNT);
        const extracted: number[] = [];

        let maxVal = 0.001;
        for (let i = 0; i < BAR_COUNT; i++) {
          const start = i * step;
          const end = Math.min(start + step, channelData.length);
          let sum = 0;
          for (let j = start; j < end; j++) {
            sum += Math.abs(channelData[j]);
          }
          const avg = sum / (end - start || 1);
          extracted.push(avg);
          if (avg > maxVal) maxVal = avg;
        }

        // Normalize between 0.12 and 1.0
        const normalized = extracted.map((val) =>
          Math.max(0.12, Math.min(1.0, val / maxVal))
        );

        setPeaks(normalized);
      } catch {
        // Fallback gracefully on audio decode error (e.g. unsupported raw codec container in WebAudio)
        if (!isCancelled) {
          setPeaks(generateFallbackPeaks());
        }
      } finally {
        if (!isCancelled) {
          setIsDecoding(false);
        }
        if (audioCtx && audioCtx.state !== 'closed') {
          audioCtx.close().catch(() => {});
        }
      }
    };

    extractPeaks();

    return () => {
      isCancelled = true;
      if (audioCtx && audioCtx.state !== 'closed') {
        audioCtx.close().catch(() => {});
      }
    };
  }, [src, generateFallbackPeaks]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current && audioRef.current.duration && !isNaN(audioRef.current.duration)) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleRestart = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      setCurrentTime(0);
    }
  };

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const seekToRatio = (ratio: number) => {
    const targetTime = Math.max(0, Math.min(duration || 1, ratio * (duration || 1)));
    setCurrentTime(targetTime);
    if (audioRef.current) {
      audioRef.current.currentTime = targetTime;
    }
  };

  const handleWaveformClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    seekToRatio(ratio);
  };

  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs) || !isFinite(secs) || secs < 0) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const progressRatio = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;
  const activeBarIndex = Math.floor(progressRatio * (peaks.length || BAR_COUNT));

  return (
    <div className="audio-waveform-component">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      <div className="waveform-header">
        <span className="waveform-title">{title}</span>
        <div className="waveform-timer">
          <span>{formatTime(currentTime)}</span>
          <span className="timer-sep">/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Visual Waveform Interactive Bar Chart */}
      <div
        className="waveform-display"
        onClick={handleWaveformClick}
        role="slider"
        aria-label="Seek waveform"
        aria-valuemin={0}
        aria-valuemax={duration || 1}
        aria-valuenow={currentTime}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') {
            seekToRatio((currentTime + 1) / (duration || 1));
          } else if (e.key === 'ArrowLeft') {
            seekToRatio((currentTime - 1) / (duration || 1));
          } else if (e.key === ' ') {
            e.preventDefault();
            togglePlay();
          }
        }}
      >
        {isDecoding && peaks.length === 0 ? (
          <div className="waveform-decoding-msg">
            <Loader2 size={16} className="spin" />
            <span>Decoding audio waveform...</span>
          </div>
        ) : (
          <div className="waveform-bars-wrap">
            {peaks.map((heightRatio, idx) => {
              const isPlayed = idx <= activeBarIndex;
              return (
                <div
                  key={`wf-bar-${idx}`}
                  className={`waveform-bar ${isPlayed ? 'is-played' : ''}`}
                  style={{
                    height: `${Math.round(heightRatio * 100)}%`,
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Player Controls Bar */}
      <div className="waveform-controls-bar">
        <div className="controls-left">
          <button
            type="button"
            onClick={togglePlay}
            className="player-btn primary"
            title={isPlaying ? 'Pause' : 'Play'}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} style={{ marginLeft: 2 }} />}
          </button>

          <button
            type="button"
            onClick={handleRestart}
            className="player-btn secondary"
            title="Restart from beginning"
            aria-label="Restart audio"
          >
            <RotateCcw size={16} />
          </button>
        </div>

        {/* Fallback & accessible scrubber range */}
        <input
          type="range"
          min="0"
          max={duration || 1}
          step="0.05"
          value={currentTime}
          onChange={handleScrubberChange}
          className="player-scrubber"
          aria-label="Audio scrubber"
        />

        <button
          type="button"
          onClick={toggleMute}
          className="player-btn secondary"
          title={isMuted ? 'Unmute' : 'Mute'}
          aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}
        >
          {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>
    </div>
  );
};
