from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "InterviewSense AI"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Audio Limits & Normalization
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1  # Mono
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MIN_DURATION_SECONDS: float = 2.0
    MAX_DURATION_SECONDS: float = 300.0
    SILENCE_RMS_THRESHOLD: float = 1e-4

    # Speech Recognition Model Configuration
    WHISPER_MODEL_SIZE: str = "small.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_CPU_THREADS: int = 4

    # Phoneme-Level Pronunciation Model Configuration (Wav2Vec2 CTC)
    PRONUNCIATION_MODEL_NAME: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    PRONUNCIATION_DEVICE: str = "cpu"
    AUTO_LOAD_PRONUNCIATION: bool = False

    # Acoustic Analysis Settings (Praat F0 & Energy)
    PITCH_FLOOR_HZ: float = 65.0
    PITCH_CEILING_HZ: float = 400.0
    PITCH_TIME_STEP_SECONDS: Optional[float] = 0.01  # 10 ms step
    RMS_FRAME_LENGTH_MS: float = 25.0               # 25 ms window
    RMS_HOP_LENGTH_MS: float = 10.0                 # 10 ms hop

    # Speech Delivery Guidelines (Configurable Reference Thresholds)
    WPM_GUIDELINE_MIN: float = 120.0
    WPM_GUIDELINE_MAX: float = 160.0
    PAUSE_RATIO_MIN: float = 0.15
    PAUSE_RATIO_MAX: float = 0.25
    MIN_PAUSE_SECONDS: float = 0.25
    SIGNIFICANT_PAUSE_SECONDS: float = 0.50
    EXTENDED_PAUSE_SECONDS: float = 2.0
    MONOTONE_SEMITONE_THRESHOLD: float = 3.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
