class InterviewSenseException(Exception):
    """Base exception for all domain-specific application errors."""

    def __init__(self, error_code: str, message: str, status_code: int = 400):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AudioFormatError(InterviewSenseException):
    def __init__(self, message: str = "Invalid or unsupported audio container format."):
        super().__init__(error_code="INVALID_AUDIO_FORMAT", message=message, status_code=400)


class AudioTooShortError(InterviewSenseException):
    def __init__(self, duration: float, min_duration: float = 2.0):
        message = f"Audio duration of {duration:.2f}s is below the minimum required duration of {min_duration:.1f}s."
        super().__init__(error_code="AUDIO_TOO_SHORT", message=message, status_code=422)


class AudioTooLongError(InterviewSenseException):
    def __init__(self, duration: float, max_duration: float = 300.0):
        message = f"Audio duration of {duration:.2f}s exceeds the maximum allowed duration of {max_duration:.1f}s."
        super().__init__(error_code="AUDIO_TOO_LONG", message=message, status_code=422)


class FileTooLargeError(InterviewSenseException):
    def __init__(self, size_bytes: int, max_bytes: int):
        message = f"Audio payload size ({size_bytes / (1024*1024):.1f} MB) exceeds limit of {max_bytes / (1024*1024):.1f} MB."
        super().__init__(error_code="FILE_TOO_LARGE", message=message, status_code=413)


class SilentAudioError(InterviewSenseException):
    def __init__(self, message: str = "Audio contains no discernible speech; energy is below audible threshold."):
        super().__init__(error_code="SILENT_AUDIO", message=message, status_code=422)


class ModelNotLoadedError(InterviewSenseException):
    def __init__(self, model_name: str):
        message = f"Required model '{model_name}' has not been initialized."
        super().__init__(error_code="MODEL_NOT_LOADED", message=message, status_code=503)


class InvalidReferenceTextError(InterviewSenseException):
    def __init__(self, message: str = "Invalid reference text provided."):
        super().__init__(error_code="INVALID_REFERENCE_TEXT", message=message, status_code=422)

