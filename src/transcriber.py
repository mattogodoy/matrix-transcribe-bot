import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "es",
        model_dir: str | None = None,
        cpu_threads: int = 0,
    ):
        self.language = language
        logger.info("Loading whisper model '%s'...", model_name)
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,
            cpu_threads=cpu_threads,
        )
        logger.info("Whisper model loaded (cpu_threads=%s).", cpu_threads or "all")

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path, language=self.language)
        text = " ".join(segment.text.strip() for segment in segments)
        return text
