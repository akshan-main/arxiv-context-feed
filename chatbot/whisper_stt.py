"""Whisper speech-to-text for voice input.

Supports any language. Lazy-loads Whisper model on first use (~500MB).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Speech-to-text using OpenAI Whisper.

    Lazy loads the model on first transcription to save RAM at startup.
    """

    def __init__(self, model_name: str = ""):
        """Initialize transcriber.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large).
        """
        self._model_name = model_name or os.getenv("WHISPER_MODEL", "small")
        self._model = None

    def _load_model(self):
        """Lazy-load Whisper model."""
        if self._model is None:
            import whisper

            logger.info(f"Loading Whisper model: {self._model_name}")
            self._model = whisper.load_model(self._model_name)
            logger.info("Whisper model loaded")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text.
        """
        self._load_model()

        try:
            result = self._model.transcribe(audio_path)
            text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            logger.info(f"Transcribed ({language}): {text[:50]}...")
            return text

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
