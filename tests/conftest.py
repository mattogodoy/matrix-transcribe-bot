import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_whisper_model():
    """Patch WhisperModel so no real model is loaded."""
    with patch("src.transcriber.WhisperModel") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_transcriber():
    """A mock Transcriber that returns a fixed transcript."""
    t = MagicMock()
    t.transcribe.return_value = "Hola mundo"
    return t
