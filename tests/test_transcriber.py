from unittest.mock import MagicMock, patch


class TestTranscriberInit:
    @patch("src.transcriber.WhisperModel")
    def test_loads_model_with_correct_params(self, mock_cls):
        from src.transcriber import Transcriber

        Transcriber(model_name="large-v3", language="es", model_dir="/models")

        mock_cls.assert_called_once_with(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root="/models",
        )

    @patch("src.transcriber.WhisperModel")
    def test_stores_language(self, mock_cls):
        from src.transcriber import Transcriber

        t = Transcriber(model_name="tiny", language="es")
        assert t.language == "es"


class TestTranscriberTranscribe:
    @patch("src.transcriber.WhisperModel")
    def test_returns_joined_segments(self, mock_cls):
        from src.transcriber import Transcriber

        seg1 = MagicMock()
        seg1.text = " Hola mundo "
        seg2 = MagicMock()
        seg2.text = " esto es una prueba "
        model = MagicMock()
        model.transcribe.return_value = (iter([seg1, seg2]), None)
        mock_cls.return_value = model

        t = Transcriber(model_name="tiny", language="es")
        result = t.transcribe("/tmp/audio.ogg")

        model.transcribe.assert_called_once_with("/tmp/audio.ogg", language="es")
        assert result == "Hola mundo esto es una prueba"

    @patch("src.transcriber.WhisperModel")
    def test_returns_empty_string_for_no_segments(self, mock_cls):
        from src.transcriber import Transcriber

        model = MagicMock()
        model.transcribe.return_value = (iter([]), None)
        mock_cls.return_value = model

        t = Transcriber(model_name="tiny", language="es")
        result = t.transcribe("/tmp/audio.ogg")

        assert result == ""
