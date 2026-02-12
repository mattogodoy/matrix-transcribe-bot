# Matrix Transcript Bot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-hosted Matrix bot that transcribes audio messages into text using faster-whisper and replies with the transcript.

**Architecture:** Single async Python process using `matrix-nio` for Matrix protocol and `faster-whisper` for transcription. Audio processing runs in a thread pool to avoid blocking the async event loop. Docker deployment with persistent volumes for encryption keys and model cache.

**Tech Stack:** Python 3.12, matrix-nio[e2e], faster-whisper, pytest, pytest-asyncio, Docker

---

### Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.pyo
.env
data/
*.egg-info/
dist/
build/
.pytest_cache/
venv/
.venv/
```

**Step 2: Create `requirements.txt`**

```
matrix-nio[e2e]>=0.24.0
faster-whisper>=1.0.0
```

**Step 3: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Step 4: Create `.env.example`**

```
MATRIX_HOMESERVER=https://matrix.example.com
MATRIX_USER_ID=@transcribe-bot:example.com
MATRIX_PASSWORD=changeme
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=es
LOG_LEVEL=INFO
```

**Step 5: Create `src/__init__.py`**

Empty file.

**Step 6: Create `tests/__init__.py`**

Empty file.

**Step 7: Create `tests/conftest.py`**

```python
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
```

**Step 8: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt .env.example src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "scaffold: project structure, dependencies, and test config"
```

---

### Task 2: Transcriber Module (TDD)

**Files:**
- Create: `tests/test_transcriber.py`
- Create: `src/transcriber.py`

**Step 1: Write failing tests**

Create `tests/test_transcriber.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && python -m pytest tests/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.transcriber'`

**Step 3: Implement `src/transcriber.py`**

```python
import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "es",
        model_dir: str | None = None,
    ):
        self.language = language
        logger.info("Loading whisper model '%s'...", model_name)
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,
        )
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path, language=self.language)
        text = " ".join(segment.text.strip() for segment in segments)
        return text
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && python -m pytest tests/test_transcriber.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/transcriber.py tests/test_transcriber.py
git commit -m "feat: add transcriber module with faster-whisper"
```

---

### Task 3: Bot Module (TDD)

**Files:**
- Create: `tests/test_bot.py`
- Create: `src/bot.py`

**Step 1: Write failing tests**

Create `tests/test_bot.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Patch nio before importing bot
with patch.dict("sys.modules", {"nio": MagicMock(), "nio.crypto.attachments": MagicMock()}):
    pass


@pytest.fixture
def mock_transcriber():
    t = MagicMock()
    t.transcribe.return_value = "Hola mundo"
    return t


@pytest.fixture
def bot(mock_transcriber):
    with patch("src.bot.AsyncClient"), patch("src.bot.AsyncClientConfig"):
        from src.bot import TranscriptBot

        b = TranscriptBot(
            homeserver="https://matrix.example.com",
            user_id="@bot:example.com",
            password="password",
            store_path="/tmp/store",
            transcriber=mock_transcriber,
        )
        b.client = AsyncMock()
        b.client.user_id = "@bot:example.com"
        return b


def make_audio_event(sender, event_id="$event1", server_timestamp=None, url="mxc://example.com/audio"):
    event = MagicMock()
    event.sender = sender
    event.event_id = event_id
    event.server_timestamp = server_timestamp or 9999999999999
    event.url = url
    event.source = {
        "content": {
            "body": "voice-message.ogg",
            "url": url,
        }
    }
    return event


def make_room(room_id="!room:example.com"):
    room = MagicMock()
    room.room_id = room_id
    return room


class TestEventFiltering:
    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, bot):
        event = make_audio_event(sender="@bot:example.com")
        room = make_room()

        await bot.on_audio_message(room, event)

        bot.client.room_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_old_messages(self, bot):
        bot._startup_ms = 5000
        event = make_audio_event(sender="@user:example.com", server_timestamp=1000)
        room = make_room()

        await bot.on_audio_message(room, event)

        bot.client.room_send.assert_not_called()


class TestReactions:
    @pytest.mark.asyncio
    async def test_reacts_with_robot_on_start(self, bot):
        from src.bot import TranscriptBot

        bot._startup_ms = 0
        event = make_audio_event(sender="@user:example.com")
        room = make_room()

        send_response = MagicMock()
        send_response.event_id = "$reaction1"
        bot.client.room_send.return_value = send_response
        bot.client.download.return_value = MagicMock(body=b"fake audio", content_type="audio/ogg")

        await bot.on_audio_message(room, event)

        # First room_send call should be the robot reaction
        first_call = bot.client.room_send.call_args_list[0]
        assert first_call[0][0] == "!room:example.com"
        assert first_call[0][1] == "m.reaction"
        content = first_call[0][2]
        assert content["m.relates_to"]["key"] == "\U0001f916"

    @pytest.mark.asyncio
    async def test_removes_robot_reaction_on_success(self, bot):
        bot._startup_ms = 0
        event = make_audio_event(sender="@user:example.com")
        room = make_room()

        send_response = MagicMock()
        send_response.event_id = "$reaction1"
        bot.client.room_send.return_value = send_response
        bot.client.download.return_value = MagicMock(body=b"fake audio", content_type="audio/ogg")

        await bot.on_audio_message(room, event)

        bot.client.room_redact.assert_called_once_with("!room:example.com", "$reaction1")


class TestReplyFormat:
    @pytest.mark.asyncio
    async def test_reply_contains_transcription_prefix(self, bot):
        bot._startup_ms = 0
        event = make_audio_event(sender="@user:example.com")
        room = make_room()

        send_response = MagicMock()
        send_response.event_id = "$reaction1"
        bot.client.room_send.return_value = send_response
        bot.client.download.return_value = MagicMock(body=b"fake audio", content_type="audio/ogg")

        await bot.on_audio_message(room, event)

        # Find the m.room.message send call (not the reaction)
        message_calls = [
            c for c in bot.client.room_send.call_args_list
            if c[0][1] == "m.room.message"
        ]
        assert len(message_calls) == 1
        content = message_calls[0][0][2]
        assert content["body"] == "Transcription:\n\nHola mundo"
        assert content["m.relates_to"]["m.in_reply_to"]["event_id"] == "$event1"

    @pytest.mark.asyncio
    async def test_replies_no_speech_for_empty_result(self, bot, mock_transcriber):
        bot._startup_ms = 0
        mock_transcriber.transcribe.return_value = ""
        event = make_audio_event(sender="@user:example.com")
        room = make_room()

        send_response = MagicMock()
        send_response.event_id = "$reaction1"
        bot.client.room_send.return_value = send_response
        bot.client.download.return_value = MagicMock(body=b"fake audio", content_type="audio/ogg")

        await bot.on_audio_message(room, event)

        message_calls = [
            c for c in bot.client.room_send.call_args_list
            if c[0][1] == "m.room.message"
        ]
        assert len(message_calls) == 1
        assert message_calls[0][0][2]["body"] == "No speech detected."


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_reacts_with_x_on_transcription_failure(self, bot, mock_transcriber):
        bot._startup_ms = 0
        mock_transcriber.transcribe.side_effect = RuntimeError("model error")
        event = make_audio_event(sender="@user:example.com")
        room = make_room()

        send_response = MagicMock()
        send_response.event_id = "$reaction1"
        bot.client.room_send.return_value = send_response
        bot.client.download.return_value = MagicMock(body=b"fake audio", content_type="audio/ogg")

        await bot.on_audio_message(room, event)

        # Should redact robot reaction
        bot.client.room_redact.assert_called()
        # Should send X reaction
        reaction_calls = [
            c for c in bot.client.room_send.call_args_list
            if c[0][1] == "m.reaction" and c[0][2]["m.relates_to"]["key"] == "\u274c"
        ]
        assert len(reaction_calls) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && python -m pytest tests/test_bot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bot'`

**Step 3: Implement `src/bot.py`**

```python
import asyncio
import logging
import os
import tempfile
import time

from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    JoinResponse,
    LoginResponse,
    RoomMessageAudio,
    RoomMessageVideo,
)
from nio.crypto.attachments import decrypt_attachment

logger = logging.getLogger(__name__)


class TranscriptBot:
    def __init__(
        self,
        homeserver: str,
        user_id: str,
        password: str,
        store_path: str,
        transcriber,
    ):
        config = AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True,
        )
        self.client = AsyncClient(
            homeserver,
            user_id,
            store_path=store_path,
            config=config,
        )
        self.password = password
        self.transcriber = transcriber
        self._startup_ms = int(time.time() * 1000)

    async def start(self):
        response = await self.client.login(self.password)
        if not isinstance(response, LoginResponse):
            logger.error("Login failed: %s", response)
            raise SystemExit(1)

        logger.info("Logged in as %s", self.client.user_id)

        self.client.add_event_callback(self.on_audio_message, RoomMessageAudio)
        self.client.add_event_callback(self.on_audio_message, RoomMessageVideo)
        self.client.add_event_callback(self._on_invite, InviteMemberEvent)

        # Trust all known devices (TOFU)
        self._trust_all_devices()

        await self.client.sync_forever(timeout=30000)

    async def stop(self):
        await self.client.close()

    def _trust_all_devices(self):
        for user_id in self.client.device_store.users:
            for device in self.client.device_store.active_user_devices(user_id):
                if not self.client.device_store.is_device_verified(device):
                    self.client.verify_device(device)

    async def _on_invite(self, room, event):
        result = await self.client.join(room.room_id)
        if isinstance(result, JoinResponse):
            logger.info("Joined room %s", room.room_id)
        else:
            logger.error("Failed to join %s: %s", room.room_id, result)

    async def on_audio_message(self, room, event):
        # Ignore own messages
        if event.sender == self.client.user_id:
            return

        # Ignore messages from before startup
        if event.server_timestamp < self._startup_ms:
            return

        logger.info(
            "Audio from %s in %s (%s)",
            event.sender,
            room.room_id,
            event.event_id,
        )

        # React with robot emoji
        reaction_event_id = await self._react(room.room_id, event.event_id, "\U0001f916")

        try:
            audio_path = await self._download_media(event)
            if not audio_path:
                await self._remove_reaction(room.room_id, reaction_event_id)
                return

            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None, self.transcriber.transcribe, audio_path
                )
            finally:
                os.unlink(audio_path)

            await self._remove_reaction(room.room_id, reaction_event_id)

            if not text or not text.strip():
                await self._reply(room.room_id, event.event_id, "No speech detected.")
            else:
                await self._reply(
                    room.room_id,
                    event.event_id,
                    f"Transcription:\n\n{text}",
                )

        except Exception:
            logger.exception("Transcription failed for %s", event.event_id)
            await self._remove_reaction(room.room_id, reaction_event_id)
            await self._react(room.room_id, event.event_id, "\u274c")

    async def _download_media(self, event):
        content = event.source.get("content", {})

        if "file" in content:
            mxc = content["file"]["url"]
        else:
            mxc = event.url

        response = await self.client.download(mxc)

        if not hasattr(response, "body"):
            logger.error("Failed to download media: %s", response)
            return None

        data = response.body

        if "file" in content:
            data = decrypt_attachment(
                data,
                content["file"]["key"]["k"],
                content["file"]["hashes"]["sha256"],
                content["file"]["iv"],
            )

        body = content.get("body", "audio.ogg")
        suffix = os.path.splitext(body)[1] or ".ogg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return tmp.name

    async def _react(self, room_id, event_id, emoji):
        content = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": emoji,
            }
        }
        response = await self.client.room_send(room_id, "m.reaction", content)
        if hasattr(response, "event_id"):
            return response.event_id
        return None

    async def _remove_reaction(self, room_id, reaction_event_id):
        if reaction_event_id:
            await self.client.room_redact(room_id, reaction_event_id)

    async def _reply(self, room_id, event_id, text):
        content = {
            "msgtype": "m.text",
            "body": text,
            "m.relates_to": {
                "m.in_reply_to": {
                    "event_id": event_id,
                }
            },
        }
        await self.client.room_send(room_id, "m.room.message", content)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && python -m pytest tests/test_bot.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: add bot module with event handling, reactions, and replies"
```

---

### Task 4: Main Entry Point

**Files:**
- Create: `src/main.py`

**Step 1: Create `src/main.py`**

```python
import asyncio
import logging
import os
import signal
import sys


def main():
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load config
    homeserver = os.environ.get("MATRIX_HOMESERVER")
    user_id = os.environ.get("MATRIX_USER_ID")
    password = os.environ.get("MATRIX_PASSWORD")
    whisper_model = os.environ.get("WHISPER_MODEL", "large-v3")
    whisper_language = os.environ.get("WHISPER_LANGUAGE", "es")
    model_dir = os.environ.get("WHISPER_MODEL_DIR", "/app/models")
    store_path = os.environ.get("STORE_PATH", "/app/store")

    if not all([homeserver, user_id, password]):
        logger.error(
            "MATRIX_HOMESERVER, MATRIX_USER_ID, and MATRIX_PASSWORD are required"
        )
        sys.exit(1)

    # Initialize transcriber
    from src.transcriber import Transcriber

    logger.info("Initializing transcriber...")
    transcriber = Transcriber(
        model_name=whisper_model,
        language=whisper_language,
        model_dir=model_dir,
    )

    # Initialize bot
    from src.bot import TranscriptBot

    bot = TranscriptBot(
        homeserver=homeserver,
        user_id=user_id,
        password=password,
        store_path=store_path,
        transcriber=transcriber,
    )

    # Graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown(sig):
        logger.info("Received %s, shutting down...", sig.name)
        loop.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(bot.start())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.run_until_complete(bot.stop())
        loop.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: add main entry point with config and graceful shutdown"
```

---

### Task 5: Docker & Deployment

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libolm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

RUN mkdir -p /app/store /app/models

CMD ["python", "-m", "src.main"]
```

**Step 2: Create `docker-compose.yml`**

```yaml
services:
  transcript-bot:
    build: .
    restart: unless-stopped
    volumes:
      - ./data/store:/app/store
      - ./data/models:/app/models
    env_file: .env
```

**Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and docker-compose for deployment"
```

---

### Task 6: Build Verification

**Step 1: Run all tests**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify Docker image builds**

Run: `cd /Users/mattog/dev/matto/matrix-transcript-bot && docker build -t matrix-transcript-bot .`
Expected: Build completes successfully

**Step 3: Commit any fixes if needed**

If tests or build failed, fix issues and commit.
