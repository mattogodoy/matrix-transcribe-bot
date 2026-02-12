# Matrix Transcript Bot — Design Document

**Date:** 2026-02-12
**Status:** Approved

## Overview

A self-hosted Matrix bot that transcribes audio messages into text using faster-whisper. When invited to a room, the bot detects audio messages, transcribes them, and replies with the text. Also works via DMs — forward an audio message to the bot and it replies with the transcript.

## Decisions

- **Language:** Python
- **Matrix library:** `matrix-nio[e2e]` (async variant)
- **Transcription:** `faster-whisper` with `large-v3` model
- **Compute:** CPU only, `int8` compute type
- **Transcription language:** Spanish (`es`), hardcoded
- **Architecture:** Single monolithic async process
- **Deployment:** Docker container via docker-compose

## Architecture

```
┌─────────────────────────────────────────┐
│           matrix-transcript-bot         │
│                                         │
│  ┌───────────┐    ┌──────────────────┐  │
│  │ MatrixBot  │───>│  Transcriber     │  │
│  │ (nio)      │    │  (faster-whisper) │  │
│  │            │<───│                  │  │
│  └───────────┘    └──────────────────┘  │
│       │                    │            │
│       v                    v            │
│  Matrix Server       Temp audio files   │
│  (homeserver)        (cleaned up after) │
└─────────────────────────────────────────┘
```

### Components

1. **`bot.py`** — Matrix client using `matrix-nio` (async). Handles login, session management, auto-joining rooms, listening for audio events, downloading media, sending replies and reactions.

2. **`transcriber.py`** — Wrapper around faster-whisper. Loads the `large-v3` model once at startup (CPU, `int8`). Accepts an audio file path, returns transcribed text. Runs in a thread pool executor to avoid blocking the async event loop.

3. **`main.py`** — Entry point. Loads configuration from environment variables, initializes the transcriber and bot, starts the event loop.

## Event Handling & Message Flow

### Trigger

The bot processes any `m.room.message` event where `msgtype` is `m.audio` or `m.video`, in any room it's a member of (including DMs). Both room messages and forwarded DM audio use the same code path.

### Flow

1. Audio message detected
2. Bot reacts with 🤖 to the message (signals "working on it")
3. Bot downloads the media file to a temp file
4. Bot transcribes the audio via faster-whisper (in thread pool)
5. **On success:** Remove 🤖 reaction, reply in-thread with transcript
6. **On failure:** Remove 🤖 reaction, react with ❌

### Reply Format

```
Transcription:

[transcribed text here]
```

Plain text, threaded reply (`m.relates_to` with `m.in_reply_to`).

### Event Filtering

- Only process `m.audio` and `m.video` message types
- Ignore messages from the bot itself (prevent loops)
- Ignore messages older than bot startup (don't re-process history)

## Encryption & Media

- E2EE supported via `matrix-nio[e2e]` with TOFU (trust on first use) device trust
- Encryption keys stored persistently in a mounted volume
- Encrypted media decrypted transparently by `matrix-nio`
- Unencrypted media downloaded directly via `mxc://` URL
- Audio formats (ogg, m4a, wav, mp3) handled by faster-whisper via ffmpeg
- Temp files cleaned up immediately after transcription using Python's `tempfile`

## Docker & Deployment

### Dockerfile

- Base: `python:3.12-slim`
- Installs `ffmpeg` via apt
- Installs Python deps: `matrix-nio[e2e]`, `faster-whisper`
- Model downloaded on first run, cached in volume

### Docker Compose

```yaml
services:
  transcript-bot:
    build: .
    restart: unless-stopped
    volumes:
      - ./data/store:/app/store        # nio encryption keys + sync token
      - ./data/models:/app/models      # faster-whisper model cache
    env_file: .env
```

### Persistent Volumes

- `store/` — matrix-nio encryption keys, device info, sync token
- `models/` — cached whisper model

### Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MATRIX_HOMESERVER` | (required) | Homeserver URL |
| `MATRIX_USER_ID` | (required) | Bot user ID |
| `MATRIX_PASSWORD` | (required) | Bot account password |
| `WHISPER_MODEL` | `large-v3` | Whisper model name |
| `WHISPER_LANGUAGE` | `es` | Transcription language |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
matrix-transcript-bot/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── bot.py
│   └── transcriber.py
└── data/                # Mounted volumes (gitignored)
    ├── store/
    └── models/
```

## Error Handling

| Error | Behavior |
|-------|----------|
| Media download fails | Log error, skip (no reaction, no reply) |
| Transcription fails | Remove 🤖, react with ❌, log error |
| Empty transcription result | Reply with "No speech detected." |
| Bot can't join a room | Log error, ignore the invite |
| Matrix send fails | Log error, retry once |
| Model loading fails at startup | Exit with clear error message |

## Testing

- Unit tests for `transcriber.py` with a short sample audio file
- Unit tests for event filtering logic (correct msgtype, ignore self, ignore old messages)
- Integration test: mock `matrix-nio` client to verify full flow (receive event → download → transcribe → reply + reactions)

## Logging

- Python `logging` module
- Level configurable via `LOG_LEVEL` env var
- Logs: room events received, transcription start/finish, errors

## Graceful Shutdown

- Handle `SIGTERM`/`SIGINT` to close Matrix client session cleanly
- Ensure temp files are cleaned up on shutdown
