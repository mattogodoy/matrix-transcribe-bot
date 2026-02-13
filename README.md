# matrix-transcript-bot

A self-hosted Matrix bot that transcribes audio messages into text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

When invited to a room, the bot listens for audio messages, transcribes them, and replies with the text. It also works via direct messages — forward an audio message to the bot and it replies with the transcript.

## How it works

1. Someone sends an audio message in a room the bot is in (or DMs the bot directly)
2. The bot reacts with a robot emoji to indicate it's working
3. The bot downloads the audio, transcribes it with faster-whisper
4. On success: removes the robot emoji and replies with the transcribed text
5. On failure: replaces the robot emoji with a red X

## Features

- Transcribes audio messages (`m.audio` and `m.video`) in any room it's invited to
- Supports end-to-end encrypted rooms
- Uses faster-whisper with the `large-v3` model for high-accuracy transcription
- Runs on CPU (no GPU required)
- Docker deployment

## Setup

### 1. Create a bot account

Create a Matrix account for the bot on your homeserver (e.g., `@transcribe-bot:example.com`).

### 2. Configure

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

```
MATRIX_HOMESERVER=https://matrix.example.com
MATRIX_USER_ID=@transcribe-bot:example.com
MATRIX_PASSWORD=your-bot-password
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=es
WHISPER_CPU_THREADS=0
LOG_LEVEL=INFO
```

### 3. Run

```bash
docker compose up -d
```

The bot will download the whisper model on first run (~3GB, cached in `data/models/`).

### 4. Invite

Invite the bot to any room — it will auto-join and start transcribing audio messages.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MATRIX_HOMESERVER` | (required) | Homeserver URL |
| `MATRIX_USER_ID` | (required) | Bot user ID |
| `MATRIX_PASSWORD` | (required) | Bot account password |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model name |
| `WHISPER_LANGUAGE` | `es` | Transcription language code |
| `WHISPER_CPU_THREADS` | `0` | Number of CPU threads for transcription (0 = all cores) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Data persistence

Two directories are mounted as Docker volumes:

- `data/store/` — Matrix encryption keys and sync state (do not delete, or the bot loses its encryption identity)
- `data/models/` — Cached whisper model (can be deleted, will re-download on next start)
