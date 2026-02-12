import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock nio modules before importing bot so we don't need matrix-nio installed
_nio_mock = MagicMock()
sys.modules.setdefault("nio", _nio_mock)
sys.modules.setdefault("nio.crypto", MagicMock())
sys.modules.setdefault("nio.crypto.attachments", MagicMock())


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
    async def test_reply_contains_transcribed_text(self, bot):
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
        assert content["body"] == "Hola mundo"
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
