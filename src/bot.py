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
