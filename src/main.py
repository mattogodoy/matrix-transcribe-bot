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
