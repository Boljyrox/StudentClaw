"""
Telegram bot application lifecycle (Module 2).

Two run modes (blueprint §7.2):

  * Webhook mode (production): the PTB Application is initialised and started
    alongside FastAPI; Telegram POSTs updates to our webhook route, which feeds
    them into `application.process_update`. See app/bot/webhook.py and the
    FastAPI lifespan wiring in main.py.

  * Polling mode (local dev): `python -m app.bot.bot` runs long-polling with no
    public URL required.

The Application is a module-level singleton so the webhook route and the
FastAPI lifespan share one instance.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, ApplicationBuilder

from app.bot.config import get_settings
from app.bot.handlers import register_handlers

logger = logging.getLogger("student_claw.bot")

_application: Application | None = None

# Default commands shown in Telegram's "/" menu (chat-scoped menus override
# these per group mode — see app.bot.modes).
_BOT_COMMANDS = [
    ("ask", "Ask Agnes anything"),
    ("summary", "Catch me up on the chat"),
    ("news", "Today's news, summarised"),
    ("joke", "An actually funny joke"),
    ("roast", "Roast someone (with love)"),
    ("exams", "Upcoming exams & deadlines"),
    ("splitbill", "Split a receipt photo"),
    ("paynow", "Set your PayNow number"),
    ("sc", "Open the Agnes menu"),
]


async def _post_init(application: Application) -> None:
    """Register the command menu + start the Redis notification listener."""
    try:
        await application.bot.set_my_commands(_BOT_COMMANDS)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not set bot command menu: %s", exc)

    from app.bot.notifications import start_notification_listener

    start_notification_listener(application)


async def _post_shutdown(application: Application) -> None:
    from app.bot.notifications import stop_notification_listener

    await stop_notification_listener()


def build_application() -> Application:
    """Construct (once) and return the shared PTB Application singleton."""
    global _application
    if _application is not None:
        return _application

    settings = get_settings()
    application = (
        ApplicationBuilder()
        .token(settings.bot_token)
        .concurrent_updates(True)  # process updates concurrently on the loop
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    register_handlers(application)
    _application = application
    logger.info("PTB Application built and handlers registered.")
    return application


def get_application() -> Application:
    """Return the already-built Application (build it lazily if needed)."""
    return _application or build_application()


# ---------------------------------------------------------------------------
# Webhook-mode lifecycle (called from FastAPI lifespan)
# ---------------------------------------------------------------------------
async def start_webhook_application() -> Application:
    """
    Initialise and start the Application for webhook mode, and register the
    webhook URL with Telegram. Does NOT start an updater/poller — updates are
    delivered via the FastAPI route.
    """
    settings = get_settings()
    application = build_application()

    await application.initialize()
    await application.start()

    if settings.webhook_base_url:
        await application.bot.set_webhook(
            url=settings.webhook_url,
            secret_token=settings.webhook_secret,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
        )
        logger.info("Webhook registered at %s", settings.webhook_url)
    else:
        logger.warning(
            "TELEGRAM_WEBHOOK_BASE_URL not set; webhook not registered. "
            "Set it in production or run in polling mode for local dev."
        )
    return application


async def stop_webhook_application() -> None:
    """Gracefully stop the Application (called on FastAPI shutdown)."""
    global _application
    if _application is None:
        return
    try:
        await _application.bot.delete_webhook(drop_pending_updates=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to delete webhook on shutdown: %s", exc)
    await _application.stop()
    await _application.shutdown()
    logger.info("PTB Application stopped.")
    _application = None


# ---------------------------------------------------------------------------
# Polling-mode entrypoint (local development)
# ---------------------------------------------------------------------------
def run_polling() -> None:
    """Blocking long-poll loop for local development."""
    application = build_application()
    logger.info("Starting bot in POLLING mode (development).")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        run_polling()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Bot polling interrupted; shutting down.")
