"""
Multi-Mode definitions: per-mode command menus (for chat-scoped
set_my_commands) and AI personas injected into the agent's system prompt.

Agnes is a group-chat companion first; the old project-management behaviour
lives on only as the "projects" legacy mode.
"""

from __future__ import annotations

import logging

from telegram import BotCommand, BotCommandScopeChat

logger = logging.getLogger("student_claw.bot.modes")

MODE_LABELS: dict[str, str] = {
    "fun": "😎 Friends / Chill",
    "expense": "💸 Bills & Makan",
    "study": "📚 Study & Exams",
    "event": "📅 Trips & Events",
    "general": "💡 A bit of everything",
    "projects": "🗂 Projects (legacy)",
}
MODE_ORDER = ["fun", "expense", "study", "event", "general", "projects"]

# Every mode gets the full companion toolkit — modes only change the vibe and
# which extras are surfaced first in the "/" menu.
_COMMON = [
    BotCommand("ask", "Ask Agnes anything"),
    BotCommand("summary", "Catch me up on the chat"),
    BotCommand("news", "Today's news, summarised"),
    BotCommand("joke", "An actually funny joke"),
    BotCommand("roast", "Roast someone (with love)"),
    BotCommand("exams", "Upcoming exams & deadlines"),
    BotCommand("splitbill", "Split a receipt photo"),
    BotCommand("paynow", "Set your PayNow number"),
    BotCommand("sc", "Open the Agnes menu"),
]

_EXPENSE_EXTRAS = [
    BotCommand("bill", "Show the open bill"),
    BotCommand("add_expense", "Log an expense: /add_expense 15 pizza"),
    BotCommand("list_expenses", "List recent expenses"),
    BotCommand("settle_up", "Who owes whom"),
]

# Commands shown in each chat's "/" menu (handlers stay globally registered).
MODE_COMMANDS: dict[str, list[BotCommand]] = {
    "uninitialized": [BotCommand("init", "Initialise this group")],
    "fun": _COMMON,
    "expense": _COMMON + _EXPENSE_EXTRAS,
    "study": _COMMON,
    "event": _COMMON,
    "general": _COMMON,
    "projects": _COMMON,
}

# Persona prepended to the agent system prompt per mode.
PERSONA: dict[str, str] = {
    "fun": (
        "PERSONA: This is a friend group and you're one of them. Be playful, "
        "quick-witted and warm — banter back, hype people up, drop the "
        "occasional emoji. Never sound like customer support."
    ),
    "expense": (
        "PERSONA: This group mostly tracks shared food and bills. Be the "
        "sharp, funny friend who's good with money — surface amounts clearly, "
        "keep receipts straight, and tease (gently) whoever always forgets to "
        "pay up."
    ),
    "study": (
        "PERSONA: This group is studying together. Be the encouraging friend "
        "who actually knows the material — explain clearly, quiz when asked, "
        "keep exam dates front of mind, and keep morale up close to papers."
    ),
    "event": (
        "PERSONA: This group plans trips and outings. Be the organised friend "
        "with hype — pin down dates, venues and logistics, remember what was "
        "agreed, and keep the excitement going."
    ),
    "general": (
        "PERSONA: Be a clear, friendly all-rounder — answer anything, keep "
        "the tone light, and adapt to whatever the group is into today."
    ),
    "projects": (
        "PERSONA (legacy projects mode): You also help track tasks and "
        "deadlines for a team project. Stay friendly, not corporate."
    ),
}


def persona_for(mode: str | None) -> str:
    return PERSONA.get(mode or "fun", PERSONA["fun"])


async def apply_chat_commands(bot, chat_id: int, mode: str) -> None:
    """Swap the visible command menu for a single chat (best-effort)."""
    commands = MODE_COMMANDS.get(mode, _COMMON)
    try:
        await bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id))
        logger.info("Set chat-scoped commands for %s (mode=%s).", chat_id, mode)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not set chat commands for %s: %s", chat_id, exc)
