"""
The single command surface and persona for Agnes.

There used to be per-group "modes" (fun / expense / study / …) that changed the
persona and which commands were visible. That split features across groups for
no real benefit, so it's gone: every group gets every feature at the same level,
with one persona. This module is now just the command catalogue.

`COMMAND_CATALOGUE` is the single source of truth — it drives both the Telegram
"/" menu and the in-chat "List commands" screen, so the two can never drift.
"""

from __future__ import annotations

import logging

from telegram import BotCommand, BotCommandScopeChat

logger = logging.getLogger("student_claw.bot.modes")


class Cmd:
    """
    One command.

    `picker` controls whether it appears in Telegram's "/" autocomplete. Only a
    handful do — a long picker list is unusable on mobile. Everything else
    still works when typed and is listed in full by /commands.
    """

    __slots__ = ("name", "blurb", "section", "admin_only", "usage", "picker")

    def __init__(
        self,
        name: str,
        blurb: str,
        section: str,
        *,
        admin_only: bool = False,
        usage: str = "",
        picker: bool = False,
    ) -> None:
        self.name = name
        self.blurb = blurb
        self.section = section
        self.admin_only = admin_only
        self.usage = usage
        self.picker = picker


# Section order is the order they appear in the "List commands" screen.
SECTIONS = ["Everyday", "Money", "Going out", "Setup"]

COMMAND_CATALOGUE: list[Cmd] = [
    # ── Everyday ──
    Cmd("mainmenu", "Settings, memory & full command list", "Everyday", picker=True),
    Cmd("summary", "Catch me up on the chat", "Everyday", picker=True),
    Cmd("exams", "Upcoming exams & deadlines", "Everyday", picker=True),
    Cmd("news", "Today's news, summarised", "Everyday", picker=True),
    Cmd("roast", "Roast someone (with love)", "Everyday", usage="/roast @handle", picker=True),
    Cmd("ask", "Ask Agnes anything", "Everyday", usage="/ask what's the plan for friday?"),
    Cmd("joke", "An actually funny joke", "Everyday"),
    Cmd("commands", "Show everything I can do", "Everyday"),
    # ── Money ──
    Cmd("splitbill", "Split a receipt photo", "Money",
        usage="reply to a receipt with /splitbill", picker=True),
    Cmd("splitexpense", "Split a cost with no receipt", "Money",
        usage="/splitexpense i paid $20 for the photo booth with raja and madhu"),
    Cmd("settle_up", "Who owes whom + mark payments", "Money"),
    Cmd("bill", "Show the open bill", "Money"),
    Cmd("paynow", "Set your PayNow number", "Money", usage="/paynow 91234567"),
    # ── Going out ──
    Cmd("meetpoint", "Find a place to meet everyone halfway", "Going out"),
    # ── Setup ──
    Cmd("sync", "Re-index anything I missed", "Setup"),
    Cmd("activate", "Wake me up in this group", "Setup", admin_only=True),
    Cmd("deactivate", "Send me to sleep", "Setup", admin_only=True),
]

# Published to Telegram's "/" autocomplete. Deliberately short — a 15-item
# picker is unusable on a phone. Everything else still works when typed and is
# listed by /commands and in the menu.
_MENU_COMMANDS = [
    BotCommand(c.name, c.blurb)
    for c in COMMAND_CATALOGUE
    if c.picker and not c.admin_only
]


def command_catalogue_text(privileged: bool = False) -> str:
    """Telegram-HTML listing of every command, grouped by section."""
    out: list[str] = ["📖 <b>What I can do</b>\n"]
    for section in SECTIONS:
        rows = [
            c
            for c in COMMAND_CATALOGUE
            if c.section == section and (privileged or not c.admin_only)
        ]
        if not rows:
            continue
        out.append(f"\n<b>{section}</b>")
        for c in rows:
            line = f"/{c.name} — {c.blurb}"
            if c.usage:
                line += f"\n   <i>e.g. {c.usage}</i>"
            out.append(line)
    out.append(
        "\n\nYou can also just talk to me normally — reply to one of my "
        "messages or mention me and I'll answer."
    )
    return "\n".join(out)


# Persona prepended to the agent system prompt. One group, one vibe.
PERSONA = (
    "PERSONA: This is a friend group and you're one of them. Be playful, "
    "quick-witted and warm — banter back, hype people up, drop the occasional "
    "emoji. You're equally happy splitting a bill, explaining a concept before "
    "an exam, or planning where to meet. Never sound like customer support."
)


def persona_for(_mode: str | None = None) -> str:
    """Kept for call-site compatibility; the persona no longer varies."""
    return PERSONA


async def apply_chat_commands(bot, chat_id: int, _mode: str | None = None) -> None:
    """
    Remove any chat-specific command list so this chat falls back to the single
    global one.

    Telegram resolves command scopes by precedence (chat > all_group_chats >
    default) and stores them server-side **indefinitely**. The old per-mode
    system wrote a chat-scoped list into every group, which then kept shadowing
    the global list forever — that's why groups still showed stale commands
    after the catalogue changed. Deleting the chat scope is what actually
    unifies the menu; `set_my_commands` here would just re-create the problem.
    """
    try:
        await bot.delete_my_commands(scope=BotCommandScopeChat(chat_id))
        logger.info("Cleared chat-scoped commands for %s (using global list).", chat_id)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not clear chat commands for %s: %s", chat_id, exc)


def default_commands() -> list[BotCommand]:
    """Global command list (used at startup)."""
    return list(_MENU_COMMANDS)
