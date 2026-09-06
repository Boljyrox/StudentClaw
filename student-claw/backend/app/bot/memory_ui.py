"""
Memory screens for the /mainmenu tree, plus natural-language forget requests.

Everything the group has asked Agnes to remember is listed as a numbered set of
facts. From there you can wipe the lot, or drill into a single entry to edit or
delete it. Deletion always asks first — including when it's triggered
conversationally ("/ask delete the memory about the DDW exam"), which resolves
candidate memories and shows the same confirmation buttons.
"""

from __future__ import annotations

import html
import logging
import re

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.ai.config import MEMORY_RETENTION_DAYS
from app.bot import services

logger = logging.getLogger("student_claw.bot.memory_ui")

_AWAIT_KEY = "memory_await"
# Candidate ids for a pending natural-language forget confirmation.
_PENDING_FORGET_KEY = "memory_pending_forget"

# Conversational "forget that" intents, caught before the message reaches the
# agent so deletion can never happen without an explicit confirmation.
_FORGET_INTENT = re.compile(
    r"\b(?:delete|forget|remove|clear|wipe)\b[^.?!]*\bmemor(?:y|ies)\b"
    r"|\bmemor(?:y|ies)\b[^.?!]*\b(?:delete|forget|remove)\b",
    re.IGNORECASE,
)
# Strip the command words to leave the topic ("... about the DDW exam").
_TOPIC_STRIP = re.compile(
    r"\b(?:please|can you|could you|delete|forget|remove|clear|wipe|the|my|our|"
    r"a|an|all|every|memor(?:y|ies)|about|regarding|concerning|of|on|that|this|"
    r"which|says?|saying|entry|entries|item|items)\b",
    re.IGNORECASE,
)
_ALL_MEMORY = re.compile(
    r"\b(?:all|every|everything|entire)\b.{0,20}\bmemor(?:y|ies)\b"
    r"|\bmemor(?:y|ies)\b.{0,20}\b(?:all|everything)\b",
    re.IGNORECASE,
)


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def _btn(label: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=data)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def memory_text(items: list[services.MemoryItem]) -> str:
    """The whole memory as one numbered block."""
    if not items:
        return (
            "🧠 <b>Memory</b>\n\n"
            "<i>I haven't saved anything yet.</i>\n\n"
            "Tell me things like \"remember that our exam is on 12 Nov\" and "
            "they'll show up here."
        )
    lines = [f"🧠 <b>Memory</b> — {len(items)} saved\n"]
    for i, item in enumerate(items, start=1):
        who = f" <i>— {_esc(item.created_by_name)}</i>" if item.created_by_name else ""
        lines.append(f"<b>{i})</b> {_esc(item.content)}{who}")
    lines.append(f"\n<i>Memories older than {MEMORY_RETENTION_DAYS} days clear themselves.</i>")
    return "\n".join(lines)


def memory_keyboard(items: list[services.MemoryItem]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if items:
        rows.append([_btn("✏️ Edit / delete one", "sc|mem|pick")])
        rows.append([_btn("🗑 Delete everything", "sc|mem|wipe")])
    rows.append([_btn("🔙 Back", "sc|main")])
    return InlineKeyboardMarkup(rows)


def picker_keyboard(items: list[services.MemoryItem]) -> InlineKeyboardMarkup:
    """One row per memory — numbered to match the listing."""
    rows = [
        [_btn(f"{i}) {item.content[:40]}", f"sc|mem|one|{item.id}")]
        for i, item in enumerate(items[:20], start=1)
    ]
    rows.append([_btn("🔙 Back", "sc|mem")])
    return InlineKeyboardMarkup(rows)


def single_keyboard(memory_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                _btn("✏️ Edit", f"sc|mem|edit|{memory_id}"),
                _btn("🗑 Delete", f"sc|mem|del|{memory_id}"),
            ],
            [_btn("🔙 Back", "sc|mem|pick")],
        ]
    )


async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Render the full numbered memory list into the current menu message."""
    chat = update.effective_chat
    query = update.callback_query
    items = await services.list_memories(chat.id) or []
    body, markup = memory_text(items), memory_keyboard(items)
    if query:
        await query.edit_message_text(body, parse_mode="HTML", reply_markup=markup)
    else:
        await update.effective_message.reply_text(
            body, parse_mode="HTML", reply_markup=markup
        )


# ---------------------------------------------------------------------------
# Callback handling (routed from the `sc|mem|…` branch)
# ---------------------------------------------------------------------------
async def handle_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE, parts: list[str]
) -> None:
    query = update.callback_query
    chat = update.effective_chat
    sub = parts[2] if len(parts) > 2 else None
    arg = parts[3] if len(parts) > 3 else None

    if sub is None:
        await query.answer()
        await show_memory(update, context)
        return

    if sub == "pick":
        await query.answer()
        items = await services.list_memories(chat.id) or []
        await query.edit_message_text(
            "Which one?" if items else "Nothing saved yet.",
            parse_mode="HTML",
            reply_markup=picker_keyboard(items),
        )
        return

    if sub == "one" and arg:
        await query.answer()
        items = await services.list_memories(chat.id) or []
        item = next((m for m in items if m.id == arg), None)
        if item is None:
            await show_memory(update, context)
            return
        await query.edit_message_text(
            f"🧠 {_esc(item.content)}", parse_mode="HTML",
            reply_markup=single_keyboard(item.id),
        )
        return

    if sub == "del" and arg:
        await query.answer()
        items = await services.list_memories(chat.id) or []
        item = next((m for m in items if m.id == arg), None)
        if item is None:
            await show_memory(update, context)
            return
        await query.edit_message_text(
            f"Delete this memory?\n\n🧠 <i>{_esc(item.content)}</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [[_btn("✅ Delete", f"sc|mem|delok|{arg}"), _btn("↩️ Keep", "sc|mem|pick")]]
            ),
        )
        return

    if sub == "delok" and arg:
        await services.delete_memory(chat.id, arg)
        await query.answer("Deleted")
        await show_memory(update, context)
        return

    if sub == "edit" and arg:
        await query.answer()
        await _prompt(update, context, {"action": "edit", "memory_id": arg},
                      "✏️ Reply with the corrected version of this memory:")
        return

    if sub == "wipe":
        await query.answer()
        items = await services.list_memories(chat.id) or []
        await query.edit_message_text(
            f"Delete <b>all {len(items)}</b> memories? This can't be undone.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [[_btn("✅ Delete all", "sc|mem|wipeok"), _btn("↩️ Cancel", "sc|mem")]]
            ),
        )
        return

    if sub == "wipeok":
        removed = await services.clear_memories(chat.id)
        await query.answer(f"Cleared {removed or 0}")
        await show_memory(update, context)
        return

    # Confirmation for a natural-language forget request. The candidate ids
    # live in chat_data because callback_data is capped at 64 bytes and UUIDs
    # blow through that after the first one.
    if sub == "nlbulk":
        ids = context.chat_data.pop(_PENDING_FORGET_KEY, [])
        removed = 0
        for mid in ids:
            if await services.delete_memory(chat.id, mid):
                removed += 1
        await query.answer(f"Forgot {removed}")
        await query.edit_message_text(f"🗑 Forgotten ({removed}).", parse_mode="HTML")
        return

    if sub == "cancel":
        await query.answer("Kept")
        await query.edit_message_text("👍 Left everything as it was.")
        return

    await query.answer()


async def _prompt(
    update: Update, context: ContextTypes.DEFAULT_TYPE, payload: dict, text: str
) -> None:
    chat = update.effective_chat
    user = update.effective_user
    sent = await context.bot.send_message(
        chat.id, text, parse_mode="HTML", reply_markup=ForceReply(selective=True)
    )
    context.chat_data[_AWAIT_KEY] = {
        **payload,
        "user_id": user.id if user else None,
        "prompt_id": sent.message_id,
    }


async def maybe_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Consume a pending memory ForceReply. Returns True when handled."""
    pending = context.chat_data.get(_AWAIT_KEY)
    msg = update.effective_message
    user = update.effective_user
    if not pending or not msg or not msg.text:
        return False
    if pending.get("user_id") and user and user.id != pending["user_id"]:
        return False
    if msg.reply_to_message and msg.reply_to_message.message_id != pending.get("prompt_id"):
        return False

    context.chat_data.pop(_AWAIT_KEY, None)
    if pending.get("action") == "edit":
        ok = await services.update_memory(
            update.effective_chat.id, pending["memory_id"], msg.text.strip()
        )
        await msg.reply_text("✅ Updated." if ok else "⚠️ Couldn't update that one.")
        return True
    return False


# ---------------------------------------------------------------------------
# Natural-language "forget X"
# ---------------------------------------------------------------------------
def looks_like_forget_request(text: str) -> bool:
    return bool(_FORGET_INTENT.search(text or ""))


def _topic_from(text: str) -> str:
    return re.sub(r"\s+", " ", _TOPIC_STRIP.sub(" ", text or "")).strip(" .,!?")


async def handle_forget_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> bool:
    """
    Resolve a conversational delete request to specific memories and ask for
    confirmation. Returns True when the request was handled here (so the agent
    never sees it). Nothing is deleted without a tap.
    """
    chat = update.effective_chat
    msg = update.effective_message
    items = await services.list_memories(chat.id) or []

    if not items:
        await msg.reply_text("I haven't saved any memories yet — nothing to forget.")
        return True

    # "delete all my memories" → the wipe confirmation.
    if _ALL_MEMORY.search(text):
        await msg.reply_text(
            f"Delete <b>all {len(items)}</b> memories? This can't be undone.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [[_btn("✅ Delete all", "sc|mem|wipeok"), _btn("↩️ Cancel", "sc|mem|cancel")]]
            ),
        )
        return True

    topic = _topic_from(text)
    matches = await services.search_memories(chat.id, topic) if topic else []
    if not matches:
        await msg.reply_text(
            f"I couldn't find a memory about <b>{_esc(topic or 'that')}</b>. "
            "Open /mainmenu → 🧠 Memory to see everything I've saved.",
            parse_mode="HTML",
        )
        return True

    listing = "\n".join(f"• <i>{_esc(m.content)}</i>" for m in matches)
    context.chat_data[_PENDING_FORGET_KEY] = [m.id for m in matches]
    plural = "these" if len(matches) > 1 else "this"
    await msg.reply_text(
        f"Forget {plural}?\n\n{listing}",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    _btn(f"✅ Forget ({len(matches)})", "sc|mem|nlbulk"),
                    _btn("↩️ Keep", "sc|mem|cancel"),
                ]
            ]
        ),
    )
    return True
