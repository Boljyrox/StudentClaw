"""
Telegram update handlers for Agnes — the group-chat companion.

Implements:
  * /start, /help, /init                   — onboarding
  * fun & useful commands                  — /ask /summary /news /joke /roast
                                             /exams /splitbill /paynow …
  * bot-added-to-group detection           — my_chat_member + new_chat_members
  * passive text/image/document listener   — RAG ingestion + @mention replies
  * legacy web-dashboard linkage           — /verify (hidden; kept for hackathons)

Handlers are deliberately thin: all DB work is delegated to app.bot.services
(and app.bot.billsplit for bills), each call being its own transaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import random
import re

from telegram import (
    Chat,
    ChatMemberUpdated,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationHandlerStop,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from app.ai import pipeline, queue, repository, storage
from app.ai.agent import run_agent
from app.bot import billsplit, events, modes, news, services
from app.bot.config import MAX_FILE_SIZE_BYTES
from app.database.models import ContentType, ProjectStatus

logger = logging.getLogger("student_claw.bot.handlers")

_SGT = ZoneInfo("Asia/Singapore")

# Texts shorter than this are logged but not embedded — "ok", "lol" and
# stickers-adjacent noise only pollute the vector index.
_MIN_EMBED_TEXT_CHARS = 12


# ---------------------------------------------------------------------------
# Message copy
# ---------------------------------------------------------------------------
PRIVACY_NOTICE = (
    "ℹ️ I remember this group's messages and files so I can recap chats, "
    "answer questions and split bills. An admin can wipe my memory anytime "
    "from the /sc menu."
)


def _welcome_text() -> str:
    return (
        "👋 <b>Hey, I'm Agnes!</b> Your group chat's resident AI.\n\n"
        "Things I'm good at:\n"
        "🧾 <b>/splitbill</b> — snap a receipt, tap what you ate, I handle "
        "GST, service charge and who PayNows whom\n"
        "📰 <b>/news</b> — today's headlines, summarised with opinions\n"
        "😂 <b>/joke</b> &amp; 🔥 <b>/roast</b> — entertainment on demand\n"
        "📋 <b>/summary</b> — catch up on what you missed\n"
        "📚 <b>/exams</b> — I remember your exam dates and timings\n"
        "💬 <b>/ask</b> anything — or just @mention me\n\n"
        f"{PRIVACY_NOTICE}"
    )


# ---------------------------------------------------------------------------
# Basic commands
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        # Ensure the group is registered (handles the case where the bot was
        # added before this handler shipped, or join events were missed).
        result = await services.get_or_create_project(chat.id, chat.title or "")
        await update.effective_message.reply_text(
            _welcome_text(), parse_mode="HTML"
        )
        if result.created:
            await update.effective_message.reply_text(
                "👉 Run /init to pick this group's vibe and become its admin."
            )
    else:
        await update.effective_message.reply_text(
            "👋 I'm Agnes — add me to a group chat with your friends and "
            "run /init there to get started."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "<b>Agnes commands</b>\n\n"
        "💬 /ask &lt;anything&gt; — questions, ideas, settle debates (or just @mention me)\n"
        "📋 /summary — recap what's been happening in the chat\n"
        "📰 /news [sg|world|tech|sport|business] — summarised headlines\n"
        "😂 /joke [topic] — an actually funny joke\n"
        "🔥 /roast &lt;name&gt; — playful roast, powered by chat receipts\n"
        "📚 /exams — upcoming exams &amp; deadlines (add: /exams add Math final 12 Aug 9am)\n\n"
        "🧾 <b>Bill splitting</b>\n"
        "/splitbill — reply to a receipt photo; tap items to claim them\n"
        "/bill — reopen the current bill's claim board\n"
        "/paynow &lt;mobile&gt; — save your PayNow so friends can pay you\n"
        "/add_expense 15 pizza · /list_expenses · /settle_up — quick ledger\n\n"
        "⚙️ /sc — full menu (sync files, mode, admin, memory)\n"
        "🕶 Legacy: /verify links the old web dashboard (hackathon feature)",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Group join detection
# ---------------------------------------------------------------------------
def _was_bot_added(cmu: ChatMemberUpdated, bot_id: int) -> bool:
    """True when this my_chat_member update represents the bot being added."""
    if cmu.new_chat_member.user.id != bot_id:
        return False
    old_status = cmu.old_chat_member.status
    new_status = cmu.new_chat_member.status
    was_present = old_status in ("member", "administrator", "creator")
    is_present = new_status in ("member", "administrator", "creator")
    return is_present and not was_present


async def _register_and_welcome(chat: Chat, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Idempotently register a project for `chat` and post the welcome message."""
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return

    result = await services.get_or_create_project(chat.id, chat.title or "")
    if result.created:
        logger.info(
            "Registered new project %s for chat_id=%s (%s)",
            result.project_id, chat.id, result.name,
        )
        # New groups start uninitialised — scope their menu to just /init.
        await modes.apply_chat_commands(context.bot, chat.id, "uninitialized")
    try:
        await context.bot.send_message(
            chat_id=chat.id,
            text=_welcome_text(),
            parse_mode="HTML",
        )
        if result.created:
            await context.bot.send_message(
                chat_id=chat.id,
                text="👉 Run /init to pick this group's vibe and become its admin.",
            )
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not send welcome message to %s: %s", chat.id, exc)


async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Primary, reliable path: fires when the bot's own membership changes."""
    cmu = update.my_chat_member
    if cmu is None:
        return
    if _was_bot_added(cmu, context.bot.id):
        await _register_and_welcome(cmu.chat, context)


async def on_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Fallback path via the `new_chat_members` service message. Some client/server
    combinations surface this even when my_chat_member is also delivered; the
    underlying get_or_create_project call is idempotent so double-firing is safe.
    """
    msg = update.effective_message
    if not msg or not msg.new_chat_members:
        return
    if any(member.id == context.bot.id for member in msg.new_chat_members):
        await _register_and_welcome(update.effective_chat, context)


# ---------------------------------------------------------------------------
# /verify {token}
# ---------------------------------------------------------------------------
_VERIFY_ERRORS = {
    "invalid_token": "❌ That token is not valid. Generate a fresh one in the dashboard.",
    "consumed": "❌ That token has already been used. Generate a new one.",
    "expired": "❌ That token has expired (tokens last 15 minutes). Generate a new one.",
    "wrong_chat": "❌ That token was issued for a different group. Run /verify in the correct group.",
    "no_username": (
        "❌ Your Telegram account has no @username set. Add one in Telegram "
        "settings, then try again."
    ),
    "unknown_student": (
        "❌ No web account is registered with your Telegram username. Register "
        "on the dashboard first, making sure your Telegram username matches."
    ),
}


async def verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    sender = update.effective_user

    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Run /verify inside the group chat (legacy web-dashboard linking).")
        return

    if not context.args:
        await msg.reply_text("Usage: /verify <token>")
        return

    token = context.args[0].strip()

    result = await services.consume_link_token(
        token=token,
        chat_id=chat.id,
        telegram_user_id=sender.id,
        telegram_username=sender.username,
    )

    if not result.ok:
        await msg.reply_text(_VERIFY_ERRORS.get(result.reason, "❌ Verification failed."))
        return

    # Success — notify the dashboard in real time (best-effort).
    await events.publish_project_event(
        project_id=result.project_id,
        event_type="member_joined",
        payload={
            "student_id": result.student_id,
            "telegram_user_id": sender.id,
            "telegram_username": sender.username,
            "already_member": result.already_member,
        },
    )

    project_label = result.project_name or "your project"
    # Reply privately when possible to avoid leaking linkage in the group.
    confirmation = f"✅ Verified! Your account is now linked to {project_label}."
    try:
        await context.bot.send_message(chat_id=sender.id, text=confirmation)
        await msg.reply_text(f"✅ @{sender.username} is now verified.")
    except Exception:
        # Bot can't DM the user (they haven't started a private chat) — reply in group.
        await msg.reply_text(confirmation)


# ---------------------------------------------------------------------------
# Passive message listener (RAG foundation)
# ---------------------------------------------------------------------------
def _classify_content(update: Update) -> tuple[ContentType, str | None, str | None]:
    """
    Map a Telegram message to (content_type, raw_text, file_mime_type).

    File *download* and OCR/parse happen in Module 3; here we only capture
    metadata. Captions on media are preserved as raw_text.
    """
    msg = update.effective_message
    if msg.photo:
        return ContentType.image, msg.caption, None
    if msg.document:
        return ContentType.document, msg.caption, msg.document.mime_type
    if msg.voice:
        return ContentType.voice, msg.caption, msg.voice.mime_type
    # Plain text (and text-only edits).
    return ContentType.text, (msg.text or msg.caption), None


async def _download_and_store(
    update: Update, context: ContextTypes.DEFAULT_TYPE, content_type: ContentType
) -> tuple[str | None, str | None]:
    """
    Download a media message from Telegram and stream it into MinIO.

    Returns (file_storage_path, file_mime_type). Photos/documents are stored;
    oversized files and voice notes are skipped (voice has no transcription
    path yet — Module 3 only embeds text/image/document). Returns (None, None)
    when nothing was stored.
    """
    msg = update.effective_message
    chat = update.effective_chat

    if content_type == ContentType.image and msg.photo:
        tg_file = await msg.photo[-1].get_file()  # largest resolution
        category, mime = "imgs", "image/jpeg"
        filename = f"{tg_file.file_unique_id}.jpg"
    elif content_type == ContentType.document and msg.document:
        doc = msg.document
        if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
            logger.info("Skipping oversized document (%s bytes).", doc.file_size)
            return None, doc.mime_type
        tg_file = await doc.get_file()
        category = "docs"
        mime = doc.mime_type
        filename = doc.file_name or tg_file.file_unique_id
    else:
        # Voice / unsupported — metadata only.
        return None, (msg.voice.mime_type if msg.voice else None)

    data = bytes(await tg_file.download_as_bytearray())
    if len(data) > MAX_FILE_SIZE_BYTES:
        logger.info("Skipping oversized download (%d bytes).", len(data))
        return None, mime

    storage_path = await storage.store_bytes(chat.id, category, filename, data, mime)
    return storage_path, mime


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Intercept all non-command text/image/document/voice messages in group chats,
    stream any media into MinIO, persist metadata to message_logs
    (is_vectorized=False), then enqueue an async embedding job.
    """
    chat = update.effective_chat
    msg = update.effective_message
    sender = update.effective_user

    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return
    if msg is None:
        return

    # Consume a pending /sc free-text input (Set Goals / Set Details) — the user
    # replying to our ForceReply prompt. Not ingested into RAG.
    awaiting = context.chat_data.get("sc_await")
    if (
        awaiting
        and sender
        and awaiting.get("user_id") == sender.id
        and msg.reply_to_message
        and msg.reply_to_message.message_id == awaiting.get("prompt_id")
    ):
        context.chat_data.pop("sc_await", None)
        value = (msg.text or "").strip()
        if awaiting["action"] == "goals":
            ok = await services.update_project_goals(chat.id, value)
            await msg.reply_text("🎯 Project goals updated." if ok else "⚠️ Couldn't update goals.")
        elif awaiting["action"] == "goal_add":
            ok = await services.add_goal_line(chat.id, value)
            await msg.reply_text("🎯 Goal added." if ok else "⚠️ Couldn't add goal.")
        elif awaiting["action"] == "details":
            ok = await services.update_project_details(chat.id, value)
            await msg.reply_text("✏️ Group name updated." if ok else "⚠️ Couldn't update.")
        elif awaiting["action"] == "member_add":
            # "Bala @balaji05" / "@balaji05 Bala" / "Bala" — the @token (if
            # any) is the handle, the rest is the display name.
            tokens = value.split()
            handle = next((t for t in tokens if t.startswith("@")), None)
            name = " ".join(t for t in tokens if not t.startswith("@")).strip()
            if not name and handle:
                name = handle.lstrip("@")
            ok = bool(name) and await services.upsert_group_member(
                chat.id, name, handle, sender.id if sender else None
            )
            if ok:
                label = name + (f" ({handle})" if handle else "")
                await msg.reply_text(f"👥 Added <b>{_md_escape_min(label)}</b>.", parse_mode="HTML")
            else:
                await msg.reply_text("⚠️ Couldn't add that. Format: Name @handle")
        return

    content_type, raw_text, file_mime_type = _classify_content(update)

    # Expense-mode capture: parse "Ashok paid $15 for pizza" into the ledger.
    if raw_text and content_type == ContentType.text:
        m = _EXPENSE_RE.match(raw_text.strip())
        if m:
            state = await services.get_group_state(chat.id)
            if state and state.group_mode == "expense":
                payer = m.group(1).strip()
                amount = float(m.group(2))
                desc = (m.group(3) or "").strip()
                await services.add_expense(chat.id, payer, None, amount, desc or None)
                await msg.reply_text(
                    f"💸 Logged: <b>{payer}</b> paid {_money(amount)}"
                    + (f" for {desc}" if desc else ""),
                    parse_mode="HTML",
                )

    file_storage_path: str | None = None
    if content_type in (ContentType.image, ContentType.document):
        try:
            file_storage_path, file_mime_type = await _download_and_store(
                update, context, content_type
            )
        except Exception as exc:  # storage/download failure must not drop the log
            logger.error("Failed to download/store media in chat %s: %s", chat.id, exc)

    received_at = msg.date or datetime.now(timezone.utc)

    # Fall back to the sender's first name so people without a public
    # @username still show up in Agnes's roster and recaps.
    sender_name = (sender.username or sender.first_name) if sender else None
    sender_name = sender_name[:50] if sender_name else None  # column limit

    log_id = await services.log_incoming_message(
        chat_id=chat.id,
        telegram_message_id=msg.message_id,
        content_type=content_type,
        sender_telegram_user_id=sender.id if sender else None,
        sender_telegram_username=sender_name,
        received_at=received_at,
        raw_text=raw_text,
        file_mime_type=file_mime_type,
        file_storage_path=file_storage_path,
    )

    if log_id is None:
        logger.debug("Message in unregistered chat_id=%s ignored.", chat.id)
        return

    # Enqueue async vectorization for content we can embed. Trivially short
    # texts ("ok", "lol") are logged for the recap window but skipped for
    # embedding — they only pollute the vector index.
    if content_type in (ContentType.text, ContentType.image, ContentType.document):
        if content_type == ContentType.text:
            has_payload = bool(raw_text) and len(raw_text.strip()) >= _MIN_EMBED_TEXT_CHARS
        else:
            has_payload = bool(file_storage_path)
        if has_payload:
            await queue.enqueue_embed_job(
                message_log_id=log_id,
                chat_id=chat.id,
                content_type=content_type.value,
            )

    # A receipt photo captioned "/splitbill" starts a bill split directly
    # (CommandHandler only sees text messages, not captions).
    if content_type == ContentType.image and (msg.caption or "").strip().lower().startswith("/splitbill"):
        await billsplit.splitbill_command(update, context)
        return

    # @mentioning Agnes (or replying to her) is the same as /ask — general
    # queries without anyone needing to remember a command.
    if content_type == ContentType.text and raw_text:
        bot_username = (context.bot.username or "").lower()
        mentioned = bool(bot_username) and f"@{bot_username}" in raw_text.lower()
        reply_to_bot = (
            msg.reply_to_message is not None
            and msg.reply_to_message.from_user is not None
            and msg.reply_to_message.from_user.id == context.bot.id
        )
        if mentioned or reply_to_bot:
            question = re.sub(
                rf"@{re.escape(context.bot.username or '')}", "", raw_text, flags=re.IGNORECASE
            ).strip()
            if question:
                # The passive listener already logged the question row.
                await _deferred_agent(
                    update, context, user_message=question, log_question=False
                )
            return

    logger.debug(
        "Logged message_log=%s chat_id=%s type=%s", log_id, chat.id, content_type.value
    )


# ---------------------------------------------------------------------------
# /ask — invoke the Agnes agent (deferred "Thinking…" pattern, Requirement 2)
# ---------------------------------------------------------------------------
async def _deferred_agent(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_message: str,
    system_directive: str | None = None,
    fallback_text: str | None = None,
    log_question: bool = True,
) -> None:
    """
    Reply with an immediate "🤔 Thinking…" placeholder, run the agent in a
    background task (so the webhook returns within Telegram's 10s window), then
    edit the placeholder with the final answer. The agent itself handles the
    OpenRouter/Gemini fallback. `fallback_text` (if given) replaces error
    replies so commands like /joke always deliver something.
    """
    chat = update.effective_chat
    msg = update.effective_message
    sender = update.effective_user
    assert chat is not None and msg is not None

    placeholder = await msg.reply_text("🤔 Thinking…")

    async def _work() -> None:
        try:
            answer = await run_agent(chat.id, user_message, system_directive=system_directive)
        except Exception as exc:  # run_agent already guards; defend anyway
            logger.exception("Agent crashed in deferred task: %s", exc)
            answer = "⚠️ Something went wrong. Please try again."
        if fallback_text and answer.startswith("⚠️"):
            answer = fallback_text

        try:
            await context.bot.edit_message_text(
                chat_id=chat.id, message_id=placeholder.message_id,
                text=answer, parse_mode="HTML",
            )
        except Exception as exc:
            # Most likely a Telegram HTML-parse rejection — retry as plain text.
            logger.warning("edit_message_text (HTML) failed: %s", exc)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat.id, message_id=placeholder.message_id,
                    text=services._strip_html(answer) or "(no response)",
                )
            except Exception as exc2:
                logger.error("edit_message_text (plain) failed: %s", exc2)

        # Persist the Q&A turn so the agent remembers it next time.
        try:
            await services.log_agent_interaction(
                chat_id=chat.id,
                asker_username=sender.username if sender else None,
                asker_user_id=sender.id if sender else None,
                question=user_message,
                answer=answer,
                q_message_id=msg.message_id,
                a_message_id=placeholder.message_id,
                include_question=log_question,
            )
        except Exception as exc:
            logger.warning("Failed to log agent interaction: %s", exc)

    context.application.create_task(_work(), update=update)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the bounded agentic loop on the user's question (deferred reply)."""
    chat = update.effective_chat
    msg = update.effective_message

    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Ask me inside your group chat.")
        return

    question = " ".join(context.args).strip() if context.args else ""
    if not question:
        await msg.reply_text("Usage: /ask <anything> — or just @mention me in chat.")
        return

    await _deferred_agent(update, context, user_message=question)


# ---------------------------------------------------------------------------
# Structured agent commands (each carries its own Agnes directive)
# ---------------------------------------------------------------------------
_SUMMARY_DIRECTIVE = (
    "The user invoked /summary — they want to catch up on the chat. Recap "
    "what's been happening: the main topics, any plans or decisions made "
    "(with who/when/where if known), anything that still needs someone's "
    "answer or action, and one funny highlight if there is one. Use short "
    "bold labels, keep it tight and scannable. Call search_chat_history when "
    "the recent window isn't enough. Base everything strictly on the actual "
    "conversation — never invent."
)
_JOKE_DIRECTIVE = (
    "The user invoked /joke. Tell ONE original, genuinely funny joke — no "
    "stale programming/dad-joke clichés unless they specifically asked for "
    "that. Best material: this group's recent chat (running gags, what people "
    "were just talking about, shared misery like exams or bills); otherwise "
    "sharp observational humour on the requested topic. 1–3 lines, no "
    "preamble, no explanation — just land the joke."
)
_ROAST_DIRECTIVE = (
    "The user invoked /roast on a named target. Deliver a playful 2–4 line "
    "comedy roast grounded in the target's ACTUAL behaviour in this chat — "
    "mine the recent messages and call search_chat_history for receipts "
    "(always late? left on read? ordered the most expensive dish and "
    "'forgot' to PayNow?). Punch at behaviour only: never appearance, "
    "identity, family, or anything genuinely hurtful — this is loving fire "
    "between friends. If the target isn't in this chat, roast the requester "
    "for pointing at ghosts. End with a wink so nobody actually cries."
)
_EXAMS_ADD_DIRECTIVE = (
    "The user invoked /exams with details of an exam or deadline. If it "
    "contains an unambiguous date, save it with save_important_date (include "
    "the timing, e.g. '9–11am', in the title), then confirm and show the "
    "updated list via list_saved_dates with a countdown for each. If the "
    "date is ambiguous, ask exactly one clarifying question instead. Never "
    "guess dates."
)

# Legacy projects-mode directives (reachable via old /sc buttons only).
_ASSIGN_WORK_DIRECTIVE = (
    "The user asked you to delegate work (legacy projects mode). Identify "
    "concrete outstanding tasks from the conversation and delegate each to the "
    "most suitable member using the delegate_task tool. Choose assignees only "
    "from people actually in the chat. After delegating, reply with a short "
    "bulleted list of who got what and why. If there is nothing concrete to "
    "assign, say so rather than inventing work."
)


async def _run_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    directive: str,
    default_message: str,
    fallback_text: str | None = None,
) -> None:
    """Shared runner for the structured agent slash-commands."""
    chat = update.effective_chat
    msg = update.effective_message
    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Run this inside your group chat.")
        return

    # Any extra words after the command become additional focus for Agnes.
    extra = " ".join(context.args).strip() if context.args else ""
    user_message = f"{default_message} {extra}".strip() if extra else default_message

    await _deferred_agent(
        update, context,
        user_message=user_message,
        system_directive=directive,
        fallback_text=fallback_text,
    )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_command(
        update, context,
        directive=_SUMMARY_DIRECTIVE,
        default_message="Catch me up — what's been happening in this chat?",
    )


async def joke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_command(
        update, context,
        directive=_JOKE_DIRECTIVE,
        default_message="Tell us a joke.",
        fallback_text=random.choice(_JOKES),
    )


async def roast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Roasts happen in the group, where everyone can watch. 🔥")
        return

    target = " ".join(context.args).strip() if context.args else ""
    # Replying to someone's message roasts them.
    if not target and msg.reply_to_message and msg.reply_to_message.from_user:
        u = msg.reply_to_message.from_user
        target = u.first_name or (f"@{u.username}" if u.username else "")
    if not target:
        await msg.reply_text("Who am I roasting? Use /roast <name> or reply to their message.")
        return

    await _deferred_agent(
        update, context,
        user_message=f"Roast {target}.",
        system_directive=_ROAST_DIRECTIVE,
    )


async def exams_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List saved exams/deadlines with countdowns; with args, save a new one."""
    msg = update.effective_message
    chat = update.effective_chat
    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Use /exams inside your group chat.")
        return

    args = " ".join(context.args).strip() if context.args else ""
    if args:
        # Natural-language add ("/exams add Math final 12 Aug 9am").
        await _deferred_agent(
            update, context,
            user_message=f"Save this exam/deadline: {args}",
            system_directive=_EXAMS_ADD_DIRECTIVE,
        )
        return

    dates = await repository.list_upcoming_dates(chat.id)
    if dates is None:
        await msg.reply_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    if not dates:
        await msg.reply_text(
            "📚 No exams or deadlines saved yet.\n"
            "Add one with /exams add <what and when>, e.g.\n"
            "/exams add Linear Algebra final, 12 Aug 9am–11am\n"
            "…or just mention it in chat and ask me to remember it."
        )
        return

    now = datetime.now(timezone.utc)
    lines = []
    for d in dates:
        local = d.due_date.astimezone(_SGT)
        delta = d.due_date - now
        days, hours = delta.days, delta.seconds // 3600
        countdown = f"{days}d {hours}h" if days > 0 else (f"{hours}h" if hours > 0 else "soon!")
        flame = " 🔥" if days <= 3 else ""
        lines.append(
            f"• <b>{services._strip_html(d.title)}</b>\n"
            f"   {local.strftime('%a %d %b %Y, %H:%M')} SGT — in {countdown}{flame}"
        )
    await msg.reply_text(
        "📚 <b>Upcoming exams &amp; deadlines</b>\n\n" + "\n".join(lines),
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Activation state engine (Multi-Mode Group Agent)
# ---------------------------------------------------------------------------
async def state_gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global middleware (handler group -10, runs first). When a group's bot is
    deactivated, swallow ALL updates — text, file uploads, and commands — so the
    bot ignores everyone, EXCEPT an admin's /activate. Raises
    ApplicationHandlerStop to halt all further handler processing for the update.
    """
    chat = update.effective_chat
    if chat is None or chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return  # DMs / channels pass through untouched

    state = await services.get_group_state(chat.id)
    if state is None or state.bot_active:
        return  # unregistered or active → normal processing

    # Deactivated: allow only an admin's /activate to wake it.
    msg = update.effective_message
    user = update.effective_user
    text = (msg.text or "") if msg else ""
    is_activate = text.split()[0].split("@")[0] == "/activate" if text else False
    if is_activate and services.can_admin(state, user.id if user else None):
        return  # let the /activate handler run

    raise ApplicationHandlerStop  # silently ignore everything else


async def activate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    state = await services.get_group_state(chat.id)
    if state is None:
        await msg.reply_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    if not services.can_admin(state, update.effective_user.id if update.effective_user else None):
        await msg.reply_text("🔒 Only the group admin can activate the bot.")
        return
    await services.set_bot_active(chat.id, True)
    await msg.reply_text("✅ Agnes is awake again. Miss me?")


async def deactivate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    state = await services.get_group_state(chat.id)
    if state is None:
        await msg.reply_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    if not services.can_admin(state, update.effective_user.id if update.effective_user else None):
        await msg.reply_text("🔒 Only the group admin can deactivate the bot.")
        return
    await services.set_bot_active(chat.id, False)
    await msg.reply_text(
        "🔕 Agnes is now <b>deactivated</b> and will ignore the group. "
        "An admin can wake me with /activate.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Interactive commands & menus (Requirement 4)
# ---------------------------------------------------------------------------
_CELEBRATE_DIRECTIVE = (
    "The user invoked /celebrate — the project is complete. Write a warm, "
    "celebratory end-of-project wrap-up in Telegram HTML with emojis: "
    "congratulate the team, summarise what was accomplished (the completed "
    "tasks), gently acknowledge anything left undone, and thank everyone. Keep "
    "it upbeat and concise. Base it strictly on the task ledger provided."
)

_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs. 🐛",
    "There are only 10 kinds of people: those who understand binary and those who don't.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I JOIN you?' 🍻",
    "I'd tell you a UDP joke, but you might not get it.",
    "Student life: 8 cups of coffee, 0 commits, infinite vibes. ☕",
    "Why was the function sad after the party? It didn't get called. 📞",
    "My code doesn't work, I have no idea why. My code works, I have no idea why. 🤷",
    "It's not a bug — it's an undocumented feature. ✨",
    "Deadlines are just suggestions delivered with anxiety. 🗓️",
    "Git commit -m 'final'. Git commit -m 'final FINAL'. Git commit -m 'final for real'. 😅",
]


def _is_group(chat: Chat | None) -> bool:
    return chat is not None and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)


async def change_details_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_group(update.effective_chat):
        await update.effective_message.reply_text("Use this inside your group chat.")
        return
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎯 Edit Goals", callback_data="cd:goals")],
            [InlineKeyboardButton("📅 Deadlines", callback_data="cd:deadlines")],
            [InlineKeyboardButton("✅ Assign Tasks", callback_data="cd:tasks")],
        ]
    )
    await update.effective_message.reply_text(
        "What would you like to change?", reply_markup=keyboard
    )


async def setgoals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    goals = " ".join(context.args).strip() if context.args else ""
    if not goals:
        await msg.reply_text("Usage: /setgoals <your project goals>")
        return
    ok = await services.update_project_goals(chat.id, goals)
    await msg.reply_text(
        "🎯 Project goals updated." if ok else "⚠️ This group isn't registered yet."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_group(update.effective_chat):
        await update.effective_message.reply_text("Use this inside your group chat.")
        return
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🕒 Upcoming", callback_data="st:upcoming"),
                InlineKeyboardButton("⚡ Active", callback_data="st:active"),
                InlineKeyboardButton("✅ Completed", callback_data="st:completed"),
            ]
        ]
    )
    await update.effective_message.reply_text(
        "Set the project status:", reply_markup=keyboard
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_group(update.effective_chat):
        await update.effective_message.reply_text("Use this inside your group chat.")
        return
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🗑️ Yes, clear", callback_data="clr:yes"),
                InlineKeyboardButton("Cancel", callback_data="clr:no"),
            ]
        ]
    )
    await update.effective_message.reply_text(
        "⚠️ Wipe my memory of this group? Files are kept, but my recall of "
        "past chats/documents is gone. This can't be undone.",
        reply_markup=keyboard,
    )


async def celebrate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return

    ledger = await services.get_task_ledger(chat.id)
    if ledger is None:
        await msg.reply_text("⚠️ This group isn't registered yet.")
        return

    # Mark the project completed, then let Agnes write the celebration.
    await services.set_project_status(chat.id, ProjectStatus.completed)

    def _block(title: str, items: list[str]) -> str:
        body = "\n".join(f"- {t}" for t in items) or "(none)"
        return f"{title}:\n{body}"

    ledger_text = (
        _block("Completed", ledger["completed"])
        + "\n\n"
        + _block("Outstanding", ledger["outstanding"])
        + "\n\n"
        + _block("Dropped", ledger["dropped"])
    )
    user_message = (
        "The project is wrapping up. Here is the final task ledger:\n"
        f"{ledger_text}\n\nWrite the celebration message now."
    )
    await _deferred_agent(update, context, user_message=user_message, system_directive=_CELEBRATE_DIRECTIVE)


async def hehe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Legacy alias for /joke."""
    await joke_command(update, context)


async def sync_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Ensure every document/image/text shared in this chat is ingested into the
    backend (OCR + vectorise). Re-enqueues anything not yet vectorised.
    """
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return

    status = await msg.reply_text("🔄 Syncing shared files…")
    pending = await services.list_unvectorized_messages(chat.id)
    if pending is None:
        await status.edit_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    if not pending:
        await status.edit_text("✅ Everything is already synced — nothing new to ingest.")
        return

    for message_log_id, content_type in pending:
        await queue.enqueue_embed_job(
            message_log_id=message_log_id, chat_id=chat.id, content_type=content_type
        )
    await status.edit_text(
        f"🔄 Queued <b>{len(pending)}</b> item(s) for OCR + indexing. "
        "Give it a minute, then ask me about them with /ask.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Inline-button callback router
# ---------------------------------------------------------------------------
_STATUS_LABELS = {
    "upcoming": "🕒 Upcoming",
    "active": "⚡ Active",
    "completed": "✅ Completed",
}


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    data = query.data or ""
    chat = update.effective_chat
    if chat is None:
        return

    # ---- /status ----
    if data.startswith("st:"):
        value = data.split(":", 1)[1]
        try:
            status = ProjectStatus(value)
        except ValueError:
            return
        ok = await services.set_project_status(chat.id, status)
        await query.edit_message_text(
            f"Project status set to <b>{_STATUS_LABELS.get(value, value)}</b>."
            if ok
            else "⚠️ This group isn't registered yet.",
            parse_mode="HTML",
        )
        return

    # ---- /clear ----
    if data.startswith("clr:"):
        if data == "clr:no":
            await query.edit_message_text("Cancelled — nothing was cleared.")
            return
        project_id = await services.resolve_project_id(chat.id)
        if project_id is None:
            await query.edit_message_text("⚠️ This group isn't registered yet.")
            return
        await query.edit_message_text("🧹 Clearing vector memory…")
        try:
            result = await pipeline.clear_project_cache(project_id, include_files=False)
            await query.edit_message_text(
                f"🧹 Vector memory cleared — {result.messages_soft_deleted} messages "
                "archived. Files were kept."
            )
        except Exception as exc:
            logger.error("clear cache failed: %s", exc)
            await query.edit_message_text("⚠️ Couldn't clear the cache. Please try again.")
        return

    # ---- /change_details ----
    if data.startswith("cd:"):
        action = data.split(":", 1)[1]
        if action == "goals":
            exists, goals = await services.get_project_goals(chat.id)
            if not exists:
                await query.edit_message_text("⚠️ This group isn't registered yet.")
                return
            current = goals or "(none set)"
            await query.edit_message_text(
                "🎯 To update the goals, send:\n"
                "<code>/setgoals your goals here</code>\n\n"
                f"<b>Current goals:</b>\n{_md_escape_min(current)}",
                parse_mode="HTML",
            )
        elif action == "deadlines":
            await query.edit_message_text("📅 Fetching saved dates…")
            await _deferred_agent(
                update, context,
                user_message="List all saved exams and deadlines with a countdown for each.",
            )
        elif action == "tasks":
            await query.edit_message_text("✅ Reviewing work to assign…")
            await _deferred_agent(
                update, context,
                user_message="Assign the outstanding work for this project to the team.",
                system_directive=_ASSIGN_WORK_DIRECTIVE,
            )
        return


def _md_escape_min(text: str) -> str:
    """Minimal HTML escaping for user-provided goals shown in an HTML message."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Multi-Mode: /init, mode selection, /admin_settings, Mode B & Mode C
# ---------------------------------------------------------------------------
_EXPENSE_RE = re.compile(
    r"^([A-Za-z][\w ]*?)\s+paid\s+\$?(\d+(?:\.\d{1,2})?)\s*(?:for\s+|on\s+|-\s*)?(.*)$",
    re.IGNORECASE,
)


def _mode_menu_keyboard() -> InlineKeyboardMarkup:
    rows = [[_btn(modes.MODE_LABELS[m], f"sc|mode|{m}")] for m in modes.MODE_ORDER]
    return InlineKeyboardMarkup(rows)


def _admin_settings_keyboard(allowed: dict) -> InlineKeyboardMarkup:
    def state(key: str) -> str:
        return "🟢 ON" if services.model_allowed(allowed, key) else "🔴 OFF"

    return InlineKeyboardMarkup(
        [
            [_btn(f"Qwen-VL (image/PDF OCR): {state('qwen_vl')}", "sc|am|qwen_vl")],
            [_btn(f"Gemini fallback: {state('gemini_fallback')}", "sc|am|gemini_fallback")],
            _back(),
        ]
    )


def _goals_editor_keyboard(lines: list[str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for i, ln in enumerate(lines[:12]):
        label = (ln[:40] + "…") if len(ln) > 40 else ln
        rows.append([_btn(f"❌ {label}", f"sc|gedit|rm|{i}")])
    rows.append([_btn("➕ Add goal", "sc|gedit|add"), _btn("🗑 Clear", "sc|gedit|clear")])
    rows.append(_back("sc|set"))
    return InlineKeyboardMarkup(rows)


async def init_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """First-time initialisation: claim admin + choose the group's mode."""
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Run /init inside your group chat.")
        return
    state = await services.get_group_state(chat.id)
    if state is None:
        await msg.reply_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    uid = update.effective_user.id if update.effective_user else None
    if state.group_admin_id is not None and state.group_admin_id != uid:
        await msg.reply_text("🔒 This group is already initialised. Only its admin can re-initialise.")
        return
    await msg.reply_text(
        "👋 <b>Welcome!</b> Pick what this group is for — you'll become its admin:",
        parse_mode="HTML",
        reply_markup=_mode_menu_keyboard(),
    )


async def admin_settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    state = await services.get_group_state(chat.id)
    if state is None:
        await msg.reply_text("⚠️ This group isn't registered yet.")
        return
    uid = update.effective_user.id if update.effective_user else None
    if not await services.is_privileged_user(chat.id, uid):
        await msg.reply_text("🔒 Leaders only.")
        return
    await msg.reply_text(
        "⚙️ <b>AI model settings</b> — tap to toggle:",
        parse_mode="HTML",
        reply_markup=_admin_settings_keyboard(state.allowed_models),
    )


# ── Fun extras ──
async def meme_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not _is_group(chat):
        await update.effective_message.reply_text("Use this inside your group chat.")
        return
    await _deferred_agent(
        update, context,
        user_message="Suggest one funny, wholesome meme idea or caption for this friend group.",
        system_directive=(
            "Reply with a single short meme idea (format + caption) in a playful tone. "
            "Two lines max. Telegram HTML, emojis welcome."
        ),
    )


# ── Mode C: Expense Tracker ──
def _money(v: float) -> str:
    return f"${v:,.2f}"


async def add_expense_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    user = update.effective_user
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    if not context.args:
        await msg.reply_text("Usage: /add_expense <amount> <description>\ne.g. /add_expense 15 pizza")
        return
    try:
        amount = float(context.args[0].lstrip("$"))
    except ValueError:
        await msg.reply_text("First argument must be an amount, e.g. /add_expense 15 pizza")
        return
    desc = " ".join(context.args[1:]).strip()
    payer = (user.first_name if user else None) or (f"@{user.username}" if user and user.username else "Someone")
    ok = await services.add_expense(chat.id, payer, user.id if user else None, amount, desc or None)
    if not ok:
        await msg.reply_text("⚠️ This group isn't registered yet.")
        return
    await msg.reply_text(
        f"💸 Logged: <b>{payer}</b> paid {_money(amount)}" + (f" for {desc}" if desc else ""),
        parse_mode="HTML",
    )


async def list_expenses_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    items = await services.list_expenses(chat.id, limit=20)
    if items is None:
        await msg.reply_text("⚠️ This group isn't registered yet.")
        return
    if not items:
        await msg.reply_text("No expenses logged yet. Add one with /add_expense 15 pizza")
        return
    total = sum(i["amount"] for i in items)
    lines = "\n".join(
        f"• <b>{i['payer']}</b> {_money(i['amount'])}" + (f" — {i['description']}" if i["description"] else "")
        for i in items
    )
    await msg.reply_text(
        f"💸 <b>Recent expenses</b> (total {_money(total)})\n{lines}", parse_mode="HTML"
    )


async def settle_up_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Use this inside your group chat.")
        return
    result = await services.compute_balances(chat.id)
    if result is None:
        await msg.reply_text("⚠️ This group isn't registered yet.")
        return
    if result["count"] == 0:
        await msg.reply_text("No expenses to settle yet.")
        return
    bal_lines = "\n".join(
        f"• {name}: {'+' if net >= 0 else ''}{_money(net)}" for name, net in result["balances"]
    )
    if result["settlements"]:
        settle_lines = "\n".join(
            f"➡️ <b>{d}</b> pays <b>{c}</b> {_money(a)}" for d, c, a in result["settlements"]
        )
    else:
        settle_lines = "Everyone's square. 🎉"
    await msg.reply_text(
        f"🧾 <b>Settle up</b> (total {_money(result['total'])})\n\n"
        f"<b>Balances</b>\n{bal_lines}\n\n<b>Who pays whom</b>\n{settle_lines}",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Unified /sc inline menu + RBAC (app-like UX)
# ---------------------------------------------------------------------------
_SC_HEADER = "🤖 <b>Agnes</b> — pick an option:"
_DENY = "🔒 Admins only. Ask the group admin."

_SC_MENU_HELP = (
    "<b>Agnes menu</b>\n\n"
    "• <b>Summary / News / Joke / Exams</b> — the daily drivers\n"
    "• <b>Bill</b> — reopen the current bill (start one by replying to a "
    "receipt photo with /splitbill)\n"
    "• <b>Sync</b> — index all shared files (OCR + memory)\n"
    "• <b>Set</b> — group mode, name, AI settings\n"
    "• <b>Clear / Activation</b> — admins only\n\n"
    "Outside the menu: <code>/ask &lt;anything&gt;</code>, "
    "<code>/roast &lt;name&gt;</code>, or just @mention me."
)
_VERIFY_HELP = (
    "<b>Legacy: link the web dashboard</b>\n"
    "The old Student Claw web app still works if you need it for a "
    "hackathon:\n"
    "1. Register on the dashboard with your Telegram @username.\n"
    "2. Submit your Project Key to get a token.\n"
    "3. Send <code>/verify &lt;token&gt;</code> here in the group."
)


def _btn(label: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=data)


def _back(to: str = "sc|main") -> list[InlineKeyboardButton]:
    return [_btn("🔙 Back", to)]


def _main_menu_keyboard(state: "services.GroupState", privileged: bool) -> InlineKeyboardMarkup:
    rows = [
        [_btn("📋 Summary", "sc|summary"), _btn("📰 News", "sc|news")],
        [_btn("😂 Joke", "sc|hehe"), _btn("📚 Exams", "sc|deadlines")],
        [_btn("🧾 Bill", "sc|bill"), _btn("🔄 Sync", "sc|sync")],
    ]
    set_row = [_btn("⚙️ Set", "sc|set")]
    if privileged:
        set_row.append(_btn("🗑️ Clear memory", "sc|clear"))
    rows.append(set_row)
    last: list[InlineKeyboardButton] = []
    if privileged:
        dot = "🟢" if state.bot_active else "🔴"
        last.append(_btn(f"{dot} Activation", "sc|act"))
    last.append(_btn("❓ Help", "sc|help"))
    rows.append(last)
    return InlineKeyboardMarkup(rows)


def _set_menu_keyboard(privileged: bool) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if privileged:
        rows.append([_btn("👥 Members", "sc|mem"), _btn("🔀 Set Mode", "sc|mode")])
        rows.append([_btn("✏️ Set Group Name", "sc|set|details"), _btn("⚙️ AI Settings", "sc|am")])
    else:
        rows.append([_btn("👥 Members", "sc|mem")])
    rows.append(_back())
    return InlineKeyboardMarkup(rows)


def _members_keyboard(members: list[dict]) -> InlineKeyboardMarkup:
    """Manual roster editor: tap a member to remove, ➕ to add."""
    rows: list[list[InlineKeyboardButton]] = []
    for m in members[:20]:
        label = m["display_name"] + (
            f" (@{m['telegram_username']})" if m["telegram_username"] else ""
        )
        rows.append([_btn(f"❌ {label[:48]}", f"sc|mem|rm|{m['id']}")])
    rows.append([_btn("➕ Add member", "sc|mem|add")])
    rows.append(_back("sc|set"))
    return InlineKeyboardMarkup(rows)


def _members_text(members: list[dict]) -> str:
    if not members:
        body = "<i>(no members added yet)</i>"
    else:
        body = "\n".join(
            f"• <b>{_md_escape_min(m['display_name'])}</b>"
            + (f" — @{_md_escape_min(m['telegram_username'])}" if m["telegram_username"] else "")
            for m in members
        )
    return (
        "👥 <b>Group members</b>\n"
        "So I know exactly who you mean — for roasts, summaries, bills…\n\n"
        f"{body}\n\n"
        "Tap ❌ to remove, or ➕ to add someone (name + @handle)."
    )


def _activation_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [_btn("🟢 Activate", "sc|act|on"), _btn("🔴 Deactivate", "sc|act|off")],
            _back(),
        ]
    )


def _roles_keyboard(members: list[dict]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for m in members:
        name = m["display_name"]
        sid = m["student_id"]
        if m["role"] == "lead":
            rows.append([_btn(f"👑 {name} (Leader) → Member", f"sc|role|{sid}|member")])
        else:
            rows.append([_btn(f"👤 {name} (Member) → Leader", f"sc|role|{sid}|lead")])
    if not rows:
        rows.append([_btn("(no verified members yet)", "sc|set|roles")])
    rows.append(_back("sc|set"))
    return InlineKeyboardMarkup(rows)


async def sc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Master command — opens the interactive menu."""
    chat = update.effective_chat
    msg = update.effective_message
    if not _is_group(chat):
        await msg.reply_text("Open the menu inside your group chat.")
        return
    state = await services.get_group_state(chat.id)
    if state is None:
        await msg.reply_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    uid = update.effective_user.id if update.effective_user else None
    privileged = await services.is_privileged_user(chat.id, uid)
    await msg.reply_text(
        _SC_HEADER, parse_mode="HTML",
        reply_markup=_main_menu_keyboard(state, privileged),
    )


async def _prompt_input(
    update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, prompt: str
) -> None:
    """Set a per-user awaiting flag + ForceReply for free-text input (Set Goals/Details)."""
    chat = update.effective_chat
    user = update.effective_user
    sent = await context.bot.send_message(
        chat_id=chat.id, text=prompt, parse_mode="HTML",
        reply_markup=ForceReply(selective=True),
    )
    context.chat_data["sc_await"] = {
        "action": action,
        "user_id": user.id if user else None,
        "prompt_id": sent.message_id,
    }


async def on_sc_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router for the /sc menu tree (callback_data prefixed `sc|`)."""
    query = update.callback_query
    chat = update.effective_chat
    user = update.effective_user
    if query is None or chat is None:
        if query:
            await query.answer()
        return

    parts = (query.data or "").split("|")  # ["sc", action, sub?, ...]
    action = parts[1] if len(parts) > 1 else ""
    sub = parts[2] if len(parts) > 2 else None

    state = await services.get_group_state(chat.id)
    if state is None:
        await query.answer("Group not registered.", show_alert=True)
        return
    uid = user.id if user else None
    privileged = await services.is_privileged_user(chat.id, uid)

    # RBAC: gate sensitive actions even if a stale button is clicked.
    denied = (
        (action in {"clear", "act", "role", "mode", "am", "mem"})
        or (action == "set" and sub in {"details", "roles"})
    )
    if denied and not privileged:
        await query.answer(_DENY, show_alert=True)
        return

    # ── Navigation ──
    if action == "main":
        await query.answer()
        await query.edit_message_text(
            _SC_HEADER, parse_mode="HTML",
            reply_markup=_main_menu_keyboard(state, privileged),
        )
        return
    if action == "set" and sub is None:
        await query.answer()
        await query.edit_message_text(
            "⚙️ <b>Settings</b>", parse_mode="HTML",
            reply_markup=_set_menu_keyboard(privileged),
        )
        return
    if action == "act" and sub is None:
        await query.answer()
        cur = "🟢 Active" if state.bot_active else "🔴 Inactive"
        await query.edit_message_text(
            f"Activation — currently <b>{cur}</b>", parse_mode="HTML",
            reply_markup=_activation_menu_keyboard(),
        )
        return

    # ── Info screens ──
    if action == "help":
        await query.answer()
        await query.edit_message_text(
            _SC_MENU_HELP, parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([_back()]),
        )
        return
    if action == "verify":
        await query.answer()
        await query.edit_message_text(
            _VERIFY_HELP, parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([_back()]),
        )
        return

    # ── Leaf actions reusing existing command logic (new message below the menu) ──
    if action == "summary":
        await query.answer("Summarising…"); await summary_command(update, context); return
    if action == "news":
        await query.answer("📰"); await news.news_command(update, context); return
    if action == "hehe":
        await query.answer("😂"); await joke_command(update, context); return
    if action == "deadlines":
        await query.answer("📚"); await exams_command(update, context); return
    if action == "bill":
        await query.answer("🧾"); await billsplit.bill_command(update, context); return
    if action == "sync":
        await query.answer("Syncing…"); await sync_command(update, context); return
    if action == "celebrate":  # legacy button
        await query.answer("🎉"); await celebrate_command(update, context); return
    if action == "assign":  # legacy button (projects mode)
        await query.answer("Assigning…")
        await _deferred_agent(
            update, context,
            user_message="Assign the outstanding work to the team.",
            system_directive=_ASSIGN_WORK_DIRECTIVE,
        )
        return
    if action == "clear":
        await query.answer(); await clear_command(update, context); return  # renders clr: confirm

    # ── Activation submenu ──
    if action == "act" and sub == "on":
        await services.set_bot_active(chat.id, True)
        await query.answer("Activated ✅")
        await query.edit_message_text(
            "✅ Agnes is <b>active</b>.", parse_mode="HTML",
            reply_markup=_activation_menu_keyboard(),
        )
        return
    if action == "act" and sub == "off":
        await services.set_bot_active(chat.id, False)
        await query.answer("Deactivated 🔕")
        await query.edit_message_text(
            "🔕 Agnes is <b>deactivated</b>. Send /activate to wake me.",
            parse_mode="HTML",
        )
        return

    # ── Mode selection (claim admin + scope commands) ──
    if action == "mode" and sub is None:
        await query.answer()
        await query.edit_message_text(
            "🔀 <b>Choose a mode</b> for this group:", parse_mode="HTML",
            reply_markup=_mode_menu_keyboard(),
        )
        return
    if action == "mode" and sub:
        ok = await services.initialise_group(chat.id, uid or 0, sub)
        if ok:
            await modes.apply_chat_commands(context.bot, chat.id, sub)
        await query.answer("Mode set ✅" if ok else "Failed.", show_alert=not ok)
        label = modes.MODE_LABELS.get(sub, sub)
        await query.edit_message_text(
            f"✅ This group is now in <b>{label}</b> mode. The command menu has been updated.",
            parse_mode="HTML",
        )
        return

    # ── Members (manual roster — replaces web-app registration) ──
    if action == "mem":
        if sub == "add":
            await query.answer()
            await _prompt_input(
                update, context, "member_add",
                "➕ Reply to this with the member's <b>name and @handle</b>, "
                "e.g. <code>Bala @balaji05</code> (handle optional):",
            )
            return
        if sub == "rm" and len(parts) >= 4:
            await services.remove_group_member(chat.id, parts[3])
            await query.answer("Removed")
        else:
            await query.answer()
        members = await services.list_group_members(chat.id) or []
        await query.edit_message_text(
            _members_text(members), parse_mode="HTML",
            reply_markup=_members_keyboard(members),
        )
        return

    # ── Admin settings (AI model toggles) ──
    if action == "am" and sub is None:
        await query.answer()
        await query.edit_message_text(
            "⚙️ <b>AI model settings</b> — tap to toggle:", parse_mode="HTML",
            reply_markup=_admin_settings_keyboard(state.allowed_models),
        )
        return
    if action == "am" and sub:
        models = await services.toggle_allowed_model(chat.id, sub)
        await query.answer("Updated ✅")
        await query.edit_message_text(
            "⚙️ <b>AI model settings</b> — tap to toggle:", parse_mode="HTML",
            reply_markup=_admin_settings_keyboard(models or {}),
        )
        return

    # ── Interactive goals editor (tree) ──
    if action == "gedit":
        if sub == "add":
            await query.answer()
            await _prompt_input(update, context, "goal_add", "➕ Reply to this with ONE goal to add:")
            return
        if sub == "clear":
            await services.clear_goals(chat.id)
            await query.answer("Cleared")
        elif sub == "rm" and len(parts) >= 4:
            await services.remove_goal_line(chat.id, int(parts[3]))
            await query.answer("Removed")
        else:
            await query.answer()
        lines = await services.get_goal_lines(chat.id) or []
        body = "🎯 <b>Project Goals</b>\n" + (
            "\n".join(f"• {ln}" for ln in lines) if lines else "<i>(no goals yet)</i>"
        )
        await query.edit_message_text(
            body, parse_mode="HTML", reply_markup=_goals_editor_keyboard(lines)
        )
        return

    # ── Set submenu ──
    if action == "set" and sub == "status":
        await query.answer(); await status_command(update, context); return  # renders st: keyboard
    if action == "set" and sub == "goals":
        # Open the interactive goals editor tree (Mode A update).
        await query.answer()
        lines = await services.get_goal_lines(chat.id) or []
        body = "🎯 <b>Project Goals</b>\n" + (
            "\n".join(f"• {ln}" for ln in lines) if lines else "<i>(no goals yet)</i>"
        )
        await query.edit_message_text(
            body, parse_mode="HTML", reply_markup=_goals_editor_keyboard(lines)
        )
        return
    if action == "set" and sub == "details":
        await query.answer()
        await _prompt_input(update, context, "details", "✏️ Reply to this with the new <b>project name</b>:")
        return
    if action == "set" and sub == "roles":
        await query.answer()
        members = await services.list_members(chat.id)
        await query.edit_message_text(
            "👑 <b>Set Roles</b> — tap to promote/demote:", parse_mode="HTML",
            reply_markup=_roles_keyboard(members),
        )
        return

    # ── Role toggle ──
    if action == "role" and len(parts) >= 4:
        ok = await services.set_member_role(chat.id, parts[2], parts[3])
        await query.answer("Updated ✅" if ok else "Couldn't update.", show_alert=not ok)
        members = await services.list_members(chat.id)
        await query.edit_message_text(
            "👑 <b>Set Roles</b> — tap to promote/demote:", parse_mode="HTML",
            reply_markup=_roles_keyboard(members),
        )
        return

    await query.answer()


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------
def register_handlers(application) -> None:
    """Attach all Module 2 handlers to a PTB Application, in priority order."""
    # State engine: runs before everything else and can swallow updates when the
    # bot is deactivated (Multi-Mode Group Agent).
    application.add_handler(TypeHandler(Update, state_gate), group=-10)

    # ── Commands ──
    # ── The daily drivers ──
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("summary", summary_command))
    application.add_handler(CommandHandler("news", news.news_command))
    application.add_handler(CommandHandler("joke", joke_command))
    application.add_handler(CommandHandler("hehe", hehe_command))  # legacy alias
    application.add_handler(CommandHandler("roast", roast_command))
    application.add_handler(CommandHandler("exams", exams_command))
    application.add_handler(CommandHandler("meme_prompt", meme_prompt_command))
    application.add_handler(CommandHandler("sc", sc_command))
    application.add_handler(CommandHandler("help", help_command))

    # ── Bills & money ──
    application.add_handler(CommandHandler("splitbill", billsplit.splitbill_command))
    application.add_handler(CommandHandler("bill", billsplit.bill_command))
    application.add_handler(CommandHandler("paynow", billsplit.paynow_command))
    application.add_handler(CommandHandler("add_expense", add_expense_command))
    application.add_handler(CommandHandler("list_expenses", list_expenses_command))
    application.add_handler(CommandHandler("settle_up", settle_up_command))

    # ── Onboarding / admin ──
    #   /activate — wakes the bot when the menu is blocked (deactivated state)
    #   /verify   — LEGACY web-dashboard linking (kept for hackathons, hidden)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("init", init_command))
    application.add_handler(CommandHandler("admin_settings", admin_settings_command))
    application.add_handler(CommandHandler("activate", activate_command))
    application.add_handler(CommandHandler("deactivate", deactivate_command))
    application.add_handler(CommandHandler("sync", sync_command))
    application.add_handler(CommandHandler("verify", verify_command))

    # ── Callback routers ──
    # /sc menu tree (callback_data prefixed `sc|`).
    application.add_handler(CallbackQueryHandler(on_sc_callback, pattern=r"^sc\|"))
    # Bill-split claim board (callback_data prefixed `bl|`).
    application.add_handler(CallbackQueryHandler(billsplit.on_bill_callback, pattern=r"^bl\|"))
    # Reused confirm/status/legacy sub-flows (colon-delimited prefixes).
    application.add_handler(
        CallbackQueryHandler(on_callback_query, pattern=r"^(st|clr|cd):")
    )

    # Bot membership changes (primary join-detection path).
    application.add_handler(
        ChatMemberHandler(on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER)
    )

    # new_chat_members service message (fallback join-detection path).
    application.add_handler(
        MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, on_new_chat_members)
    )

    # Passive content listener: text / photo / document / voice, excluding
    # commands and service messages. Lowest priority so commands win.
    content_filter = (
        (filters.TEXT & ~filters.COMMAND)
        | filters.PHOTO
        | filters.Document.ALL
        | filters.VOICE
    )
    application.add_handler(MessageHandler(content_filter, on_message))
