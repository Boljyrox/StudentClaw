"""
/splitexpense — split a cost with no receipt photo.

Deliberately built on the SAME Bill/BillItem/BillClaim tables as /splitbill, so
there is one split engine, one claim board and one settle-up flow. An expense
is simply a bill with `kind="expense"` and (usually) a single line item.

Two ways in:

  1. Natural language, parsed in one shot:
         /splitexpense i paid $20 for a photo booth with raja, pravin and madhu
     Named people are pre-ticked so nobody has to tap anything.

  2. Guided, when the text is missing something:
         who paid  →  what for  →  how much  →  claim board
"""

from __future__ import annotations

import logging
import re
import zlib
from dataclasses import dataclass, field
from typing import Optional

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.bot import billsplit

logger = logging.getLogger("student_claw.bot.expenses")

_DRAFT_KEY = "expense_draft"
_AWAIT_KEY = "expense_await"


# ---------------------------------------------------------------------------
# Natural-language parsing
# ---------------------------------------------------------------------------
_AMOUNT = re.compile(r"(?:[$€£]\s*)?(\d+(?:\.\d{1,2})?)\b")
_PAYER = re.compile(
    r"^\s*(?:i|i've|ive)\b|^\s*([A-Z][\w'-]*)\s+(?:paid|spent|covered|got)\b",
    re.IGNORECASE,
)
_FOR = re.compile(
    r"\b(?:for|on)\s+(?:a|an|the|some)?\s*(.+?)"
    r"(?=\s+\b(?:with|for|and\s+split|among|between|w/)\b|$)",
    re.IGNORECASE,
)
_WITH = re.compile(r"\b(?:with|among|between|w/)\s+(.+?)$", re.IGNORECASE)
_SPLIT_NAMES = re.compile(r"\s*(?:,|&|\band\b|\+)\s*", re.IGNORECASE)
_SELF_WORDS = {"me", "myself", "i", "us", "everyone", "all"}


@dataclass
class ParsedExpense:
    amount: Optional[float] = None
    description: Optional[str] = None
    payer_is_sender: bool = True
    payer_name: Optional[str] = None
    participants: list[str] = field(default_factory=list)
    everyone: bool = False
    # "…with me and mei" — the sender joins in even though they didn't pay.
    include_sender: bool = False

    @property
    def complete(self) -> bool:
        return self.amount is not None and bool(self.description)


def parse_expense(text: str) -> ParsedExpense:
    """
    Pull amount, description, payer and participants out of free text.

    Deliberately regex rather than an LLM call: this handles money, so being
    predictable and auditable beats being clever, and anything it can't parse
    falls through to the guided prompts instead of being guessed at.
    """
    out = ParsedExpense()
    text = (text or "").strip()
    if not text:
        return out

    # Who paid — "I paid ..." (default) or "Raja paid ...".
    payer_match = _PAYER.match(text)
    if payer_match and payer_match.group(1):
        out.payer_is_sender = False
        out.payer_name = payer_match.group(1).strip()

    # Participants after "with" — captured before the description so the
    # trailing name list doesn't get swallowed into it.
    tail = ""
    with_match = _WITH.search(text)
    if with_match:
        tail = with_match.group(1)
        text_for_desc = text[: with_match.start()]
        for raw in _SPLIT_NAMES.split(tail):
            name = raw.strip(" .,!?").strip()
            if not name:
                continue
            low = name.lower()
            if low in _SELF_WORDS:
                if low in {"everyone", "all", "us"}:
                    out.everyone = True
                else:  # "me" / "myself" / "i"
                    out.include_sender = True
                continue
            out.participants.append(name[:60])
    else:
        text_for_desc = text

    # Amount — search the whole string so "$20" after the description still
    # counts, but prefer one that isn't part of a name.
    amount_match = _AMOUNT.search(text_for_desc) or _AMOUNT.search(text)
    if amount_match:
        try:
            out.amount = float(amount_match.group(1))
        except ValueError:
            out.amount = None

    # Description — the "for <thing>" clause, minus any stray amount.
    for_match = _FOR.search(text_for_desc)
    if for_match:
        desc = for_match.group(1).strip(" .,!?")
        desc = _AMOUNT.sub("", desc).strip(" .,!?$")
        out.description = re.sub(r"\s{2,}", " ", desc)[:120] or None

    return out


def pseudo_user_id(name: str) -> int:
    """
    Stable negative id for someone we can't map to a Telegram account.

    Claims require a user_id, and real Telegram ids are always positive, so a
    negative CRC of the name is collision-resistant enough here and — unlike
    Python's `hash()` — identical across restarts.
    """
    return -(zlib.crc32(name.strip().lower().encode()) % 1_000_000_000 + 1)


def is_pseudo(user_id: Optional[int]) -> bool:
    return user_id is not None and user_id < 0


# ---------------------------------------------------------------------------
# Creating the expense
# ---------------------------------------------------------------------------
async def _resolve_people(chat_id: int, names: list[str]) -> list[tuple[int, str]]:
    """Map written names onto real chat members where possible."""
    from app.bot import services

    candidates = await services.list_payer_candidates(chat_id)
    lookup = {c.display_name.strip().lower(): c for c in candidates}

    resolved: list[tuple[int, str]] = []
    for name in names:
        match = lookup.get(name.strip().lower())
        if match and match.telegram_user_id:
            resolved.append((match.telegram_user_id, match.display_name))
        else:
            display = match.display_name if match else name
            resolved.append((pseudo_user_id(display), display))
    return resolved


async def create_expense(
    chat_id: int,
    payer_user_id: Optional[int],
    payer_name: str,
    description: str,
    amount: float,
    participants: list[tuple[int, str]],
) -> Optional[billsplit.BillView]:
    """Create the expense as a single-item bill and pre-claim the participants."""
    receipt = billsplit.ParsedReceipt(
        merchant=description[:200],
        currency="SGD",
        items=[{"name": description[:200], "qty": 1.0, "total_price": round(amount, 2)}],
        subtotal=round(amount, 2),
        service_charge=0.0,
        gst=0.0,
        gst_inclusive=True,
        other_charges=0.0,
        discount=0.0,
        total=round(amount, 2),
        warnings=[],
    )
    view = await billsplit.create_bill(
        chat_id, payer_user_id, payer_name, receipt, kind="expense"
    )
    if view is None:
        return None

    # Pre-tick everyone the user named, plus the payer (they ate too).
    seen: set[int] = set()
    people = list(participants)
    if payer_user_id is not None:
        people.append((payer_user_id, payer_name))
    for uid, uname in people:
        if uid in seen:
            continue
        seen.add(uid)
        updated = await billsplit.toggle_claim(view.id, 0, uid, uname)
        if updated is not None:
            view = updated
    return view


# ---------------------------------------------------------------------------
# Guided flow
# ---------------------------------------------------------------------------
def _who_paid_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🙋 I paid", callback_data="xp|payer|me")],
            [InlineKeyboardButton("👥 Someone else paid", callback_data="xp|payer|other")],
            [InlineKeyboardButton("✖️ Cancel", callback_data="xp|cancel")],
        ]
    )


async def _prompt(
    update: Update, context: ContextTypes.DEFAULT_TYPE, step: str, text: str
) -> None:
    chat, user = update.effective_chat, update.effective_user
    sent = await context.bot.send_message(
        chat.id, text, parse_mode="HTML", reply_markup=ForceReply(selective=True)
    )
    context.chat_data[_AWAIT_KEY] = {
        "step": step,
        "user_id": user.id if user else None,
        "prompt_id": sent.message_id,
    }


async def _advance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask for whatever the draft is still missing, else build the expense."""
    draft = context.chat_data.get(_DRAFT_KEY) or {}

    if not draft.get("payer_name"):
        await update.effective_message.reply_text(
            "💸 <b>New expense</b> — who paid?",
            parse_mode="HTML",
            reply_markup=_who_paid_keyboard(),
        )
        return
    if not draft.get("description"):
        await _prompt(
            update, context, "description",
            "📝 What was it for? <i>(e.g. photo booth, Grab, movie tickets)</i>",
        )
        return
    if draft.get("amount") is None:
        await _prompt(
            update, context, "amount",
            "💵 How much was it in total? <i>(e.g. 20 or 20.50)</i>",
        )
        return
    await _finish(update, context)


async def _finish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Build the bill and post the claim board."""
    chat = update.effective_chat
    draft = context.chat_data.pop(_DRAFT_KEY, {}) or {}
    context.chat_data.pop(_AWAIT_KEY, None)

    participants = await _resolve_people(chat.id, draft.get("participants") or [])
    participants += [tuple(p) for p in (draft.get("extra_people") or [])]
    view = await create_expense(
        chat.id,
        draft.get("payer_user_id"),
        draft.get("payer_name") or "Someone",
        draft.get("description") or "Expense",
        float(draft.get("amount") or 0),
        participants,
    )
    if view is None:
        await update.effective_message.reply_text(
            "⚠️ This group isn't registered yet. Send /start first."
        )
        return

    hint = (
        "\n\n<i>I've pre-ticked the people you named — everyone else can tap "
        "to join in.</i>"
        if participants
        else ""
    )
    sent = await update.effective_message.reply_text(
        billsplit.bill_text(view) + hint,
        parse_mode="HTML",
        reply_markup=billsplit.bill_keyboard(view),
    )
    try:
        await billsplit.set_menu_message_id(view.id, sent.message_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def splitexpense_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, user, chat = update.effective_message, update.effective_user, update.effective_chat
    if chat is None or chat.type == "private":
        await msg.reply_text("Use /splitexpense inside your group chat.")
        return

    raw = " ".join(context.args) if context.args else ""
    parsed = parse_expense(raw)
    sender_name = billsplit._display_name(user)

    draft: dict = {
        "participants": parsed.participants,
        "description": parsed.description,
        "amount": parsed.amount,
    }
    # "X paid … with me and Y" — the sender is in the split even though
    # someone else footed it.
    if parsed.include_sender and not parsed.payer_is_sender and user:
        draft["extra_people"] = [(user.id, sender_name)]
    if parsed.payer_is_sender:
        draft["payer_user_id"] = user.id if user else None
        draft["payer_name"] = sender_name
    elif parsed.payer_name:
        resolved = await _resolve_people(chat.id, [parsed.payer_name])
        uid, display = resolved[0]
        draft["payer_user_id"] = None if is_pseudo(uid) else uid
        draft["payer_name"] = display
    context.chat_data[_DRAFT_KEY] = draft

    await _advance(update, context)


# ---------------------------------------------------------------------------
# Free-text + callback handling
# ---------------------------------------------------------------------------
async def maybe_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Consume a pending /splitexpense ForceReply. True when handled."""
    pending = context.chat_data.get(_AWAIT_KEY)
    msg, user = update.effective_message, update.effective_user
    if not pending or not msg or not msg.text:
        return False
    if pending.get("user_id") and user and user.id != pending["user_id"]:
        return False
    if msg.reply_to_message and msg.reply_to_message.message_id != pending.get("prompt_id"):
        return False

    context.chat_data.pop(_AWAIT_KEY, None)
    draft = context.chat_data.setdefault(_DRAFT_KEY, {})
    value = msg.text.strip()
    step = pending.get("step")

    if step == "description":
        draft["description"] = value[:120]
    elif step == "amount":
        match = _AMOUNT.search(value)
        if not match:
            await msg.reply_text("That's not a number I recognise — try e.g. <code>20</code>.", parse_mode="HTML")
            await _prompt(update, context, "amount", "💵 How much was it in total?")
            return True
        draft["amount"] = float(match.group(1))
    elif step == "payer_name":
        resolved = await _resolve_people(update.effective_chat.id, [value])
        uid, display = resolved[0]
        draft["payer_user_id"] = None if is_pseudo(uid) else uid
        draft["payer_name"] = display
    else:
        return False

    await _advance(update, context)
    return True


async def on_expense_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router for the guided flow (`xp|…`)."""
    query = update.callback_query
    user, chat = update.effective_user, update.effective_chat
    if query is None or chat is None:
        if query:
            await query.answer()
        return

    parts = (query.data or "").split("|")
    action = parts[1] if len(parts) > 1 else ""
    sub = parts[2] if len(parts) > 2 else None

    if action == "cancel":
        context.chat_data.pop(_DRAFT_KEY, None)
        context.chat_data.pop(_AWAIT_KEY, None)
        await query.answer("Cancelled")
        await query.edit_message_text("💸 Expense cancelled.")
        return

    if action == "payer":
        draft = context.chat_data.setdefault(_DRAFT_KEY, {})
        if sub == "me":
            draft["payer_user_id"] = user.id if user else None
            draft["payer_name"] = billsplit._display_name(user)
            await query.answer()
            await query.edit_message_text(
                f"💸 <b>New expense</b> — paid by {billsplit._esc(draft['payer_name'])}.",
                parse_mode="HTML",
            )
            await _advance(update, context)
            return
        if sub == "other":
            from app.bot import services

            candidates = await services.list_payer_candidates(chat.id)
            if not candidates:
                await query.answer()
                await query.edit_message_text("💸 <b>New expense</b>", parse_mode="HTML")
                await _prompt(update, context, "payer_name", "👤 Who paid? Reply with their name:")
                return
            context.chat_data["xp_cands"] = [
                (c.display_name, c.telegram_user_id) for c in candidates
            ]
            rows = [
                [InlineKeyboardButton(c.display_name[:44], callback_data=f"xp|pick|{i}")]
                for i, c in enumerate(candidates)
            ]
            rows.append([InlineKeyboardButton("✖️ Cancel", callback_data="xp|cancel")])
            await query.answer()
            await query.edit_message_text(
                "👤 <b>Who paid?</b>", parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return

    if action == "pick" and sub is not None:
        cands = context.chat_data.get("xp_cands") or []
        try:
            name, uid = cands[int(sub)]
        except (ValueError, IndexError):
            await query.answer("That expired — run /splitexpense again.", show_alert=True)
            return
        draft = context.chat_data.setdefault(_DRAFT_KEY, {})
        draft["payer_user_id"] = uid
        draft["payer_name"] = name
        await query.answer()
        await query.edit_message_text(
            f"💸 <b>New expense</b> — paid by {billsplit._esc(name)}.", parse_mode="HTML"
        )
        await _advance(update, context)
        return

    await query.answer()
