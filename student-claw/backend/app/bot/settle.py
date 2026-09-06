"""
Settling up — who owes whom across every bill, and marking debts paid.

When a bill is finalised, `record_debts` freezes one row per person per bill.
This module aggregates those rows across *all* bills in a chat into a single
pending-expenses view, and drives the pay/confirm lifecycle:

    created  →  paid_at        debtor pressed "I've paid"
             →  confirmed_at   creditor verified it — fully settled

Telegram constraint worth knowing: a group message has ONE keyboard shared by
everyone, so you cannot show different buttons to different people in the same
message. What you *can* do is see who tapped. So the shared summary carries
generic buttons, and tapping one posts a NEW message addressed to that person
containing only their own debts. Only their taps on it are honoured.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.database.connection import session_scope
from app.database.models import Bill, BillDebt

logger = logging.getLogger("student_claw.bot.settle")


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _money(v: float, cur: str = "SGD") -> str:
    sym = "$" if cur in ("SGD", "USD", "AUD") else f"{cur} "
    return f"{sym}{v:,.2f}"


def _btn(label: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=data)


# ---------------------------------------------------------------------------
# Writing debts
# ---------------------------------------------------------------------------
async def record_debts(
    bill_id: str,
    chat_id: int,
    creditor_user_id: Optional[int],
    creditor_name: str,
    shares: list,
) -> int:
    """
    Freeze one debt row per person for a finalised bill.

    The payer's own share is skipped (you can't owe yourself). Re-finalising a
    bill replaces its debts, so a corrected split doesn't double-count.
    """
    async with session_scope() as session:
        existing = (
            await session.scalars(
                select(BillDebt).where(BillDebt.bill_id == uuid.UUID(bill_id))
            )
        ).all()
        for row in existing:
            await session.delete(row)

        written = 0
        for share in shares:
            if creditor_user_id is not None and share.user_id == creditor_user_id:
                continue  # the payer doesn't owe themselves
            if share.total <= 0:
                continue
            session.add(
                BillDebt(
                    id=uuid.uuid4(),
                    bill_id=uuid.UUID(bill_id),
                    chat_id=chat_id,
                    debtor_user_id=share.user_id,
                    debtor_name=share.name[:100],
                    creditor_user_id=creditor_user_id,
                    creditor_name=creditor_name[:100],
                    amount=round(share.total, 2),
                )
            )
            written += 1
        return written


# ---------------------------------------------------------------------------
# Reading debts
# ---------------------------------------------------------------------------
@dataclass
class DebtRow:
    id: str
    bill_id: str
    debtor_user_id: Optional[int]
    debtor_name: str
    creditor_user_id: Optional[int]
    creditor_name: str
    amount: float
    paid: bool
    confirmed: bool
    merchant: Optional[str] = None


@dataclass
class PairTotal:
    """All outstanding debt from one person to another, across every bill."""

    debtor_user_id: Optional[int]
    debtor_name: str
    creditor_user_id: Optional[int]
    creditor_name: str
    total: float = 0.0
    claimed_paid: float = 0.0
    debt_ids: list[str] = field(default_factory=list)

    @property
    def fully_claimed(self) -> bool:
        return self.total > 0 and abs(self.claimed_paid - self.total) < 0.01


async def list_open_debts(chat_id: int) -> list[DebtRow]:
    """Every unconfirmed debt in the chat (newest bill first)."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(BillDebt, Bill.merchant)
                .join(Bill, Bill.id == BillDebt.bill_id)
                .where(
                    BillDebt.chat_id == chat_id,
                    BillDebt.confirmed_at.is_(None),
                )
                .order_by(BillDebt.created_at.desc())
            )
        ).all()
    return [
        DebtRow(
            id=str(d.id),
            bill_id=str(d.bill_id),
            debtor_user_id=d.debtor_user_id,
            debtor_name=d.debtor_name,
            creditor_user_id=d.creditor_user_id,
            creditor_name=d.creditor_name,
            amount=float(d.amount),
            paid=d.paid_at is not None,
            confirmed=d.confirmed_at is not None,
            merchant=merchant,
        )
        for d, merchant in rows
    ]


def aggregate(debts: list[DebtRow]) -> list[PairTotal]:
    """Collapse per-bill debts into one line per (debtor → creditor) pair."""
    pairs: dict[tuple, PairTotal] = {}
    for d in debts:
        key = (d.debtor_name.lower(), d.creditor_name.lower())
        pair = pairs.get(key)
        if pair is None:
            pair = PairTotal(
                debtor_user_id=d.debtor_user_id,
                debtor_name=d.debtor_name,
                creditor_user_id=d.creditor_user_id,
                creditor_name=d.creditor_name,
            )
            pairs[key] = pair
        pair.total = round(pair.total + d.amount, 2)
        if d.paid:
            pair.claimed_paid = round(pair.claimed_paid + d.amount, 2)
        pair.debt_ids.append(d.id)
    return sorted(pairs.values(), key=lambda p: -p.total)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------
async def mark_paid(debt_ids: list[str]) -> int:
    """Debtor claims they've paid — awaits the creditor's confirmation."""
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        n = 0
        for did in debt_ids:
            row = await session.get(BillDebt, uuid.UUID(did))
            if row is not None and row.paid_at is None:
                row.paid_at = now
                n += 1
        return n


async def confirm_paid(debt_ids: list[str]) -> int:
    """Creditor verifies receipt — the debt is now fully settled."""
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        n = 0
        for did in debt_ids:
            row = await session.get(BillDebt, uuid.UUID(did))
            if row is not None and row.confirmed_at is None:
                row.paid_at = row.paid_at or now
                row.confirmed_at = now
                n += 1
        return n


async def close_all_owed_to(chat_id: int, creditor_user_id: int) -> int:
    """Creditor closes every debt owed to them ('everyone has settled')."""
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(BillDebt).where(
                    BillDebt.chat_id == chat_id,
                    BillDebt.creditor_user_id == creditor_user_id,
                    BillDebt.confirmed_at.is_(None),
                )
            )
        ).all()
        for row in rows:
            row.paid_at = row.paid_at or now
            row.confirmed_at = now
        return len(rows)


# ---------------------------------------------------------------------------
# Rendering — the shared summary
# ---------------------------------------------------------------------------
def summary_text(pairs: list[PairTotal], currency: str = "SGD") -> str:
    if not pairs:
        return (
            "💰 <b>Pending expenses</b>\n\n"
            "🎉 Everyone's square — nothing outstanding.\n\n"
            "<i>Start one with /splitbill (receipt photo) or /splitexpense.</i>"
        )

    # Group by creditor so each person sees what they're owed in one block.
    by_creditor: dict[str, list[PairTotal]] = {}
    for p in pairs:
        by_creditor.setdefault(p.creditor_name, []).append(p)

    out = ["💰 <b>Pending expenses</b>\n"]
    grand = 0.0
    for creditor, rows in by_creditor.items():
        owed = round(sum(r.total for r in rows), 2)
        grand += owed
        out.append(f"\n<b>{_esc(creditor)}</b> is owed {_money(owed, currency)}")
        for r in rows:
            flag = " ⏳ <i>says paid</i>" if r.fully_claimed else ""
            out.append(f"   • {_esc(r.debtor_name)} — {_money(r.total, currency)}{flag}")
    out.append(f"\n<b>Total outstanding: {_money(round(grand, 2), currency)}</b>")
    out.append(
        "\n<i>Tap a button below — I'll show you only your own debts.</i>"
    )
    return "\n".join(out)


def summary_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [_btn("💸 I've paid someone", "sx|mine")],
            [_btn("✔️ Confirm payments to me", "sx|owed")],
            [_btn("🔄 Refresh", "sx|refresh")],
        ]
    )


async def show_summary(update: Update, context: ContextTypes.DEFAULT_TYPE, edit: bool = False) -> None:
    chat = update.effective_chat
    pairs = aggregate(await list_open_debts(chat.id))
    text, markup = summary_text(pairs), summary_keyboard()
    if edit and update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text, parse_mode="HTML", reply_markup=markup
            )
        except Exception:  # "message is not modified"
            pass
    else:
        await update.effective_message.reply_text(
            text, parse_mode="HTML", reply_markup=markup
        )


# ---------------------------------------------------------------------------
# Rendering — the per-person views
# ---------------------------------------------------------------------------
def _matches(user_id: Optional[int], name: str, row_uid: Optional[int], row_name: str) -> bool:
    """
    Match the tapping user to a debt row.

    Prefer the Telegram id, but fall back to the name when the row carries a
    *pseudo* id — the stable negative placeholder /splitexpense assigns to
    someone named in text who has no Telegram account we can see. Real ids are
    always positive, so the sign is a safe discriminator.
    """
    if row_uid is not None and row_uid > 0 and user_id is not None:
        return row_uid == user_id
    return row_name.strip().lower() == (name or "").strip().lower()


async def show_my_debts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A new message addressed to the tapping user, listing what THEY owe."""
    query = update.callback_query
    chat, user = update.effective_chat, update.effective_user
    name = user.first_name or (f"@{user.username}" if user.username else "You")

    pairs = [
        p
        for p in aggregate(await list_open_debts(chat.id))
        if _matches(user.id, name, p.debtor_user_id, p.debtor_name)
    ]
    unpaid = [p for p in pairs if not p.fully_claimed]

    if not pairs:
        await query.answer("You don't owe anyone right now. 🎉", show_alert=True)
        return
    if not unpaid:
        await query.answer("You've already marked everything paid — waiting on confirmation.", show_alert=True)
        return

    await query.answer()
    context.chat_data.setdefault("sx_mine", {})[str(user.id)] = {
        str(i): p.debt_ids for i, p in enumerate(unpaid)
    }
    rows = [
        [
            _btn(
                f"✅ Paid {p.creditor_name} {_money(p.total)}",
                f"sx|paid|{user.id}|{i}",
            )
        ]
        for i, p in enumerate(unpaid)
    ]
    total = round(sum(p.total for p in unpaid), 2)
    await context.bot.send_message(
        chat.id,
        f"💸 <b>{_esc(name)}</b>, you owe {_money(total)} in total.\n"
        "Tap once you've actually transferred the money — the other person "
        "then confirms it.",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def show_owed_to_me(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A new message addressed to the tapping user, listing what they're OWED."""
    query = update.callback_query
    chat, user = update.effective_chat, update.effective_user
    name = user.first_name or (f"@{user.username}" if user.username else "You")

    pairs = [
        p
        for p in aggregate(await list_open_debts(chat.id))
        if _matches(user.id, name, p.creditor_user_id, p.creditor_name)
    ]
    if not pairs:
        await query.answer("Nobody owes you anything right now.", show_alert=True)
        return

    await query.answer()
    context.chat_data.setdefault("sx_owed", {})[str(user.id)] = {
        str(i): p.debt_ids for i, p in enumerate(pairs)
    }
    rows = []
    for i, p in enumerate(pairs):
        mark = "⏳ says paid" if p.fully_claimed else "not yet"
        rows.append(
            [_btn(f"✔️ {p.debtor_name} {_money(p.total)} ({mark})", f"sx|conf|{user.id}|{i}")]
        )
    rows.append([_btn("✅ Confirm all & close", f"sx|closeall|{user.id}")])
    total = round(sum(p.total for p in pairs), 2)
    await context.bot.send_message(
        chat.id,
        f"📥 <b>{_esc(name)}</b>, you're owed {_money(total)}.\n"
        "Confirm each person once the money has actually landed.",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(rows),
    )


# ---------------------------------------------------------------------------
# Callback router (`sx|…`)
# ---------------------------------------------------------------------------
async def on_settle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat, user = update.effective_chat, update.effective_user
    if query is None or user is None or chat is None:
        if query:
            await query.answer()
        return

    parts = (query.data or "").split("|")  # ["sx", action, owner?, idx?]
    action = parts[1] if len(parts) > 1 else ""

    if action == "refresh":
        await query.answer("Refreshed")
        await show_summary(update, context, edit=True)
        return
    if action == "open":
        # Posts a NEW summary rather than editing — the caller's message (e.g.
        # a finalised bill breakdown) must survive.
        await query.answer()
        await show_summary(update, context, edit=False)
        return
    if action == "mine":
        await show_my_debts(update, context)
        return
    if action == "owed":
        await show_owed_to_me(update, context)
        return

    # The remaining actions live on a message addressed to ONE person. The
    # owner id is baked into callback_data so nobody can settle someone
    # else's debts by tapping their message.
    owner = parts[2] if len(parts) > 2 else ""
    if owner != str(user.id):
        await query.answer("That's not your list — tap the buttons on the summary.", show_alert=True)
        return

    if action == "closeall":
        n = await close_all_owed_to(chat.id, user.id)
        await query.answer(f"Closed {n}")
        await query.edit_message_text(
            f"✅ Closed {n} debt{'s' if n != 1 else ''} — everyone who owed you is settled."
        )
        await show_summary(update, context)
        return

    idx = parts[3] if len(parts) > 3 else None
    if action == "paid":
        store = (context.chat_data.get("sx_mine") or {}).get(owner) or {}
        ids = store.get(idx or "")
        if not ids:
            await query.answer("That list expired — open /settle_up again.", show_alert=True)
            return
        n = await mark_paid(ids)
        await query.answer(f"Marked {n} as paid")
        await query.edit_message_text(
            "💸 Marked as paid — waiting for them to confirm.\n"
            "<i>Open /settle_up for the latest.</i>",
            parse_mode="HTML",
        )
        return

    if action == "conf":
        store = (context.chat_data.get("sx_owed") or {}).get(owner) or {}
        ids = store.get(idx or "")
        if not ids:
            await query.answer("That list expired — open /settle_up again.", show_alert=True)
            return
        n = await confirm_paid(ids)
        await query.answer(f"Confirmed {n}")
        await query.edit_message_text("✅ Confirmed — that debt is settled.")
        await show_summary(update, context)
        return

    await query.answer()


async def settle_up_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The cross-bill pending-expenses summary."""
    chat = update.effective_chat
    if chat is None or chat.type == "private":
        await update.effective_message.reply_text("Use /settle_up inside your group chat.")
        return
    await show_summary(update, context)
