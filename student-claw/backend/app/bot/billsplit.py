"""
Receipt-based bill splitting (the overhauled flow).

How it works
────────────
1. Someone pays the bill, then sends the receipt photo and replies to it with
   /splitbill (or runs /splitbill right after posting it).
2. The receipt is OCR'd by a vision model with a *structured* prompt that
   returns JSON: line items, subtotal, service charge, GST (and whether it is
   already included in prices), discounts and the grand total.
3. Agnes posts an interactive message: everyone taps the items they ordered.
   Multiple people tapping the same item share it equally.
4. The payer hits ✅ Split it — each person owes their items plus a share of
   the service charge/GST *proportional to what they ordered*. The breakdown
   is posted with the payer's PayNow number and a scannable PayNow QR code.

The finalised breakdown is also logged into the chat history so you can later
ask Agnes "who ordered what at Haidilao?".
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType
from telegram.ext import ContextTypes

from app.database.connection import session_scope
from app.database.models import (
    Bill,
    BillClaim,
    BillItem,
    BillRating,
    ContentType,
    PayProfile,
    Project,
)

logger = logging.getLogger("student_claw.bot.billsplit")


# ---------------------------------------------------------------------------
# Receipt OCR (structured)
# ---------------------------------------------------------------------------
_RECEIPT_PROMPT = """You are reading a restaurant/shop receipt photographed by a user.
Extract it into STRICT JSON (no markdown fences, no commentary) with exactly this shape:

{
  "merchant": "restaurant name or null",
  "currency": "3-letter code, default SGD",
  "items": [{"name": "item name", "qty": 1, "total_price": 12.90}],
  "subtotal": 45.80,
  "service_charge": 4.58,
  "gst": 4.03,
  "gst_inclusive": false,
  "other_charges": 0,
  "discount": 0,
  "total": 54.41
}

Rules:
- "total_price" is the LINE total (qty x unit price), after any line discount.
- Merge duplicate lines only if they are truly the same item at the same price.
- "service_charge" is the 10% svc chg line if present, else 0.
- "gst" is the GST/tax line if present, else 0. Set "gst_inclusive": true if
  the receipt says GST is included in prices (e.g. "GST incl." / "Prices are
  inclusive of GST") — in that case gst is informational, not an add-on.
- "discount" is positive for money taken off the bill.
- "other_charges" covers takeaway/packaging/delivery fees etc.
- Use null for subtotal/total ONLY if truly unreadable. Numbers as plain
  decimals, never strings. Output ONLY the JSON object."""


class ReceiptOCRError(Exception):
    """Raised when the receipt cannot be read into structured data."""


@dataclass
class ParsedReceipt:
    merchant: Optional[str]
    currency: str
    items: list[dict[str, Any]]  # {"name", "qty", "total_price"}
    subtotal: float
    service_charge: float
    gst: float
    gst_inclusive: bool
    other_charges: float
    discount: float
    total: float
    warnings: list[str] = field(default_factory=list)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _parse_receipt_json(raw: str) -> ParsedReceipt:
    """Parse the model's JSON (tolerating markdown fences) and sanity-check it."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Last resort: grab the outermost object.
    if not text.startswith("{"):
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReceiptOCRError(f"Model returned unparseable JSON: {exc}") from exc

    items = []
    for it in data.get("items") or []:
        name = str(it.get("name") or "").strip()[:200]
        price = _to_float(it.get("total_price"))
        if not name or price <= 0:
            continue
        items.append({"name": name, "qty": _to_float(it.get("qty"), 1.0) or 1.0, "total_price": price})
    if not items:
        raise ReceiptOCRError("No priced line items found on the receipt.")

    warnings: list[str] = []
    items_sum = round(sum(i["total_price"] for i in items), 2)
    subtotal = _to_float(data.get("subtotal")) or items_sum
    if abs(subtotal - items_sum) > 0.05:
        warnings.append(
            f"Item lines add to {items_sum:.2f} but the receipt subtotal reads "
            f"{subtotal:.2f} — going with the item lines."
        )
        subtotal = items_sum

    gst_inclusive = bool(data.get("gst_inclusive"))
    service_charge = _to_float(data.get("service_charge"))
    gst = _to_float(data.get("gst"))
    other = _to_float(data.get("other_charges"))
    discount = abs(_to_float(data.get("discount")))

    addon_gst = 0.0 if gst_inclusive else gst
    computed_total = round(subtotal + service_charge + addon_gst + other - discount, 2)
    total = _to_float(data.get("total")) or computed_total
    if abs(total - computed_total) > 0.05:
        warnings.append(
            f"Receipt total reads {total:.2f} but items + charges add to "
            f"{computed_total:.2f} — using {computed_total:.2f}."
        )
        total = computed_total

    return ParsedReceipt(
        merchant=(str(data.get("merchant")).strip()[:200] if data.get("merchant") else None),
        currency=(str(data.get("currency") or "SGD").upper()[:8]),
        items=items,
        subtotal=subtotal,
        service_charge=service_charge,
        gst=gst,
        gst_inclusive=gst_inclusive,
        other_charges=other,
        discount=discount,
        total=total,
        warnings=warnings,
    )


async def ocr_receipt(image_bytes: bytes, mime: str = "image/jpeg") -> ParsedReceipt:
    """
    Run structured receipt OCR. Primary: the OpenRouter VLM (Qwen 2.5 VL).
    Fallback: the Agnes vision model. Raises ReceiptOCRError if both fail.
    """
    raw: Optional[str] = None

    # Primary: OpenRouter VLM.
    try:
        from app.ai.config import get_ai_settings
        from app.ai.parser import _build_vlm_client

        cfg = get_ai_settings()
        client = _build_vlm_client()
        import base64

        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        resp = await client.chat.completions.create(
            model=cfg.openrouter_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": _RECEIPT_PROMPT},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("OpenRouter receipt OCR failed (%s); trying Agnes vision.", exc)

    if not raw:
        try:
            from app.ai import vision

            raw = await vision.describe_image(image_bytes, mime, prompt=_RECEIPT_PROMPT)
        except Exception as exc:
            raise ReceiptOCRError(f"All vision models failed: {exc}") from exc

    return _parse_receipt_json(raw)


# ---------------------------------------------------------------------------
# PayNow QR (EMVCo / SGQR payload)
# ---------------------------------------------------------------------------
def _crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def _tlv(tag: str, value: str) -> str:
    return f"{tag}{len(value):02d}{value}"


def normalize_paynow_id(raw: str) -> Optional[tuple[str, str]]:
    """
    Normalise a user-entered PayNow identifier.
    Returns (proxy_type, proxy_value) — proxy_type '0'=mobile, '2'=UEN —
    or None if it doesn't look like either.
    """
    value = raw.strip().replace(" ", "").replace("-", "")
    if re.fullmatch(r"(\+65)?[89]\d{7}", value):
        return ("0", value if value.startswith("+65") else f"+65{value}")
    if re.fullmatch(r"[0-9]{8,10}[A-Za-z][0-9A-Za-z]{0,3}", value):  # UEN-ish
        return ("2", value.upper())
    return None


def paynow_qr_payload(
    proxy_type: str,
    proxy_value: str,
    *,
    name: str = "NA",
    amount: Optional[float] = None,
    reference: Optional[str] = None,
) -> str:
    """Build an EMVCo PayNow QR payload (amount stays editable in the bank app)."""
    merchant_info = (
        _tlv("00", "SG.PAYNOW")
        + _tlv("01", proxy_type)
        + _tlv("02", proxy_value)
        + _tlv("03", "1")  # amount editable by the payer
    )
    payload = (
        _tlv("00", "01")
        + _tlv("01", "11")  # static QR, reusable
        + _tlv("26", merchant_info)
        + _tlv("52", "0000")
        + _tlv("53", "702")  # SGD
    )
    if amount is not None and amount > 0:
        payload += _tlv("54", f"{amount:.2f}")
    payload += _tlv("58", "SG") + _tlv("59", (name or "NA")[:25]) + _tlv("60", "Singapore")
    if reference:
        payload += _tlv("62", _tlv("01", reference[:25]))
    payload += "6304"
    payload += f"{_crc16_ccitt(payload.encode('ascii')):04X}"
    return payload


def render_qr_png(payload: str) -> Optional[bytes]:
    """Render a QR PNG for the payload; None if the qrcode lib is missing."""
    try:
        import qrcode
    except ImportError:
        logger.info("qrcode library not installed; skipping QR render.")
        return None
    img = qrcode.make(payload)
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")  # PIL backend
    except TypeError:
        img.save(buf)  # PyPNG backend (no PIL installed)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------
async def set_pay_profile(user_id: int, display_name: str, paynow_id: str) -> None:
    async with session_scope() as session:
        row = await session.scalar(
            select(PayProfile).where(PayProfile.telegram_user_id == user_id)
        )
        if row is None:
            session.add(
                PayProfile(
                    telegram_user_id=user_id,
                    display_name=display_name[:100],
                    paynow_id=paynow_id[:32],
                )
            )
        else:
            row.display_name = display_name[:100]
            row.paynow_id = paynow_id[:32]


async def get_pay_profile(user_id: Optional[int]) -> Optional[str]:
    if user_id is None:
        return None
    async with session_scope() as session:
        return await session.scalar(
            select(PayProfile.paynow_id).where(PayProfile.telegram_user_id == user_id)
        )


@dataclass
class BillView:
    """Detached snapshot of a bill + items + claims for rendering/maths."""

    id: str
    chat_id: int
    kind: str
    payer_user_id: Optional[int]
    payer_name: str
    merchant: Optional[str]
    currency: str
    items_subtotal: float
    service_charge: float
    gst: float
    other_charges: float
    discount: float
    total: float
    status: str
    menu_message_id: Optional[int]
    # items: (position, name, qty, total_price, [(user_id, user_name), ...])
    items: list[tuple[int, str, float, float, list[tuple[int, str]]]]


def _bill_view(bill: Bill) -> BillView:
    return BillView(
        id=str(bill.id),
        chat_id=bill.chat_id,
        kind=bill.kind,
        payer_user_id=bill.payer_user_id,
        payer_name=bill.payer_name,
        merchant=bill.merchant,
        currency=bill.currency,
        items_subtotal=float(bill.items_subtotal),
        service_charge=float(bill.service_charge),
        gst=float(bill.gst),
        other_charges=float(bill.other_charges),
        discount=float(bill.discount),
        total=float(bill.total),
        status=bill.status,
        menu_message_id=bill.menu_message_id,
        items=[
            (
                item.position,
                item.name,
                float(item.quantity),
                float(item.total_price),
                [(c.user_id, c.user_name) for c in item.claims],
            )
            for item in bill.items
        ],
    )


_BILL_LOAD = (
    select(Bill)
    .options(selectinload(Bill.items).selectinload(BillItem.claims))
)


async def set_rating(bill_id: str, user_id: int, user_name: str, stars: int) -> None:
    """Record (or change) one person's star vote for a bill."""
    async with session_scope() as session:
        existing = await session.scalar(
            select(BillRating).where(
                BillRating.bill_id == uuid.UUID(bill_id),
                BillRating.user_id == user_id,
            )
        )
        if existing is not None:
            existing.stars = stars
            existing.user_name = user_name[:100]
            return
        session.add(
            BillRating(
                id=uuid.uuid4(),
                bill_id=uuid.UUID(bill_id),
                user_id=user_id,
                user_name=user_name[:100],
                stars=stars,
            )
        )


async def get_ratings(bill_id: str) -> tuple[float, int, list[tuple[str, int]]]:
    """(average, count, [(name, stars)]) for a bill."""
    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(BillRating).where(BillRating.bill_id == uuid.UUID(bill_id))
            )
        ).all()
    if not rows:
        return 0.0, 0, []
    votes = [(r.user_name, int(r.stars)) for r in rows]
    return round(sum(s for _, s in votes) / len(votes), 2), len(votes), votes


async def create_bill(
    chat_id: int,
    payer_user_id: Optional[int],
    payer_name: str,
    receipt: ParsedReceipt,
    kind: str = "receipt",
) -> Optional[BillView]:
    """Create a new open bill (cancelling any previous open one). None if the
    chat isn't registered."""
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return None

        stale = (
            await session.scalars(
                select(Bill).where(Bill.chat_id == chat_id, Bill.status == "open")
            )
        ).all()
        for b in stale:
            b.status = "cancelled"

        bill = Bill(
            id=uuid.uuid4(),
            project_id=project.id,
            chat_id=chat_id,
            kind=kind,
            payer_user_id=payer_user_id,
            payer_name=payer_name[:100],
            merchant=receipt.merchant,
            currency=receipt.currency,
            items_subtotal=receipt.subtotal,
            service_charge=receipt.service_charge,
            gst=0.0 if receipt.gst_inclusive else receipt.gst,
            other_charges=receipt.other_charges,
            discount=receipt.discount,
            total=receipt.total,
            status="open",
        )
        session.add(bill)
        for pos, it in enumerate(receipt.items):
            session.add(
                BillItem(
                    id=uuid.uuid4(),
                    bill_id=bill.id,
                    position=pos,
                    name=it["name"],
                    quantity=it["qty"],
                    total_price=it["total_price"],
                )
            )
        await session.flush()
        loaded = await session.scalar(_BILL_LOAD.where(Bill.id == bill.id))
        return _bill_view(loaded)


async def get_bill(bill_id: str) -> Optional[BillView]:
    try:
        bid = uuid.UUID(bill_id)
    except ValueError:
        return None
    async with session_scope() as session:
        bill = await session.scalar(_BILL_LOAD.where(Bill.id == bid))
        return _bill_view(bill) if bill else None


async def get_open_bill(chat_id: int) -> Optional[BillView]:
    async with session_scope() as session:
        bill = await session.scalar(
            _BILL_LOAD.where(Bill.chat_id == chat_id, Bill.status == "open")
            .order_by(Bill.created_at.desc())
        )
        return _bill_view(bill) if bill else None


async def set_menu_message_id(bill_id: str, message_id: int) -> None:
    async with session_scope() as session:
        bill = await session.get(Bill, uuid.UUID(bill_id))
        if bill is not None:
            bill.menu_message_id = message_id


async def toggle_claim(
    bill_id: str, position: int, user_id: int, user_name: str
) -> Optional[BillView]:
    """Toggle a user's claim on one item; returns the refreshed view."""
    try:
        bid = uuid.UUID(bill_id)
    except ValueError:
        return None
    async with session_scope() as session:
        bill = await session.scalar(_BILL_LOAD.where(Bill.id == bid))
        if bill is None or bill.status != "open":
            return _bill_view(bill) if bill else None
        item = next((i for i in bill.items if i.position == position), None)
        if item is None:
            return _bill_view(bill)
        existing = next((c for c in item.claims if c.user_id == user_id), None)
        if existing is not None:
            await session.delete(existing)
            item.claims.remove(existing)
        else:
            claim = BillClaim(
                id=uuid.uuid4(),
                bill_item_id=item.id,
                user_id=user_id,
                user_name=user_name[:100],
            )
            session.add(claim)
            item.claims.append(claim)
        await session.flush()
        return _bill_view(bill)


async def set_bill_status(bill_id: str, status: str) -> None:
    async with session_scope() as session:
        bill = await session.get(Bill, uuid.UUID(bill_id))
        if bill is not None:
            bill.status = status


# ---------------------------------------------------------------------------
# Split maths
# ---------------------------------------------------------------------------
@dataclass
class PersonShare:
    user_id: int
    name: str
    items: list[tuple[str, float]]  # (label, share amount)
    items_total: float = 0.0
    charges: float = 0.0
    total: float = 0.0


def compute_split(view: BillView) -> tuple[list[PersonShare], list[str]]:
    """
    Split the bill. Each claimed item is divided equally among its claimants;
    unclaimed items are divided among everyone who claimed anything; the net
    charges (service charge + GST + other − discount) are apportioned
    proportionally to each person's food subtotal. Rounding residue lands on
    the payer (or the largest share) so the shares sum exactly to the total.
    """
    notes: list[str] = []
    people: dict[int, PersonShare] = {}
    for _, _, _, _, claims in view.items:
        for uid, uname in claims:
            people.setdefault(uid, PersonShare(user_id=uid, name=uname, items=[]))
    if not people:
        return [], ["Nobody has claimed any items yet."]

    unclaimed: list[tuple[str, float]] = []
    for _, name, qty, price, claims in view.items:
        label = f"{name}" + (f" ×{qty:g}" if qty and qty != 1 else "")
        if claims:
            share = price / len(claims)
            for uid, _ in claims:
                p = people[uid]
                suffix = f" (÷{len(claims)})" if len(claims) > 1 else ""
                p.items.append((label + suffix, share))
                p.items_total += share
        else:
            unclaimed.append((label, price))

    if unclaimed:
        n = len(people)
        for label, price in unclaimed:
            share = price / n
            for p in people.values():
                p.items.append((f"{label} (unclaimed ÷{n})", share))
                p.items_total += share
        notes.append(
            f"{len(unclaimed)} unclaimed item(s) were split equally among everyone."
        )

    charges = view.service_charge + view.gst + view.other_charges - view.discount
    base = sum(p.items_total for p in people.values()) or 1.0
    for p in people.values():
        p.charges = charges * (p.items_total / base)
        p.total = round(p.items_total + p.charges, 2)
        p.items_total = round(p.items_total, 2)
        p.charges = round(p.charges, 2)

    # Pin the rounding residue on the payer (or the biggest share).
    residue = round(view.total - sum(p.total for p in people.values()), 2)
    if abs(residue) >= 0.01:
        target = people.get(view.payer_user_id or -1) or max(
            people.values(), key=lambda p: p.total
        )
        target.total = round(target.total + residue, 2)

    ordered = sorted(people.values(), key=lambda p: -p.total)
    return ordered, notes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _money(v: float, cur: str = "SGD") -> str:
    sym = "$" if cur in ("SGD", "USD", "AUD") else f"{cur} "
    return f"{sym}{v:,.2f}"


def unit_price(qty: float, total_price: float) -> float:
    """Per-unit cost. Receipts list line totals, but people recognise the
    per-item price ('$12 each'), so that's what we lead with."""
    return total_price / qty if qty else total_price


def _item_price_label(qty: float, price: float, cur: str) -> str:
    """'2 × $12.00 = $24.00' for multiples, plain price for singles."""
    if qty and qty != 1:
        return f"{qty:g} × {_money(unit_price(qty, price), cur)} = {_money(price, cur)}"
    return _money(price, cur)


def bill_text(view: BillView) -> str:
    cur = view.currency
    kind_icon = "🧾" if view.kind == "receipt" else "💸"
    head = (
        f"{kind_icon} <b>{_esc(view.merchant or 'Receipt')}</b> — "
        f"paid by <b>{_esc(view.payer_name)}</b>\n"
    )
    lines = []
    for pos, name, qty, price, claims in view.items:
        who = ", ".join(_esc(n) for _, n in claims) if claims else "—"
        lines.append(
            f"{pos + 1}. {_esc(name)} · {_item_price_label(qty, price, cur)}"
            f"  <i>[{who}]</i>"
        )
    charges = []
    if view.service_charge:
        charges.append(f"Svc charge {_money(view.service_charge, cur)}")
    if view.gst:
        charges.append(f"GST {_money(view.gst, cur)}")
    if view.other_charges:
        charges.append(f"Other {_money(view.other_charges, cur)}")
    if view.discount:
        charges.append(f"Discount −{_money(view.discount, cur)}")
    charge_line = (" · ".join(charges) + "\n") if charges else ""
    return (
        head
        + "\n".join(lines)
        + f"\n\nSubtotal {_money(view.items_subtotal, cur)}\n"
        + charge_line
        + f"<b>Total {_money(view.total, cur)}</b>\n\n"
        "👇 <b>Tap the items you ordered.</b> Tap again to un-claim. "
        "Shared dish? Everyone who ate it taps it — the price splits between you. "
        "Service charge &amp; GST get split in proportion to your food."
    )


def bill_keyboard(view: BillView) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for pos, name, qty, price, claims in view.items:
        # Per-unit price on the button: it's what people compare against.
        each = unit_price(qty, price)
        label = f"{pos + 1}. {name[:22]} · {each:.2f}"
        if qty and qty != 1:
            label += f" ea ×{qty:g}"
        if claims:
            label += f" ✅{len(claims)}"
        rows.append(
            [InlineKeyboardButton(label, callback_data=f"bl|c|{view.id}|{pos}")]
        )
    rows.append([InlineKeyboardButton("⭐ Rate the food", callback_data=f"bl|rate|{view.id}")])
    rows.append(
        [
            InlineKeyboardButton("✅ Split it", callback_data=f"bl|f|{view.id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"bl|x|{view.id}"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def stars_keyboard(bill_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("⭐" * n, callback_data=f"bl|star|{bill_id}|{n}")
                for n in (1, 2, 3)
            ],
            [
                InlineKeyboardButton("⭐" * n, callback_data=f"bl|star|{bill_id}|{n}")
                for n in (4, 5)
            ],
            [InlineKeyboardButton("🔙 Back to the bill", callback_data=f"bl|back|{bill_id}")],
        ]
    )


def render_stars(avg: float) -> str:
    """Round to the nearest half star and draw it."""
    full = int(avg)
    half = 1 if (avg - full) >= 0.5 else 0
    return "★" * full + ("½" if half else "") + "☆" * (5 - full - half)


def breakdown_text(
    view: BillView,
    shares: list[PersonShare],
    notes: list[str],
    paynow: Optional[str],
    rating: Optional[tuple[float, int]] = None,
) -> str:
    cur = view.currency
    out = [f"💸 <b>Bill split — {_esc(view.merchant or 'receipt')}</b> (total {_money(view.total, cur)})\n"]
    for p in shares:
        out.append(f"<b>{_esc(p.name)} → {_money(p.total, cur)}</b>")
        for label, amt in p.items:
            out.append(f"   • {_esc(label)} — {_money(round(amt, 2), cur)}")
        if p.charges:
            out.append(f"   • svc + GST share — {_money(p.charges, cur)}")
    if notes:
        out.append("")
        out.extend(f"ℹ️ {_esc(n)}" for n in notes)

    # Group verdict on the food.
    if rating and rating[1]:
        avg, count = rating
        out.append(
            f"\n⭐ <b>Group rating: {render_stars(avg)} {avg:.1f}/5</b> "
            f"<i>({count} vote{'s' if count != 1 else ''})</i>"
        )
    elif view.kind == "receipt":
        out.append("\n⭐ <i>No ratings yet — tap ⭐ on the bill to rate the food.</i>")

    out.append("")
    payer = _esc(view.payer_name)
    if paynow:
        out.append(f"💳 Pay <b>{payer}</b> via PayNow: <code>{_esc(paynow)}</code>")
    else:
        out.append(
            f"💳 Pay <b>{payer}</b> directly — they haven't set a PayNow number "
            "yet (they can with /paynow &lt;mobile&gt;)."
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
def _is_group(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)


def _display_name(user) -> str:
    if user is None:
        return "Someone"
    return user.first_name or (f"@{user.username}" if user.username else "Someone")


async def _find_receipt(update: Update) -> Optional[tuple[bytes, str, Optional[object]]]:
    """
    Locate the receipt image and, crucially, WHO SENT IT.

    Anyone can run /splitbill, so the person invoking the command is not
    necessarily the person who paid. The uploader is the better default guess,
    and we confirm it before charging anyone.

    Returns (image_bytes, mime, uploader_user) — uploader may be None for
    forwarded or anonymous posts.
    """
    msg = update.effective_message
    for candidate in (msg.reply_to_message, msg):
        if candidate is None:
            continue
        if candidate.photo:
            tg_file = await candidate.photo[-1].get_file()
            data = bytes(await tg_file.download_as_bytearray())
            return data, "image/jpeg", candidate.from_user
        doc = candidate.document
        if doc is not None and (doc.mime_type or "").startswith("image/"):
            tg_file = await doc.get_file()
            data = bytes(await tg_file.download_as_bytearray())
            return data, doc.mime_type, candidate.from_user
    return None


# Pending bills awaiting a "who paid?" answer, keyed by a short token so the
# id fits comfortably inside Telegram's 64-byte callback_data limit.
_PENDING_KEY = "splitbill_pending"


def _payer_confirm_keyboard(token: str, has_guess: bool) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if has_guess:
        rows.append([InlineKeyboardButton("✅ Yes, they paid", callback_data=f"bl|pyes|{token}")])
    rows.append(
        [InlineKeyboardButton("👥 Someone else paid", callback_data=f"bl|pwho|{token}")]
    )
    rows.append([InlineKeyboardButton("✖️ Cancel", callback_data=f"bl|pcan|{token}")])
    return InlineKeyboardMarkup(rows)


def _payer_picker_keyboard(token: str, candidates: list) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                cand.display_name[:44], callback_data=f"bl|ppick|{token}|{i}"
            )
        ]
        for i, cand in enumerate(candidates)
    ]
    rows.append([InlineKeyboardButton("✖️ Cancel", callback_data=f"bl|pcan|{token}")])
    return InlineKeyboardMarkup(rows)


def _receipt_summary(receipt: ParsedReceipt) -> str:
    where = f" at <b>{_esc(receipt.merchant)}</b>" if receipt.merchant else ""
    return (
        f"🧾 Read the receipt{where} — "
        f"<b>{_money(receipt.total, receipt.currency)}</b> "
        f"across {len(receipt.items)} item{'s' if len(receipt.items) != 1 else ''}."
    )


async def _open_claim_board(
    context: ContextTypes.DEFAULT_TYPE,
    message,
    chat_id: int,
    payer_user_id: Optional[int],
    payer_name: str,
    receipt: ParsedReceipt,
) -> None:
    """Create the bill for the confirmed payer and render the claim board."""
    view = await create_bill(chat_id, payer_user_id, payer_name, receipt)
    if view is None:
        await message.edit_text("⚠️ This group isn't registered yet. Send /start first.")
        return
    warn = ("\n\n⚠️ " + "\n⚠️ ".join(receipt.warnings)) if receipt.warnings else ""
    sent = await message.edit_text(
        bill_text(view) + warn, parse_mode="HTML", reply_markup=bill_keyboard(view)
    )
    try:
        await set_menu_message_id(view.id, sent.message_id)
    except Exception:  # best-effort bookkeeping
        pass


async def splitbill_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Entry point: OCR the receipt, confirm who actually paid, then open the
    claim board.

    Anyone in the group can run this — the payer is whoever the group confirms,
    defaulting to the person who uploaded the photo rather than the person who
    typed the command.
    """
    msg = update.effective_message
    chat = update.effective_chat
    if not _is_group(update):
        await msg.reply_text("Use /splitbill inside your group chat.")
        return

    found = await _find_receipt(update)
    if found is None:
        await msg.reply_text(
            "📸 Reply to a receipt photo with /splitbill (or send the photo, "
            "then reply to it). I'll read it and set up the split."
        )
        return

    image_bytes, mime, uploader = found
    placeholder = await msg.reply_text("🧾 Reading the receipt… give me a few seconds.")

    async def _work() -> None:
        try:
            receipt = await ocr_receipt(image_bytes, mime)
        except ReceiptOCRError as exc:
            logger.warning("Receipt OCR failed: %s", exc)
            await placeholder.edit_text(
                "😵 I couldn't read that receipt. Try a sharper, straight-on "
                "photo with the totals visible."
            )
            return

        token = uuid.uuid4().hex[:8]
        guess_id = uploader.id if uploader else None
        guess_name = _display_name(uploader) if uploader else None
        context.chat_data.setdefault(_PENDING_KEY, {})[token] = {
            "receipt": receipt,
            "guess_id": guess_id,
            "guess_name": guess_name,
            "chat_id": chat.id,
        }

        if guess_name:
            question = (
                f"{_receipt_summary(receipt)}\n\n"
                f"💳 <b>{_esc(guess_name)}</b> posted the receipt — did they pay for it?"
            )
        else:
            question = (
                f"{_receipt_summary(receipt)}\n\n"
                "💳 I can't tell who posted the receipt. <b>Who paid?</b>"
            )
        await placeholder.edit_text(
            question,
            parse_mode="HTML",
            reply_markup=_payer_confirm_keyboard(token, bool(guess_name)),
        )

    context.application.create_task(_work(), update=update)


async def _handle_payer_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, token: str, arg: Optional[str]
) -> None:
    """The 'who paid?' step, before any bill row exists."""
    query = update.callback_query
    chat = update.effective_chat
    pending = (context.chat_data.get(_PENDING_KEY) or {}).get(token)

    if pending is None:
        await query.answer("That bill setup expired — run /splitbill again.", show_alert=True)
        return

    if action == "pcan":
        context.chat_data[_PENDING_KEY].pop(token, None)
        await query.answer("Cancelled")
        await query.edit_message_text("🧾 Bill split cancelled.")
        return

    if action == "pwho":
        # Local import: app.bot.services pulls in the ORM layer, and this
        # module is imported during handler registration.
        from app.bot import services

        candidates = await services.list_payer_candidates(chat.id)
        if not candidates:
            await query.answer(
                "I don't know anyone in this group yet — add people via "
                "/mainmenu → Set → Set names.",
                show_alert=True,
            )
            return
        pending["candidates"] = [(c.display_name, c.telegram_user_id) for c in candidates]
        await query.answer()
        await query.edit_message_text(
            f"{_receipt_summary(pending['receipt'])}\n\n💳 <b>Who paid?</b>",
            parse_mode="HTML",
            reply_markup=_payer_picker_keyboard(token, candidates),
        )
        return

    if action == "pyes":
        payer_id, payer_name = pending["guess_id"], pending["guess_name"]
    elif action == "ppick":
        stored = pending.get("candidates") or []
        try:
            payer_name, payer_id = stored[int(arg or -1)]
        except (ValueError, IndexError):
            await query.answer("That option expired — try again.", show_alert=True)
            return
    else:
        await query.answer()
        return

    context.chat_data[_PENDING_KEY].pop(token, None)
    await query.answer("Setting up…")
    await _open_claim_board(
        context, query.message, chat.id, payer_id, payer_name, pending["receipt"]
    )


async def bill_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-post the current open bill's claim board."""
    msg = update.effective_message
    if not _is_group(update):
        await msg.reply_text("Use /bill inside your group chat.")
        return
    view = await get_open_bill(update.effective_chat.id)
    if view is None:
        await msg.reply_text("No open bill. Reply to a receipt photo with /splitbill to start one.")
        return
    sent = await msg.reply_text(
        bill_text(view), parse_mode="HTML", reply_markup=bill_keyboard(view)
    )
    await set_menu_message_id(view.id, sent.message_id)


async def paynow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Register (or show) the caller's PayNow number, used when bills settle."""
    msg = update.effective_message
    user = update.effective_user
    if user is None:
        return
    if not context.args:
        current = await get_pay_profile(user.id)
        await msg.reply_text(
            f"Your PayNow is set to: {current}" if current
            else "No PayNow set. Register with /paynow <mobile or UEN>, e.g. /paynow 91234567"
        )
        return
    parsed = normalize_paynow_id(" ".join(context.args))
    if parsed is None:
        await msg.reply_text(
            "That doesn't look like a PayNow mobile (8 digits starting 8/9) or UEN. Try again."
        )
        return
    _, proxy_value = parsed
    display = user.first_name or (f"@{user.username}" if user.username else "Someone")
    await set_pay_profile(user.id, display, proxy_value)
    await msg.reply_text(f"✅ PayNow saved: {proxy_value}. Bill splits will now point people at it.")


async def on_bill_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router for bill buttons (callback_data prefixed `bl|`)."""
    query = update.callback_query
    user = update.effective_user
    if query is None or user is None:
        return
    parts = (query.data or "").split("|")  # ["bl", action, bill_id|token, pos?]
    if len(parts) < 3:
        await query.answer()
        return
    action, bill_id = parts[1], parts[2]

    # "Who paid?" runs before a bill row exists, so it's keyed by a pending
    # token rather than a bill id — route it out before any DB lookup.
    if action in {"pyes", "pwho", "ppick", "pcan"}:
        await _handle_payer_callback(
            update, context, action, bill_id, parts[3] if len(parts) > 3 else None
        )
        return

    if action == "c" and len(parts) >= 4:
        name = user.first_name or (f"@{user.username}" if user.username else "Someone")
        view = await toggle_claim(bill_id, int(parts[3]), user.id, name)
        if view is None:
            await query.answer("Bill not found.", show_alert=True)
            return
        if view.status != "open":
            await query.answer("This bill is closed.", show_alert=True)
            return
        await query.answer("Updated ✅")
        try:
            await query.edit_message_text(
                bill_text(view), parse_mode="HTML", reply_markup=bill_keyboard(view)
            )
        except Exception:  # "message is not modified" and friends
            pass
        return

    view = await get_bill(bill_id)
    if view is None:
        await query.answer("Bill not found.", show_alert=True)
        return

    # ── Ratings: open to everyone, not just the payer ──
    if action == "rate":
        await query.answer()
        avg, count, _ = await get_ratings(view.id)
        header = (
            f"⭐ Current: <b>{render_stars(avg)} {avg:.1f}/5</b> from {count} "
            f"vote{'s' if count != 1 else ''}\n\n"
            if count
            else "⭐ No votes yet.\n\n"
        )
        await query.edit_message_text(
            f"{header}<b>How was the food at "
            f"{_esc(view.merchant or 'this place')}?</b>\nEveryone can vote once.",
            parse_mode="HTML",
            reply_markup=stars_keyboard(view.id),
        )
        return

    if action == "star" and len(parts) >= 4:
        try:
            stars = max(1, min(5, int(parts[3])))
        except ValueError:
            await query.answer()
            return
        voter = user.first_name or (f"@{user.username}" if user.username else "Someone")
        await set_rating(view.id, user.id, voter, stars)
        avg, count, _ = await get_ratings(view.id)
        await query.answer(f"You rated {stars}★")
        await query.edit_message_text(
            f"⭐ <b>{render_stars(avg)} {avg:.1f}/5</b> from {count} "
            f"vote{'s' if count != 1 else ''}\n\n"
            f"<b>How was the food at {_esc(view.merchant or 'this place')}?</b>\n"
            "Tap to change your vote.",
            parse_mode="HTML",
            reply_markup=stars_keyboard(view.id),
        )
        return

    if action == "back":
        await query.answer()
        try:
            await query.edit_message_text(
                bill_text(view), parse_mode="HTML", reply_markup=bill_keyboard(view)
            )
        except Exception:
            pass
        return

    # Only the payer may finalise/cancel.
    if view.payer_user_id is not None and user.id != view.payer_user_id:
        await query.answer(f"Only {view.payer_name} (who paid) can do that.", show_alert=True)
        return

    if action == "x":
        await set_bill_status(view.id, "cancelled")
        await query.answer("Cancelled")
        await query.edit_message_text("🧾 Bill split cancelled.")
        return

    if action == "f":
        shares, notes = compute_split(view)
        if not shares:
            await query.answer("Nobody has claimed anything yet!", show_alert=True)
            return
        await set_bill_status(view.id, "finalized")
        await query.answer("Splitting…")

        # Freeze who owes whom so it shows up in the pending-expenses summary
        # and can be marked paid / confirmed later.
        from app.bot import settle

        await settle.record_debts(
            view.id, view.chat_id, view.payer_user_id, view.payer_name, shares
        )

        paynow = await get_pay_profile(view.payer_user_id)
        avg, count, _ = await get_ratings(view.id)
        text = breakdown_text(view, shares, notes, paynow, (avg, count))
        await query.edit_message_text(
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("⭐ Rate the food", callback_data=f"bl|rate|{view.id}")],
                    [InlineKeyboardButton("💰 Pending expenses", callback_data="sx|open")],
                ]
            ),
        )

        chat = update.effective_chat
        # PayNow QR so people can just scan and pay.
        if paynow and chat is not None:
            parsed = normalize_paynow_id(paynow)
            if parsed:
                png = await asyncio.to_thread(
                    render_qr_png,
                    paynow_qr_payload(parsed[0], parsed[1], name=view.payer_name),
                )
                if png:
                    try:
                        await context.bot.send_photo(
                            chat_id=chat.id,
                            photo=png,
                            caption=f"📲 Scan to PayNow {view.payer_name} ({paynow}) — "
                            "enter your own amount from the breakdown above.",
                        )
                    except Exception as exc:
                        logger.warning("Could not send PayNow QR: %s", exc)

        # Log the breakdown into chat history so Agnes can answer
        # "who ordered what" later.
        if chat is not None:
            try:
                from app.ai import queue as embed_queue
                from app.bot import services

                plain = re.sub(r"<[^>]+>", "", text)
                log_id = await services.log_incoming_message(
                    chat_id=chat.id,
                    telegram_message_id=query.message.message_id if query.message else 0,
                    content_type=ContentType.text,
                    sender_telegram_user_id=None,
                    sender_telegram_username="Agnes",
                    received_at=datetime.now(timezone.utc),
                    raw_text=plain,
                )
                if log_id:
                    await embed_queue.enqueue_embed_job(
                        message_log_id=log_id, chat_id=chat.id, content_type="text"
                    )
            except Exception as exc:
                logger.warning("Could not log bill breakdown: %s", exc)
        return

    await query.answer()
