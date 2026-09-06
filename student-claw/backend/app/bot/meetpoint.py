"""
/meetpoint — find somewhere fair to meet, then route everyone there.

The flow
────────
1. Everyone adds where they're coming from (postal code or address). OneMap
   turns that into coordinates.
2. The group picks what they want to do — lepak, eat (specific cuisine),
   adventure, shopping, movie, or "surprise me". Free text is fine too.
3. Agnes proposes 5 concrete spots near the group's centroid. Don't like them?
   Generate a fresh list.
4. Someone replies to the suggestions with the option number to lock it in.
5. Each member gets their OWN public-transport directions — a separate,
   personally-addressed message, not one shared wall of text.

Session state (activity, current suggestions, chosen venue) lives in chat_data:
it's short-lived and per-chat, so there's no reason to persist it. Locations
DO persist, in member_locations, so nobody re-types their postal code weekly.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass

from telegram import (
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import ContextTypes

from app.ai import onemap
from app.bot import services

logger = logging.getLogger("student_claw.bot.meetpoint")

# chat_data keys
_SESSION_KEY = "meetpoint_session"
_AWAIT_KEY = "meetpoint_await"

SUGGESTION_COUNT = 5

# Preset activities. "Surprise me" deliberately has no cuisine follow-up.
ACTIVITIES: list[tuple[str, str]] = [
    ("lepak", "🧋 Lepak / chill"),
    ("eat", "🍜 Eat"),
    ("adventure", "🎢 Adventure"),
    ("shopping", "🛍 Shopping"),
    ("movie", "🎬 Movie"),
    ("surprise", "🎲 Surprise me"),
]
_ACTIVITY_LABELS = dict(ACTIVITIES)


@dataclass
class Suggestion:
    name: str
    area: str
    why: str

    @property
    def search_term(self) -> str:
        return f"{self.name} {self.area}".strip()


def _btn(label: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=data)


def _session(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.chat_data.setdefault(_SESSION_KEY, {})


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------
def _main_keyboard(has_locations: bool, has_activity: bool) -> InlineKeyboardMarkup:
    rows = [
        [_btn("📍 Add my location", "mp|loc|add")],
        [_btn("👥 Who's coming", "mp|loc|list")],
        [_btn("🎯 What shall we do?", "mp|act")],
    ]
    if has_locations and has_activity:
        rows.append([_btn("🚀 Find us a spot", "mp|find")])
    return InlineKeyboardMarkup(rows)


def _activity_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(ACTIVITIES), 2):
        rows.append([_btn(label, f"mp|act|{key}") for key, label in ACTIVITIES[i : i + 2]])
    rows.append([_btn("✍️ Something else", "mp|act|custom")])
    rows.append([_btn("🔙 Back", "mp|main")])
    return InlineKeyboardMarkup(rows)


def _locations_keyboard(locations: list) -> InlineKeyboardMarkup:
    rows = [
        [_btn(f"❌ {loc.display_name[:40]}", f"mp|loc|rm|{loc.id}")] for loc in locations[:15]
    ]
    rows.append([_btn("➕ Add mine", "mp|loc|add")])
    rows.append([_btn("🔙 Back", "mp|main")])
    return InlineKeyboardMarkup(rows)


def _header(session: dict, locations: list) -> str:
    activity = session.get("activity_label") or "<i>not chosen yet</i>"
    who = (
        ", ".join(_esc(loc.display_name) for loc in locations)
        if locations
        else "<i>nobody yet</i>"
    )
    return (
        "📍 <b>Meet point</b>\n\n"
        f"<b>Coming:</b> {who}\n"
        f"<b>Plan:</b> {activity}\n\n"
        "Add where you're travelling from, pick what you feel like doing, "
        "and I'll find somewhere that works for everyone."
    )


async def _render_main(update: Update, context: ContextTypes.DEFAULT_TYPE, edit: bool) -> None:
    chat = update.effective_chat
    session = _session(context)
    locations = await services.list_locations(chat.id) or []
    text = _header(session, locations)
    markup = _main_keyboard(bool(locations), bool(session.get("activity")))
    if edit and update.callback_query:
        await update.callback_query.edit_message_text(
            text, parse_mode="HTML", reply_markup=markup
        )
    else:
        await update.effective_message.reply_text(
            text, parse_mode="HTML", reply_markup=markup
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def meetpoint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None or chat.type == "private":
        await update.effective_message.reply_text("Use /meetpoint inside your group chat.")
        return
    if await services.get_group_state(chat.id) is None:
        await update.effective_message.reply_text(
            "⚠️ This group isn't registered yet. Send /start first."
        )
        return
    await _render_main(update, context, edit=False)


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------
def _centroid(locations: list) -> tuple[float, float] | None:
    pts = [(l.latitude, l.longitude) for l in locations if l.has_coords]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


_SUGGESTION_LINE = re.compile(r"^\s*\d+[).\-]\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")


def _parse_suggestions(raw: str) -> list[Suggestion]:
    """Parse the model's `name | area | why` lines, tolerating stray prose."""
    out: list[Suggestion] = []
    for line in (raw or "").splitlines():
        match = _SUGGESTION_LINE.match(line.strip())
        if match:
            out.append(
                Suggestion(
                    name=match.group(1)[:80],
                    area=match.group(2)[:60],
                    why=match.group(3)[:160],
                )
            )
    return out[:SUGGESTION_COUNT]


async def _generate_suggestions(
    chat_id: int, session: dict, locations: list, exclude: list[str]
) -> list[Suggestion]:
    """Ask Agnes for venue ideas near the group's midpoint."""
    from app.ai.agent import run_agent

    centre = _centroid(locations)
    areas = ", ".join(
        f"{l.display_name} ({l.address or l.raw_input})" for l in locations
    ) or "unknown"
    activity = session.get("activity_text") or session.get("activity_label") or "anything"

    centre_hint = (
        f"Their rough midpoint is latitude {centre[0]:.4f}, longitude {centre[1]:.4f}. "
        if centre
        else ""
    )
    avoid = (
        f"Do NOT suggest any of these again: {', '.join(exclude)}. "
        if exclude
        else ""
    )

    directive = (
        "You are picking places to meet in Singapore. "
        f"The group is travelling from: {areas}. {centre_hint}"
        f"They want: {activity}. {avoid}"
        f"Suggest exactly {SUGGESTION_COUNT} real, specific, currently-operating "
        "places in Singapore that are convenient by MRT/bus for everyone and fit "
        "what they want to do. Prefer well-known venues near MRT stations. "
        "Reply with ONLY the list, one per line, in exactly this format:\n"
        "1) Place name | Area or nearest MRT | one short reason\n"
        "No preamble, no closing text, no HTML."
    )

    try:
        raw = await run_agent(
            chat_id,
            "Suggest places for us to meet.",
            system_directive=directive,
            force_complex=True,
        )
    except Exception as exc:
        logger.warning("Suggestion generation failed for %s: %s", chat_id, exc)
        return []
    return _parse_suggestions(re.sub(r"<[^>]+>", "", raw or ""))


def _suggestions_text(session: dict, suggestions: list[Suggestion]) -> str:
    activity = session.get("activity_label") or "your plan"
    lines = [f"🎯 <b>{_esc(activity)}</b> — here's what I'd pick:\n"]
    for i, s in enumerate(suggestions, start=1):
        lines.append(
            f"<b>{i})</b> {_esc(s.name)} — <i>{_esc(s.area)}</i>\n     {_esc(s.why)}"
        )
    lines.append(
        "\n<b>Reply to this message with the number</b> you want (e.g. <code>3</code>), "
        "and I'll send everyone their own directions."
    )
    return "\n".join(lines)


def _suggestions_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [_btn("🔄 Give me 5 new ones", "mp|regen")],
            [_btn("🔙 Back", "mp|main")],
        ]
    )


async def _send_suggestions(
    update: Update, context: ContextTypes.DEFAULT_TYPE, *, regenerate: bool
) -> None:
    chat = update.effective_chat
    session = _session(context)
    locations = await services.list_locations(chat.id) or []

    if not locations:
        await context.bot.send_message(
            chat.id, "Nobody's added a location yet — tap 📍 Add my location first."
        )
        return
    if not session.get("activity"):
        await context.bot.send_message(
            chat.id, "Pick what you want to do first (🎯)."
        )
        return

    seen: list[str] = session.get("seen", []) if regenerate else []
    notice = await context.bot.send_message(
        chat.id, "🔍 Finding spots that work for everyone…"
    )

    suggestions = await _generate_suggestions(chat.id, session, locations, seen)
    if not suggestions:
        await notice.edit_text(
            "😕 Couldn't come up with suggestions just now. Try again in a moment."
        )
        return

    session["suggestions"] = [s.__dict__ for s in suggestions]
    session["seen"] = seen + [s.name for s in suggestions]

    sent = await notice.edit_text(
        _suggestions_text(session, suggestions),
        parse_mode="HTML",
        reply_markup=_suggestions_keyboard(),
    )
    # Remember which message the number-reply should be attached to.
    session["prompt_message_id"] = sent.message_id


# ---------------------------------------------------------------------------
# Confirmation → per-member directions
# ---------------------------------------------------------------------------
async def _directions_for(
    location, dest: onemap.Place
) -> str:
    """One member's journey, or a maps link when routing isn't available."""
    if not location.has_coords:
        return (
            "I don't have your coordinates — re-add your location and I'll route you.\n"
            f"📍 <a href=\"{dest.maps_url}\">Open the destination in Maps</a>"
        )
    route = await onemap.public_transport_route(
        location.latitude, location.longitude, dest.latitude, dest.longitude
    )
    if route is None:
        return (
            "Public-transport routing isn't set up, so here's the map instead:\n"
            f"📍 <a href=\"{dest.maps_url}\">Open in Maps</a>"
        )
    return onemap.render_route(route)


async def _confirm_choice(
    update: Update, context: ContextTypes.DEFAULT_TYPE, index: int
) -> None:
    """Lock in a numbered suggestion and fan out individual directions."""
    chat = update.effective_chat
    session = _session(context)
    raw_suggestions = session.get("suggestions") or []

    if not (1 <= index <= len(raw_suggestions)):
        await update.effective_message.reply_text(
            f"Pick a number between 1 and {len(raw_suggestions)}."
        )
        return

    chosen = Suggestion(**raw_suggestions[index - 1])
    locations = await services.list_locations(chat.id) or []

    status = await update.effective_message.reply_text(
        f"📌 <b>{_esc(chosen.name)}</b> it is — working out everyone's route…",
        parse_mode="HTML",
    )

    dest = await onemap.geocode(chosen.search_term) or await onemap.geocode(chosen.name)
    if dest is None:
        await status.edit_text(
            f"📌 <b>{_esc(chosen.name)}</b> ({_esc(chosen.area)})\n\n"
            "I couldn't pin that on the map, so I can't route everyone. "
            "Try another option.",
            parse_mode="HTML",
        )
        return

    session["chosen"] = chosen.__dict__
    header = (
        f"📌 <b>{_esc(chosen.name)}</b>\n"
        f"{_esc(dest.address or chosen.area)}\n"
        f"<a href=\"{dest.maps_url}\">Open in Maps</a>"
    )
    await status.edit_text(header, parse_mode="HTML", disable_web_page_preview=True)

    # Route everyone concurrently, then post each person their own message.
    routes = await asyncio.gather(
        *(_directions_for(loc, dest) for loc in locations), return_exceptions=True
    )
    for loc, result in zip(locations, routes):
        if isinstance(result, Exception):
            logger.warning("Routing failed for %s: %s", loc.display_name, result)
            body = "Couldn't work out your route — check the map link above."
        else:
            body = result
        mention = f"@{loc.telegram_username}" if loc.telegram_username else loc.display_name
        await context.bot.send_message(
            chat.id,
            f"🧭 <b>{_esc(mention)}</b> → {_esc(chosen.name)}\n"
            f"<i>from {_esc(loc.address or loc.raw_input)}</i>\n\n{body}",
            parse_mode="HTML",
            disable_web_page_preview=True,
        )


async def maybe_handle_number_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """
    If this message is a bare number replying to the suggestions, treat it as
    the group's choice. Returns True when consumed.
    """
    msg = update.effective_message
    session = context.chat_data.get(_SESSION_KEY) or {}
    prompt_id = session.get("prompt_message_id")
    if not prompt_id or not msg or not msg.text:
        return False
    if not (msg.reply_to_message and msg.reply_to_message.message_id == prompt_id):
        return False
    match = re.fullmatch(r"\s*(\d{1,2})\s*", msg.text)
    if not match:
        return False
    await _confirm_choice(update, context, int(match.group(1)))
    return True


# ---------------------------------------------------------------------------
# Free-text input (location / custom activity)
# ---------------------------------------------------------------------------
async def maybe_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Consume a pending ForceReply for /meetpoint. Returns True when handled."""
    pending = context.chat_data.get(_AWAIT_KEY)
    msg = update.effective_message
    user = update.effective_user
    if not pending or not msg or not msg.text:
        return False
    # Only the person who tapped the button, replying to that prompt.
    if pending.get("user_id") and user and user.id != pending["user_id"]:
        return False
    if msg.reply_to_message and msg.reply_to_message.message_id != pending.get("prompt_id"):
        return False

    context.chat_data.pop(_AWAIT_KEY, None)
    action = pending.get("action")
    text = msg.text.strip()

    if action == "location":
        await _save_location(update, context, text)
        return True
    if action == "activity":
        session = _session(context)
        session["activity"] = "custom"
        session["activity_text"] = text[:120]
        session["activity_label"] = text[:120]
        session.pop("seen", None)
        await msg.reply_text(
            f"Got it — <b>{_esc(text[:120])}</b>. Tap 🚀 in /meetpoint when everyone's in.",
            parse_mode="HTML",
        )
        return True
    return False


async def _save_location(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> None:
    chat = update.effective_chat
    user = update.effective_user
    name = (user.full_name if user else None) or "Someone"

    place = await onemap.geocode(text)
    ok = await services.upsert_location(
        chat.id,
        name,
        text,
        telegram_user_id=user.id if user else None,
        telegram_username=user.username if user else None,
        address=place.address if place else None,
        postal_code=place.postal_code if place else None,
        latitude=place.latitude if place else None,
        longitude=place.longitude if place else None,
    )
    if not ok:
        await update.effective_message.reply_text("Couldn't save that — try again.")
        return
    if place is None:
        await update.effective_message.reply_text(
            f"Saved <b>{_esc(text)}</b>, but I couldn't find it on the map — "
            "a 6-digit postal code works best, and I need one to route you.",
            parse_mode="HTML",
        )
        return
    await update.effective_message.reply_text(
        f"📍 Got you at <b>{_esc(place.address or place.name)}</b>.",
        parse_mode="HTML",
    )


async def _prompt(
    update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, text: str
) -> None:
    chat = update.effective_chat
    user = update.effective_user
    sent = await context.bot.send_message(
        chat.id, text, parse_mode="HTML", reply_markup=ForceReply(selective=True)
    )
    context.chat_data[_AWAIT_KEY] = {
        "action": action,
        "user_id": user.id if user else None,
        "prompt_id": sent.message_id,
    }


# ---------------------------------------------------------------------------
# Callback router (`mp|…`)
# ---------------------------------------------------------------------------
async def on_meetpoint_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    chat = update.effective_chat
    if query is None or chat is None:
        if query:
            await query.answer()
        return

    parts = (query.data or "").split("|")  # ["mp", action, sub?, id?]
    action = parts[1] if len(parts) > 1 else ""
    sub = parts[2] if len(parts) > 2 else None
    session = _session(context)

    if action == "main":
        await query.answer()
        await _render_main(update, context, edit=True)
        return

    if action == "loc":
        if sub == "add":
            await query.answer()
            await _prompt(
                update, context, "location",
                "📍 Reply with your <b>postal code</b> (or address), "
                "e.g. <code>487372</code>:",
            )
            return
        if sub == "rm" and len(parts) >= 4:
            await services.remove_location(chat.id, parts[3])
            await query.answer("Removed")
        else:
            await query.answer()
        locations = await services.list_locations(chat.id) or []
        body = "👥 <b>Who's coming</b>\n\n" + (
            "\n".join(
                f"• <b>{_esc(l.display_name)}</b> — {_esc(l.address or l.raw_input)}"
                + ("" if l.has_coords else " ⚠️ <i>not on map</i>")
                for l in locations
            )
            or "<i>Nobody yet.</i>"
        )
        await query.edit_message_text(
            body, parse_mode="HTML", reply_markup=_locations_keyboard(locations)
        )
        return

    if action == "act":
        if sub is None:
            await query.answer()
            await query.edit_message_text(
                "🎯 <b>What are we doing?</b>", parse_mode="HTML",
                reply_markup=_activity_keyboard(),
            )
            return
        if sub == "custom":
            await query.answer()
            await _prompt(
                update, context, "activity",
                "✍️ Reply with what you feel like doing, "
                "e.g. <code>korean bbq</code> or <code>quiet cafe to study</code>:",
            )
            return
        if sub == "eat":
            await query.answer()
            await _prompt(
                update, context, "activity",
                "🍜 What are we eating? Reply with a cuisine "
                "(e.g. <code>japanese</code>), or <code>anything</code>:",
            )
            return

        session["activity"] = sub
        session["activity_label"] = _ACTIVITY_LABELS.get(sub, sub)
        session["activity_text"] = (
            "anything fun — surprise them" if sub == "surprise" else _ACTIVITY_LABELS.get(sub, sub)
        )
        session.pop("seen", None)
        await query.answer("Set ✅")
        await _render_main(update, context, edit=True)
        return

    if action == "find":
        await query.answer("On it…")
        await _send_suggestions(update, context, regenerate=False)
        return

    if action == "regen":
        await query.answer("Fresh batch…")
        await _send_suggestions(update, context, regenerate=True)
        return

    await query.answer()
