"""
/news — latest headlines, fetched from public RSS feeds and summarised by
Agnes into a chatty group-friendly briefing.

No API keys needed: headlines come from public RSS (CNA, BBC, The Verge, BBC
Sport), and Agnes turns them into a short, personable summary. If the AI is
down, the raw headlines are posted instead so the command always works.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from app.ai.agent import run_agent

logger = logging.getLogger("student_claw.bot.news")

# category → list of RSS feed URLs (first that works wins).
FEEDS: dict[str, list[str]] = {
    "sg": [
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
        "https://www.straitstimes.com/news/singapore/rss.xml",
    ],
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
    ],
    "tech": [
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
    "sport": [
        "https://feeds.bbci.co.uk/sport/rss.xml",
    ],
    "business": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
}
_DEFAULT_CATEGORY = "sg"
_MAX_HEADLINES = 12

_NEWS_DIRECTIVE = (
    "The user asked for a news briefing. You are given today's raw headlines. "
    "Write a fun, scannable group-chat briefing in Telegram HTML: a one-line "
    "opener, then the 5-8 most interesting stories as short lines — <b>bold "
    "the topic</b>, one sentence of what happened, add your own quick take or "
    "quip where it fits naturally. End with a one-liner sign-off. Only use "
    "facts from the provided headlines; do not invent details beyond them."
)

_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return html.unescape(_TAG_STRIP_RE.sub("", text)).strip()


def _parse_feed(xml_text: str) -> list[tuple[str, str]]:
    """Parse RSS 2.0 or Atom into (title, description) pairs."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[tuple[str, str]] = []
    # RSS 2.0
    for item in root.iter("item"):
        title = _clean(item.findtext("title"))
        desc = _clean(item.findtext("description"))
        if title:
            out.append((title, desc))
    if out:
        return out
    # Atom
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        title = _clean(entry.findtext("a:title", namespaces=ns))
        desc = _clean(entry.findtext("a:summary", namespaces=ns))
        if title:
            out.append((title, desc))
    return out


async def fetch_headlines(category: str) -> list[tuple[str, str]]:
    """Fetch up to _MAX_HEADLINES (title, description) pairs for a category."""
    urls = FEEDS.get(category, FEEDS[_DEFAULT_CATEGORY])
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=True, headers={"User-Agent": "AgnesBot/1.0"}
    ) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("News feed %s failed: %s", url, exc)
                continue
            items = _parse_feed(resp.text)
            if items:
                return items[:_MAX_HEADLINES]
    return []


def _headlines_block(items: list[tuple[str, str]]) -> str:
    lines = []
    for title, desc in items:
        lines.append(f"- {title}" + (f" — {desc[:200]}" if desc else ""))
    return "\n".join(lines)


def _fallback_text(category: str, items: list[tuple[str, str]]) -> str:
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = "\n".join(f"• {esc(t)}" for t, _ in items[:8])
    return f"📰 <b>Top headlines ({category})</b>\n{lines}"


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/news [sg|world|tech|sport|business] — deferred summary reply."""
    msg = update.effective_message
    chat = update.effective_chat
    if msg is None or chat is None:
        return

    category = (context.args[0].lower() if context.args else _DEFAULT_CATEGORY)
    if category not in FEEDS:
        await msg.reply_text(
            "Pick a category: " + ", ".join(FEEDS) + f" (default {_DEFAULT_CATEGORY})"
        )
        return

    placeholder = await msg.reply_text("📰 Grabbing the headlines…")

    async def _work() -> None:
        items = await fetch_headlines(category)
        if not items:
            await placeholder.edit_text(
                "😕 Couldn't reach the news feeds right now. Try again in a bit."
            )
            return
        prompt = (
            f"Here are the latest '{category}' headlines:\n\n"
            f"{_headlines_block(items)}\n\nWrite the briefing now."
        )
        try:
            answer = await run_agent(
                chat.id, prompt, system_directive=_NEWS_DIRECTIVE
            )
        except Exception as exc:
            logger.warning("News summarisation failed: %s", exc)
            answer = ""
        if not answer or answer.startswith("⚠️"):
            answer = _fallback_text(category, items)
        try:
            await placeholder.edit_text(answer, parse_mode="HTML")
        except Exception:
            await placeholder.edit_text(_fallback_text(category, items), parse_mode="HTML")

    context.application.create_task(_work(), update=update)
