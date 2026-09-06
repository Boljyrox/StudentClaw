"""
Agnes AI orchestration & bounded agentic loop (blueprint §3.1, §3.2, §3.4).

The orchestration state lives here, not inside Agnes: we build the system
prompt, run the tool-calling loop (max 5 rounds, 30s budget), execute tools
server-side, feed results back as `role: "tool"` messages, and return the final
Telegram-HTML-safe reply.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

_SGT = ZoneInfo("Asia/Singapore")

from app.ai import repository, routing, tools
from app.ai.clients import get_agnes_client, get_openrouter_client
from app.ai.observability import logged_chat
from app.ai.config import (
    AGENT_MAX_ITERATIONS,
    AGENT_TIMEOUT_SECONDS,
    MEMORY_PROMPT_LIMIT,
    MEMORY_TURNS,
    RECENT_MESSAGE_WINDOW,
    get_ai_settings,
)

FALLBACK_NOTE = "\n\n<i>⚡ Processed via Gemini Fallback</i>"

logger = logging.getLogger("student_claw.ai.agent")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_ROLE = (
    "You are Agnes, the resident AI companion of a Telegram group chat of "
    "friends. You live in the chat like one of the gang: you remember what "
    "was said, you're funny without trying too hard, and you're genuinely "
    "useful when it matters. You can: answer any general question, recap "
    "what's been happening in the chat, remember exams/deadlines/plans, "
    "recall anything from shared photos and files, summarise the news, "
    "crack original jokes, and (when explicitly asked) roast people. Match "
    "the group's energy — casual by default, precise when the question is "
    "serious."
)

# Telegram HTML constraints — Telegram's sendMessage(parse_mode=HTML) supports
# only a small tag set. Markdown headers/syntax cause delivery failures.
_FORMATTING = (
    "OUTPUT FORMAT — STRICT. Your reply is sent to Telegram with parse_mode=HTML. "
    "You MUST output ONLY plain text plus this exact set of Telegram-supported "
    "HTML tags: <b>bold</b>, <i>italic</i>, <u>underline</u>, <s>strike</s>, "
    "<code>inline code</code>, <pre>code block</pre>, and "
    "<a href=\"https://...\">links</a>. "
    "NEVER use Markdown: no '#' headers, no '**', no '__', no '*' bullets, no "
    "'```' fences, no '[text](url)' links. "
    "Escape any literal '<', '>' or '&' in normal prose as &lt; &gt; &amp;. "
    "Do not nest unsupported tags. Keep replies concise and scannable."
)

_BEHAVIOR = (
    "BEHAVIORAL CONSTRAINTS: "
    "(1) Never invent dates, quotes, amounts, or facts about the group — if "
    "you don't know, say so or call search_chat_history first. General world "
    "knowledge is fine to use freely. "
    "(2) Before answering questions about things said or shared outside the "
    "recent window, call search_chat_history. "
    "(3) The chat_id is supplied by the system; use the value provided and "
    "never guess it. "
    "(4) Humour rules: jokes and roasts must punch at behaviour visible in "
    "the chat (always-late, spams stickers, ghosts the group), never at "
    "appearance, identity, or anything genuinely hurtful. Only roast someone "
    "when explicitly asked to. "
    "(5) Keep replies chat-sized: a few sentences for casual questions, "
    "short structured lists only when genuinely needed."
)


def build_system_prompt(
    ctx: repository.ProjectContext,
    recent: list[repository.RecentMessage],
    memory: list[repository.RecentMessage] | None = None,
    participants: list[str] | None = None,
    roster: list[tuple[str, str | None]] | None = None,
    facts: list[str] | None = None,
) -> str:
    """Assemble the system prompt with roster, clock and short-term memory."""
    # Manual roster first (name → handle, admin-curated), then anyone else
    # spotted chatting who isn't already covered by a roster entry.
    roster = roster or []
    roster_lines = [
        f"- {name}" + (f" (@{handle})" if handle else "") for name, handle in roster
    ]
    known = {name.lower() for name, _ in roster} | {
        (handle or "").lower() for _, handle in roster
    }
    extra = [p for p in (participants or []) if p.lower() not in known]
    people_lines = "\n".join(roster_lines) or "- (no members added yet)"
    if extra:
        people_lines += "\nAlso seen chatting (not in the member list): " + ", ".join(extra)
    if not roster_lines and not extra:
        people_lines = "- (nobody has chatted yet)"

    recent_lines = (
        "\n".join(f"[{r.telegram_message_id}] {r.sender}: {r.text}" for r in recent)
        or "(no recent messages)"
    )

    # SHORT-TERM MEMORY — the last few turns, so the agent remembers prior
    # questions/answers and stays consistent across the conversation.
    memory_lines = (
        "\n".join(f"{m.sender}: {m.text}" for m in (memory or []))
        or "(no prior turns)"
    )

    # LONG-TERM MEMORY — curated facts the group asked Agnes to remember.
    # Numbered so the agent can refer to "memory 3" the same way the user sees it.
    fact_lines = (
        "\n".join(f"{i}) {f}" for i, f in enumerate(facts or [], start=1))
        or "(nothing saved yet)"
    )

    from app.bot.modes import PERSONA

    persona_block = f"{PERSONA}\n\n"

    now_sg = datetime.now(timezone.utc).astimezone(_SGT)

    return (
        f"{_ROLE}\n\n"
        f"{persona_block}"
        f"GROUP CONTEXT\n"
        f"Group name: {ctx.name}\n"
        f"chat_id: {ctx.chat_id}\n"
        f"Current date/time: {now_sg.strftime('%A, %d %B %Y, %H:%M')} Singapore "
        f"time (UTC+8) — use this for countdowns and anything time-relative.\n\n"
        f"PEOPLE IN THIS CHAT (member list is admin-curated — when someone "
        f"refers to a person by name, match them via this list)\n{people_lines}\n\n"
        f"SAVED MEMORY (facts this group asked you to remember — treat as "
        f"true and current)\n{fact_lines}\n\n"
        f"SHORT-TERM MEMORY (most recent turns — use to stay consistent with the "
        f"ongoing conversation)\n{memory_lines}\n\n"
        f"RECENT MESSAGES (oldest first)\n{recent_lines}\n\n"
        f"{_BEHAVIOR}\n\n"
        f"{_FORMATTING}"
    )


# ---------------------------------------------------------------------------
# Bounded agentic loop
# ---------------------------------------------------------------------------
def _signature(name: str, raw_args: str) -> str:
    """Stable signature for circular-call detection."""
    try:
        normalized = json.dumps(json.loads(raw_args or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        normalized = raw_args or ""
    return f"{name}:{normalized}"


async def _run_loop(
    chat_id: int,
    messages: list[dict[str, Any]],
    *,
    provider: str = "agnes",
) -> str:
    """
    Bounded tool-calling loop against the chosen provider.

    Agnes is the default; `provider="openrouter"` is used only when the router
    judged the request complex enough to be worth paying for.
    """
    settings = get_ai_settings()
    openrouter = get_openrouter_client() if provider == "openrouter" else None
    if openrouter is not None:
        # Complex request and OpenRouter is configured — use the stronger model.
        client, model = openrouter, settings.openrouter_reasoning_model
    else:
        # Default path, and the safety net when OpenRouter isn't configured.
        client, model = get_agnes_client(), settings.chat_model

    seen_signatures: set[str] = set()
    toolset = tools.all_tools()

    for iteration in range(1, AGENT_MAX_ITERATIONS + 1):
        response = await logged_chat(
            client,
            model=model,
            messages=messages,
            tools=toolset,
            tool_choice="auto",
            temperature=0.4,
            chat_id=chat_id,
        )
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            # finish_reason == "stop": final answer.
            return (msg.content or "").strip()

        # Append the assistant turn (with its tool_calls) before tool results.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            sig = _signature(name, raw_args)

            # Circular-call detection (§3.4): drop exact duplicate calls.
            if sig in seen_signatures:
                logger.info("Dropping duplicate tool call %s", sig)
                messages.append(
                    _tool_message(tc.id, json.dumps({"ok": False, "detail": "duplicate call skipped"}))
                )
                continue
            seen_signatures.add(sig)

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed tool arguments for %s: %s", name, exc)
                messages.append(_tool_message(tc.id, json.dumps({"ok": False, "detail": f"invalid JSON arguments: {exc}"})))
                continue

            try:
                result = await tools.execute_tool(name, args, chat_id=chat_id)
            except tools.ToolExecutionError as exc:
                logger.warning("Tool execution error (%s): %s", name, exc)
                result = json.dumps({"ok": False, "detail": str(exc)})
            except Exception as exc:  # defensive: never crash the loop
                logger.exception("Unexpected tool failure (%s): %s", name, exc)
                result = json.dumps({"ok": False, "detail": "internal tool error"})

            messages.append(_tool_message(tc.id, result))

    # Exhausted iterations without a clean stop — ask for a final summary.
    logger.info("Agent hit max iterations (%d); requesting final answer.", AGENT_MAX_ITERATIONS)
    messages.append(
        {
            "role": "system",
            "content": "Tool budget exhausted. Reply now using what you have, in Telegram HTML.",
        }
    )
    final = await logged_chat(
        client, model=model, messages=messages, temperature=0.2, chat_id=chat_id
    )
    return (final.choices[0].message.content or "").strip()


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


async def _openrouter_fallback(
    base_messages: list[dict[str, Any]], *, chat_id: int
) -> Optional[str]:
    """
    Fallback to OpenRouter (google/gemini-3.5-flash) when Agnes fails or times
    out (Requirement 2). Uses the clean base prompt (no tool transcript); the
    answer is grounded in the project context + memory already in the prompt.
    Returns None if OpenRouter is unconfigured or also fails.
    """
    client = get_openrouter_client()
    if client is None:
        return None
    model = get_ai_settings().openrouter_model
    try:
        resp = await logged_chat(
            client, model=model, messages=base_messages, chat_id=chat_id, temperature=0.3
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        logger.error("OpenRouter fallback failed for chat_id=%s: %s", chat_id, exc)
        return None


async def run_agent(
    chat_id: int,
    user_message: str,
    *,
    history: Optional[list[dict[str, Any]]] = None,
    system_directive: Optional[str] = None,
    force_complex: bool = False,
) -> str:
    """
    Top-level entrypoint. Loads context, roster and saved memory, builds the
    prompt, picks a provider (Agnes unless the request looks genuinely complex),
    runs the bounded loop under a 30s budget, and returns Telegram-HTML text.

    `force_complex` skips the heuristic and goes straight to the stronger model.
    Never raises.
    """
    ctx = await repository.load_project_context(chat_id)
    if ctx is None:
        return "⚠️ This group isn't registered yet. Send /start to wake me up."

    recent = await repository.load_recent_messages(chat_id, RECENT_MESSAGE_WINDOW)
    memory = recent[-MEMORY_TURNS:] if recent else []
    participants = await repository.load_chat_participants(chat_id)
    roster = await repository.load_group_roster(chat_id)

    # Curated long-term facts (bounded so the prompt can't grow without limit).
    from app.bot import services as bot_services

    facts = [
        m.content for m in (await bot_services.list_memories(chat_id) or [])
    ][-MEMORY_PROMPT_LIMIT:]

    base_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                ctx, recent, memory, participants, roster, facts
            ),
        }
    ]
    if system_directive:
        base_messages.append({"role": "system", "content": system_directive})
    if history:
        base_messages.extend(history)
    base_messages.append({"role": "user", "content": user_message})

    # Route: Agnes handles everything unless the request is genuinely hard.
    decision = routing.choose_provider(
        user_message,
        allowed=ctx.allowed_models,
        openrouter_available=get_openrouter_client() is not None,
        force_complex=force_complex,
    )
    logger.info(
        "Routing chat_id=%s -> %s (%s)", chat_id, decision.provider, decision.reason
    )

    try:
        answer = await asyncio.wait_for(
            _run_loop(chat_id, list(base_messages), provider=decision.provider),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        if answer:
            return answer + (FALLBACK_NOTE if decision.is_openrouter else "")
        logger.warning("Empty answer from %s for chat_id=%s.", decision.provider, chat_id)
    except asyncio.TimeoutError:
        logger.warning("Agent timed out (%ss) for chat_id=%s; trying fallback.", AGENT_TIMEOUT_SECONDS, chat_id)
    except Exception as exc:
        logger.exception("Agent error for chat_id=%s; trying fallback: %s", chat_id, exc)

    # Fallback. If the primary attempt was already OpenRouter there's nothing
    # better to escalate to, so retry plainly on Agnes instead.
    if decision.is_openrouter:
        try:
            answer = await asyncio.wait_for(
                _run_loop(chat_id, list(base_messages), provider="agnes"),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            if answer:
                return answer
        except Exception as exc:
            logger.warning("Agnes retry also failed for chat_id=%s: %s", chat_id, exc)
    elif ctx.allowed_models.get("gemini_fallback", True):
        fallback = await _openrouter_fallback(base_messages, chat_id=chat_id)
        if fallback:
            return f"{fallback}{FALLBACK_NOTE}"
    return "⚠️ Something went wrong while processing that. Please try again."
