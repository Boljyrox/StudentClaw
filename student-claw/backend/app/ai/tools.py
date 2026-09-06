"""
Agnes AI tool definitions + executors.

Holds the OpenAI-standard function schemas and the server-side handlers that
execute them. Every group gets the same toolset: chat-history search, saving
and listing important dates, and saving/listing long-term memory. The older
task-delegation and contribution-scoring executors are kept reachable for the
legacy web dashboard but are no longer advertised to the model.

Security invariant: the authoritative `chat_id` is injected by the backend on
every call; any `chat_id` the model emits in its arguments is IGNORED and
overwritten. This makes cross-group access via prompt injection impossible.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.ai import pipeline, repository

logger = logging.getLogger("student_claw.ai.tools")


# ---------------------------------------------------------------------------
# Tool JSON schemas
# ---------------------------------------------------------------------------
_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_chat_history",
        "description": (
            "Semantically search everything this group has ever shared — chat "
            "messages, photos (OCR'd), PDFs and slides — to recall specific "
            "past information. Use this before answering any question about "
            "things that happened outside the recent-messages window (old "
            "plans, who said what, contents of shared files, past bills)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Provided by the system context; never infer it.",
                },
                "query": {
                    "type": "string",
                    "description": "Semantic search query. Rephrase the user's question for similarity search.",
                    "maxLength": 500,
                },
                "content_type_filter": {
                    "type": "string",
                    "enum": ["all", "text", "image", "document"],
                    "description": "Optionally restrict to a content type. Default 'all'.",
                    "default": "all",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of results. Default 8.", "default": 8},
            },
            "required": ["chat_id", "query"],
            "additionalProperties": False,
        },
    },
}

_SAVE_DATE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_important_date",
        "description": (
            "Save an exam, deadline, or other dated event the group cares "
            "about (e.g. 'Linear Algebra final on 12 Aug, 9am–11am'). Only "
            "call when a date is unambiguously stated — never guess dates. "
            "Upsert behavior: an entry with the same title has its date "
            "updated instead of duplicating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Provided by the system context; never infer it.",
                },
                "title": {
                    "type": "string",
                    "description": "Concise title, e.g. 'Linear Algebra Final (9–11am)'. Include the timing in the title when known. Max 200 chars.",
                    "maxLength": 200,
                },
                "due_date": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO 8601 with timezone. For exams use the START time. If only a date is known, default to 09:00 Singapore Time (UTC+8) for exams and 23:59 for deadlines.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence this is a genuine, explicitly stated date. Below 0.7 → don't call.",
                },
            },
            "required": ["chat_id", "title", "due_date", "confidence"],
            "additionalProperties": False,
        },
    },
}

_LIST_DATES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_saved_dates",
        "description": (
            "List the group's saved exams/deadlines/events, earliest first. "
            "Call this whenever asked about upcoming exams, papers, deadlines "
            "or 'when is X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Provided by the system context; never infer it.",
                },
                "include_past": {
                    "type": "boolean",
                    "description": "Include dates that already passed. Default false.",
                    "default": False,
                },
            },
            "required": ["chat_id"],
            "additionalProperties": False,
        },
    },
}

# ── Legacy (projects mode only) ────────────────────────────────────────────
_DELEGATE_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": (
            "LEGACY (projects mode). Create and assign a task to a specific group member. Only call when: "
            "(a) a member explicitly volunteers, (b) a member is explicitly assigned "
            "by another member, or (c) a member's demonstrated expertise makes them "
            "the unambiguous choice AND delegation was requested. Never assign "
            "tasks speculatively."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer", "description": "The Telegram group chat ID from system context."},
                "telegram_username": {
                    "type": "string",
                    "description": "The Telegram username (without @) of the member receiving the task. Must be a member listed in the group context.",
                },
                "task_description": {
                    "type": "string",
                    "description": "A detailed, actionable description: what to do, relevant context, and completion criteria if discernible.",
                    "maxLength": 1000,
                },
                "task_title": {"type": "string", "description": "Short title for the task card. Maximum 200 characters.", "maxLength": 200},
                "priority": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "1=High (deadline within 48h or explicitly urgent), 2=Medium (default), 3=Low.",
                },
                "related_deadline_title": {
                    "type": "string",
                    "description": "Optional: the title of an existing deadline this task contributes to.",
                    "nullable": True,
                },
                "delegation_rationale": {
                    "type": "string",
                    "description": "Brief explanation of why this person was selected. Stored for transparency.",
                },
            },
            "required": ["chat_id", "telegram_username", "task_description", "task_title", "priority", "delegation_rationale"],
            "additionalProperties": False,
        },
    },
}

_CONTRIBUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "log_contribution_metric",
        "description": (
            "LEGACY (projects mode). Score a member's contribution over an evaluation window. Only call "
            "when explicitly requested by a group member. Never call autonomously. "
            "Scoring must be evidence-based."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer", "description": "The Telegram group chat ID from system context."},
                "telegram_username": {"type": "string", "description": "The Telegram username (without @) being evaluated."},
                "score_value": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "description": "Contribution score 0.00-10.00 (one decimal). Calibrate against the group: equal distribution = 5.0 each.",
                },
                "score_reason": {
                    "type": "string",
                    "description": "Detailed, objective, evidence-based justification citing specific actions.",
                    "maxLength": 2000,
                },
                "scoring_window_start": {"type": "string", "format": "date-time", "description": "Start of evaluation period (ISO 8601)."},
                "scoring_window_end": {"type": "string", "format": "date-time", "description": "End of evaluation period (ISO 8601). Defaults to now."},
                "evidence_message_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Telegram message_ids used as evidence (audit trail).",
                    "maxItems": 50,
                },
            },
            "required": ["chat_id", "telegram_username", "score_value", "score_reason", "scoring_window_start", "scoring_window_end"],
            "additionalProperties": False,
        },
    },
}

_REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember_fact",
        "description": (
            "Save one durable fact about this group so you can recall it in "
            "future conversations — someone's exam date, a recurring plan, a "
            "preference, an inside joke. Use it when the group explicitly asks "
            "you to remember something, or when a clearly durable fact is "
            "stated. Do NOT use it for passing chatter; ordinary messages are "
            "already searchable via search_chat_history. Facts expire after "
            "one month."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Provided by the system context; never infer it.",
                },
                "fact": {
                    "type": "string",
                    "description": (
                        "The fact as one self-contained sentence, e.g. "
                        "\"Bala's DDW exam is on 12 November\". Include names "
                        "so it makes sense on its own."
                    ),
                    "maxLength": 300,
                },
            },
            "required": ["chat_id", "fact"],
            "additionalProperties": False,
        },
    },
}

_RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "list_saved_memory",
        "description": (
            "List the facts you've saved about this group. Use it when asked "
            "what you remember, or to check whether something is already saved "
            "before saving it again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Provided by the system context; never infer it.",
                },
            },
            "required": ["chat_id"],
            "additionalProperties": False,
        },
    },
}

# One group, one toolset — there are no modes any more.
_CORE_TOOLS: list[dict[str, Any]] = [
    _SEARCH_TOOL,
    _SAVE_DATE_TOOL,
    _LIST_DATES_TOOL,
    _REMEMBER_TOOL,
    _RECALL_TOOL,
]

# Kept for backwards compatibility with existing imports.
TOOLS: list[dict[str, Any]] = _CORE_TOOLS


def all_tools() -> list[dict[str, Any]]:
    """The toolset exposed to the agent (identical for every group)."""
    return _CORE_TOOLS


def tools_for_mode(_mode: str | None = None) -> list[dict[str, Any]]:
    """Deprecated alias kept so older call sites keep working."""
    return _CORE_TOOLS


class ToolExecutionError(Exception):
    """Raised on a malformed tool-call signature."""


def _parse_dt(value: str) -> datetime:
    """Parse an ISO 8601 string into an aware datetime (assume UTC if naive)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
async def execute_tool(name: str, arguments: dict[str, Any], *, chat_id: int) -> str:
    """
    Execute a tool call and return a string suitable for a `role: "tool"`
    message. `chat_id` is authoritative and always overrides arguments.

    Raises ToolExecutionError for unknown tools or malformed required args.
    """
    args = dict(arguments or {})
    args["chat_id"] = chat_id  # security override

    try:
        if name == "search_chat_history":
            results = await pipeline.semantic_search(
                chat_id=chat_id,
                query=args["query"],
                top_k=int(args.get("top_k", 8)),
                content_type_filter=args.get("content_type_filter", "all"),
            )
            return _format_search_results(results)

        if name == "save_important_date":
            res = await repository.upsert_deadline(
                chat_id=chat_id,
                task_title=args["title"],
                due_date=_parse_dt(args["due_date"]),
                confidence=float(args.get("confidence", 0.0)),
            )
            return _format_write(res)

        if name == "list_saved_dates":
            dates = await repository.list_upcoming_dates(
                chat_id, include_past=bool(args.get("include_past", False))
            )
            if dates is None:
                return json.dumps({"ok": False, "detail": "Group not registered."})
            return json.dumps(
                {
                    "ok": True,
                    "dates": [
                        {"title": d.title, "due_date": d.due_date.isoformat()}
                        for d in dates
                    ],
                }
            )

        # Memory writes are additive only. Deletion deliberately has no tool:
        # it always goes through an explicit in-chat confirmation instead.
        if name == "remember_fact":
            from app.bot import services as bot_services

            memory_id = await bot_services.add_memory(
                chat_id, args["fact"], source="auto"
            )
            if memory_id is None:
                return json.dumps({"ok": False, "detail": "Group not registered."})
            return json.dumps({"ok": True, "detail": "Saved to memory.", "id": memory_id})

        if name == "list_saved_memory":
            from app.bot import services as bot_services

            items = await bot_services.list_memories(chat_id)
            if items is None:
                return json.dumps({"ok": False, "detail": "Group not registered."})
            return json.dumps(
                {
                    "ok": True,
                    "memories": [
                        {"index": i, "fact": m.content}
                        for i, m in enumerate(items, start=1)
                    ],
                }
            )

        if name == "delegate_task":
            res = await repository.delegate_task(
                chat_id=chat_id,
                telegram_username=args["telegram_username"],
                task_title=args["task_title"],
                task_description=args["task_description"],
                priority=int(args.get("priority", 2)),
                delegation_rationale=args["delegation_rationale"],
                related_deadline_title=args.get("related_deadline_title"),
            )
            return _format_write(res)

        if name == "log_contribution_metric":
            res = await repository.log_contribution_metric(
                chat_id=chat_id,
                telegram_username=args["telegram_username"],
                score_value=float(args["score_value"]),
                score_reason=args["score_reason"],
                scoring_window_start=_parse_dt(args["scoring_window_start"]),
                scoring_window_end=_parse_dt(args["scoring_window_end"]),
                evidence_message_ids=args.get("evidence_message_ids"),
            )
            return _format_write(res)

    except KeyError as exc:
        raise ToolExecutionError(f"Tool {name} missing required argument: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise ToolExecutionError(f"Tool {name} received a malformed argument: {exc}") from exc

    raise ToolExecutionError(f"Unknown tool: {name!r}")


def _format_write(res: repository.ToolWriteResult) -> str:
    return json.dumps({"ok": res.ok, "detail": res.detail, "id": res.entity_id})


def _format_search_results(results: list[pipeline.SearchResult]) -> str:
    if not results:
        return json.dumps({"results": [], "note": "No relevant chat history found."})
    return json.dumps(
        {
            "results": [
                {
                    "rank": i + 1,
                    "snippet": r.text_snippet,
                    "sender": r.sender_username,
                    "when": r.received_at,
                    "source": r.source_filename,
                    "score": round(r.score, 4),
                }
                for i, r in enumerate(results)
            ]
        }
    )
