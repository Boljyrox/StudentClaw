"""
Model routing — decide whether a request goes to Agnes or to OpenRouter.

Agnes is free and handles the overwhelming majority of group chatter, so it is
always the default. OpenRouter costs money and is reserved for the small set of
requests that genuinely benefit from a stronger reasoning model.

The group can override the router from the AI settings menu:
    auto       — use the heuristic below (default)
    agnes      — never leave Agnes (free-only mode)
    openrouter — send everything to OpenRouter

Vision/OCR is deliberately NOT routed here: receipt and document extraction go
straight to the Qwen-VL model because Agnes has no comparable vision path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("student_claw.ai.routing")

# Setting key stored in projects.allowed_models.
MODEL_CHOICE_KEY = "model_choice"
CHOICE_AUTO = "auto"
CHOICE_AGNES = "agnes"
CHOICE_OPENROUTER = "openrouter"
VALID_CHOICES = (CHOICE_AUTO, CHOICE_AGNES, CHOICE_OPENROUTER)

CHOICE_LABELS = {
    CHOICE_AUTO: "Auto (recommended)",
    CHOICE_AGNES: "Agnes only (free)",
    CHOICE_OPENROUTER: "Always OpenRouter",
}

# ---------------------------------------------------------------------------
# Complexity heuristic
# ---------------------------------------------------------------------------
# Anything longer than this is likely a real problem statement, not banter.
_LONG_PROMPT_CHARS = 420

# Phrases that signal multi-step reasoning rather than recall or chit-chat.
_REASONING_PATTERNS = re.compile(
    r"\b("
    r"explain why|why (?:does|do|is|are|did)|how (?:does|do|come)|"
    r"compare|contrast|trade[- ]?offs?|pros and cons|"
    r"analys[ei]|analyz[ei]|evaluate|assess|critique|"
    r"step[- ]by[- ]step|walk me through|derive|prove|"
    r"plan (?:out|a|the)|strategy|strategise|strategize|"
    r"debug|troubleshoot|root cause|optimi[sz]e|refactor|"
    r"write (?:me )?(?:a |the )?(?:code|script|program|function|essay|report)|"
    r"calculate|compute|solve|equation|integral|derivative|probability"
    r")\b",
    re.IGNORECASE,
)

# Study/academic help is the main legitimately-hard workload in a student group.
_ACADEMIC_PATTERNS = re.compile(
    r"\b("
    r"proof|theorem|algorithm|complexity|big[- ]?o|"
    r"thermodynamics|calculus|linear algebra|statistics|"
    r"past year paper|pyp|revision notes|model answer|"
    r"summari[sz]e this (?:paper|pdf|document|chapter)"
    r")\b",
    re.IGNORECASE,
)

# Cheap, obviously-simple intents that must never escalate.
_TRIVIAL_PATTERNS = re.compile(
    r"^\s*(?:"
    r"hi|hello|hey|yo|sup|lol|lmao|haha|ok(?:ay)?|thanks|thank you|ty|"
    r"good (?:morning|night|evening)|gm|gn|"
    r"joke|roast|news|summary|who|what time|when is"
    r")\b",
    re.IGNORECASE,
)

# Code fences / stack traces are a strong signal of a technical question.
_CODE_HINT = re.compile(r"```|def \w+\(|class \w+|Traceback \(most recent|SELECT .+ FROM ", re.IGNORECASE)


@dataclass(frozen=True)
class RouteDecision:
    provider: str  # "agnes" | "openrouter"
    reason: str
    complex: bool

    @property
    def is_openrouter(self) -> bool:
        return self.provider == "openrouter"


def score_complexity(text: str) -> tuple[int, list[str]]:
    """
    Rough complexity score for a user request. Higher = more reasoning needed.
    Returns (score, reasons) so the decision is explainable in logs and in the
    admin UI.
    """
    text = (text or "").strip()
    score = 0
    reasons: list[str] = []

    if not text:
        return 0, reasons

    # A short greeting/one-liner is never complex, regardless of keywords.
    if len(text) < 25 and _TRIVIAL_PATTERNS.search(text):
        return 0, ["trivial"]

    if len(text) >= _LONG_PROMPT_CHARS:
        score += 2
        reasons.append("long prompt")
    elif len(text) >= 200:
        score += 1
        reasons.append("medium prompt")

    # Several distinct reasoning markers compound — "explain why … step by
    # step" is a stronger signal than either phrase alone.
    reasoning_hits = len({m.group(0).lower() for m in _REASONING_PATTERNS.finditer(text)})
    if reasoning_hits:
        score += 2 + (reasoning_hits - 1)
        reasons.append(
            "reasoning verb" if reasoning_hits == 1 else f"{reasoning_hits} reasoning verbs"
        )

    if _ACADEMIC_PATTERNS.search(text):
        score += 2
        reasons.append("academic topic")

    if _CODE_HINT.search(text):
        score += 2
        reasons.append("code/technical")

    # Several distinct questions in one message = multi-part answer.
    if text.count("?") >= 3:
        score += 1
        reasons.append("multi-question")

    # Dense numeric content suggests a calculation rather than chat.
    digits = sum(c.isdigit() for c in text)
    if digits >= 12 and len(text) > 60:
        score += 1
        reasons.append("numeric")

    return score, reasons


# Escalate only when the signal is clear; 3 means at least one strong marker
# plus corroboration, which keeps casual chat on Agnes.
COMPLEXITY_THRESHOLD = 3


def choose_provider(
    text: str,
    *,
    allowed: dict | None = None,
    openrouter_available: bool = True,
    force_complex: bool = False,
) -> RouteDecision:
    """
    Pick the provider for a chat request.

    `allowed` is the group's projects.allowed_models dict; `force_complex` lets
    callers (e.g. a deliberate "think harder" path) skip the heuristic.
    """
    allowed = allowed or {}
    choice = str(allowed.get(MODEL_CHOICE_KEY, CHOICE_AUTO)).lower()
    if choice not in VALID_CHOICES:
        choice = CHOICE_AUTO

    if choice == CHOICE_AGNES:
        return RouteDecision("agnes", "group set Agnes-only", False)

    if not openrouter_available:
        return RouteDecision("agnes", "OpenRouter unavailable", False)

    if choice == CHOICE_OPENROUTER:
        return RouteDecision("openrouter", "group set OpenRouter-always", True)

    score, reasons = score_complexity(text)
    is_complex = force_complex or score >= COMPLEXITY_THRESHOLD
    if is_complex:
        why = "forced" if force_complex else f"score={score} ({', '.join(reasons)})"
        return RouteDecision("openrouter", why, True)
    return RouteDecision("agnes", f"score={score}", False)


def get_model_choice(allowed: dict | None) -> str:
    choice = str((allowed or {}).get(MODEL_CHOICE_KEY, CHOICE_AUTO)).lower()
    return choice if choice in VALID_CHOICES else CHOICE_AUTO
