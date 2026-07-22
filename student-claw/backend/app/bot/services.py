"""
Database service layer for the bot.

These functions encapsulate every DB mutation the Telegram handlers perform,
each inside a single `session_scope` transaction (Module 1 utility). Handlers
stay thin and free of SQLAlchemy details. Return values are plain dataclasses
of primitives so callers never touch detached ORM instances after the session
closes.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select

from app.bot.keys import derive_project_key, vector_namespace_for
from app.database.connection import session_scope
from app.database.models import (
    ContentType,
    Expense,
    GroupMember,
    LinkedVia,
    MemberRole,
    MessageLog,
    Project,
    ProjectLinkToken,
    ProjectStatus,
    Student,
    StudentProject,
    Task,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class ProjectResult:
    project_id: str
    chat_id: int
    name: str
    project_key: str
    vector_namespace: str
    created: bool


@dataclass
class VerifyResult:
    ok: bool
    # One of: "verified", "invalid_token", "expired", "consumed",
    # "wrong_chat", "no_username", "unknown_student"
    reason: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    student_id: Optional[str] = None
    already_member: bool = False


# ---------------------------------------------------------------------------
# Project registration (Phase 2)
# ---------------------------------------------------------------------------
async def get_or_create_project(chat_id: int, name: str) -> ProjectResult:
    """
    Idempotently fetch-or-create the Project for a Telegram chat_id.

    The project_key and vector_namespace are derived deterministically so this
    is safe to call repeatedly (e.g. bot re-added to the same group).
    """
    async with session_scope() as session:
        existing = await session.scalar(
            select(Project).where(Project.chat_id == chat_id)
        )
        if existing is not None:
            return ProjectResult(
                project_id=str(existing.id),
                chat_id=existing.chat_id,
                name=existing.name,
                project_key=existing.project_key,
                vector_namespace=existing.vector_namespace,
                created=False,
            )

        project = Project(
            chat_id=chat_id,
            name=name or f"Project {chat_id}",
            project_key=derive_project_key(chat_id),
            vector_namespace=vector_namespace_for(chat_id),
        )
        session.add(project)
        await session.flush()  # populate project.id within the transaction
        return ProjectResult(
            project_id=str(project.id),
            chat_id=project.chat_id,
            name=project.name,
            project_key=project.project_key,
            vector_namespace=project.vector_namespace,
            created=True,
        )


# ---------------------------------------------------------------------------
# Verification (Phase 4)
# ---------------------------------------------------------------------------
async def consume_link_token(
    *,
    token: str,
    chat_id: int,
    telegram_user_id: int,
    telegram_username: Optional[str],
) -> VerifyResult:
    """
    Validate and consume a project_link_token, then link the sender's student
    account to the project. Single transaction; rolls back on any failure.

    Matching strategy (per blueprint Phase 4): the sender is matched to a
    `students` row by `telegram_username` captured at web registration, then
    their `telegram_user_id` is recorded.
    """
    now = datetime.now(timezone.utc)

    async with session_scope() as session:
        token_row = await session.scalar(
            select(ProjectLinkToken).where(ProjectLinkToken.token == token)
        )
        if token_row is None:
            return VerifyResult(ok=False, reason="invalid_token")
        if token_row.consumed_at is not None:
            return VerifyResult(ok=False, reason="consumed")
        if token_row.expires_at <= now:
            return VerifyResult(ok=False, reason="expired")
        if token_row.chat_id != chat_id:
            # Token must be redeemed inside the group it was issued for.
            return VerifyResult(ok=False, reason="wrong_chat")

        if not telegram_username:
            return VerifyResult(ok=False, reason="no_username")

        # Telegram usernames are case-insensitive; match on lowercase.
        student = await session.scalar(
            select(Student).where(
                Student.telegram_username.ilike(telegram_username)
            )
        )
        if student is None:
            return VerifyResult(ok=False, reason="unknown_student")

        # Bind the Telegram numeric identity to the account.
        student.telegram_user_id = telegram_user_id

        project = await session.get(Project, token_row.project_id)
        project_name = project.name if project else None

        # Idempotent membership insert.
        existing_membership = await session.scalar(
            select(StudentProject).where(
                StudentProject.student_id == student.id,
                StudentProject.project_id == token_row.project_id,
            )
        )
        already_member = existing_membership is not None
        if not already_member:
            session.add(
                StudentProject(
                    student_id=student.id,
                    project_id=token_row.project_id,
                    role=MemberRole.member,
                    linked_via=LinkedVia.telegram,
                )
            )

        # Single-use enforcement.
        token_row.consumed_at = now
        token_row.consumed_by_student_id = student.id

        return VerifyResult(
            ok=True,
            reason="verified",
            project_id=str(token_row.project_id),
            project_name=project_name,
            student_id=str(student.id),
            already_member=already_member,
        )


# ---------------------------------------------------------------------------
# Passive message logging (RAG foundation)
# ---------------------------------------------------------------------------
async def log_incoming_message(
    *,
    chat_id: int,
    telegram_message_id: int,
    content_type: ContentType,
    sender_telegram_user_id: Optional[int],
    sender_telegram_username: Optional[str],
    received_at: datetime,
    raw_text: Optional[str] = None,
    file_mime_type: Optional[str] = None,
    file_storage_path: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a message into `message_logs` (is_vectorized=False) and keep the
    project roster current.

    Returns the new message_log id (str), or None if no project exists for the
    chat (the bot has not been registered in this group yet).

    Roster maintenance: if the sender maps to a known student (by
    telegram_user_id) who is not yet a project member, a membership row is
    created so the dashboard reflects active participants.
    """
    async with session_scope() as session:
        project = await session.scalar(
            select(Project).where(Project.chat_id == chat_id)
        )
        if project is None:
            return None

        # Keep the roster up to date for already-verified students.
        if sender_telegram_user_id is not None:
            student = await session.scalar(
                select(Student).where(
                    Student.telegram_user_id == sender_telegram_user_id
                )
            )
            if student is not None:
                membership = await session.scalar(
                    select(StudentProject).where(
                        StudentProject.student_id == student.id,
                        StudentProject.project_id == project.id,
                    )
                )
                if membership is None:
                    session.add(
                        StudentProject(
                            student_id=student.id,
                            project_id=project.id,
                            role=MemberRole.member,
                            linked_via=LinkedVia.telegram,
                        )
                    )

        log = MessageLog(
            id=uuid.uuid4(),
            chat_id=chat_id,
            project_id=project.id,
            telegram_message_id=telegram_message_id,
            sender_telegram_username=sender_telegram_username,
            sender_telegram_user_id=sender_telegram_user_id,
            content_type=content_type,
            raw_text=raw_text,
            file_mime_type=file_mime_type,
            file_storage_path=file_storage_path,
            is_vectorized=False,
            received_at=received_at,
        )
        session.add(log)
        await session.flush()
        return str(log.id)


async def list_unvectorized_messages(chat_id: int) -> Optional[list[tuple[str, str]]]:
    """
    Message logs for the chat that still need embedding (for /sync). Returns
    (message_log_id, content_type) pairs, or None if the chat isn't registered.
    Already-vectorized rows are skipped; vectorize_message is idempotent anyway.
    """
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return None
        rows = (
            await session.scalars(
                select(MessageLog).where(
                    MessageLog.chat_id == chat_id,
                    MessageLog.deleted_at.is_(None),
                    MessageLog.is_vectorized.is_(False),
                    MessageLog.content_type.in_(
                        [ContentType.text, ContentType.image, ContentType.document]
                    ),
                )
            )
        ).all()

    out: list[tuple[str, str]] = []
    for m in rows:
        if m.content_type == ContentType.text and not m.raw_text:
            continue
        if m.content_type in (ContentType.image, ContentType.document) and not m.file_storage_path:
            continue
        out.append((str(m.id), m.content_type.value))
    return out


# ---------------------------------------------------------------------------
# Multi-Mode Group Agent — state engine
# ---------------------------------------------------------------------------
@dataclass
class GroupState:
    project_id: str
    chat_id: int
    group_admin_id: Optional[int]
    group_mode: str
    bot_active: bool
    allowed_models: dict


async def get_group_state(chat_id: int) -> Optional[GroupState]:
    """Current control state for a chat's project, or None if not registered."""
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return None
        return GroupState(
            project_id=str(p.id),
            chat_id=p.chat_id,
            group_admin_id=p.group_admin_id,
            group_mode=p.group_mode,
            bot_active=p.bot_active,
            allowed_models=dict(p.allowed_models or {}),
        )


def is_group_admin(state: GroupState, user_id: Optional[int]) -> bool:
    """Strict admin check — the immutable group admin only."""
    return (
        state.group_admin_id is not None
        and user_id is not None
        and state.group_admin_id == user_id
    )


def can_admin(state: GroupState, user_id: Optional[int]) -> bool:
    """
    Admin gate with a transitional rule: until the admin is claimed (via /init,
    next step) the group is unclaimed and anyone may run admin actions; once
    claimed it locks to that single admin.
    """
    return state.group_admin_id is None or is_group_admin(state, user_id)


async def set_bot_active(chat_id: int, active: bool) -> bool:
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return False
        p.bot_active = active
        return True


# ---------------------------------------------------------------------------
# Role-Based Access Control (RBAC)
# ---------------------------------------------------------------------------
async def get_member_role_by_telegram(
    chat_id: int, telegram_user_id: Optional[int]
) -> Optional[str]:
    """The caller's project role (member|lead|observer) by Telegram id, or None."""
    if telegram_user_id is None:
        return None
    async with session_scope() as session:
        role = await session.scalar(
            select(StudentProject.role)
            .join(Student, Student.id == StudentProject.student_id)
            .join(Project, Project.id == StudentProject.project_id)
            .where(
                Project.chat_id == chat_id,
                Student.telegram_user_id == telegram_user_id,
            )
        )
    return role.value if role is not None else None


async def is_privileged_user(chat_id: int, telegram_user_id: Optional[int]) -> bool:
    """
    Leader/admin gate for sensitive menu actions. True when the user is the
    group admin OR has the 'lead' role. Transitional rule: an unclaimed group
    (no admin yet) is open until /init claims it.
    """
    state = await get_group_state(chat_id)
    if state is None:
        return False
    if state.group_admin_id is None:
        return True  # unclaimed group → open
    if state.group_admin_id == telegram_user_id:
        return True
    return (await get_member_role_by_telegram(chat_id, telegram_user_id)) == "lead"


async def list_members(chat_id: int) -> list[dict]:
    """Members of the chat's project (for the Set Roles screen)."""
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return []
        rows = (
            await session.execute(
                select(Student, StudentProject.role)
                .join(StudentProject, StudentProject.student_id == Student.id)
                .where(StudentProject.project_id == project.id)
                .order_by(StudentProject.joined_at.asc())
            )
        ).all()
    return [
        {
            "student_id": str(s.id),
            "display_name": s.display_name,
            "telegram_username": s.telegram_username,
            "role": role.value,
        }
        for s, role in rows
    ]


async def set_member_role(chat_id: int, student_id: str, role: str) -> bool:
    """Promote/demote a member (Leader ↔ Member)."""
    try:
        sid = uuid.UUID(student_id)
        new_role = MemberRole(role)
    except (ValueError, KeyError):
        return False
    async with session_scope() as session:
        membership = await session.scalar(
            select(StudentProject)
            .join(Project, Project.id == StudentProject.project_id)
            .where(Project.chat_id == chat_id, StudentProject.student_id == sid)
        )
        if membership is None:
            return False
        membership.role = new_role
        return True


async def update_project_details(chat_id: int, name: str) -> bool:
    """Set the project display name (Set Details)."""
    name = name.strip()
    if not name:
        return False
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return False
        p.name = name[:200]
        return True


# ---------------------------------------------------------------------------
# Mode selection & model toggles
# ---------------------------------------------------------------------------
async def initialise_group(chat_id: int, admin_id: int, mode: str) -> bool:
    """Claim the admin (if unclaimed) and set the group mode."""
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return False
        if p.group_admin_id is None:
            p.group_admin_id = admin_id
        p.group_mode = mode
        return True


def model_allowed(allowed: dict, key: str) -> bool:
    """A model is allowed unless explicitly toggled off."""
    return bool(allowed.get(key, True))


async def toggle_allowed_model(chat_id: int, key: str) -> Optional[dict]:
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return None
        models = dict(p.allowed_models or {})
        models[key] = not model_allowed(models, key)
        p.allowed_models = models  # reassign so SQLAlchemy flushes the JSONB
        return models


# ---------------------------------------------------------------------------
# Manual group roster (Settings → Members — replaces web-app registration)
# ---------------------------------------------------------------------------
async def list_group_members(chat_id: int) -> Optional[list[dict]]:
    """Manually added members, or None if the chat isn't registered."""
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return None
        rows = (
            await session.scalars(
                select(GroupMember)
                .where(GroupMember.project_id == p.id)
                .order_by(GroupMember.display_name.asc())
            )
        ).all()
    return [
        {
            "id": str(m.id),
            "display_name": m.display_name,
            "telegram_username": m.telegram_username,
        }
        for m in rows
    ]


async def upsert_group_member(
    chat_id: int, display_name: str, telegram_username: Optional[str], added_by: Optional[int]
) -> bool:
    """Add a member (or update their handle if the name already exists)."""
    display_name = display_name.strip()[:100]
    if not display_name:
        return False
    handle = (telegram_username or "").lstrip("@").strip()[:50] or None
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return False
        existing = await session.scalar(
            select(GroupMember).where(
                GroupMember.project_id == p.id,
                GroupMember.display_name.ilike(display_name),
            )
        )
        if existing is not None:
            existing.telegram_username = handle
        else:
            session.add(
                GroupMember(
                    project_id=p.id,
                    display_name=display_name,
                    telegram_username=handle,
                    added_by_user_id=added_by,
                )
            )
        return True


async def remove_group_member(chat_id: int, member_id: str) -> bool:
    try:
        mid = uuid.UUID(member_id)
    except ValueError:
        return False
    async with session_scope() as session:
        member = await session.scalar(
            select(GroupMember)
            .join(Project, Project.id == GroupMember.project_id)
            .where(Project.chat_id == chat_id, GroupMember.id == mid)
        )
        if member is None:
            return False
        await session.delete(member)
        return True


# ---------------------------------------------------------------------------
# Mode C — Expense Tracker
# ---------------------------------------------------------------------------
async def add_expense(
    chat_id: int, payer_name: str, payer_uid: Optional[int], amount: float, description: Optional[str]
) -> bool:
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return False
        session.add(
            Expense(
                project_id=p.id,
                payer_name=payer_name[:100],
                payer_telegram_user_id=payer_uid,
                amount=round(amount, 2),
                description=(description or "")[:300] or None,
            )
        )
        return True


async def list_expenses(chat_id: int, limit: int = 20) -> Optional[list[dict]]:
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return None
        rows = (
            await session.scalars(
                select(Expense)
                .where(Expense.project_id == p.id)
                .order_by(Expense.created_at.desc())
                .limit(limit)
            )
        ).all()
    return [
        {"payer": e.payer_name, "amount": float(e.amount), "description": e.description}
        for e in rows
    ]


async def compute_balances(chat_id: int) -> Optional[dict]:
    """Split every expense equally across members ∪ payers; return net + settlements."""
    async with session_scope() as session:
        p = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if p is None:
            return None
        expenses = (
            await session.scalars(select(Expense).where(Expense.project_id == p.id))
        ).all()
        member_names = (
            await session.execute(
                select(Student.display_name)
                .join(StudentProject, StudentProject.student_id == Student.id)
                .where(StudentProject.project_id == p.id)
            )
        ).all()

    paid: dict[str, float] = {}
    total = 0.0
    for e in expenses:
        paid[e.payer_name] = paid.get(e.payer_name, 0.0) + float(e.amount)
        total += float(e.amount)

    participants = {r[0] for r in member_names} | set(paid.keys())
    if not participants or total == 0:
        return {"total": round(total, 2), "balances": [], "settlements": [], "count": len(expenses)}

    share = total / len(participants)
    balances = {name: round(paid.get(name, 0.0) - share, 2) for name in participants}

    # Greedy settlement: debtors pay creditors.
    debtors = sorted(([n, -b] for n, b in balances.items() if b < -0.009), key=lambda x: x[1])
    creditors = sorted(([n, b] for n, b in balances.items() if b > 0.009), key=lambda x: -x[1])
    settlements: list[tuple[str, str, float]] = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        owe, recv = debtors[i], creditors[j]
        amt = round(min(owe[1], recv[1]), 2)
        if amt > 0:
            settlements.append((owe[0], recv[0], amt))
        owe[1] -= amt
        recv[1] -= amt
        if owe[1] <= 0.009:
            i += 1
        if recv[1] <= 0.009:
            j += 1

    return {
        "total": round(total, 2),
        "balances": sorted(balances.items(), key=lambda x: x[1]),
        "settlements": settlements,
        "count": len(expenses),
    }


# ---------------------------------------------------------------------------
# Goals as an editable list (interactive /project_goals tree)
# ---------------------------------------------------------------------------
async def get_goal_lines(chat_id: int) -> Optional[list[str]]:
    exists, goals = await get_project_goals(chat_id)
    if not exists:
        return None
    return [ln.strip(" •-\t") for ln in (goals or "").splitlines() if ln.strip(" •-\t")]


async def _save_goal_lines(chat_id: int, lines: list[str]) -> bool:
    text = "\n".join(f"• {ln}" for ln in lines)
    return await update_project_goals(chat_id, text)


async def add_goal_line(chat_id: int, line: str) -> bool:
    lines = await get_goal_lines(chat_id)
    if lines is None:
        return False
    line = line.strip(" •-\t")
    if line:
        lines.append(line)
    return await _save_goal_lines(chat_id, lines)


async def remove_goal_line(chat_id: int, index: int) -> bool:
    lines = await get_goal_lines(chat_id)
    if lines is None or not (0 <= index < len(lines)):
        return False
    lines.pop(index)
    return await _save_goal_lines(chat_id, lines)


async def clear_goals(chat_id: int) -> bool:
    return await update_project_goals(chat_id, "")


# ---------------------------------------------------------------------------
# Project mutation helpers (Requirement 4 commands)
# ---------------------------------------------------------------------------
async def resolve_project_id(chat_id: int) -> Optional[str]:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        return str(project.id) if project else None


async def get_project_goals(chat_id: int) -> tuple[bool, Optional[str]]:
    """(exists, goals) for the chat's project."""
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return (False, None)
        return (True, project.goals)


async def update_project_goals(chat_id: int, goals: str) -> bool:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return False
        project.goals = goals.strip() or None
        return True


async def set_project_status(chat_id: int, status: ProjectStatus) -> bool:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return False
        project.status = status
        return True


async def get_task_ledger(chat_id: int) -> Optional[dict]:
    """Completed vs outstanding vs dropped task titles for /celebrate."""
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return None
        tasks = (
            await session.scalars(
                select(Task).where(Task.project_id == project.id, Task.deleted_at.is_(None))
            )
        ).all()
    ledger: dict[str, list[str]] = {"completed": [], "outstanding": [], "dropped": []}
    for t in tasks:
        if t.status == TaskStatus.done:
            ledger["completed"].append(t.title)
        elif t.status == TaskStatus.dropped:
            ledger["dropped"].append(t.title)
        else:
            ledger["outstanding"].append(t.title)
    return ledger


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text or "").strip()


async def log_agent_interaction(
    *,
    chat_id: int,
    asker_username: Optional[str],
    asker_user_id: Optional[int],
    question: str,
    answer: str,
    q_message_id: int,
    a_message_id: int,
    include_question: bool = True,
) -> None:
    """
    Persist an agent Q&A turn into message_logs so it becomes part of the
    short-term memory window. Slash-command questions are otherwise dropped
    (commands aren't captured by the passive listener), which is why the agent
    used to "forget" previous questions. NOT enqueued for embedding.

    include_question=False skips the question row — used when the passive
    listener already logged it (e.g. @mention questions).
    """
    now = datetime.now(timezone.utc)
    async with session_scope() as session:
        project = await session.scalar(
            select(Project).where(Project.chat_id == chat_id)
        )
        if project is None:
            return

        if include_question:
            session.add(
                MessageLog(
                    id=uuid.uuid4(),
                    chat_id=chat_id,
                    project_id=project.id,
                    telegram_message_id=q_message_id,
                    sender_telegram_username=asker_username,
                    sender_telegram_user_id=asker_user_id,
                    content_type=ContentType.text,
                    raw_text=question,
                    is_vectorized=False,
                    received_at=now,
                )
            )
        # Agnes answer (plain text; HTML stripped for memory readability).
        session.add(
            MessageLog(
                id=uuid.uuid4(),
                chat_id=chat_id,
                project_id=project.id,
                telegram_message_id=a_message_id,
                sender_telegram_username="Agnes",
                content_type=ContentType.text,
                raw_text=_strip_html(answer),
                is_vectorized=False,
                received_at=now,
            )
        )
