"""
Async DB access for the AI subsystem.

Two concerns:
  * Reads that build the agent's prompt context (project, roster, recent logs)
    and feed the embedding worker.
  * Writes that persist the agent's structural tool calls (deadlines, tasks,
    contribution metrics) and mark messages as vectorized.

Everything goes through the Module 1 session_scope. Tool-write functions resolve
`telegram_username` to a real `students` row scoped to the project, so the agent
can never attribute work to a non-member.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc, select, update

from app.database.connection import session_scope
from app.database.models import (
    ContentType,
    ContributionMetric,
    Deadline,
    DeadlineExtractedBy,
    MessageLog,
    Project,
    Student,
    StudentProject,
    Task,
    TaskSource,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Context reads
# ---------------------------------------------------------------------------
@dataclass
class MemberInfo:
    student_id: str
    display_name: str
    telegram_username: Optional[str]
    role: str


@dataclass
class ProjectContext:
    project_id: str
    chat_id: int
    name: str
    module_code: Optional[str]
    status: str
    group_mode: str = "projects"
    allowed_models: dict = field(default_factory=dict)
    members: list[MemberInfo] = field(default_factory=list)


async def load_project_context(chat_id: int) -> Optional[ProjectContext]:
    async with session_scope() as session:
        project = await session.scalar(
            select(Project).where(Project.chat_id == chat_id)
        )
        if project is None:
            return None

        rows = (
            await session.execute(
                select(StudentProject, Student)
                .join(Student, Student.id == StudentProject.student_id)
                .where(StudentProject.project_id == project.id)
            )
        ).all()
        members = [
            MemberInfo(
                student_id=str(s.id),
                display_name=s.display_name,
                telegram_username=s.telegram_username,
                role=sp.role.value,
            )
            for sp, s in rows
        ]
        return ProjectContext(
            project_id=str(project.id),
            chat_id=project.chat_id,
            name=project.name,
            module_code=project.module_code,
            status=project.status.value,
            group_mode=project.group_mode,
            allowed_models=dict(project.allowed_models or {}),
            members=members,
        )


async def get_allowed_models(chat_id: int) -> dict:
    """AI-model allow-list for a chat (empty = all allowed)."""
    async with session_scope() as session:
        models = await session.scalar(
            select(Project.allowed_models).where(Project.chat_id == chat_id)
        )
    return dict(models or {})


async def load_group_roster(chat_id: int) -> list[tuple[str, Optional[str]]]:
    """
    Manually added members (Settings → Members) as (display_name, handle)
    pairs. This replaces the legacy web-app registration flow.
    """
    from app.database.models import GroupMember

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(GroupMember.display_name, GroupMember.telegram_username)
                .join(Project, Project.id == GroupMember.project_id)
                .where(Project.chat_id == chat_id)
                .order_by(GroupMember.display_name.asc())
            )
        ).all()
    return [(name, handle) for name, handle in rows]


async def load_chat_participants(chat_id: int, limit: int = 400) -> list[str]:
    """
    Distinct sender names seen in the chat's recent history (newest first).
    This is the real roster of a friend group — nobody has to 'verify' via the
    legacy web dashboard for Agnes to know who's in the chat.
    """
    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(MessageLog.sender_telegram_username)
                .where(
                    MessageLog.chat_id == chat_id,
                    MessageLog.deleted_at.is_(None),
                    MessageLog.sender_telegram_username.is_not(None),
                )
                .order_by(desc(MessageLog.received_at))
                .limit(limit)
            )
        ).all()
    seen: list[str] = []
    for name in rows:
        if name and name != "Agnes" and name not in seen:
            seen.append(name)
    return seen


@dataclass
class UpcomingDate:
    title: str
    due_date: datetime


async def list_upcoming_dates(
    chat_id: int, *, include_past: bool = False
) -> Optional[list[UpcomingDate]]:
    """
    Saved exams/deadlines for a chat, earliest first. Returns None when the
    chat isn't registered. By default past dates are dropped.
    """
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return None
        rows = (
            await session.scalars(
                select(Deadline)
                .where(Deadline.project_id == project.id)
                .order_by(Deadline.due_date.asc())
            )
        ).all()
    now = datetime.now(timezone.utc)
    out = [
        UpcomingDate(title=d.title, due_date=d.due_date)
        for d in rows
        if include_past or d.due_date >= now
    ]
    return out


@dataclass
class RecentMessage:
    sender: str
    text: str
    telegram_message_id: int


async def load_recent_messages(chat_id: int, limit: int) -> list[RecentMessage]:
    """Most recent text-bearing logs (oldest→newest) for the prompt window."""
    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(MessageLog)
                .where(
                    MessageLog.chat_id == chat_id,
                    MessageLog.deleted_at.is_(None),
                )
                .order_by(desc(MessageLog.received_at))
                .limit(limit)
            )
        ).all()

    out: list[RecentMessage] = []
    for m in reversed(rows):  # chronological
        text = (m.raw_text or m.extracted_text or "").strip()
        if not text:
            continue
        out.append(
            RecentMessage(
                sender=m.sender_telegram_username or "unknown",
                text=text,
                telegram_message_id=m.telegram_message_id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Embedding-worker reads/writes
# ---------------------------------------------------------------------------
@dataclass
class ChatMessageRow:
    message_log_id: str
    sender: str
    text: str


async def load_chat_messages_for_reindex(chat_id: int) -> list[ChatMessageRow]:
    """All non-deleted text-bearing messages (chronological) for bulk re-index."""
    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(MessageLog)
                .where(
                    MessageLog.chat_id == chat_id,
                    MessageLog.deleted_at.is_(None),
                )
                .order_by(MessageLog.received_at)
            )
        ).all()
    out: list[ChatMessageRow] = []
    for m in rows:
        text = (m.raw_text or m.extracted_text or "").strip()
        if text:
            out.append(
                ChatMessageRow(
                    message_log_id=str(m.id),
                    sender=m.sender_telegram_username or "unknown",
                    text=text,
                )
            )
    return out


@dataclass
class MessageForVectorization:
    id: str
    chat_id: int
    project_id: str
    content_type: str
    raw_text: Optional[str]
    extracted_text: Optional[str]
    file_storage_path: Optional[str]
    file_mime_type: Optional[str]
    sender_username: Optional[str]
    received_at: datetime
    is_vectorized: bool
    deleted: bool


async def get_message_for_vectorization(message_log_id: str) -> Optional[MessageForVectorization]:
    async with session_scope() as session:
        m = await session.get(MessageLog, uuid.UUID(message_log_id))
        if m is None:
            return None
        return MessageForVectorization(
            id=str(m.id),
            chat_id=m.chat_id,
            project_id=str(m.project_id),
            content_type=m.content_type.value,
            raw_text=m.raw_text,
            extracted_text=m.extracted_text,
            file_storage_path=m.file_storage_path,
            file_mime_type=m.file_mime_type,
            sender_username=m.sender_telegram_username,
            received_at=m.received_at,
            is_vectorized=m.is_vectorized,
            deleted=m.deleted_at is not None,
        )


async def save_extracted_text(message_log_id: str, extracted_text: str) -> None:
    async with session_scope() as session:
        await session.execute(
            update(MessageLog)
            .where(MessageLog.id == uuid.UUID(message_log_id))
            .values(extracted_text=extracted_text)
        )


async def mark_vectorized(
    message_log_id: str, project_id: str, point_ids: list[uuid.UUID]
) -> None:
    """Flag the message as vectorized and bump the project's point counter."""
    async with session_scope() as session:
        await session.execute(
            update(MessageLog)
            .where(MessageLog.id == uuid.UUID(message_log_id))
            .values(is_vectorized=True, qdrant_point_ids=point_ids)
        )
        await session.execute(
            update(Project)
            .where(Project.id == uuid.UUID(project_id))
            .values(qdrant_point_count=Project.qdrant_point_count + len(point_ids))
        )


# ---------------------------------------------------------------------------
# Tool persistence (agent structural mutations)
# ---------------------------------------------------------------------------
async def _resolve_member(
    session, project_id: uuid.UUID, telegram_username: str
) -> Optional[Student]:
    """Resolve a project member by Telegram username (case-insensitive)."""
    return await session.scalar(
        select(Student)
        .join(StudentProject, StudentProject.student_id == Student.id)
        .where(
            StudentProject.project_id == project_id,
            Student.telegram_username.ilike(telegram_username.lstrip("@")),
        )
    )


@dataclass
class ToolWriteResult:
    ok: bool
    detail: str
    entity_id: Optional[str] = None


async def upsert_deadline(
    *, chat_id: int, task_title: str, due_date: datetime, confidence: float,
    source_message_id: Optional[int] = None,
) -> ToolWriteResult:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return ToolWriteResult(False, "No project for this chat.")

        existing = await session.scalar(
            select(Deadline).where(
                Deadline.project_id == project.id, Deadline.title == task_title
            )
        )
        if existing is not None:
            existing.due_date = due_date
            return ToolWriteResult(True, "Updated existing deadline.", str(existing.id))

        deadline = Deadline(
            project_id=project.id,
            title=task_title,
            due_date=due_date,
            extracted_by=DeadlineExtractedBy.ai,
            is_confirmed=False,  # AI-extracted deadlines require user confirmation
        )
        session.add(deadline)
        await session.flush()
        return ToolWriteResult(True, "Created deadline (pending confirmation).", str(deadline.id))


async def delegate_task(
    *, chat_id: int, telegram_username: str, task_title: str, task_description: str,
    priority: int, delegation_rationale: str, related_deadline_title: Optional[str] = None,
) -> ToolWriteResult:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return ToolWriteResult(False, "No project for this chat.")

        member = await _resolve_member(session, project.id, telegram_username)
        if member is None:
            return ToolWriteResult(
                False, f"@{telegram_username} is not a verified member of this project."
            )

        task = Task(
            project_id=project.id,
            assigned_to_student_id=member.id,
            title=task_title,
            description=task_description,
            source=TaskSource.ai_delegated,
            status=TaskStatus.pending,
            priority=priority,
            task_metadata={
                "delegation_rationale": delegation_rationale,
                "related_deadline_title": related_deadline_title,
            },
        )
        session.add(task)
        await session.flush()
        task_id, project_id = str(task.id), str(project.id)

    # Published AFTER commit so the web's refetch sees the new task (bidirectional
    # sync — Agnes delegations appear live on the web board, Requirement 3).
    from app.bot import events

    await events.publish_project_event(
        project_id=project_id,
        event_type="task_created",
        payload={"task_id": task_id},
        triggered_by="ai_agent",
    )
    return ToolWriteResult(True, f"Assigned to @{telegram_username}.", task_id)


async def log_contribution_metric(
    *, chat_id: int, telegram_username: str, score_value: float, score_reason: str,
    scoring_window_start: datetime, scoring_window_end: datetime,
    evidence_message_ids: Optional[list[int]] = None,
) -> ToolWriteResult:
    async with session_scope() as session:
        project = await session.scalar(select(Project).where(Project.chat_id == chat_id))
        if project is None:
            return ToolWriteResult(False, "No project for this chat.")

        member = await _resolve_member(session, project.id, telegram_username)
        if member is None:
            return ToolWriteResult(
                False, f"@{telegram_username} is not a verified member of this project."
            )

        metric = ContributionMetric(
            project_id=project.id,
            student_id=member.id,
            score_value=score_value,
            score_reason=score_reason,
            scoring_window_start=scoring_window_start,
            scoring_window_end=scoring_window_end,
            score_metadata={"evidence_message_ids": evidence_message_ids or []},
        )
        session.add(metric)
        await session.flush()
        return ToolWriteResult(True, f"Logged contribution score for @{telegram_username}.", str(metric.id))
