"""
SQLAlchemy ORM models for Student Claw — Module 1: Database & Core Models.

These models are an exact, production-grade mapping of the PostgreSQL schema
defined in the architectural blueprint (§2.3). Conventions:

  * SQLAlchemy 2.0 typed declarative style (Mapped / mapped_column).
  * UUID primary keys with server-side `gen_random_uuid()` (pgcrypto/PG13+).
  * All timestamps are TIMESTAMPTZ (DateTime(timezone=True)).
  * Native PostgreSQL ENUM types, created once and reused.
  * JSONB for flexible AI-generated metadata.
  * Soft deletes via `deleted_at` on `tasks` and `message_logs`.

Enum design note: `create_type=False` is NOT used here — the enum types are
created/dropped alongside the tables by `init_db`. We set `native_enum=True`
(default) so the DB enforces the allowed values.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for all Student Claw ORM models."""
    pass


# ---------------------------------------------------------------------------
# Reusable column helpers
# ---------------------------------------------------------------------------
def _uuid_pk() -> Mapped[uuid.UUID]:
    """UUID primary key generated server-side via gen_random_uuid()."""
    return mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# Python-side enums (mirror the DB ENUM types)
# ---------------------------------------------------------------------------
class ProjectStatus(str, enum.Enum):
    upcoming = "upcoming"
    active = "active"
    completed = "completed"
    archived = "archived"
    cleared = "cleared"
    # transient state used during cache cleansing (§5.3 Step 2)
    clearing = "clearing"


class GroupMode(str, enum.Enum):
    """Operating mode for a group (Multi-Mode Group Agent)."""
    uninitialized = "uninitialized"
    projects = "projects"
    fun = "fun"
    expense = "expense"
    event = "event"
    study = "study"
    general = "general"


class MemberRole(str, enum.Enum):
    member = "member"
    lead = "lead"
    observer = "observer"


class LinkedVia(str, enum.Enum):
    telegram = "telegram"
    web_key = "web_key"


class TaskSource(str, enum.Enum):
    ai_delegated = "ai_delegated"
    manual = "manual"
    deadline_extracted = "deadline_extracted"


class TaskStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    dropped = "dropped"


class DeadlineExtractedBy(str, enum.Enum):
    ai = "ai"
    manual = "manual"


class ContentType(str, enum.Enum):
    text = "text"
    image = "image"
    document = "document"
    voice = "voice"


# Reusable SAEnum instances. `name` fixes the PostgreSQL type name so the same
# type is reused across columns (and created exactly once by create_all).
project_status_enum = SAEnum(ProjectStatus, name="project_status", validate_strings=True)
member_role_enum = SAEnum(MemberRole, name="member_role", validate_strings=True)
linked_via_enum = SAEnum(LinkedVia, name="linked_via", validate_strings=True)
task_source_enum = SAEnum(TaskSource, name="task_source", validate_strings=True)
task_status_enum = SAEnum(TaskStatus, name="task_status", validate_strings=True)
deadline_extracted_by_enum = SAEnum(
    DeadlineExtractedBy, name="deadline_extracted_by", validate_strings=True
)
content_type_enum = SAEnum(ContentType, name="content_type", validate_strings=True)


# ===========================================================================
# students
# ===========================================================================
class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = _uuid_pk()
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    telegram_username: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, nullable=True
    )
    telegram_user_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, unique=True, nullable=True
    )
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)

    # OAuth2 offline token, encrypted at rest (AES-256-GCM) before storage.
    google_calendar_refresh_token: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    google_calendar_linked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )

    # Relationships
    memberships: Mapped[list["StudentProject"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    assigned_tasks: Mapped[list["Task"]] = relationship(
        back_populates="assignee", foreign_keys="Task.assigned_to_student_id"
    )
    contribution_metrics: Mapped[list["ContributionMetric"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Student id={self.id} username={self.username!r}>"


# ===========================================================================
# projects
# ===========================================================================
class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()

    # Public-facing linkage key derived from chat_id via HMAC-SHA256 (first 16
    # hex chars). Never exposes raw chat_id.
    project_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    module_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Free-text project goals/objectives (Requirement 3).
    goals: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Multi-Mode Group Agent state ──────────────────────────────────────
    # Telegram user-id of the immutable admin (set once at /init).
    group_admin_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    # Operating mode (stored as text; see GroupMode). New groups start
    # 'uninitialized' and pick a mode during onboarding.
    group_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'uninitialized'")
    )
    # Activate/deactivate switch (the global state engine).
    bot_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )
    # Admin-toggled AI model allow-list, e.g. {"qwen_vl": true, "gemini_fallback": true}.
    allowed_models: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    status: Mapped[ProjectStatus] = mapped_column(
        project_status_enum,
        nullable=False,
        server_default=text("'active'"),
    )

    # Maps 1:1 to the Qdrant collection name. Format: project_{chat_id}
    vector_namespace: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False
    )
    qdrant_point_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    created_at: Mapped[datetime] = _created_at()
    cleared_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    memberships: Mapped[list["StudentProject"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    deadlines: Mapped[list["Deadline"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    message_logs: Mapped[list["MessageLog"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    contribution_metrics: Mapped[list["ContributionMetric"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    link_tokens: Mapped[list["ProjectLinkToken"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Project id={self.id} chat_id={self.chat_id} name={self.name!r}>"


# ===========================================================================
# student_projects (junction / many-to-many)
# ===========================================================================
class StudentProject(Base):
    __tablename__ = "student_projects"
    __table_args__ = (
        UniqueConstraint(
            "student_id", "project_id", name="uq_student_project_membership"
        ),
        Index("ix_student_projects_student_id", "student_id"),
        Index("ix_student_projects_project_id", "project_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[MemberRole] = mapped_column(
        member_role_enum, nullable=False, server_default=text("'member'")
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    linked_via: Mapped[LinkedVia] = mapped_column(linked_via_enum, nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship(back_populates="memberships")
    project: Mapped["Project"] = relationship(back_populates="memberships")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<StudentProject student={self.student_id} "
            f"project={self.project_id} role={self.role}>"
        )


# ===========================================================================
# tasks
# ===========================================================================
class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_project_id", "project_id"),
        Index("ix_tasks_assigned_to_student_id", "assigned_to_student_id"),
        Index("ix_tasks_status", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    assigned_to_student_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source: Mapped[TaskSource] = mapped_column(task_source_enum, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        task_status_enum, nullable=False, server_default=text("'pending'")
    )
    # 1=High, 2=Medium, 3=Low
    priority: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("2")
    )

    google_calendar_event_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # AI-generated context, e.g. {"confidence": 0.92, "source_message_id": 12345}
    task_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="tasks")
    assignee: Mapped[Optional["Student"]] = relationship(
        back_populates="assigned_tasks", foreign_keys=[assigned_to_student_id]
    )
    deadlines: Mapped[list["Deadline"]] = relationship(back_populates="task")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Task id={self.id} title={self.title!r} status={self.status}>"


# ===========================================================================
# deadlines
# ===========================================================================
class Deadline(Base):
    __tablename__ = "deadlines"
    __table_args__ = (
        Index("ix_deadlines_project_id", "project_id"),
        Index("ix_deadlines_due_date", "due_date"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    due_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    extracted_by: Mapped[DeadlineExtractedBy] = mapped_column(
        deadline_extracted_by_enum, nullable=False
    )
    source_message_log_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("message_logs.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    google_calendar_event_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="deadlines")
    task: Mapped[Optional["Task"]] = relationship(back_populates="deadlines")
    source_message_log: Mapped[Optional["MessageLog"]] = relationship(
        back_populates="extracted_deadlines"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Deadline id={self.id} title={self.title!r} due={self.due_date}>"


# ===========================================================================
# message_logs
# ===========================================================================
class MessageLog(Base):
    """
    Stores raw + extracted content for every Telegram message in a project.

    Blueprint note: production deployments may use declarative range/list
    partitioning on chat_id. For portability across small deployments this
    model uses a plain chat_id index (per the blueprint's "smaller deployments"
    fallback). To enable partitioning, add `postgresql_partition_by` here and
    manage partitions out-of-band.
    """

    __tablename__ = "message_logs"
    __table_args__ = (
        Index("ix_message_logs_chat_id", "chat_id"),
        Index("ix_message_logs_project_id", "project_id"),
        Index("ix_message_logs_is_vectorized", "is_vectorized"),
        # Speeds up the common "un-vectorized rows for this chat" worker scan.
        Index(
            "ix_message_logs_chat_pending_vectorization",
            "chat_id",
            postgresql_where=text("is_vectorized = FALSE AND deleted_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    telegram_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_telegram_username: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    sender_telegram_user_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )

    content_type: Mapped[ContentType] = mapped_column(content_type_enum, nullable=False)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    file_storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    is_vectorized: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    # Qdrant point IDs for traceability and deletion.
    qdrant_point_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)),
        nullable=False,
        server_default=text("'{}'::uuid[]"),
    )

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = _created_at()
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="message_logs")
    extracted_deadlines: Mapped[list["Deadline"]] = relationship(
        back_populates="source_message_log"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MessageLog id={self.id} chat_id={self.chat_id} "
            f"type={self.content_type}>"
        )


# ===========================================================================
# contribution_metrics
# ===========================================================================
class ContributionMetric(Base):
    __tablename__ = "contribution_metrics"
    __table_args__ = (
        CheckConstraint(
            "score_value >= 0 AND score_value <= 10",
            name="ck_contribution_score_range",
        ),
        Index("ix_contribution_metrics_project_id", "project_id"),
        Index("ix_contribution_metrics_student_id", "student_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
    )

    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # 0.00 - 10.00
    score_value: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    score_reason: Mapped[str] = mapped_column(Text, nullable=False)

    scoring_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    scoring_window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Breakdown: {"messages_sent": 42, "tasks_completed": 3, "files_uploaded": 1}
    score_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="contribution_metrics")
    student: Mapped["Student"] = relationship(back_populates="contribution_metrics")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ContributionMetric student={self.student_id} "
            f"score={self.score_value}>"
        )


# ===========================================================================
# project_link_tokens
# ===========================================================================
class ProjectLinkToken(Base):
    """Short-lived, single-use verification tokens for the onboarding flow (§4.1)."""

    __tablename__ = "project_link_tokens"
    __table_args__ = (
        Index("ix_project_link_tokens_chat_id", "chat_id"),
        Index("ix_project_link_tokens_project_id", "project_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_at: Mapped[datetime] = _created_at()
    # Application sets this to created_at + 15 minutes on creation.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    consumed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    consumed_by_student_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="link_tokens")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ProjectLinkToken token={self.token[:8]}… chat_id={self.chat_id}>"


# ===========================================================================
# ai_request_logs  (observability — SUTD_Admin dashboard)
# ===========================================================================
class AIRequestLog(Base):
    """
    Lightweight audit log of every call made to Agnes AI (chat, embedding,
    vision). Written best-effort by app.ai.observability; never blocks the
    request path. Payloads are trimmed summaries, not full transcripts.
    """

    __tablename__ = "ai_request_logs"
    __table_args__ = (
        Index("ix_ai_request_logs_created_at", "created_at"),
        Index("ix_ai_request_logs_chat_id", "chat_id"),
        Index("ix_ai_request_logs_kind", "kind"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = _created_at()

    chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    project_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)   # chat|embedding|vision
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # success|error
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    request_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    response_payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AIRequestLog kind={self.kind} status={self.status} model={self.model}>"


# ===========================================================================
# expenses  (Mode C — Expense Tracker)
# ===========================================================================
class Expense(Base):
    """A shared expense logged in an expense-mode group."""

    __tablename__ = "expenses"
    __table_args__ = (Index("ix_expenses_project_id", "project_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    payer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    payer_telegram_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = _created_at()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Expense {self.payer_name} {self.amount} {self.description!r}>"


# ===========================================================================
# bills / bill_items / bill_claims  (receipt-based bill splitting)
# ===========================================================================
class Bill(Base):
    """
    One receipt being split in a group. Created from a receipt photo via
    /splitbill (VLM OCR), claimed item-by-item with inline buttons, then
    finalised into a per-person breakdown (GST + service charge split
    proportionally to what each person ordered).
    """

    __tablename__ = "bills"
    __table_args__ = (
        Index("ix_bills_project_id", "project_id"),
        Index("ix_bills_chat_id", "chat_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # "receipt" — OCR'd from a photo via /splitbill.
    # "expense" — typed by hand via /splitexpense (no image).
    # Both live in this table so there is exactly ONE split + settle engine.
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'receipt'")
    )

    # The person who paid the bill and should be reimbursed.
    payer_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    payer_name: Mapped[str] = mapped_column(String(100), nullable=False)

    merchant: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    currency: Mapped[str] = mapped_column(
        String(8), nullable=False, server_default=text("'SGD'")
    )

    items_subtotal: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    service_charge: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False, server_default=text("0")
    )
    gst: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False, server_default=text("0")
    )
    other_charges: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False, server_default=text("0")
    )
    discount: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False, server_default=text("0")
    )
    total: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    # open | finalized | cancelled
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'open'")
    )
    # Telegram message id of the interactive claim keyboard (for re-rendering).
    menu_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped[datetime] = _created_at()

    items: Mapped[list["BillItem"]] = relationship(
        back_populates="bill", cascade="all, delete-orphan", order_by="BillItem.position"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Bill id={self.id} total={self.total} status={self.status}>"


class BillItem(Base):
    """A single line item on a receipt (e.g. '2x Chicken Rice — $9.00')."""

    __tablename__ = "bill_items"
    __table_args__ = (Index("ix_bill_items_bill_id", "bill_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    bill_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("bills.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    quantity: Mapped[float] = mapped_column(
        Numeric(6, 2), nullable=False, server_default=text("1")
    )
    total_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    bill: Mapped["Bill"] = relationship(back_populates="items")
    claims: Mapped[list["BillClaim"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BillItem {self.name!r} {self.total_price}>"


class BillClaim(Base):
    """A person claiming (part of) a bill item. Shared items have many claims."""

    __tablename__ = "bill_claims"
    __table_args__ = (
        UniqueConstraint("bill_item_id", "user_id", name="uq_bill_claim_item_user"),
        Index("ix_bill_claims_item_id", "bill_item_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    bill_item_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("bill_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = _created_at()

    item: Mapped["BillItem"] = relationship(back_populates="claims")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BillClaim user={self.user_name!r}>"


# ===========================================================================
# group_members  (manual roster — replaces web-app registration)
# ===========================================================================
class GroupMember(Base):
    """
    A group member added manually via the bot's Settings → Members menu
    (name + Telegram handle). Lets Agnes resolve real names to handles and
    know about lurkers who never text — no web-app registration required.
    """

    __tablename__ = "group_members"
    __table_args__ = (
        UniqueConstraint("project_id", "display_name", name="uq_group_member_name"),
        Index("ix_group_members_project_id", "project_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Without the leading @; nullable for members who have no username.
    telegram_username: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    added_by_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<GroupMember {self.display_name!r} @{self.telegram_username}>"


# ===========================================================================
# pay_profiles  (PayNow details, one per Telegram user, shared across groups)
# ===========================================================================
class PayProfile(Base):
    """A user's PayNow identifier so settle-ups can say exactly where to pay."""

    __tablename__ = "pay_profiles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    telegram_user_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, nullable=False
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Mobile number in +65XXXXXXXX form (or NRIC/UEN as entered).
    paynow_id: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PayProfile user={self.telegram_user_id} paynow={self.paynow_id!r}>"


# ===========================================================================
# bill_ratings  (how the group rated the food, 1-5 stars)
# ===========================================================================
class BillRating(Base):
    """One person's star rating for a bill. Averaged into the final summary."""

    __tablename__ = "bill_ratings"
    __table_args__ = (
        UniqueConstraint("bill_id", "user_id", name="uq_bill_rating_user"),
        Index("ix_bill_ratings_bill_id", "bill_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    bill_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("bills.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    stars: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BillRating {self.user_name!r} {self.stars}★>"


# ===========================================================================
# bill_debts  (who owes whom, frozen at finalise time)
# ===========================================================================
class BillDebt(Base):
    """
    One person's debt to the payer of one bill.

    Written when a bill is finalised rather than derived on the fly: the split
    must be frozen at that moment (later claim edits shouldn't silently rewrite
    a debt someone has already settled), and payment state needs somewhere to
    live. Aggregating these across bills is what powers the pending-expenses
    summary.

    Lifecycle:  created → paid_at (debtor says they paid)
                        → confirmed_at (creditor verifies) = settled
    """

    __tablename__ = "bill_debts"
    __table_args__ = (
        Index("ix_bill_debts_chat_id", "chat_id"),
        Index("ix_bill_debts_bill_id", "bill_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    bill_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("bills.id", ondelete="CASCADE"), nullable=False
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    debtor_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    debtor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    creditor_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    creditor_name: Mapped[str] = mapped_column(String(100), nullable=False)

    amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    # Debtor pressed "I've paid" — awaiting the creditor's confirmation.
    paid_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Creditor confirmed receipt. Non-null == fully settled.
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = _created_at()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BillDebt {self.debtor_name}→{self.creditor_name} {self.amount}>"


# ===========================================================================
# group_memories  (long-term facts Agnes has been told to remember)
# ===========================================================================
class GroupMemory(Base):
    """
    A single discrete fact about the group ("Bala's DDW exam is on 12 Nov").

    Kept separate from message_logs on purpose: message logs are raw history,
    these are curated facts the group can list, edit and delete individually.
    Entries older than MEMORY_RETENTION_DAYS are pruned automatically.
    """

    __tablename__ = "group_memories"
    __table_args__ = (Index("ix_group_memories_project_id", "project_id"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # "user" (explicitly asked to remember) or "auto" (agent-extracted).
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    created_by_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_by_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<GroupMemory {self.content[:40]!r}>"


# ===========================================================================
# member_locations  (home locations for /meetpoint)
# ===========================================================================
class MemberLocation(Base):
    """
    Where a member is travelling from, used to find a fair meeting point.

    Stored per project so the same person can have different starting points in
    different groups (home vs hall, say).
    """

    __tablename__ = "member_locations"
    __table_args__ = (
        UniqueConstraint("project_id", "display_name", name="uq_member_location_name"),
        Index("ix_member_locations_project_id", "project_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    telegram_user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # What the user typed (postal code or free-text address).
    raw_input: Mapped[str] = mapped_column(String(255), nullable=False)
    # Resolved by OneMap geocoding.
    address: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    postal_code: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MemberLocation {self.display_name!r} @ {self.postal_code}>"


__all__ = [
    "Base",
    # enums (python)
    "ProjectStatus",
    "GroupMode",
    "MemberRole",
    "LinkedVia",
    "TaskSource",
    "TaskStatus",
    "DeadlineExtractedBy",
    "ContentType",
    # models
    "Student",
    "Project",
    "StudentProject",
    "Task",
    "Deadline",
    "MessageLog",
    "ContributionMetric",
    "ProjectLinkToken",
    "AIRequestLog",
    "Expense",
    "Bill",
    "BillItem",
    "BillClaim",
    "GroupMember",
    "PayProfile",
    "GroupMemory",
    "MemberLocation",
    "BillRating",
    "BillDebt",
]
