"""
AI / RAG subsystem configuration (blueprint §3, §5, §7.3).

Everything is env-driven. Loaders are lazy + cached so that importing a single
piece (e.g. the lightweight queue used by the bot process) never forces the
whole ML stack or a populated AI environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Qdrant collection naming — must match app.bot.keys.vector_namespace_for.
COLLECTION_TEMPLATE = "project_{chat_id}"

# Redis keys for the async ingestion pipeline (§5.1).
EMBED_QUEUE_KEY = "embed_queue"
EMBED_DEAD_LETTER_KEY = "embed_dead_letter"

# Embedding batch ceiling (§5.1: "up to 20 chunks per API call").
EMBED_BATCH_SIZE = 20

# Agentic loop bounds (§3.4).
AGENT_MAX_ITERATIONS = 5
AGENT_TIMEOUT_SECONDS = 30

# How many recent text logs to inject into the system prompt window (§3.2).
RECENT_MESSAGE_WINDOW = 40

# Short-term memory: number of most recent turns injected as conversation memory.
MEMORY_TURNS = 8

# Long-term group memory: discrete facts Agnes has been told to remember.
# Anything older than this is pruned automatically (one month, per product spec).
MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
# Ceiling on how many memories get injected into the prompt.
MEMORY_PROMPT_LIMIT = 40

# Semantic search: drop hits below this cosine score — low-score matches are
# noise that misleads the agent more than it helps.
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.25"))

# Chunking constants (§2.4).
PDF_CHUNK_TOKENS = 512
PDF_CHUNK_OVERLAP = 64
IMAGE_CHUNK_TOKENS = 256
IMAGE_CHUNK_OVERLAP = 32
CHAT_WINDOW_MESSAGES = 10
CHAT_WINDOW_OVERLAP = 2

# Known document MIME types.
MIME_PDF = "application/pdf"
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set (§7.3).")
    return value


@dataclass(frozen=True)
class AISettings:
    agnes_api_key: str
    agnes_base_url: str
    chat_model: str
    vision_model: str
    embed_model: str
    embed_dim: int
    # Embedding provider: "fastembed" (local, default) or "agnes".
    embed_provider: str
    fastembed_model: str
    # OpenRouter. THREE distinct roles — do not collapse them into one var:
    #   openrouter_model            cheap text model for the chat fallback
    #   openrouter_reasoning_model  stronger text model for complex requests
    #   openrouter_vision_model     multimodal model for receipt/document OCR
    # These were previously all one setting, which meant ordinary text replies
    # were being answered by a large (expensive) vision model.
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_reasoning_model: str
    openrouter_vision_model: str


@dataclass(frozen=True)
class OneMapSettings:
    """OneMap SG — geocoding is public; routing needs a token."""

    base_url: str
    token: str
    email: str
    password: str


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    api_key: str | None


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool


# Default local embedding model (no API needed). 384-dim.
_FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"
_FASTEMBED_DIM = 384


@lru_cache(maxsize=1)
def get_ai_settings() -> AISettings:
    chat_model = os.getenv("AGNES_CHAT_MODEL", "agnes-2.0-flash")
    # Default to the local provider since Agnes exposes no embeddings endpoint.
    embed_provider = os.getenv("EMBED_PROVIDER", "fastembed").lower()
    fastembed_model = os.getenv("FASTEMBED_MODEL", _FASTEMBED_DEFAULT)
    embed_dim = (
        _FASTEMBED_DIM
        if embed_provider == "fastembed"
        else int(os.getenv("EMBED_DIM", "1536"))
    )
    return AISettings(
        agnes_api_key=_require("AGNES_AI_API_KEY"),
        agnes_base_url=os.getenv("AGNES_AI_BASE_URL", "https://apihub.agnes-ai.com/v1"),
        chat_model=chat_model,
        # Vision-capable model for image/scanned-PDF understanding. Defaults to
        # the chat model (agnes-2.0-flash supports vision inputs).
        vision_model=os.getenv("AGNES_VISION_MODEL", chat_model),
        embed_model=os.getenv("AGNES_EMBED_MODEL", "agnes-embeddings"),
        embed_dim=embed_dim,
        embed_provider=embed_provider,
        fastembed_model=fastembed_model,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        # Cheap, fast text model — the everyday fallback when Agnes is down.
        openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
        # Stronger text model, used only for requests the router deems complex.
        openrouter_reasoning_model=os.getenv(
            "OPENROUTER_REASONING_MODEL", "deepseek/deepseek-v3.2"
        ),
        # Multimodal model for receipt photos and scanned PDFs. MUST support
        # image input — DeepSeek text models do not.
        openrouter_vision_model=os.getenv(
            "OPENROUTER_VISION_MODEL", "qwen/qwen3-vl-32b-instruct"
        ),
    )


@lru_cache(maxsize=1)
def get_onemap_settings() -> OneMapSettings:
    return OneMapSettings(
        base_url=os.getenv("ONEMAP_BASE_URL", "https://www.onemap.gov.sg"),
        token=os.getenv("ONEMAP_TOKEN", ""),
        email=os.getenv("ONEMAP_EMAIL", ""),
        password=os.getenv("ONEMAP_PASSWORD", ""),
    )


@lru_cache(maxsize=1)
def get_qdrant_settings() -> QdrantSettings:
    return QdrantSettings(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


@lru_cache(maxsize=1)
def get_minio_settings() -> MinioSettings:
    return MinioSettings(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=_require("MINIO_ACCESS_KEY"),
        secret_key=_require("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes"},
    )


def collection_name(chat_id: int) -> str:
    return COLLECTION_TEMPLATE.format(chat_id=chat_id)
