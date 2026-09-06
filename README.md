# Agnes — your group chat's AI companion

Agnes is a Telegram bot that lives in a group chat with your friends and is
actually fun *and* useful. She remembers everything shared in the chat
(messages, photos, PDFs — all OCR'd and indexed for semantic recall) and turns
it into features people use daily:

| Command | What it does |
|---|---|
| 🧾 `/splitbill` | Reply to a **receipt photo** — Agnes OCRs the items, everyone taps what they ordered (shared dishes split automatically), service charge + GST are split **proportionally to what you ate**, and the payer's **PayNow number + QR code** is posted with the breakdown. |
| 💳 `/paynow <mobile>` | Save your PayNow so bill splits point friends at it. |
| 📋 `/summary` | Catch-up recap of the chat: topics, plans made, things needing your reply, one funny highlight. |
| 📰 `/news [sg\|world\|tech\|sport\|business]` | Live headlines from public RSS feeds, summarised by Agnes with opinions. |
| 😂 `/joke [topic]` | Original AI-generated humour, tuned to the group's running jokes (no lame programming jokes). |
| 🔥 `/roast <name>` | A playful roast grounded in the target's actual chat behaviour — receipts included, feelings intact. |
| 📚 `/exams` | Saved exam/deadline dates **with timings and countdowns** (SGT). Add naturally: `/exams add Linear Algebra final, 12 Aug 9–11am`. |
| 💬 `/ask <anything>` | General questions — or just **@mention Agnes / reply to her** in chat. |
| 💸 `/add_expense`, `/settle_up` | Quick shared-expense ledger with who-pays-whom. |
| 📍 `/meetpoint` | Everyone drops a postal code, picks what they feel like doing, and Agnes proposes 5 spots — then sends **each person their own** public-transport directions. |
| 🧠 `/mainmenu` → Memory | See, edit or delete everything Agnes remembers. Also works conversationally: *"/ask forget the memory about the DDW exam"* (always confirms first). |

There are **no modes** — every group gets every feature. `/init` just claims
the admin and switches Agnes on. `/commands` lists everything she can do.

**Model routing.** Agnes AI handles all everyday chat because it's free. A
heuristic (`app/ai/routing.py`) escalates only genuinely hard requests —
multi-step reasoning, code, analysis, calculations — to OpenRouter. Admins can
force this either way under `/mainmenu` → 🤖 AI (Auto / Agnes-only / Always
OpenRouter), and it defaults to Auto.

```
Telegram group ──► Bot (python-telegram-bot)
                        │
                        ▼
              FastAPI backend ──► Agnes AI (chat) + fastembed (local vectors)
              │   │   │   │
   PostgreSQL ┘   │   │   └─ MinIO (files)
   Qdrant (vectors)   └─ Redis (queue + pub/sub)
```

Image, receipt and scanned-PDF OCR is handled by **Qwen3-VL 32B** via the
[OpenRouter](https://openrouter.ai) API — no local Tesseract needed. OpenRouter
is used in three distinct roles, each its own env var: a cheap text model for
the fallback, a reasoning model for complex questions, and this vision model
for OCR.

> **Legacy web dashboard** — this repo also contains the original "Student
> Claw" Next.js project-management dashboard (`student-claw/frontend/` plus the
> `/verify` linking flow). It is no longer part of the core experience but is
> kept working for hackathon demos; see §7 and the legacy notes below.

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| **Python** | **3.11 or 3.12** | ⚠️ Not 3.13+ — `python-telegram-bot` v20 is incompatible. |
| **Node.js** | 18.18+ or 20+ | For the Next.js frontend. |
| **uv** | latest | Fast Python package manager. `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker** + Docker Compose | any recent | Easiest way to run Postgres / Qdrant / Redis / MinIO. |

> No Tesseract binary needed — OCR is handled by a vision model via OpenRouter.

---

## 2. Repository layout

```
student-claw/
├── backend/            FastAPI + Telegram bot + AI/RAG pipeline (Python)
│   ├── app/
│   │   ├── database/   SQLAlchemy models, connection, init
│   │   ├── bot/        Telegram handlers, billsplit, news, services
│   │   ├── ai/         Parsing/OCR, embeddings, agent, pipeline
│   │   ├── api/        FastAPI routers (legacy dashboard API)
│   │   └── core/       Auth, security, config
│   ├── main.py         FastAPI entrypoint
│   ├── requirements.txt
│   └── .env.example
├── frontend/           LEGACY Next.js dashboard (kept for hackathons)
└── docker-compose.yml  Local infra (Postgres/Qdrant/Redis/MinIO)
```

---

## 3. Obtain your API keys

You need **four** external credentials. Collect them before filling in `.env`.

### 3a. Agnes AI (required — chat + embeddings)
1. Sign in to the Agnes AI hub at **https://apihub.agnes-ai.com**.
2. Create an API key.
3. You'll use:
   - `AGNES_AI_API_KEY` = your key
   - `AGNES_AI_BASE_URL` = `https://apihub.agnes-ai.com/v1`
   - `AGNES_CHAT_MODEL` / `AGNES_EMBED_MODEL` — set to the model names your
     account exposes (defaults: `agnes-2.0-flash`, `agnes-embeddings`).
   - `EMBED_DIM` — the embedding dimensionality of your embed model (default `1536`).

### 3b. Telegram bot (required)
1. In Telegram, message **@BotFather** → `/newbot` → follow prompts.
2. Copy the **HTTP API token** → `TELEGRAM_BOT_TOKEN`.
3. **Disable privacy mode** so the bot can read group messages (this is essential
   for the passive RAG listener): BotFather → `/setprivacy` → select your bot →
   **Disable**.
4. Set `TELEGRAM_BOT_USERNAME` to the bot's @username (without the `@`).

### 3c. OpenRouter (required — VLM image & PDF OCR)
1. Sign up at **https://openrouter.ai** and add a credit balance.
2. Go to **Keys** → **Create key** → copy the key.
3. Set `OPENROUTER_API_KEY` in `backend/.env`.
4. The defaults are pre-configured — **three separate roles**, don't collapse
   them into one variable:
   - `OPENROUTER_MODEL` = `google/gemini-2.5-flash-lite` — cheap, fast text
     fallback. Must be non-reasoning; a reasoning model burns tokens thinking
     before every casual reply.
   - `OPENROUTER_REASONING_MODEL` = `deepseek/deepseek-v4-flash` — used only
     for requests `app/ai/routing.py` judges complex.
   - `OPENROUTER_VISION_MODEL` = `qwen/qwen3-vl-32b-instruct` — receipt and
     document OCR. **Must accept image input** (DeepSeek text models do not).

   Any OpenRouter model can be swapped in without code changes.

### 3d. Google Calendar OAuth2 (optional — deadline sync)
1. Go to **https://console.cloud.google.com** → create/select a project.
2. **APIs & Services → Library →** enable **Google Calendar API**.
3. **OAuth consent screen** → External → add yourself as a test user.
4. **Credentials → Create credentials → OAuth client ID → Web application.**
5. Add an **Authorized redirect URI**:
   `http://localhost:3000/api/integrations/google/callback`
6. Copy the **Client ID** → `GOOGLE_CLIENT_ID` and **Client secret** →
   `GOOGLE_CLIENT_SECRET` (set in **both** backend and frontend `.env`).

---

## 4. Generate the shared secrets

Several secrets must be **identical in the backend and frontend** `.env` files.
Generate them once and paste the same value into both:

```bash
# 32-byte hex secrets (use a fresh one for each line)
openssl rand -hex 32     # JWT_SECRET            (shared)
openssl rand -hex 32     # JWT_REFRESH_SECRET    (shared, different from above)
openssl rand -hex 32     # ENCRYPTION_KEY        (shared — must be 64 hex chars)
openssl rand -hex 32     # OAUTH_COOKIE_SECRET   (frontend)
openssl rand -hex 32     # PROJECT_KEY_HMAC_SECRET (backend)
openssl rand -hex 32     # TELEGRAM_WEBHOOK_SECRET (backend)
```

> **`ENCRYPTION_KEY` must decode to exactly 32 bytes** (AES-256). `openssl rand -hex 32`
> gives exactly 64 hex chars = 32 bytes. The backend and frontend use it to
> encrypt/decrypt the Google refresh token — if they differ, calendar sync breaks.

| Secret | Backend | Frontend | Must match? |
|---|:---:|:---:|:---:|
| `JWT_SECRET` | ✅ | ✅ | **Yes** |
| `JWT_REFRESH_SECRET` | ✅ | ✅ | **Yes** |
| `ENCRYPTION_KEY` | ✅ | ✅ | **Yes** |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | ✅ | ✅ | **Yes** |
| `REDIS_URL` | ✅ | ✅ | Yes (same instance) |
| `OAUTH_COOKIE_SECRET` | — | ✅ | n/a |
| `PROJECT_KEY_HMAC_SECRET`, `TELEGRAM_WEBHOOK_SECRET` | ✅ | — | n/a |

---

## 5. Start the infrastructure

The fastest path uses Docker:

```bash
cd student-claw
docker compose up -d
```

This launches:
- **PostgreSQL** on `localhost:5432` (db/user/pass = `student_claw`)
- **Qdrant** on `localhost:6333`
- **Redis** on `localhost:6379`
- **MinIO** on `localhost:9000` (console `:9001`, login `minioadmin` / `minioadmin`)

> Prefer your own managed services? Skip Compose and point the `DATABASE_URL`,
> `QDRANT_URL`, `REDIS_URL`, and `MINIO_*` env vars at them instead.

---

## 6. Backend setup

```bash
cd student-claw/backend

# 1) Create a virtual environment with uv (Python 3.12 recommended)
uv venv --python 3.12 .venv

# 2) Install dependencies
uv pip install -r requirements.txt --python .venv/bin/python

# 3) Environment
cp .env.example .env
#    → open .env and fill in every value from sections 3 & 4 above

# 4) Create the database tables (installs pgcrypto, creates all tables + enums)
.venv/bin/python -m app.database.init_db
```

---

## 7. Frontend setup (LEGACY — optional)

The Next.js dashboard is a legacy feature; skip this section unless you want
the old web app for a hackathon demo.

```bash
cd student-claw/frontend

# 1) Install dependencies
npm install

# 2) Environment
cp .env.example .env
#    → fill in FASTAPI_BASE_URL, the SHARED secrets (JWT_SECRET,
#      JWT_REFRESH_SECRET, ENCRYPTION_KEY), REDIS_URL, OAUTH_COOKIE_SECRET,
#      and the Google client id/secret + redirect URI
```

---

## 8. Run everything — ONE command 🎉

```bash
cd student-claw
docker compose up --build        # add -d to run detached
```

That single command builds the backend image and starts **seven services** in
one terminal: Postgres, Qdrant, Redis, MinIO, plus the FastAPI **api** (with
`init_db` run automatically — new tables just appear), the embedding
**worker**, and the Telegram **bot** in polling mode. Requirements: Docker and
a filled-in `backend/.env` (sections 3–4). No Python/uv/Node needed on the host.

Useful follow-ups:

```bash
docker compose logs -f bot       # follow one service's logs (api / worker / bot)
docker compose restart bot       # pick up backend code changes (api hot-reloads itself)
docker compose down              # stop everything, keep data
```

Notes:
- The backend source is bind-mounted, so the **api** hot-reloads on save;
  restart `worker`/`bot` after editing their code.
- First `worker` start downloads the ~100 MB fastembed model into a cached
  volume (one-time).
- The legacy Next.js dashboard is not in compose — run it manually with
  `cd frontend && npm run dev` if you need it.

<details>
<summary><b>Alternative: native multi-terminal dev workflow</b> (fastest iteration)</summary>

```bash
# Terminal 1 — infra only
docker compose up -d postgres qdrant redis minio

# Terminal 2 — FastAPI web API            → http://localhost:8000  (docs: /docs)
cd backend && .venv/bin/python -m uvicorn main:app --reload --port 8000

# Terminal 3 — embedding worker (drains the Redis embed_queue)
cd backend && .venv/bin/python -m app.ai.pipeline

# Terminal 4 — Telegram bot (POLLING mode for local dev — no public URL needed)
cd backend && .venv/bin/python -m app.bot.bot
```
</details>

> In local dev the bot runs in **polling** mode and FastAPI logs that no
> `TELEGRAM_WEBHOOK_BASE_URL` is set — that's expected. In production you set
> `TELEGRAM_WEBHOOK_BASE_URL` to your public HTTPS host and drop the bot
> process; FastAPI registers the webhook and receives updates directly.

### First run — try it out
1. Add your bot to a Telegram **group** → Agnes introduces herself.
2. Run `/init` and pick the group's vibe (you become its admin).
3. Save your PayNow: `/paynow 91234567`.
4. Chat for a bit, then try `/summary`, `/news`, `/joke`, `/roast <friend>`.
5. Post a receipt photo, reply to it with `/splitbill`, tap what you ordered,
   hit **✅ Split it** — breakdown + PayNow QR appear.
6. Share a PDF/image, run `/sync`, then `/ask` about its contents.

> **Legacy dashboard flow:** register at `http://localhost:3000` with your
> Telegram username, link the group via its Project Key, then `/verify <token>`
> in the group. Only needed if you're using the old web app.

---

## 9. Environment variable reference

### Backend (`backend/.env`)
| Variable | Required | Description |
|---|:---:|---|
| `DATABASE_URL` | ✅ | PostgreSQL async connection string. |
| `AGNES_AI_API_KEY` | ✅ | Agnes AI key. |
| `AGNES_AI_BASE_URL` | | Default `https://apihub.agnes-ai.com/v1`. |
| `AGNES_CHAT_MODEL` / `AGNES_EMBED_MODEL` | | Model names. |
| `EMBED_DIM` | | Embedding dimension (default `1536`). |
| `QDRANT_URL` / `QDRANT_API_KEY` | ✅/ | Vector DB. |
| `REDIS_URL` | ✅ | Queue + pub/sub. |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_SECURE` | ✅ | File storage. |
| `TELEGRAM_BOT_TOKEN` | ✅ | From BotFather. |
| `TELEGRAM_BOT_USERNAME` | | Bot @username. |
| `TELEGRAM_WEBHOOK_SECRET` | ✅ | Random secret. |
| `TELEGRAM_WEBHOOK_BASE_URL` | prod | Public HTTPS host (prod webhook only). |
| `PROJECT_KEY_HMAC_SECRET` | ✅ | Derives public project keys. |
| `JWT_SECRET` / `JWT_REFRESH_SECRET` | ✅ | Token signing (shared w/ frontend). |
| `ENCRYPTION_KEY` | ✅ | AES-256 key, 32 bytes (shared w/ frontend). |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | calendar | Google OAuth client. |
| `CORS_ORIGINS` | | Comma-separated allowed browser origins. |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter key for VLM image/PDF OCR. |
| `OPENROUTER_BASE_URL` | | Default `https://openrouter.ai/api/v1`. |
| `OPENROUTER_MODEL` | | Cheap text fallback. Default `google/gemini-2.5-flash-lite`. |
| `OPENROUTER_REASONING_MODEL` | | Complex requests only. Default `deepseek/deepseek-v4-flash`. |
| `OPENROUTER_VISION_MODEL` | | OCR; must accept images. Default `qwen/qwen3-vl-32b-instruct`. |

### Frontend (`frontend/.env`)
| Variable | Required | Description |
|---|:---:|---|
| `FASTAPI_BASE_URL` | ✅ | e.g. `http://localhost:8000`. |
| `JWT_SECRET` / `JWT_REFRESH_SECRET` | ✅ | Shared with backend. |
| `ENCRYPTION_KEY` | ✅ | Shared with backend. |
| `REDIS_URL` | ✅ | Same Redis as backend (SSE pub/sub). |
| `OAUTH_COOKIE_SECRET` | ✅ | Signs the PKCE/state cookie (defaults to `JWT_SECRET`). |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | calendar | Same client as backend. |
| `GOOGLE_REDIRECT_URI` | calendar | `http://localhost:3000/api/integrations/google/callback`. |

---

## 10. Troubleshooting

- **`ModuleNotFoundError: telegram` / bot won't `.build()`** — you're on Python
  3.13+. Use 3.11 or 3.12.
- **Bot doesn't see group messages** — privacy mode is on. BotFather →
  `/setprivacy` → Disable, then remove & re-add the bot to the group.
- **VLM OCR returns nothing** — check that `OPENROUTER_API_KEY` is set and your
  OpenRouter account has credit. The pipeline logs errors per-page; look for
  `VLM extraction failed` in the `app.ai.pipeline` output.
- **Calendar sync silently does nothing** — `ENCRYPTION_KEY` differs between
  backend and frontend, or the user hasn't connected Google in Settings.
- **401 loops in the dashboard** — `JWT_SECRET` differs between backend and
  frontend; they must match exactly.
- **No real-time updates** — the frontend and backend must point at the *same*
  `REDIS_URL`.
- **Redis `BRPOP` timeout** — ensure the Redis client has `socket_timeout=None`
  in `app/ai/queue.py` (already set). Blocking pops need an unlimited socket timeout.

---

## 11. Production notes
- Run the bot in **webhook** mode (set `TELEGRAM_WEBHOOK_BASE_URL`) behind HTTPS;
  serve FastAPI with `gunicorn -k uvicorn.workers.UvicornWorker`.
- Use **managed** Postgres/Redis/Qdrant or harden the Compose stack; enable
  MinIO TLS (`MINIO_SECURE=true`).
- `python -m app.database.init_db` is a dev bootstrap — adopt **Alembic**
  migrations for schema changes in production.
- Rotate all secrets; never commit `.env` (it's git-ignored).
- OpenRouter costs are per-token. For high-volume deployments consider caching
  OCR results in MinIO alongside the original files to avoid re-processing.
