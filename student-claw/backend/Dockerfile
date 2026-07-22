# ── Agnes backend image ─────────────────────────────────────────────────────
# One image, three roles (chosen by the compose service's `command`):
#   api    → uvicorn main:app
#   worker → python -m app.ai.pipeline
#   bot    → python -m app.bot.bot
#
# python:3.12 — python-telegram-bot v20 is incompatible with 3.13+.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits don't bust the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY app ./app

# No CMD — each docker-compose service supplies its own command.
