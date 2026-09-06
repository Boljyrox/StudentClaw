# Deploying Agnes to Oracle Cloud (Always Free ARM)

This runs the whole stack (bot + api + worker + Postgres + Qdrant + Redis +
MinIO) on a free Oracle Cloud ARM VM so your Mac doesn't have to. The bot uses
**long-polling**, so there's **no domain, no TLS, and no inbound ports** to open
beyond SSH.

**Time:** ~45–60 min (most of it waiting on the first ARM image build).
**Cost:** $0, permanently (Oracle "Always Free" A1 shapes).

---

## 0. Before you start

You need, from your local machine:
- Your working `student-claw/backend/.env` (with all the real keys).
- Access to the GitHub repo (`Boljyrox/AgnesHackathon`).

> **Single-poller rule:** Telegram allows exactly **one** process polling a bot
> token. Before the cloud bot goes live, you'll stop the one on your Mac —
> otherwise you get `Conflict: terminated by other getUpdates request`. That's
> covered in step 6.

---

## 1. Create the Oracle VM

1. Sign up at <https://cloud.oracle.com> (needs a card for identity
   verification — Always Free resources are never charged). Pick a home region
   close to you (e.g. Singapore).
2. **Compute → Instances → Create instance.**
3. **Image and shape:**
   - Image: **Canonical Ubuntu 22.04** (make sure it's the **aarch64/ARM**
     build).
   - Shape: **Ampere → VM.Standard.A1.Flex**. Set **2 OCPUs / 12 GB RAM**
     (plenty for this — idle is ~650 MB, peak ~2 GB). You may go up to 4/24 for
     free; 2/12 leaves half your free allocation spare.
   - > If you see **"Out of host capacity"**, that's Oracle's well-known free
     > A1 shortage. Try a different Availability Domain in the dropdown, or
     > retry over a few hours. It's a one-time annoyance.
4. **SSH keys:** choose **Generate a key pair for me** and download the private
   key (or paste your own `~/.ssh/id_ed25519.pub`). Save the private key as
   `~/.ssh/oracle_agnes` and `chmod 600 ~/.ssh/oracle_agnes`.
5. Leave networking at defaults (it creates a VCN whose security list allows
   only inbound SSH — exactly what we want). **Create.**
6. When it's **Running**, copy the **Public IP address**.

---

## 2. First login + install Docker

From your Mac (replace the IP):

```bash
ssh -i ~/.ssh/oracle_agnes ubuntu@<PUBLIC_IP>
```

On the server:

```bash
# Install Docker Engine + Compose plugin (the script supports arm64 and
# enables the docker service so containers come back after a reboot).
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker version
docker compose version
```

---

## 3. Get the code onto the box

The repo is private, so give the VM a **read-only deploy key**:

```bash
# On the server:
ssh-keygen -t ed25519 -C "oracle-agnes-deploy" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Copy that public key, then on GitHub go to **the repo → Settings → Deploy keys
→ Add deploy key**, paste it, leave "Allow write access" **unchecked**, save.

Back on the server:

```bash
git clone git@github.com:Boljyrox/AgnesHackathon.git
cd AgnesHackathon/student-claw
```

> Prefer HTTPS? `git clone https://github.com/Boljyrox/AgnesHackathon.git` and
> enter a GitHub Personal Access Token when prompted.

---

## 4. Copy your secrets across

`.env` is gitignored (correctly — it holds your keys), so copy it up from your
Mac. Run this **on your Mac**, not the server:

```bash
scp -i ~/.ssh/oracle_agnes \
  "student-claw/backend/.env" \
  ubuntu@<PUBLIC_IP>:~/AgnesHackathon/student-claw/backend/.env
```

The `localhost` URLs inside `.env` don't matter on the server — `docker-compose.yml`
overrides `DATABASE_URL`, `QDRANT_URL`, `REDIS_URL` and `MINIO_ENDPOINT` with the
internal service names. Everything else (Agnes key, OpenRouter, Telegram token,
admin token, MinIO creds, OneMap) is read from this file.

> Want `/meetpoint` to give real public-transport directions? Make sure the
> `ONEMAP_*` values are set in this `.env` (geocoding works without them; step
> routing needs a token or email/password). See `backend/.env.example`.

---

## 5. Build and launch

On the server, from `~/AgnesHackathon/student-claw`:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

- The **first build is slow on ARM** (compiling/installing onnxruntime,
  tokenizers, etc.) — give it 5–15 min.
- On its first run the **worker downloads the ~100 MB fastembed model** into a
  volume; it's cached after that.
- The **api** container creates every table on startup (`init_db`), so there's
  no separate migration step.

Watch it come up:

```bash
docker compose ps
docker compose logs -f bot          # Ctrl-C to stop following
```

You want to see the bot log that it has started polling, with no `Conflict`
errors.

---

## 6. Cut over from your Mac

1. **Stop the Mac stack** so only the cloud bot polls Telegram:
   ```bash
   # On your Mac, in student-claw/
   docker compose down
   ```
   (Then you can quit OrbStack entirely — that's the whole point.)
2. In Telegram, send your group a `/ask hello` (or open `/menu`). If it
   answers, the cloud deploy is live.

---

## 7. Day-to-day operations

All on the server, from `~/AgnesHackathon/student-claw`:

```bash
# Follow logs
docker compose logs -f bot
docker compose logs -f worker

# Restart everything (e.g. after editing .env)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Deploy new code you've pushed to GitHub
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Stop (keeps data) / start
docker compose stop
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Reboots take care of themselves:** `restart: unless-stopped` (from
`docker-compose.prod.yml`) plus the enabled Docker service means the whole stack
returns automatically after Oracle's maintenance reboots.

**Backups (optional but wise):** the data lives in Docker volumes. To snapshot
Postgres:
```bash
docker compose exec postgres pg_dump -U student_claw student_claw > ~/agnes-backup-$(date +%F).sql
```

---

## Hardening notes (optional)

- **Nothing is internet-exposed.** The published container ports bind on the
  VM, but Oracle's VCN security list only permits inbound SSH, so Postgres,
  MinIO, etc. are unreachable from outside. To reach the admin API from your
  Mac, tunnel it: `ssh -i ~/.ssh/oracle_agnes -L 8000:localhost:8000 ubuntu@<IP>`
  then open `http://localhost:8000`.
- **Change the defaults** if you want defence-in-depth: `POSTGRES_PASSWORD` and
  `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` in `docker-compose.yml` are
  `student_claw` / `minioadmin`. If you change the MinIO ones, update the
  matching `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` in `.env` too.
- **Do NOT open extra ports** in the Oracle VCN security list. Docker manages
  its own host iptables and can bypass a host firewall — the VCN list is your
  reliable outer wall, so keep it SSH-only.
- **Swap:** unnecessary at 12 GB. If you ever downsize to a 1–2 GB shape, add a
  2 GB swapfile first.
