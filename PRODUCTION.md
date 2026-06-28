# Production Deployment Runbook

This service is a multi-tenant WhatsApp bot (Baileys + Express + Socket.io + SQLite)
with an AI reply engine. It is designed to run on **Fly.io** but the container is
portable to any host that provides a persistent volume.

---

## 1. Required configuration

All secrets are supplied via environment variables — see [`.env.example`](.env.example)
for the full list. The application **refuses to start in production** (`NODE_ENV=production`)
without `JWT_SECRET`.

| Variable | Required | Purpose |
|---|---|---|
| `JWT_SECRET` | ✅ | Signs auth tokens. Generate: `openssl rand -hex 32` |
| `CORS_ORIGINS` | ✅ (prod) | Comma-separated allowlist of front-end origins |
| `ADMIN_EMAIL` | ✅ | Email granted the admin role |
| `DEEPSEEK_API_KEY` | ✅ | Primary text/agentic engine — direct DeepSeek paid API |
| `OPENROUTER_API_KEY` | optional | Fallback route if DeepSeek is unreachable |
| `GEMINI_API_KEYS` | optional | Vision/image understanding only. Comma-separated for rotation |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | optional | "Sign in with Google" |
| `GROQ_API_KEY`, `QWEN_ENDPOINT` | optional | Alternate AI providers / failover |

### AI routing (fully managed — no user-facing engine selection)
- **Text replies** → DeepSeek via the **direct DeepSeek API**, model `deepseek-chat`
  (non-thinking V4 Flash — ~10x cheaper/faster than thinking mode for chat).
  Failover chain: DeepSeek → OpenRouter → Chaka.
  ⚠️ `deepseek-chat` retires **2026-07-24**; before then, re-verify a non-thinking
  call on `deepseek-v4-flash` (or pick the successor) and update `DEEPSEEK_MODEL`.
  For future agentic/tool-calling, route those calls to `deepseek-reasoner`.
- **Vision (incoming images)** → Gemini (`describeImage()`).
- **Audio (voice notes)** → Gemini transcription (`transcribeAudio()`), then the
  transcript flows into DeepSeek as normal text. No local STT model (chosen over
  Vosk to keep the Fly machine light).
- All API keys are **platform-hosted via env** — users never upload their own keys,
  and the dashboard exposes no engine/model/key controls.

---

## 2. First-time Fly.io setup

```bash
fly apps create chaka-wap            # if it doesn't exist yet
fly volumes create chaka_data --size 3 --region lhr   # persistent storage for DB + sessions

# Set secrets (never put these in fly.toml or git):
fly secrets set \
  JWT_SECRET="$(openssl rand -hex 32)" \
  ADMIN_EMAIL="you@example.com" \
  CORS_ORIGINS="https://chaka-wap.fly.dev" \
  PUBLIC_URL="https://chaka-wap.fly.dev" \
  DEEPSEEK_API_KEY="sk-..." \
  GEMINI_API_KEYS="AIza...,AIza..." \
  GOOGLE_CLIENT_ID="..." \
  GOOGLE_CLIENT_SECRET="..."

fly deploy
```

`fly.toml` keeps `min_machines_running = 1` and `auto_stop_machines = false` on
purpose: the bot must stay online to receive WhatsApp messages and run the cron
scheduler. Do not enable scale-to-zero.

---

## 3. ⚠️ Rotate previously-exposed credentials

The old source code committed real secrets to git history. Before going live,
**revoke and regenerate** anything that was ever in the repo:

1. **Gemini API key** that was hardcoded in `server.js` — delete it in Google AI
   Studio and issue new keys, then set `GEMINI_API_KEYS`.
2. **Google OAuth client secret** — rotate in Google Cloud Console if it was ever committed.
3. **JWT secret** — the old default (`chaka_super_secret_dev_key_2026`) is burned;
   generate a fresh `JWT_SECRET`. (This invalidates all existing login tokens — users re-login.)

### Scrub git history (if the repo was ever pushed/shared)
Removing files from tracking does **not** remove them from history. If this repo
was ever pushed anywhere, scrub the leaked blobs:

```bash
# using git-filter-repo (recommended)
git filter-repo --invert-paths \
  --path .wwebjs_auth --path database.sqlite --path server_log.txt
```

Then force-push and have all collaborators re-clone. Treat any WhatsApp session
that was committed (`.wwebjs_auth/`) as compromised — log it out and re-pair.

---

## 4. Security posture (implemented)

- `helmet` security headers; JSON body capped at 5 MB.
- Socket.io / CORS restricted to `CORS_ORIGINS` in production.
- Rate limiting on `/api/login` and `/api/register` (30 / 15 min / IP).
- Secrets required from env; no hardcoded fallbacks in production.
- Graceful shutdown on `SIGTERM`/`SIGINT` (clean DB + HTTP close).
- `.gitignore` blocks `.env`, DBs, logs, and all session state.

---

## 5. Planned: Supabase (Postgres) migration

Decision: move the **database** to Supabase Postgres; **keep auth on Firebase**.
This is a deliberate future phase — not done yet. Recommended sequence:

1. **Schema** — port the SQLite tables (`users`, `global_settings`, `contacts`,
   `messages`, `stickers`, `scheduled_messages`, `request_logs`) to Postgres.
   Replace SQLite-isms (`INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`, integer
   unix timestamps are fine, `AUTOINCREMENT` → `bigserial`).
2. **Data access** — the app uses ~one `db.get/all/run` helper throughout. Wrap
   `pg` (or `@supabase/supabase-js`) behind the same `db.get/all/run` interface so
   the call sites barely change. Convert `?` placeholders to `$1,$2,...`.
3. **Tenant isolation** — enforce per-user **Row-Level Security** in Postgres so
   isolation no longer depends on every query remembering `WHERE user_id = ?`.
4. **WhatsApp sessions stay on disk** — Baileys `auth_baileys_*` state remains on
   the Fly volume (`/app/data`); Supabase does not replace it.
5. **Cutover** — export current `data/chaka_data.db` → load into Supabase → run both
   in parallel briefly → switch `DATABASE_URL`.

### Known follow-ups (not yet done)
- Container runs as root; move to a non-root user once volume ownership is verified.
- `uncaughtException` is logged-and-continued (chosen for bot uptime) rather than
  crash-and-restart — revisit if you prefer fail-fast semantics.
- Front-end ships inline scripts, so CSP is disabled in helmet; tighten after a
  front-end asset refactor.
- Add automated volume backups (`fly volumes snapshots`) for the SQLite DB.
