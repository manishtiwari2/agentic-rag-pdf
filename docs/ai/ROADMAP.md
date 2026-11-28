# Roadmap: one small task per prompt

Rules: do tasks in order inside a phase. One task = one prompt = one commit.
One phase = one branch and one PR. Tick a task only when CHECKLIST.md passes.

Target stack (Manish's decisions, 2026-11-26): Docker Compose on one GPU box,
FastAPI with an htmx web UI, Ollama by default plus a vLLM option, PostgreSQL
with pgvector, Redis with arq for jobs, uv with a lockfile. English only.

## P1 Safety fixes (current code)
- [x] 1.1 Verifier fails closed: "unavailable" leads to a refusal by default (config flag). Bug #1.
- [x] 1.2 Add `DOCUMENT_TEXT_IS_DATA` to the conflict and chat-rewrite prompts, and a test that every prompt has it. Bug #6.
- [x] 1.3 Per-call LLM timeout plus a wall-clock budget per question. Bug #4.
- [x] 1.4 Upload limits: max MB, max pages, `%PDF` magic bytes, reject encrypted PDFs. Bug #5.
- [x] 1.5 stdlib logging (JSON lines, request id). Replace silent excepts. Bugs #2, #3.
- [x] 1.6 Fix the `tests/test_config.py` reference. Bug #10.

## P2 Data layer
- [x] 2.1 Postgres + pgvector schema with SQLAlchemy 2 and Alembic (LF line endings).
- [x] 2.2 A pgvector store behind the existing store interface.
- [x] 2.3 Index keyed by (file sha256, config fingerprint, code version). No re-embedding of the same file.

## P3 API + auth
- [x] 3.1 FastAPI app: /health, /ready, /metrics.
- [x] 3.2 Register, login and logout: argon2, httponly cookie, CSRF, roles.
- [x] 3.3 Upload, list, delete and version documents. Ingestion runs as an arq job.
- [x] 3.4 Ask with SSE streaming, and chat history. Every query filtered by user_id.
- [x] 3.5 Rate limits, quotas, question length cap, CORS allowlist, JSON errors.

## P4 Web UI
- [x] 4.1 Jinja + htmx pages: login, library, chat.
- [x] 4.2 Clickable citations that open the PDF page with the text highlighted and a conflict banner.
- [x] 4.3 Remove Gradio once the new UI works.

## P5 Agent quality
- [x] 5.1 Reranker-score floor as a free grader, and set `min_score`. Bug #9.
- [x] 5.2 Missing-term hint (a name, number or year missing from all evidence).
- [x] 5.3 Adaptive strategy: answer, verify, and escalate only on failure. Bug #8.
- [x] 5.4 Table calculator: code does the arithmetic on cited cells.
- [x] 5.5 Adaptive vs agentic paired bootstrap on dev, with the rule written first.

## P6 Performance
- [x] 6.1 Load models once per process; batch embedding in the worker.
- [x] 6.2 Redis answer cache with invalidation; token streaming; p50, p95 and first-token metrics.

## P7 Deployment
- [x] 7.1 Multi-stage Dockerfile: non-root user, HEALTHCHECK, pinned base image.
- [x] 7.2 Compose: api, worker, postgres, redis, ollama, plus vllm profile and CPU override.
- [x] 7.3 Pinned HF model revisions, uv lockfile, `.env.example`, backup and restore script.

## P8 CI and docs
- [ ] 8.1 CI: ruff, mypy, coverage gate, pip-audit, bandit, Docker build, fake-backend benchmark smoke.
- [ ] 8.2 Integration tests with testcontainers Postgres and Redis.
- [ ] 8.3 Docs: README quick start, DEPLOY runbook, SECURITY threat model. Update FLOW.md and MENTAL-MODEL.md.
