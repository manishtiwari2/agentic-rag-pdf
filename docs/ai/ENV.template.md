# Environment: everything the app needs

## Standalone CLI

Model and retrieval settings live in `src/config.py` and can be overridden
with CLI flags. Local models do not require an API key.

| Need | Required? | What it is |
|---|---|---|
| Python 3.10 to 3.13 | yes | CI tests 3.10 and 3.13 |
| NVIDIA GPU, 16 GB (T4 or better) | for real models | The real stack used 6.15 GB in the T4 smoke run. Without a GPU, use `--offline` stand-in models. |
| Tesseract binary | only for scanned PDFs | System install. `pytesseract` only calls it. |
| Hugging Face cache (`HF_HOME`) | optional | Where models download to. Default `~/.cache/huggingface`. |
| `HF_TOKEN` | no | Only needed if a model becomes gated. Never commit it. |

## Web service

Set `RAG_RERANKER_MIN_SCORE` to a finite raw score to enable or override the
reranker floor for the API. Use a score calibrated for the active reranker.
The model-backed preset leaves the floor off when this variable is unset.

Set `RAG_RETRIEVAL_STRATEGY=adaptive` to try a verified hybrid answer before
running the full agent loop. The API defaults to `agentic`.

Set `WORKER_EMBEDDING_BATCH_SIZE` to the number of PDF chunks embedded and
written at a time. It defaults to 8. Changing it preserves the stored
index key.

## Planned (from ROADMAP P3 to P7; goes into `.env.example`)
Never put real values in git. Use `.env` locally (gitignored) and secrets on the server.

| Variable | Secret? | Plain-English meaning | Example |
|---|---|---|---|
| `DATABASE_URL` | yes | Where Postgres is and how to log in | `postgresql+psycopg://rag:CHANGE_ME@postgres:5432/rag` |
| `REDIS_URL` | no | Where Redis is | `redis://redis:6379/0` |
| `SECRET_KEY` | yes | Signs login cookies. Long and random. Changing it logs everyone out. | `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `LLM_BASE_URL` | no | The OpenAI-compatible model server | `http://ollama:11434/v1` |
| `LLM_MODEL` | no | Which model name to ask for | `qwen3:4b-instruct` |
| `LLM_TIMEOUT_S` | no | Max seconds for one model call | `60` |
| `ASK_BUDGET_S` | no | Max seconds for a whole question | `120` |
| `MAX_UPLOAD_MB` | no | Biggest PDF allowed | `50` |
| `MAX_PAGES` | no | Most pages allowed per PDF | `500` |
| `CORS_ORIGINS` | no | Which websites may call the API | `https://rag.example.org` |
| `HF_HOME` | no | Model cache folder (a Docker volume) | `/data/hf` |
| `LOG_LEVEL` | no | How much to log | `INFO` |
