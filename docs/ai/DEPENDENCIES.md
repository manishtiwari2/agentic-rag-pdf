# Approved dependencies

Only these packages may be installed or imported. Anything else needs Manish's
OK and a new row here, with its licence. No AGPL code at runtime.

## In use today
| Package | File | Licence | Why |
|---|---|---|---|
| pdfplumber | requirements.txt | MIT | Default PDF parser |
| numpy | requirements.txt | BSD-3 | Vectors, exact index fallback |
| faiss-cpu | requirements.txt | MIT | Optional fast vector index |
| torch | requirements-models.txt | BSD-3 | Runs the models |
| transformers | requirements-models.txt | Apache-2.0 | Loads bge-m3, the reranker and Qwen3 |
| accelerate | requirements-models.txt | Apache-2.0 | `device_map="auto"` |
| bitsandbytes | requirements-models.txt | MIT | 4-bit NF4 on a T4 |
| sentencepiece | requirements-models.txt | Apache-2.0 | BGE tokenizer |
| pytesseract | requirements-colab.txt | Apache-2.0 | OCR for scanned pages |
| pytest | requirements-dev.txt | MIT | Tests |
| nbformat, nbclient, ipykernel | requirements-dev.txt | BSD-3 | Run the notebooks in tests |
| pymupdf (<2) | requirements-dev.txt | AGPL-3.0 | **Test fixtures only.** Never import it in runtime code. |

## Approved for the roadmap (add when the phase starts)
| Package | Phase | Licence | Why |
|---|---|---|---|
| fastapi, uvicorn | P3 | MIT, BSD-3 | API server |
| sqlalchemy, alembic | P2 | MIT | Database access and migrations |
| psycopg[binary] | P2 | LGPL-3.0 | Postgres driver (dynamic linking is fine) |
| pgvector | P2 | MIT | Vector type for SQLAlchemy |
| redis, arq | P3 | MIT | Job queue, rate limits, cache |
| argon2-cffi | P3 | MIT | Password hashing |
| jinja2 | P4 | BSD-3 | HTML templates |
| htmx (vendored JS file) | P4 | BSD-2 | UI interactions without a Node build |
| openai | P1/P3 | Apache-2.0 | One client for Ollama and vLLM (OpenAI-compatible) |
| prometheus-client | P3 | Apache-2.0 | /metrics |
| httpx | P3/P6 | BSD-3 | API tests and model stream transport errors |
| testcontainers | P8 | Apache-2.0 | Postgres and Redis in tests |
| ruff, mypy, pip-audit, bandit, pytest-cov | P8 | MIT / Apache-2.0 | CI quality gates |
| locust | P8 | MIT | Small load test |
| uv | P7 | MIT / Apache-2.0 | Lockfile and installs |

## Services (Docker images, pin by tag)
postgres 16 + pgvector, redis 7, ollama, vllm (optional profile).

## Never add
- LangChain or LlamaIndex. The pipeline is our own and is tested.
- Any package with `trust_remote_code` models.
- Packages an AI suggests that you cannot find on PyPI. They may be hallucinated.
