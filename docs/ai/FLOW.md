# Flow: how data moves today

The authenticated web workspace uses FastAPI, Postgres, Redis, and an
OpenAI-compatible model server. The CLI and notebook can also run a local
pipeline for experiments.

## 1. Entry points
```mermaid
flowchart LR
  U[User] -->|python -m src.cli ask| CLI[src/cli.py]
  U -->|Sign in| UI[FastAPI workspace]
  UI --> LIB[Owner-scoped document library]
  LIB --> BP[build_pipeline]
  CLI --> BP
  BP --> D[Dense system]
  BP --> H[Hybrid system]
  BP --> A[Agentic system]
```
The web workspace uses the agentic strategy. The CLI selects a strategy
through its existing configuration.

## 2. Ingestion (once per PDF)
```mermaid
flowchart LR
  PDF[PDF upload] --> J[Postgres upload job]
  J --> A[Redis and arq worker]
  A --> P[Parser: pdfplumber]
  P -->|image-only page| OCR[Tesseract eng, 300 DPI]
  P --> N[Normalise, strip headers/footers]
  OCR --> N
  N --> C[Structure chunker: 1200 chars, 150 overlap]
  C --> E[bge-m3 embeddings]
  C --> S[(Postgres chunks and pgvector)]
  E --> S
```
The web library reuses saved vectors by owner, file hash, config fingerprint,
and code version. It builds BM25 from the versions selected for a question.
The local CLI and notebook keep their indexes in memory.

## 3. Answering a question (agentic strategy)
```mermaid
flowchart TD
  Q[Question] --> PL[Planner: split multi-hop/comparison into up to 3]
  PL --> R[Hybrid retrieve: dense + BM25, RRF k=60]
  R --> RR[Reranker bge-reranker-v2-m3]
  RR --> EV{Evidence coverage >= 0.5?}
  EV -->|no, rounds left| RF[Refine: missing-term queries] --> R
  EV -->|sufficient| CF[Check conflicting sources]
  EV -->|insufficient, no rounds left| OUT[Answer or refusal]
  CF --> G[Generate with C1..Cn labels]
  G --> V{Verifier}
  V -->|failed| RG[Regenerate once without blamed evidence] --> V
  V -->|passed| CT[Attach resolved pages to the answer]
  V -->|unavailable by default| OUT
  CT --> SSE[Save the exchange and stream the answer]
  SSE --> VIEW[Open a cited page and highlight the stored passage]
```

## 4. State
- The API and worker read connection and deployment settings from the environment.
- Postgres stores upload state, sessions, document versions, and chat history under their owner.
- Redis holds the ingestion queue and request counters.
- Each API process shares its embedder, reranker, and model client.
- The pipeline records its per-question decisions in `AgentState`.
