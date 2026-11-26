# Flow: how data moves today

Today there is no API server, no database and no login. Data moves through a
CLI or a Gradio UI, inside one Python process. The planned production flow
(FastAPI, Postgres, Ollama) is in `ROADMAP.md`. Update this file when it lands.

## 1. Entry points
```mermaid
flowchart LR
  U[User] -->|python -m src.cli ask| CLI[src/cli.py]
  U -->|python -m src.cli chat| UI[Gradio UI src/chat/ui.py]
  CLI --> BP[build_pipeline src/pipeline.py:1045]
  UI --> BP
  BP --> D[Dense system]
  BP --> H[Hybrid system]
  BP --> A[Agentic system]
```
The strategy is fixed by config (`retrieval.strategy`, default `dense`). There is no
routing at runtime.

## 2. Ingestion (once per PDF)
```mermaid
flowchart LR
  PDF[PDF file] --> P[Parser: pdfplumber]
  P -->|image-only page| OCR[Tesseract eng, 300 DPI]
  P --> N[Normalise, strip headers/footers]
  OCR --> N
  N --> C[Structure chunker: 1200 chars, 150 overlap]
  C --> E[bge-m3 embeddings]
  C --> B[BM25 index]
  E --> S[(In-memory store: FAISS or numpy)]
```
The index lives in memory only. Restarting the app or clicking Index again rebuilds it.

## 3. Answering a question (agentic strategy)
```mermaid
flowchart TD
  Q[Question] --> PL[Planner: split multi-hop/comparison into up to 3]
  PL --> R[Hybrid retrieve: dense + BM25, RRF k=60]
  R --> RR[Reranker bge-reranker-v2-m3]
  RR --> EV{Evidence coverage >= 0.5?}
  EV -->|no, rounds left| RF[Refine: missing-term queries] --> R
  EV -->|yes or out of rounds| G[Generate with C1..Cn labels]
  G --> V{Verifier}
  V -->|failed| RG[Regenerate once without blamed evidence] --> V
  V -->|ok| CF[Conflict check, chat only]
  CF --> CT[Resolve Cn to real pages in code]
  CT --> OUT[Answer + citations or refusal]
```

## 4. State
- Settings: one frozen config object (`src/config.py`). Nothing is read from env vars.
- Per question: `AgentState` counts rounds and model calls (`src/pipeline.py`).
- Per chat: `src/chat/session.py` keeps the history and rewrites follow-up questions.
- Models: loaded lazily per pipeline instance (`src/generation/backends.py:181`).
- Auth: none. `--share` makes a public Gradio link with no login.
