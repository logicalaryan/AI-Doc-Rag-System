# RAG Document Q&A — Project Workflow

> A comprehensive map of every pipeline, data flow, and component in this project.

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG DOCUMENT Q&A SYSTEM                      │
│                                                                 │
│   ┌──────────┐     ┌──────────────┐     ┌───────────────────┐  │
│   │  DATA/   │────▶│  INGESTION   │────▶│   VECTORSTORE     │  │
│   │  DOCS    │     │  PIPELINE    │     │   (ChromaDB)      │  │
│   └──────────┘     └──────────────┘     └────────┬──────────┘  │
│                                                  │              │
│   ┌──────────┐     ┌──────────────┐     ┌────────▼──────────┐  │
│   │   USER   │────▶│  RETRIEVER   │────▶│   LLM CHAIN       │  │
│   │ QUESTION │     │  (top-k)     │     │  (Gemini Flash)   │  │
│   └──────────┘     └──────────────┘     └────────┬──────────┘  │
│                                                  │              │
│              ┌─────────────┬────────────────────┘              │
│              ▼             ▼                                     │
│       ┌──────────┐  ┌──────────┐                                │
│       │Streamlit │  │ FastAPI  │  ← Two frontend surfaces        │
│       │   UI     │  │   API    │                                │
│       └──────────┘  └──────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline A — Document Ingestion

**Entry point:** `scripts/ingest_docs.py` → calls `app/ingest.py`

```
data/ directory
│
│  Supported formats: .pdf  .txt  .md
│
▼
┌─────────────────────────────────────────────────────┐
│  STEP 1: LOAD  (app/ingest.py → load_documents())   │
│                                                     │
│  PDF  ──▶ PyPDFLoader                               │
│  TXT  ──▶ TextLoader (utf-8)                        │
│  MD   ──▶ TextLoader (utf-8)                        │
│                                                     │
│  Output: List[LangChain Document]                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: CHUNK  (chunk_documents())                  │
│                                                     │
│  Splitter: RecursiveCharacterTextSplitter           │
│  chunk_size    = 1000 chars  (env: CHUNK_SIZE)      │
│  chunk_overlap = 200  chars  (env: CHUNK_OVERLAP)   │
│                                                     │
│  Split order: \n\n → \n → ". " → " " → ""          │
│  (preserves semantic boundaries)                    │
│                                                     │
│  Output: List[Document chunks]                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: EMBED  (get_embedding_model())              │
│                                                     │
│  Model: all-MiniLM-L6-v2  (HuggingFace, local CPU) │
│  No API key required — runs entirely offline        │
│  normalize_embeddings = True                        │
│                                                     │
│  Output: Dense vector per chunk (384-dim)           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4: PERSIST  (build_vectorstore())              │
│                                                     │
│  Database: ChromaDB  (disk-based)                   │
│  Location: vectorstore/chroma_db  (env: CHROMA_…)  │
│  Collection: "rag_documents"                        │
│                                                     │
│  Behaviour: NEW chunks are ADDED to existing store  │
│  (delete vectorstore/ to start fresh)               │
└─────────────────────────────────────────────────────┘
```

---

## 3. Pipeline B — Query & Answer (RAG Chain)

**Entry point:** `app/chain.py → ask()` or `stream()`

```
User Question  (plain string)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1: LOAD VECTORSTORE                            │
│  app/retriever.py → load_vectorstore()               │
│  Reads ChromaDB from disk using same embedding model │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: RETRIEVE  (retrieve())                      │
│                                                     │
│  Question ──▶ embed ──▶ similarity_search(k=3)      │
│  Returns: top-3 Document chunks by cosine similarity │
│  Also available: retrieve_with_scores()              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: LCEL CHAIN  (build_rag_chain())             │
│                                                     │
│  RunnableParallel                                   │
│  ├── context  branch:                               │
│  │   question ──▶ retriever ──▶ _format_context()  │
│  │   (chunks labelled with [Excerpt N — source])   │
│  │                                                  │
│  └── question branch:                               │
│      RunnablePassthrough()  (identity — no change) │
│                                                     │
│  ──▶ QA_PROMPT  (app/prompts.py)                    │
│  ──▶ ChatGoogleGenerativeAI  (gemini-2.0-flash)     │
│  ──▶ StrOutputParser()                              │
│                                                     │
│  Config: temperature=0.0 (deterministic)            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT: AnswerResult dataclass                      │
│                                                     │
│  .answer   — generated answer string                │
│  .sources  — deduplicated [{source, page}, …]       │
│  .question — original question                      │
└─────────────────────────────────────────────────────┘
```

---

## 4. Frontend Surfaces

### 4A — Streamlit UI  (`ui/streamlit_app.py`)

```
Browser
  │
  ▼
streamlit run ui/streamlit_app.py
  │
  ├── Sidebar: document upload / settings
  │
  ├── Chat history (st.session_state)
  │
  └── User input ──▶ stream() ──▶ tokens streamed to screen
                          │
                          └── sources displayed below answer
```

### 4B — FastAPI REST API  (`api/main.py`)

```
HTTP Client
  │
  ▼
uvicorn api.main:app --reload
  │
  ├── GET  /health          — liveness check
  │
  ├── POST /ask             — calls ask()  → returns AnswerResult JSON
  │
  └── POST /ingest          — calls ingest() → ingests uploaded files
```

---

## 5. Evaluation Pipeline

**Entry point:** `eval/eval_report.py`

```
eval/test_cases.json  (5 ground-truth Q&A pairs)
         │
         ├─────────────────────────────────────────────┐
         ▼                                             ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│  eval_retrieval.py       │             │  eval_generation.py      │
│                         │             │                         │
│  For each question:     │             │  For each question:     │
│  retrieve(q, k=3)       │             │  ask(q) → LLM answer    │
│                         │             │                         │
│  Metrics:               │             │  Metrics:               │
│  ┌─ Hit Rate            │             │  ┌─ Faithfulness         │
│  │  correct chunk in    │             │  │  expected keyword     │
│  │  top-k?  (0–1)       │             │  │  in answer?  (0–1)   │
│  │                      │             │  │                      │
│  ├─ MRR                 │             │  └─ Answer Relevancy     │
│  │  avg 1/rank of first │             │     question words in   │
│  │  relevant chunk      │             │     answer?  (0–1)      │
│  │                      │             │                         │
│  └─ Precision@k         │             └────────────┬────────────┘
│     relevant/k per Q    │                          │
└─────────────┬───────────┘                          │
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
                  ┌─────────────────────────┐
                  │  eval_report.py          │
                  │                         │
                  │  Overall Score =         │
                  │  avg(HitRate + MRR +    │
                  │  Precision@k +          │
                  │  Faithfulness +         │
                  │  Relevancy) / 5         │
                  │                         │
                  │  Saves: last_report.json│
                  └─────────────────────────┘
```

---

## 6. Test Suite

**Entry point:** `pytest tests/`

```
tests/
├── test_ingest.py    — unit tests for load/chunk/embed/persist
├── test_retriever.py — unit tests for load_vectorstore / retrieve
└── test_chain.py     — unit tests for ask() / stream() / LCEL chain
         │
         │  All tests use: unittest.mock / MagicMock
         │  LLM and vectorstore are MOCKED — no API calls in tests
         │
         ▼
pytest --cov=app --cov-report=term-missing
```

---

## 7. Configuration Map

| Variable | Default | File | Purpose |
|---|---|---|---|
| `GOOGLE_API_KEY` | — | `.env` | Gemini API key |
| `MODEL_NAME` | `gemini-2.0-flash` | `.env` | LLM model |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | `.env` | ChromaDB location |
| `CHUNK_SIZE` | `1000` | `.env` | Chars per chunk |
| `CHUNK_OVERLAP` | `200` | `.env` | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `.env` | HuggingFace model |
| `TOP_K` | `3` | `.env` | Chunks retrieved per query |

---

## 8. Full Project Folder Map

```
rag-document-qa/
│
├── data/                    ← Your source documents (PDF, TXT, MD)
│
├── app/
│   ├── ingest.py            ← Pipeline A: load → chunk → embed → persist
│   ├── retriever.py         ← Pipeline B step 1: similarity search
│   ├── chain.py             ← Pipeline B step 2: LCEL RAG chain + ask()
│   └── prompts.py           ← QA prompt template
│
├── ui/
│   └── streamlit_app.py     ← Chat UI (streamlit run)
│
├── api/
│   └── main.py              ← REST API (uvicorn)
│
├── scripts/
│   └── ingest_docs.py       ← CLI entry point for ingestion
│
├── eval/
│   ├── test_cases.json      ← Ground truth (5 Q&A pairs)
│   ├── eval_retrieval.py    ← Hit Rate, MRR, Precision@k
│   ├── eval_generation.py   ← Faithfulness, Answer Relevancy
│   └── eval_report.py       ← Orchestrator → last_report.json
│
├── tests/
│   ├── test_ingest.py
│   ├── test_retriever.py
│   └── test_chain.py
│
├── vectorstore/             ← ChromaDB (auto-created on ingest)
├── .env                     ← API keys + config (not committed)
├── .env.example             ← Template for .env
├── requirements.txt
└── workflow/
    └── WorkFlow.md          ← This file
```

---

## 9. End-to-End Flow Summary

```
1. ADD DOCS    →  drop files into data/
2. INGEST      →  python scripts/ingest_docs.py --source data/
3. ASK         →  streamlit run ui/streamlit_app.py
                  OR  POST /ask  via FastAPI
4. EVALUATE    →  python eval/eval_report.py
5. TEST        →  pytest tests/ --cov=app
```
