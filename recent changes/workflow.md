# RAG Document Q&A — Project Workflow

> Local reference doc. Not pushed to GitHub (folder is gitignored).

---

## What This Project Does

You upload documents (PDF, TXT, MD). The system reads them, breaks them into
chunks, converts chunks to vector embeddings, and stores them in ChromaDB.
When you ask a question, it finds the most relevant chunks and sends them to
Google Gemini (the LLM) along with your question. Gemini answers using ONLY
what's in those chunks — no hallucinations from outside knowledge.

---

## The Full Pipeline (Step by Step)

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                          │
│  (run once, or whenever you add new documents)                  │
│                                                                 │
│  data/ folder                                                   │
│    │  PDF / TXT / MD files                                      │
│    ▼                                                            │
│  app/ingest.py → load_documents()                               │
│    │  Reads all files via PyPDFLoader / TextLoader              │
│    ▼                                                            │
│  app/ingest.py → chunk_documents()                              │
│    │  RecursiveCharacterTextSplitter                            │
│    │  chunk_size=1000, chunk_overlap=200                        │
│    │  Splits on: paragraphs → sentences → words                 │
│    ▼                                                            │
│  app/ingest.py → build_vectorstore()                            │
│    │  HuggingFace: all-MiniLM-L6-v2 (local, no API key needed) │
│    │  Embeds each chunk → 384-dim vector                        │
│    ▼                                                            │
│  vectorstore/  (ChromaDB persisted to disk)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY / ANSWER PHASE                        │
│  (runs on every user question)                                  │
│                                                                 │
│  User Question (string)                                         │
│    │                                                            │
│    ▼                                                            │
│  app/retriever.py → retrieve()                                  │
│    │  Embeds question with same HuggingFace model               │
│    │  Runs similarity_search() against ChromaDB                 │
│    │  Returns top-k=3 most relevant chunks                      │
│    ▼                                                            │
│  app/chain.py → build_rag_chain() [LCEL pipe]                   │
│    │                                                            │
│    ├── context branch:                                          │
│    │     question → retriever_runnable → _format_context()      │
│    │     Prefixes each chunk: [Excerpt N — source, page]        │
│    │                                                            │
│    └── question branch:                                         │
│          RunnablePassthrough() — passes question through as-is  │
│                                                                 │
│    Both merge into → QA_PROMPT (from app/prompts.py)            │
│    {context} + {question} filled in                             │
│    │                                                            │
│    ▼                                                            │
│  Google Gemini (gemini-2.0-flash, temperature=0.0)              │
│    │  temperature=0 → deterministic, factual answers            │
│    ▼                                                            │
│  StrOutputParser() → plain text answer                          │
│    │                                                            │
│    ▼                                                            │
│  AnswerResult { answer, sources, question }                     │
│    Sources = deduplicated (source file, page number) list       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Configuration Values

| Setting            | Default            | Where Set         |
|--------------------|--------------------|-------------------|
| Chunk size         | 1000 characters    | `.env` / ingest.py |
| Chunk overlap      | 200 characters     | `.env` / ingest.py |
| Embedding model    | all-MiniLM-L6-v2   | `.env` / ingest.py |
| Vectorstore path   | `./vectorstore/`   | `.env` / ingest.py |
| LLM model          | gemini-2.0-flash   | `.env` / chain.py  |
| Top-k chunks       | 3                  | `.env` / retriever.py |
| LLM temperature    | 0.0 (fixed)        | chain.py           |

All values can be overridden in your `.env` file (copy `.env.example` → `.env`).

---

## Entry Points

### 1. Streamlit UI (recommended for demos)
```bash
streamlit run ui/streamlit_app.py
```
- Sidebar → drop files in `data/` → click **Ingest**
- Chat input → ask questions → see answers + source chips
- Supports streaming (tokens appear as they arrive)
- Adjustable `k` slider (1–8 chunks)

### 2. FastAPI (for programmatic / API access)
```bash
uvicorn api.main:app --reload
```
Interactive docs → http://localhost:8000/docs

| Endpoint       | Method | What it does                            |
|---------------|--------|-----------------------------------------|
| `/health`     | GET    | Check if API + vectorstore are ready    |
| `/ask`        | POST   | Send a question, get answer + sources   |
| `/ingest`     | POST   | Trigger ingestion (runs in background)  |

### 3. CLI script (for manual ingestion)
```bash
python scripts/ingest_docs.py
```

---

## Prompt Behaviour (app/prompts.py)

The LLM is explicitly instructed to:
- Answer **ONLY** from the provided context — no outside knowledge
- Say `"I don't have enough information..."` if the answer isn't in the docs
- Be concise and cite which part of the context it used

A second prompt (`CONDENSE_PROMPT`) exists for rephrasing follow-up questions
into standalone questions — used when conversation history is passed in.

---

## Evaluation Workflow

Run these after making changes to check retrieval/answer quality:

```bash
# Retrieval metrics: Hit Rate, MRR, Precision@k
python eval/eval_retrieval.py
python eval/eval_retrieval.py --k 5 --cases eval/test_cases.json

# Generation quality (answer faithfulness, relevance)
python eval/eval_generation.py

# Full report
python eval/eval_report.py
```

Test cases live in `eval/test_cases.json` — add new Q&A pairs there.

---

## Unit Tests

```bash
pytest tests/
```

| File                    | What it tests                       |
|------------------------|-------------------------------------|
| `tests/test_ingest.py`  | Load, chunk, embed pipeline         |
| `tests/test_retriever.py` | Vectorstore load, similarity search |
| `tests/test_chain.py`   | RAG chain, ask(), stream(), sources |

Tests use mock vectorstores and mock LLMs — no real API calls needed.

---

## Important Gotchas

- **Vectorstore is local only** — it's gitignored. After cloning on a new machine,
  you must run ingestion before querying.
- **`data/` files are gitignored** — only `.gitkeep` is tracked. Add your actual
  documents locally and ingest them.
- **HuggingFace model is downloaded on first run** — needs internet the first time,
  then cached locally.
- **Rate limits** — The free Gemini tier has limits. If you hit a 429 error, wait
  ~1 minute. The Streamlit UI shows a friendly message for this.
- **Fresh vectorstore** — If you want to clear all chunks and re-ingest, delete the
  `vectorstore/` directory manually, then re-run ingestion.

---

## File Map (Quick Reference)

```
rag-document-qa/
├── app/
│   ├── ingest.py       ← load → chunk → embed → persist
│   ├── retriever.py    ← load vectorstore → similarity search
│   ├── chain.py        ← LCEL pipe → Gemini → AnswerResult
│   └── prompts.py      ← all prompt templates (single source of truth)
├── api/
│   └── main.py         ← FastAPI: /health, /ask, /ingest
├── ui/
│   └── streamlit_app.py ← chat UI, sidebar ingest, source chips
├── eval/
│   ├── eval_retrieval.py  ← Hit Rate, MRR, Precision@k
│   ├── eval_generation.py ← answer quality metrics
│   ├── eval_report.py     ← combined report
│   └── test_cases.json    ← Q&A test data
├── tests/
│   ├── test_chain.py
│   ├── test_ingest.py
│   └── test_retriever.py
├── scripts/
│   └── ingest_docs.py  ← CLI ingestion trigger
├── data/               ← put your documents here (gitignored)
├── vectorstore/        ← ChromaDB store (gitignored, auto-created)
├── .env.example        ← copy to .env, fill in your API key
└── requirements.txt
```
