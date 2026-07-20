# RAG Document Q&A — Project Structure

## Tech Stack (100% Free)

| Layer | Tool | Cost | Why |
|---|---|---|---|
| Language | Python 3.11+ | Free | Industry standard for AI/ML |
| Framework | LangChain | Free | Abstracts RAG plumbing (loaders, splitters, chains) |
| LLM | Google Gemini (`gemini-2.0-flash`) | Free tier — 15 RPM, 1M TPM | Fast, high quality, generous free quota |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free — runs locally | No API key needed, runs on CPU, 384-dim vectors |
| Vector Store | ChromaDB | Free | Zero-config, file-based — no server to manage |
| API | FastAPI | Free | Async, auto-docs at `/docs`, beginner-friendly |
| Frontend | Streamlit | Free | One-file UI, zero JS required |

> **Free tier limits for Gemini:** 15 requests/minute, 1 million tokens/minute, 1,500 requests/day.
> This is more than enough for development and portfolio demos.

### 🧠 Embedding Model Details (`all-MiniLM-L6-v2`)

*   **Why we use it:**
    *   **100% Free & Local:** Runs on CPU without needing API keys or incurring token costs.
    *   **Fast & Efficient:** Produces 384-dimensional vectors, which are quick to generate and search.
*   **Language Support (English Only):**
    *   It is trained exclusively on English data.
    *   The tokenizer is optimized for English, meaning it will perform poorly on other languages.
*   **Multilingual Alternatives (if needed later):**
    *   **Lightweight (CPU-friendly):** Swap to `paraphrase-multilingual-MiniLM-L12-v2` or `intfloat/multilingual-e5-small`. These are small enough to run fast on a standard CPU.
    *   **State-of-the-Art (Heavier):** Use `BAAI/bge-m3` (supports 100+ languages) or `intfloat/multilingual-e5-large`. *Note: These are much larger models (1GB+) and will be noticeably slower on a CPU without a dedicated GPU.*

---

## Folder Layout

```
rag-document-qa/
│
├── app/                        # ← Application core (all business logic)
│   ├── __init__.py
│   ├── ingest.py               #    Document loading + chunking + embedding
│   ├── retriever.py            #    Vector search logic
│   ├── chain.py                #    LLM prompt + answer generation
│   └── prompts.py              #    All prompt templates (single source of truth)
│
├── api/                        # ← REST API layer
│   ├── __init__.py
│   └── main.py                 #    FastAPI app with /ask endpoint
│
├── ui/                         # ← Streamlit frontend
│   └── streamlit_app.py        #    Chat-style interface
│
├── data/                       # ← Raw documents go here (user-provided)
│   └── .gitkeep                #    Placeholder so Git tracks the empty folder
│
├── vectorstore/                # ← ChromaDB persistent storage (auto-generated)
│   └── .gitkeep
│
├── tests/                      # ← Unit + integration tests
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_retriever.py
│   └── test_chain.py
│
├── eval/                       # ← RAG evaluation metrics
│   ├── __init__.py
│   ├── test_cases.json         #    Known question → expected chunk pairs
│   ├── eval_retrieval.py       #    Hit Rate, MRR, Precision@k
│   ├── eval_generation.py      #    Faithfulness, Answer Relevancy
│   └── eval_report.py          #    Runs all evals, generates a scored report
│
├── notebooks/                  # ← Jupyter experiments & debugging
│   └── exploration.ipynb       #    Prototype chunking strategies, prompt tuning
│
├── scripts/                    # ← One-off CLI utilities
│   └── ingest_docs.py          #    CLI to ingest documents: python scripts/ingest_docs.py
│
├── .env.example                # ← Template for environment variables
├── .gitignore                  # ← Ignore .env, vectorstore/, __pycache__, etc.
├── requirements.txt            # ← Pinned dependencies
├── pyproject.toml              # ← Project metadata (optional, modern Python)
├── README.md                   # ← Setup guide, architecture diagram, usage
└── DECISIONS.md                # ← Log of chunking/prompt design decisions
```

---

## File-by-File Responsibilities

### `app/` — The Brain

> This folder contains **all RAG logic**. Everything else (API, UI, scripts) is just a thin wrapper that calls into `app/`.

#### `app/ingest.py` — Document Ingestion Pipeline

**What it does:**
1. Loads documents from `data/` (PDFs, text files, markdown)
2. Splits them into chunks using a text splitter
3. Generates embeddings for each chunk
4. Stores embeddings in ChromaDB

**Why it exists:**
Ingestion is a distinct pipeline that runs *before* any user asks a question. Separating it means you can re-ingest documents without touching query logic. This is the "write" side of your system.

**Key concepts inside:**
- `DocumentLoader` — reads raw files
- `RecursiveCharacterTextSplitter` — breaks documents into overlapping chunks
- `Chroma.from_documents()` — persists embeddings to disk

---

#### `app/retriever.py` — Vector Search

**What it does:**
1. Accepts a user question as input
2. Converts the question into an embedding
3. Performs similarity search against ChromaDB
4. Returns the top-k most relevant document chunks

**Why it exists:**
Retrieval is the **R** in RAG. Isolating it lets you:
- Swap ChromaDB for Pinecone/Weaviate later without touching anything else
- Tune `k` (number of results), similarity thresholds, and filtering independently
- Test retrieval quality in isolation (does the right chunk come back?)

---

#### `app/chain.py` — LLM Answer Generation

**What it does:**
1. Takes the user question + retrieved chunks
2. Constructs a prompt using templates from `prompts.py`
3. Sends the prompt to OpenAI
4. Returns the generated answer (with source references)

**Why it exists:**
This is the **G** in RAG — the generation step. Keeping it separate from retrieval means you can:
- Change the LLM model without affecting search
- Experiment with chain types (stuff, map-reduce, refine)
- Add answer validation or post-processing here

---

#### `app/prompts.py` — Prompt Templates

**What it does:**
Stores all prompt templates as named constants or functions.

**Why it exists:**
Prompts are the #1 thing you'll iterate on. Having them scattered across files makes debugging hallucinations painful. A single file means:
- One place to review all instructions sent to the LLM
- Easy A/B testing of different prompt strategies
- Clear documentation of what the LLM is being asked to do

**Example prompts to define here:**
- `QA_PROMPT` — main question-answering prompt with context injection
- `CONDENSE_PROMPT` — (optional) rephrases follow-up questions into standalone ones

---

### `api/` — REST Interface

#### `api/main.py` — FastAPI Application

**What it does:**
Exposes a `POST /ask` endpoint that accepts a question and returns an answer.

**Why it exists:**
- **Keeps the brain separate from the face** — Your RAG logic (the "brain") doesn't live inside your UI (the "face"). This means you can swap out Streamlit for React or a mobile app later without rewriting any of the core logic.
- **Anyone can talk to it** — Since it's a standard web API, any app (React, mobile, even a simple `curl` command in the terminal) can send it a question and get an answer back. One backend serves all.
- **Free interactive docs** — FastAPI automatically creates a visual, clickable documentation page at `/docs` where you can test your API live in the browser. No extra work needed — perfect for portfolio demos and screenshots.

**Endpoints to define:**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Send a question, get an answer + sources |
| `POST` | `/ingest` | Trigger document ingestion |
| `GET` | `/health` | Health check (useful for deployment) |

---

### `ui/` — Chat Frontend

#### `ui/streamlit_app.py` — Streamlit Chat Interface

**What it does:**
Provides a browser-based chat UI where users type questions and see answers with source citations.

**Why it exists:**
- Streamlit requires zero frontend knowledge — pure Python
- Gives you a polished demo for your portfolio with minimal effort
- Can call the API or import `app/` directly

---

### `data/` — Document Storage

**What it does:**
Holds the raw documents (PDFs, `.txt`, `.md`) that users want to query.

**Why it exists:**
- Clean separation of code from data
- Makes it obvious where to drop new documents
- `.gitkeep` ensures the folder is tracked even when empty
- Listed in `.gitignore` so actual documents (potentially sensitive) aren't committed

---

### `vectorstore/` — Persisted Embeddings

**What it does:**
ChromaDB writes its index files here after ingestion.

**Why it exists:**
- Persists embeddings across restarts (no re-embedding on every launch)
- Gitignored — embeddings are derived data, regenerated from `data/`
- Keeps the project root clean

---

### `tests/` — Automated Tests

#### `tests/test_ingest.py`
Tests that documents are loaded, chunked correctly, and chunk sizes stay within limits.

#### `tests/test_retriever.py`
Tests that a known question retrieves the expected chunk from a small test corpus.

#### `tests/test_chain.py`
Tests that the chain produces an answer containing expected keywords (integration test with mocked LLM) . 

**Why tests exist:** 
They show structured engineering practice, not just scripting. Testing across varied query types and documenting failure patterns demonstrates real engineering maturity.

---

### `eval/` — RAG Evaluation Metrics

> Most beginners build RAG and stop. Adding evaluation is what makes this portfolio-grade.

#### `eval/test_cases.json` — Ground Truth Dataset

**What it does:**
Stores known question → expected answer/chunk pairs that you write manually.

```json
[
  {
    "question": "When was the company founded?",
    "expected_chunk_contains": "founded in 2015",
    "expected_answer_contains": "2015"
  },
  {
    "question": "What is the revenue?",
    "expected_chunk_contains": "500 crore",
    "expected_answer_contains": "500 crore"
  }
]
```

**Why it exists:**
You can't measure quality without ground truth. These are the "correct answers" you compare against. Start with 10–15 test cases — that's enough for a portfolio demo.

---

#### `eval/eval_retrieval.py` — Retrieval Quality Metrics

**What it measures:**

| Metric | Question it answers | Formula |
|---|---|---|
| **Hit Rate** | Is the correct chunk in the top-k? | (questions where correct chunk is in top-k) / total questions |
| **MRR** | Where in the results is the correct chunk? | Average of 1/rank across all questions |
| **Precision@k** | Of k chunks returned, how many are relevant? | relevant chunks in top-k / k |

**Example output:**
```
Retrieval Evaluation (k=3):
  Hit Rate:     0.85  (85% of questions found the right chunk)
  MRR:          0.72  (correct chunk is usually at position 1–2)
  Precision@3:  0.47  (about half the returned chunks are relevant)
```

**Why it exists:**
If retrieval is bad, the LLM gets wrong context → wrong answers. This catches the problem at the source. It also helps you tune `chunk_size`, `chunk_overlap`, and `k`.

---

#### `eval/eval_generation.py` — Answer Quality Metrics

**What it measures:**

| Metric | Question it answers | How |
|---|---|---|
| **Faithfulness** | Did the LLM hallucinate? | Check if every claim in the answer exists in the provided chunks |
| **Answer Relevancy** | Does the answer address the question? | Check if the answer actually relates to what was asked |

**How Faithfulness works (simplified):**
```
Context:  "Revenue was ₹500 crore in Q3 2025."
Answer:   "The revenue was ₹500 crore in Q3 2025."
→ Faithful ✅ (answer matches context)

Context:  "Revenue was ₹500 crore in Q3 2025."
Answer:   "The revenue was ₹800 crore and growing rapidly."
→ NOT faithful ❌ (₹800 crore is made up, "growing rapidly" not in context)
```

**Implementation approach:**
For a free project, use a simple string-matching check (does the answer contain keywords from the expected answer?) rather than paid tools like RAGAS.

**Why it exists:**
Hallucination is the #1 risk in RAG. Measuring it shows you take quality seriously.

---

#### `eval/eval_report.py` — Full Evaluation Runner

**What it does:**
1. Loads test cases from `test_cases.json`
2. Runs retrieval evaluation → scores
3. Runs generation evaluation → scores
4. Prints a combined report

**Example output:**
```
========== RAG EVALUATION REPORT ==========
Test cases: 15
Date: 2026-07-12

RETRIEVAL (k=3):
  Hit Rate:      0.85
  MRR:           0.72
  Precision@3:   0.47

GENERATION:
  Faithfulness:     0.90
  Answer Relevancy: 0.88

OVERALL SCORE:  0.76 / 1.00
============================================
```

**Why it exists:**
One command gives you a full quality scorecard. Perfect for `DECISIONS.md` entries like: *"After changing chunk_size from 500 to 1000, Hit Rate improved from 0.70 to 0.85."*

---

### `notebooks/exploration.ipynb` — Experimentation Space

**What it does:**
A Jupyter notebook for prototyping and debugging:
- Testing different chunk sizes and overlap values
- Visualizing retrieval results
- Comparing prompt variations
- Documenting failure patterns

**Why it exists:**
Iterating on chunking strategy and prompt design happens here. Keep it messy — it's a lab notebook, not production code.

---

### `scripts/ingest_docs.py` — CLI Ingestion Script

**What it does:**
A simple CLI script that calls `app/ingest.py` to process all documents in `data/`.

```
python scripts/ingest_docs.py --data-dir ./data --chunk-size 1000
```

**Why it exists:**
- Gives a clean entry point for ingestion without starting the API
- Lets you parameterize chunk size and overlap from the command line
- Useful for batch processing and CI/CD pipelines

---

### Root Config Files

#### `.env.example`
```
GOOGLE_API_KEY=your-gemini-api-key-here
CHROMA_PERSIST_DIR=./vectorstore
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```
**Why:** Shows collaborators (and portfolio reviewers) exactly what config is needed without exposing secrets.

> **Getting a free Gemini API key:** Go to [Google AI Studio](https://aistudio.google.com/apikey) → Create API key → Done. No billing setup required.

#### `.gitignore`
Ignores: `.env`, `vectorstore/`, `__pycache__/`, `data/*.pdf`, `.ipynb_checkpoints/`

**Why:** Keeps the repo clean. Secrets, derived data, and large binaries stay local.

#### `requirements.txt`
Pinned versions of: `langchain`, `langchain-google-genai`, `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `streamlit`, `python-dotenv`, `pypdf`

**Why:** Reproducible environments. Anyone cloning your repo gets the exact same versions.

#### `README.md`
**Why:** First thing portfolio reviewers see. Should contain:
- One-line project description
- Architecture diagram (text-based with Mermaid)
- Setup instructions (3 steps max)
- Example usage with screenshot
- Design decisions summary

#### `DECISIONS.md`
**Why:** Documents *why* you made specific choices:
- Why chunk size = 1000? What did you try first?
- Why ChromaDB over FAISS?
- What prompt changes reduced hallucinations?

This supports practising structured problem diagnosis and evidence-based write-up of system behaviour.

---

## Data Flow

Two distinct pipelines:

### 1. Ingestion (offline)
```
Documents (data/) → Chunking (ingest.py) → Embedding (ingest.py) → Vector Store (vectorstore/)
```

### 2. Query (real-time)
```
User Question → Retriever (retriever.py) → Relevant Chunks → LLM Chain (chain.py) → Answer + Sources
                       ↑
               Vector Store (vectorstore/)
```

---

## Why This Structure Works for Beginners

| Principle | How It's Applied |
|---|---|
| **Single Responsibility** | Each file does one thing. `ingest.py` doesn't answer questions. `chain.py` doesn't load documents. |
| **Separation of Concerns** | Core logic (`app/`) is independent of how it's accessed (`api/`, `ui/`, `scripts/`). |
| **Swappability** | Want to replace ChromaDB with Pinecone? Change only `retriever.py` and `ingest.py`. |
| **Testability** | Each module can be tested independently with clear inputs and outputs. |
| **Portfolio-Ready** | `DECISIONS.md` + `notebooks/` + `tests/` demonstrate engineering maturity, not just "it works." |

---

## Suggested Build Order

Build in this order to always have something working:

1. **`app/ingest.py`** — Get documents into the vector store
2. **`app/retriever.py`** — Verify the right chunks come back
3. **`app/prompts.py`** → **`app/chain.py`** — Wire up the LLM
4. **`eval/test_cases.json`** → **`eval/eval_retrieval.py`** — Measure retrieval quality early
5. **`eval/eval_generation.py`** → **`eval/eval_report.py`** — Full evaluation pipeline
6. **`scripts/ingest_docs.py`** — CLI to ingest easily
7. **`ui/streamlit_app.py`** — Visual demo
8. **`api/main.py`** — REST interface
9. **`tests/`** — Lock down behavior
10. **`notebooks/exploration.ipynb`** — Document your experiments
11. **`README.md`** + **`DECISIONS.md`** — Polish for portfolio
