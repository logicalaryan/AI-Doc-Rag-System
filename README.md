# RAG Document Q&A

> Ask questions about your own documents. Answers are grounded in your content — no hallucinations.

Built with **LangChain · Google Gemini · ChromaDB · HuggingFace Embeddings · FastAPI · Streamlit**.  
100% free to run.

---

## Architecture

```
Documents (data/)
      │
      ▼
 app/ingest.py  ──►  ChromaDB (vectorstore/)
                              │
User Question ──► app/retriever.py ──► top-k chunks
                              │
                    app/chain.py + app/prompts.py
                              │
                              ▼
                         Answer + Sources
                              │
               ┌─────────────┴──────────────┐
          api/main.py              ui/streamlit_app.py
          (FastAPI REST)           (Browser chat UI)
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Set your API key

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env` and paste your free [Google Gemini API key](https://aistudio.google.com/apikey).

### 3. Add documents & ingest

Drop PDFs, `.txt`, or `.md` files into `data/`, then:

```bash
python scripts/ingest_docs.py
```

### 4. Run the UI

```bash
streamlit run ui/streamlit_app.py
```

Or start the REST API:

```bash
uvicorn api.main:app --reload
# Interactive docs: http://localhost:8000/docs
```

---

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

```bash
# Edit eval/test_cases.json with your ground-truth Q&A pairs first
python eval/eval_report.py
```

---

## Project Structure

```text
rag-document-qa/
│
├── app/                        # ← Application core (all business logic)
│   ├── ingest.py               #    Document loading + chunking + embedding
│   ├── retriever.py            #    Vector search logic
│   ├── chain.py                #    LLM prompt + answer generation
│   └── prompts.py              #    All prompt templates
│
├── api/                        # ← REST API layer
│   └── main.py                 #    FastAPI app with /ask endpoint
│
├── ui/                         # ← Streamlit frontend
│   └── streamlit_app.py        #    Chat-style interface
│
├── data/                       # ← Raw documents go here (user-provided)
├── vectorstore/                # ← ChromaDB persistent storage
├── tests/                      # ← Unit + integration tests
├── eval/                       # ← RAG evaluation metrics
├── notebooks/                  # ← Jupyter experiments & debugging
├── scripts/                    # ← One-off CLI utilities
│
├── .env.example                # ← Template for environment variables
├── requirements.txt            # ← Pinned dependencies
├── README.md                   # ← Setup guide, architecture diagram, usage
├── PROJECT_STRUCTURE.md        # ← Detailed file responsibilities
└── DECISIONS.md                # ← Log of chunking/prompt design decisions
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a detailed breakdown of every file.

---

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for the log of chunking, prompt, and architecture choices.
