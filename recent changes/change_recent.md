# Recent Changes — RAG Document Q&A Project

> Local notes only. Not pushed to GitHub (folder is gitignored).

---

## Commit: `dc52fe4` — "added Rag" (main bulk of work)

### Core App (`app/`)
- `app/chain.py` — Built the full RAG chain: prompt assembly, LLM call, response generation
- `app/ingest.py` — Document loader: reads PDFs/TXTs, chunks them, stores in vectorstore
- `app/retriever.py` — Loads ChromaDB vectorstore, runs similarity search
- `app/prompts.py` — Prompt templates for QA chain

### API (`api/`)
- `api/main.py` — FastAPI endpoints to expose the RAG pipeline (ingest + query)

### Evaluation (`eval/`)
- `eval/eval_retrieval.py` — Computes Hit Rate, MRR, Precision@k metrics
- `eval/eval_generation.py` — Evaluates LLM answer quality
- `eval/eval_report.py` — Aggregates and prints evaluation summary
- `eval/test_cases.json` — Sample test questions + expected content for evaluation

### Tests (`tests/`)
- `tests/test_chain.py` — Unit tests for the RAG chain logic
- `tests/test_ingest.py` — Unit tests for document ingestion pipeline
- `tests/test_retriever.py` — Unit tests for retrieval and vectorstore loading

### UI
- `ui/streamlit_app.py` — Streamlit frontend: upload docs, ask questions, see answers

### Scripts
- `scripts/ingest_docs.py` — CLI script to manually trigger document ingestion

### Docs & Config
- `README.md` — Full project overview, setup, and usage guide
- `DECISIONS.md` — Architecture decisions and rationale
- `Specdrive.md` — Detailed spec/design document (453 lines)
- `error_logs/error_logs.md` — Running log of bugs encountered and fixes applied
- `edges cases/edges.md` — Documented edge cases: bad retrieval, hallucination, empty docs, etc.
- `.env.example` — Template for environment variables (API keys, model config)
- `pyrightconfig.json` — Pyright type-checker config (fixed type-hint issues)
- `requirements.txt` — All Python dependencies pinned

### Data
- `data/.gitkeep` — Placeholder so the `data/` folder is tracked without actual docs
- `notebooks/exploration.ipynb` — Jupyter notebook for exploratory testing

---

## Commit: `17ebd63` — "Initial commit"
- Added `LICENSE` file only (MIT)

---

## Key Things to Remember
- Vectorstore lives in `vectorstore/` — gitignored, must be rebuilt locally after cloning
- Real docs go in `data/` — also gitignored (only `.gitkeep` tracked)
- Run app via: `streamlit run ui/streamlit_app.py`
- API via: `uvicorn api.main:app --reload`
- Eval via: `python eval/eval_retrieval.py`
