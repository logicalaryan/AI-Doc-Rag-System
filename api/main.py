"""
api/main.py — FastAPI REST interface for the RAG system.

Endpoints:
  GET  /health       → Health check
  POST /ask          → Ask a question, get an answer + sources
  POST /ingest       → Trigger document ingestion

Run with:
    uvicorn api.main:app --reload
Interactive docs available at: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from app.chain import ask as rag_ask
from app.ingest import ingest
from app.retriever import load_vectorstore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Ask questions about your documents. "
        "Powered by Google Gemini + ChromaDB + HuggingFace embeddings."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared vectorstore (loaded once at startup, reused across requests)
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = load_vectorstore()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Vectorstore not ready. "
                    "Run ingestion first: POST /ingest or "
                    "python scripts/ingest_docs.py"
                ),
            )
    return _vectorstore


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, example="What is the annual revenue?")
    k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")


class SourceRef(BaseModel):
    source: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceRef]


class IngestRequest(BaseModel):
    data_dir: str = Field(default="./data", example="./data")
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)


class IngestResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check whether the API and vectorstore are ready."""
    try:
        vs = load_vectorstore()
        ready = vs is not None
    except Exception:
        ready = False
    return HealthResponse(status="ok", vectorstore_ready=ready)


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(request: AskRequest):
    """
    Ask a question about the ingested documents.

    Returns a natural language answer with source citations.
    """
    vs = get_vectorstore()

    try:
        result = rag_ask(
            question=request.question,
            vectorstore=vs,
            k=request.k,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    sources = [
        SourceRef(source=s["source"], page=s.get("page") or None)
        for s in result.sources
    ]
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Administration"])
def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger document ingestion.

    Loads all PDFs, TXTs, and MDs from *data_dir*, chunks them,
    embeds them, and persists to ChromaDB. Resets the shared vectorstore
    cache so the next /ask picks up new documents.
    """
    global _vectorstore

    def _run_ingest():
        global _vectorstore
        _vectorstore = None  # force reload after ingestion
        ingest(
            data_dir=request.data_dir,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

    background_tasks.add_task(_run_ingest)
    return IngestResponse(
        status="accepted",
        message=(
            f"Ingestion started from '{request.data_dir}'. "
            "The vectorstore will be ready in a few seconds."
        ),
    )
