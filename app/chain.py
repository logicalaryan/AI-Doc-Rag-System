"""
app/chain.py — LLM Answer Generation.

Responsibilities:
1. Accept a user question + retrieved context chunks
2. Construct a prompt using templates from prompts.py
3. Call Google Gemini via LangChain
4. Return the generated answer with source metadata

LCEL Chain Design
-----------------
This module uses the full LCEL (LangChain Expression Language) pattern:

    RunnableParallel(
        context = retriever | RunnableLambda(format_docs),  ← calls vectorstore
        question = RunnablePassthrough(),                    ← passes question unchanged
    )
    | QA_PROMPT
    | llm
    | StrOutputParser()

RunnablePassthrough is the key piece that lets the pipeline accept a plain
question string and route it to both the retriever AND the prompt simultaneously.
"""

import os
from dataclasses import dataclass, field
from typing import Generator, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,  # ← passes the input question through unchanged
)
from langchain_google_genai import ChatGoogleGenerativeAI

from app.prompts import QA_PROMPT
from app.retriever import load_vectorstore, retrieve

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """Structured output from the RAG chain."""
    answer: str
    sources: List[dict] = field(default_factory=list)
    question: str = ""

    def __str__(self) -> str:
        source_lines = "\n".join(
            f"  - {s.get('source', 'unknown')} (page {s.get('page', '?')})"
            for s in self.sources
        )
        return (
            f"Q: {self.question}\n\n"
            f"A: {self.answer}\n\n"
            f"Sources:\n{source_lines if source_lines else '  (none)'}"
        )


# ---------------------------------------------------------------------------
# Context helpers (also used directly in tests)
# ---------------------------------------------------------------------------

def _format_context(docs: List[Document]) -> str:
    """
    Concatenate document chunks into a single labelled context string.

    Each chunk is prefixed with its source and page so the LLM (and
    faithfulness eval) can trace which excerpt each claim came from.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = f"[Excerpt {i} — {source}" + (f", page {page}" if page else "") + "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[dict]:
    """Pull deduplicated source metadata from retrieved documents."""
    seen: set = set()
    sources = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", ""),
            })
    return sources


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(model_name: str = MODEL_NAME) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini LLM."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )
    return ChatGoogleGenerativeAI(  # type: ignore[abstract]
        model=model_name,
        api_key=GOOGLE_API_KEY,
        temperature=0.0,   # deterministic — better for Q&A
    )


# ---------------------------------------------------------------------------
# LCEL chain builder
# ---------------------------------------------------------------------------

def build_rag_chain(vectorstore, llm=None, k: int = 3):
    """
    Build a full LCEL RAG chain that accepts a plain question string.

    Chain structure:
        question (str)
            │
            ├── retriever ──► [doc1, doc2, …] ──► _format_context ──► context str
            │                                                                │
            └── RunnablePassthrough ──────────────────────────────────► question str
                                                                             │
                                              {"context": "…", "question": "…"}
                                                                             │
                                                                       QA_PROMPT
                                                                             │
                                                                           llm
                                                                             │
                                                                   StrOutputParser
                                                                             │
                                                                       answer str

    RunnablePassthrough is what lets us send a bare question string into the
    pipe while simultaneously routing it to the retriever (for context) and
    keeping it available as the "question" key for the prompt template.

    Args:
        vectorstore: A loaded Chroma instance.
        llm:         Optional pre-built LLM (pass a mock in tests).
        k:           Number of chunks to retrieve.

    Returns:
        A LangChain Runnable that accepts a question string.
    """
    if llm is None:
        llm = build_llm()

    # Wrap similarity_search as a Runnable so it fits in the LCEL pipe
    retriever_runnable = RunnableLambda(
        lambda question: vectorstore.similarity_search(question, k=k)
    )

    chain = (
        RunnableParallel(
            # context branch: question → retriever → format docs into a string
            context=retriever_runnable | RunnableLambda(_format_context),
            # question branch: question passes through unchanged
            # RunnablePassthrough() is the identity function for Runnables —
            # it receives the question string and returns it as-is so the
            # prompt template can reference {question}.
            question=RunnablePassthrough(),
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    question: str,
    vectorstore=None,
    llm=None,
    k: Optional[int] = None,
    persist_dir: Optional[str] = None,
) -> AnswerResult:
    """
    Full RAG pipeline: retrieve → prompt → LLM → structured result.

    Uses build_rag_chain() internally so retrieval and generation run
    through the proper LCEL pipe with RunnablePassthrough.

    Args:
        question:     The user's natural language question.
        vectorstore:  Pre-loaded Chroma instance (optional).
        llm:          Pre-built LLM (optional — inject a mock in tests).
        k:            Number of chunks to retrieve.
        persist_dir:  Path to ChromaDB store.

    Returns:
        AnswerResult with .answer (str), .sources (list), .question (str).
    """
    from app.retriever import CHROMA_PERSIST_DIR, TOP_K

    k = k or TOP_K
    persist_dir = persist_dir or CHROMA_PERSIST_DIR

    # 1. Load vectorstore if not provided
    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir)

    # 2. Retrieve docs separately so we can extract source metadata
    #    (the LCEL chain handles retrieval internally too, but we need
    #    the raw docs for _extract_sources — so we do one explicit call)
    docs = retrieve(question, vectorstore=vectorstore, k=k)

    # 3. Build and invoke the LCEL chain
    #    RunnablePassthrough inside the chain passes the question through
    #    while the retriever_runnable populates context in parallel.
    chain = build_rag_chain(vectorstore=vectorstore, llm=llm, k=k)
    answer_text = chain.invoke(question)   # ← just a plain string, not a dict

    return AnswerResult(
        answer=answer_text.strip(),
        sources=_extract_sources(docs),
        question=question,
    )


def stream(
    question: str,
    vectorstore=None,
    llm=None,
    k: Optional[int] = None,
    persist_dir: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming variant of ask() — yields answer tokens as they arrive.

    Because the full LCEL chain is used, .stream() works out of the box.
    The Streamlit UI can call this to show the answer token-by-token.

    Usage:
        for token in stream("What is the revenue?", vectorstore=vs):
            print(token, end="", flush=True)
    """
    from app.retriever import CHROMA_PERSIST_DIR, TOP_K

    k = k or TOP_K
    persist_dir = persist_dir or CHROMA_PERSIST_DIR

    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir)

    chain = build_rag_chain(vectorstore=vectorstore, llm=llm, k=k)
    yield from chain.stream(question)
