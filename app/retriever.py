"""
app/retriever.py — Vector Search Logic.

Responsibilities:
1. Load an existing ChromaDB vectorstore from disk
2. Accept a user question
3. Convert the question to an embedding and do similarity search
4. Return the top-k most relevant document chunks
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# HuggingFaceEmbeddings moved to langchain_huggingface in newer versions.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

from app.ingest import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_embedding_model,
)

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "3"))


def load_vectorstore(
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_model=None,
) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.

    Raises RuntimeError if the store doesn't exist (run ingest first).
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def retrieve(
    question: str,
    vectorstore: Optional[Chroma] = None,
    k: int = TOP_K,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for *question*.

    Args:
        question:     Natural language question from the user.
        vectorstore:  Pre-loaded Chroma instance (optional — loaded from
                      disk if not provided).
        k:            Number of chunks to return.
        persist_dir:  Path to the ChromaDB persist directory.

    Returns:
        List of LangChain Document objects ordered by relevance (most
        relevant first).
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir)

    docs = vectorstore.similarity_search(question, k=k)
    return docs


def retrieve_with_scores(
    question: str,
    vectorstore: Optional[Chroma] = None,
    k: int = TOP_K,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> List[tuple[Document, float]]:
    """
    Like retrieve() but also returns the similarity score for each chunk.

    Returns:
        List of (Document, score) tuples. Higher score = more similar.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir)

    return vectorstore.similarity_search_with_relevance_scores(question, k=k)
