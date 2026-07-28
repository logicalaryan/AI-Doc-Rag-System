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
from langchain_qdrant import QdrantVectorStore
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
    VECTOR_DB_TYPE,
    QDRANT_URL,
    QDRANT_API_KEY,
    get_embedding_model,
    get_qdrant_client,
)

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "3"))


def load_qdrant_vectorstore(
    url: str = QDRANT_URL,
    api_key: str = QDRANT_API_KEY,
    embedding_model=None,
) -> QdrantVectorStore:
    """Load an existing Qdrant vectorstore (Cloud URL or local disk path)."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    client = get_qdrant_client(url, api_key)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )


def load_vectorstore(
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_model=None,
    db_type: str = VECTOR_DB_TYPE,
):
    """
    Load an existing ChromaDB or Qdrant vectorstore from disk/cloud.

    Raises RuntimeError if the store doesn't exist (run ingest first).
    """
    if db_type == "qdrant":
        return load_qdrant_vectorstore(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            embedding_model=embedding_model,
        )
    else:
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
    vectorstore = None,
    k: int = TOP_K,
    persist_dir: str = CHROMA_PERSIST_DIR,
    db_type: str = VECTOR_DB_TYPE,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for *question*.

    Args:
        question:     Natural language question from the user.
        vectorstore:  Pre-loaded Chroma or Qdrant instance (optional).
        k:            Number of chunks to return.
        persist_dir:  Path to the ChromaDB persist directory.
        db_type:      Vector DB engine ('chroma' or 'qdrant').

    Returns:
        List of LangChain Document objects ordered by relevance.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir, db_type=db_type)

    docs = vectorstore.similarity_search(question, k=k)
    return docs


def retrieve_with_scores(
    question: str,
    vectorstore = None,
    k: int = TOP_K,
    persist_dir: str = CHROMA_PERSIST_DIR,
    db_type: str = VECTOR_DB_TYPE,
) -> List[tuple[Document, float]]:
    """
    Like retrieve() but also returns the similarity score for each chunk.

    Returns:
        List of (Document, score) tuples. Higher score = more similar.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir, db_type=db_type)

    return vectorstore.similarity_search_with_relevance_scores(question, k=k)

