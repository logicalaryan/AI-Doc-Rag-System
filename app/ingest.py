"""
app/ingest.py — Document Ingestion Pipeline.

Responsibilities:
1. Load documents from a directory (PDF, TXT, MD)
2. Split them into overlapping chunks
3. Generate embeddings (HuggingFace, runs locally — no API key)
4. Persist embeddings to ChromaDB on disk
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

# HuggingFaceEmbeddings moved to langchain_huggingface in newer versions.
# We try the new package first, then fall back to langchain_community.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

load_dotenv()

# ---------------------------------------------------------------------------
# Config (reads from .env, falls back to sensible defaults)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "rag_documents"


def load_documents(data_dir: str) -> List[Document]:
    """
    Load all supported documents from *data_dir*.

    Supported formats: PDF, TXT, MD
    Returns a flat list of LangChain Document objects.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Document] = []

    # PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    # Plain text
    txt_files = list(data_path.glob("**/*.txt"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())

    # Markdown
    md_files = list(data_path.glob("**/*.md"))
    for md_file in md_files:
        loader = TextLoader(str(md_file), encoding="utf-8")
        documents.extend(loader.load())

    print(f"[ingest] Loaded {len(documents)} document(s) from '{data_dir}'")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    The recursive splitter tries to split on paragraphs, then sentences,
    then words — preserving semantic boundaries wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunk(s) "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model (downloaded once, cached locally)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_model=None,
) -> Chroma:
    """
    Embed *chunks* and persist them to ChromaDB.

    If a store already exists at *persist_dir*, the new chunks are ADDED
    (not replaced). To start fresh, delete the vectorstore/ directory.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    print(f"[ingest] Vectorstore persisted to '{persist_dir}' "
          f"({len(chunks)} chunk(s))")
    return vectorstore


def ingest(
    data_dir: str = "./data",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Full ingestion pipeline: load → chunk → embed → persist.

    This is the single public entry point used by scripts and tests.
    """
    documents = load_documents(data_dir)
    if not documents:
        print("[ingest] No documents found. Add files to the data/ directory.")
        return None

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    vectorstore = build_vectorstore(chunks, persist_dir)
    return vectorstore
