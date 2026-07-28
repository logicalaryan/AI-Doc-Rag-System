# Vector Database (ChromaDB & Qdrant) — Documentation & Architecture

This document provides a comprehensive overview of the Vector Database architecture, configuration, ingestion, and search workflow used in the **RAG Document Q&A System**, including comparison and integration guide for **Qdrant**.

---

## 1. What is a Vector Database?

A **Vector Database** is a specialized database designed to store, manage, and query high-dimensional vector embeddings efficiently.

- **Traditional Databases (SQL/NoSQL):** Perform exact matches or keyword filters on structured data (strings, integers, timestamps).
- **Vector Databases:** Store mathematical representations (dense floating-point vectors) of unstructured data (text, images, audio) and perform **semantic similarity searches** (e.g., Cosine Similarity, Dot Product, Euclidean Distance).

In a RAG (Retrieval-Augmented Generation) system, the vector database serves as the **long-term memory**, enabling fast retrieval of relevant document passages based on the conceptual meaning of a user's prompt rather than exact keyword matches.

---

## 2. Primary Engine: ChromaDB

This project currently uses **ChromaDB** as the default local vector database.

### Key Reasons for Choosing ChromaDB:
- **Zero-Config & Embedded:** Runs in-process alongside Python; no separate server installation or container management required.
- **Local Persistence:** Automatically persists collections and indexes to local disk storage (`vectorstore/`).
- **LangChain Native Integration:** Built-in seamless integration with LangChain's `Chroma` wrapper.
- **Cost Effective:** 100% open-source and free to run locally without cloud API subscription costs.

---

## 3. Alternative & Production Engine: Qdrant

**Qdrant** is an enterprise-grade, high-performance vector search engine written in **Rust**. It provides open-source, local, and managed cloud options.

### Key Features of Qdrant:
- **Written in Rust:** High performance, memory-efficient vector operations with HNSW indexing.
- **Advanced Payload Filtering:** Supports complex JSON payload filtering alongside vector similarity search without performance degradation.
- **Flexible Deployment Modes:**
  - **In-Memory / Local Disk:** Run directly in Python process via `qdrant-client` (no server needed).
  - **Docker Container:** Run locally via `docker run -p 6333:6333 qdrant/qdrant`.
  - **Qdrant Cloud:** Managed cloud cluster with free tier support.
- **Quantization:** Supports Scalar and Product Quantization for up to 4x RAM footprint reduction.

---

## 4. Comparison: ChromaDB vs. Qdrant

| Feature / Metric | ChromaDB | Qdrant |
|---|---|---|
| **Core Language** | Python / C++ | Rust |
| **In-Memory / Local Disk** | Yes (`./vectorstore`) | Yes (`:memory:` or `./qdrant_db`) |
| **Docker / Server Mode** | Yes | Yes (gRPC + HTTP APIs) |
| **Managed Cloud Option** | Chroma Cloud (beta) | Qdrant Cloud (Production-ready) |
| **Filtering Capability** | Basic metadata filtering | Advanced nested payload filtering |
| **Performance & Scale** | Great for small-to-medium datasets | Ultra-high performance, multi-million vector datasets |
| **Quantization Support** | Limited | Scalar & Product Quantization |
| **Primary Use Case** | Local prototypes & lightweight apps | Production, multi-tenant RAG applications |

---

## 5. Integrating Qdrant into this RAG Pipeline

To use **Qdrant** instead of ChromaDB in this project, follow these steps:

### Step 5.1: Install Dependencies
```bash
pip install qdrant-client langchain-qdrant
```

### Step 5.2: Ingestion Code (`app/ingest.py` with Qdrant)

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def build_qdrant_vectorstore(chunks, embedding_model, location="./qdrant_db"):
    """
    Ingest document chunks into Qdrant local disk storage.
    """
    # Initialize Qdrant Client (Local storage mode)
    client = QdrantClient(path=location)

    collection_name = "rag_documents"

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Wrap in LangChain QdrantVectorStore
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # Add documents
    qdrant_store.add_documents(chunks)
    print(f"[ingest] Ingested {len(chunks)} chunks into Qdrant at '{location}'")
    return qdrant_store
```

### Step 5.3: Retrieval Code (`app/retriever.py` with Qdrant)

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def retrieve_qdrant(question, k=3, location="./qdrant_db", embedding_model=None):
    """
    Query Qdrant vector database with similarity search.
    """
    client = QdrantClient(path=location)
    
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name="rag_documents",
        embedding=embedding_model,
    )

    # Perform similarity search
    results = qdrant_store.similarity_search(question, k=k)
    return results
```

---

## 6. Data Schema & Metadata Tracking

- **Collection Name:** `rag_documents`
- **Default Storage Location:** 
  - ChromaDB: `./vectorstore`
  - Qdrant: `./qdrant_db` or Qdrant Cloud URL
- **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace, 384 dimensions)
- **Metadata/Payload Fields Tracked:**
  - `source`: File path of origin document (e.g., `data/sample.pdf`)
  - `page`: Page number (for PDF documents)

---

## 7. Ingestion & Retrieval Flow Overview

```
[Raw Documents] -> [Text Splitter (1000 chars, 200 overlap)] 
                -> [HuggingFace Embeddings (384-dim)] 
                -> [Vector DB (ChromaDB / Qdrant)] 
                -> [Similarity Search (Top-k=3)] -> [LLM Context Prompt]
```

---

## 8. Configuration Parameters (`.env`)

| Parameter | Environment Variable | Default Value | Description |
|---|---|---|---|
| Vector Store Type | `VECTOR_DB_TYPE` | `chroma` | `chroma` or `qdrant` |
| Persist Directory | `CHROMA_PERSIST_DIR` | `./vectorstore` | Path for ChromaDB disk storage |
| Qdrant Directory / Host | `QDRANT_LOCATION` | `./qdrant_db` | Local path or Cloud URL for Qdrant |
| Qdrant API Key | `QDRANT_API_KEY` | `` | Optional API Key for Qdrant Cloud |
| Embedding Model | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| Top-K Results | `TOP_K` | `3` | Number of chunks returned per query |

---

## 9. Vector Database Management & Maintenance

### Resetting / Clearing ChromaDB
```bash
# PowerShell
Remove-Item -Recurse -Force vectorstore
python scripts/ingest_docs.py
```

### Resetting / Clearing Qdrant (Local Path)
```bash
# PowerShell
Remove-Item -Recurse -Force qdrant_db
python scripts/ingest_docs.py
```
