# Vector Database (ChromaDB) — Documentation & Architecture

This document provides a comprehensive overview of the Vector Database architecture, configuration, ingestion, and search workflow used in the **RAG Document Q&A System**.

---

## 1. What is a Vector Database?

A **Vector Database** is a specialized database designed to store, manage, and query high-dimensional vector embeddings efficiently.

- **Traditional Databases (SQL/NoSQL):** Perform exact matches or keyword filters on structured data (strings, integers, timestamps).
- **Vector Databases:** Store mathematical representations (dense floating-point vectors) of unstructured data (text, images, audio) and perform **semantic similarity searches** (e.g., Cosine Similarity, Dot Product, Euclidean Distance).

In a RAG (Retrieval-Augmented Generation) system, the vector database serves as the **long-term memory**, enabling fast retrieval of relevant document passages based on the conceptual meaning of a user's prompt rather than exact keyword matches.

---

## 2. Primary Engine: ChromaDB

This project uses **ChromaDB** as the vector database.

### Key Reasons for Choosing ChromaDB:
- **Zero-Config & Embedded:** Runs in-process alongside Python; no separate server installation or container management required.
- **Local Persistence:** Automatically persists collections and indexes to local disk storage (`vectorstore/`).
- **LangChain Native Integration:** Built-in seamless integration with LangChain's `Chroma` wrapper.
- **Cost Effective:** 100% open-source and free to run locally without cloud API subscription costs.

---

## 3. Data Schema & Metadata Tracking

- **Collection Name:** `rag_documents`
- **Default Storage Location:** `./vectorstore`
- **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace, 384 dimensions)
- **Metadata Fields Tracked:**
  - `source`: File path of origin document (e.g., `data/sample.pdf`)
  - `page`: Page number (for PDF documents)

---

## 4. Ingestion & Retrieval Flow Overview

```
[Raw Documents] -> [Text Splitter (1000 chars, 200 overlap)] 
                -> [HuggingFace Embeddings (384-dim)] 
                -> [ChromaDB (./vectorstore)] 
                -> [Similarity Search (Top-k=3)] -> [LLM Context Prompt]
```

---

## 5. Configuration Parameters (`.env`)

| Parameter | Environment Variable | Default Value | Description |
|---|---|---|---|
| Persist Directory | `CHROMA_PERSIST_DIR` | `./vectorstore` | Path for ChromaDB disk storage |
| Embedding Model | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| Top-K Results | `TOP_K` | `3` | Number of chunks returned per query |

---

## 6. Vector Database Management & Maintenance

### Resetting / Clearing ChromaDB
```bash
# PowerShell
Remove-Item -Recurse -Force vectorstore
python scripts/ingest_docs.py
```
