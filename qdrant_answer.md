# Qdrant — Answers & Plan

> This file answers all questions from `ins_qdrant.md`. No code changes were made. This is a planning document only.

---

## Q1. How Will Data Be Stored in Qdrant?

Qdrant organises data into **Collections** (like a table in SQL). Each collection holds **Points**.

A **Point** has 3 parts:

```
Point {
  id       → a unique auto-generated ID (UUID or integer)
  vector   → [0.12, -0.43, 0.91, ... ] — 384 numbers (your chunk embedding)
  payload  → { "source": "data/report.pdf", "page": 3, "text": "..." }
}
```

### So in your RAG system, this is exactly what gets stored:

| Field | Value Example | What It Is |
|---|---|---|
| `id` | `a1b2c3d4-...` | Auto UUID per chunk |
| `vector` | `[0.12, -0.43, ...]` | 384-dim HuggingFace embedding of the text chunk |
| `payload.source` | `data/Aryan_consultant.pdf` | Which file the chunk came from |
| `payload.page` | `3` | Page number (PDFs only) |
| `payload.text` | `"Aryan has 5 years of..."` | The actual raw chunk text |

### Storage Location (depends on mode):
- **Local disk mode:** stored as files inside `./qdrant_db/` on your machine
- **Docker mode:** stored inside a Docker volume on your machine
- **Qdrant Cloud:** stored on Qdrant's managed servers in the cloud

---

## Q2. Will Your Codebase Exceed the Qdrant Free Tier?

### Your Current Scale (analysed from codebase):
- **Chunk size:** 1000 characters
- **Chunk overlap:** 200 characters
- **Data currently:** `Aryan_consultant.pdf` (126 KB) + `company_info.txt` (0.2 KB)
- **Embedding dimension:** 384 (all-MiniLM-L6-v2)
- **Estimated chunks from current data:** ~50–100 chunks total

### Qdrant Cloud Free Tier Limits:

| Limit | Free Tier Value |
|---|---|
| **Vectors (Points)** | 1,000,000 vectors |
| **Collections** | Unlimited |
| **RAM** | ~1 GB |
| **Disk Storage** | ~4 GB |
| **API requests** | Unlimited (no rate limit on free tier) |
| **Clusters** | 1 |

### Verdict:

> **No, your project will NOT come close to exceeding the free tier.**

- 100 chunks × 384 floats × 4 bytes = **~154 KB** of vector data.
- Even if you scale to 50 PDFs of 100 pages each (~5,000 chunks), you'd use only **~7.3 MB** of vector storage.
- The free tier allows up to **1 million vectors** — you would need to ingest thousands of large documents to hit that.
- **Conclusion:** The free Qdrant Cloud tier is more than enough for this portfolio project.

---

## Q3. Should You Use Qdrant Website (Cloud) or Qdrant Docker?

### Option A: Qdrant Cloud (qdrant.io)
- Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
- Get a cluster URL + API key
- Connect from Python with `QdrantClient(url="...", api_key="...")`
- **No local setup required**

| Pros | Cons |
|---|---|
| Works on Render (no disk dependency) | Requires internet connection to query |
| Free 1M vector tier | Data leaves your machine |
| Persistent — survives redeployments | Slightly higher latency vs local |
| No Docker or server to manage | Qdrant cloud login required |

### Option B: Qdrant Docker (local server)
- Run `docker run -p 6333:6333 qdrant/qdrant` on your PC
- Connect from Python with `QdrantClient(host="localhost", port=6333)`
- **Requires Docker installed on machine**

| Pros | Cons |
|---|---|
| Data stays local | Doesn't work on Render (ephemeral containers) |
| No API key needed | Need Docker running on your PC |
| Full control over data | Not portable to cloud without migration |

### Option C: Qdrant Local Path (no server, no Docker)
- `QdrantClient(path="./qdrant_db")` — runs directly inside Python, stores on disk
- Easiest for local development — no extra software needed

| Pros | Cons |
|---|---|
| Zero setup (no Docker, no signup) | Same Render problem as ChromaDB (disk wiped on redeploy) |
| Same behaviour as ChromaDB today | Not suitable as long-term production store |

---

## Q4. The Easiest Way (Recommendation)

### For Local Development (right now):
**Use Qdrant Local Path mode** — `QdrantClient(path="./qdrant_db")`
- Zero setup. No Docker. No account.
- Works exactly like your current ChromaDB setup.
- Just swap `build_vectorstore` to `build_qdrant_vectorstore`.

### For Production / Render Deployment (when you deploy):
**Use Qdrant Cloud (free tier)**
- Sign up takes 2 minutes at [cloud.qdrant.io](https://cloud.qdrant.io).
- Add `QDRANT_URL` and `QDRANT_API_KEY` to your `.env` and Render env vars.
- Unlike ChromaDB, vectors persist across Render redeployments (because they live in Qdrant's cloud, not on Render's disk).
- This solves the current pain point: **you won't need to re-ingest after every Render restart**.

### Summary Recommendation:

| Phase | Use This |
|---|---|
| Local dev & testing | Qdrant Local Path (`./qdrant_db`) |
| Production (Render) | Qdrant Cloud free tier |

---

## Overall Action Plan (No Execution Yet)

> [!NOTE]
> All steps below are **planned only**. No files have been changed.

- [ ] **Step 1:** Create a Qdrant Cloud account at [cloud.qdrant.io](https://cloud.qdrant.io) and note down the **Cluster URL** and **API Key**.
- [ ] **Step 2:** Add `langchain-qdrant` and `qdrant-client` to `requirements.txt`.
- [ ] **Step 3:** Add `QDRANT_URL` and `QDRANT_API_KEY` to `.env.example` and your local `.env`.
- [ ] **Step 4:** Add a `build_qdrant_vectorstore()` function to `app/ingest.py`.
- [ ] **Step 5:** Add a `retrieve_qdrant()` function to `app/retriever.py`.
- [ ] **Step 6:** Add a `VECTOR_DB_TYPE` env variable (`chroma` or `qdrant`) so you can switch between them without code changes.
- [ ] **Step 7:** Update Render env vars with `QDRANT_URL` and `QDRANT_API_KEY`.
- [ ] **Step 8:** Test locally with local path mode first, then switch to cloud URL for production.
