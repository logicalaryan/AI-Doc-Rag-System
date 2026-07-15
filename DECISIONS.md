# DECISIONS.md — Design Decision Log

> Record every meaningful choice here with the date, the alternatives considered,
> and the outcome/evidence. This is what turns a "it works" project into a
> portfolio-grade one.

---

## 2026-07-15 — Initial architecture

### LLM: Google Gemini (`gemini-2.0-flash`) over OpenAI GPT
- **Why:** Generous free tier (15 RPM, 1M TPM, 1500 req/day). No billing setup.
- **Alternative considered:** OpenAI GPT-4o-mini — requires credit card even on free tier.
- **Outcome:** Gemini free tier is sufficient for development and portfolio demos.

### Embeddings: HuggingFace `all-MiniLM-L6-v2` over OpenAI Ada
- **Why:** Runs 100% locally (CPU), no API key, no cost, 384-dim vectors are fast.
- **Alternative considered:** OpenAI `text-embedding-3-small` — costs money per token.
- **Outcome:** Local model reduces latency for bulk ingestion, zero cost.

### Vector Store: ChromaDB over FAISS / Pinecone
- **Why:** Zero-config, file-based persistence, no server to manage.
- **Alternative considered:** FAISS — in-memory only (no persistence without extra code). Pinecone — requires account/API key.
- **Outcome:** ChromaDB persists to disk automatically, simplest for a portfolio project.

### Chunk Size: 1000 chars, Overlap: 200 chars (default)
- **Why:** 1000 chars ≈ 1–2 paragraphs. Small enough to be specific, large enough to hold context.
- **Alternative considered:** 500 chars — too granular, breaks mid-sentence context.
- **Evidence:** Not yet measured. Update after running `eval/eval_report.py`.

### 2026-07-15 — Switch to Gemini 3.5 Flash
- **Why:** The Google AI Studio free tier limits for new `AQ.` keys set a hard quota limit (0 requests) for `gemini-2.0-flash` by default in this project's configuration, leading to `429 ResourceExhausted` errors.
- **Alternative considered:** Enabling billing on the GCP project, which defeats the 100% free tech-stack objective.
- **Outcome:** Swapped to `gemini-3.5-flash` which has active free-tier quota, restoring LLM functionality completely for zero cost.

---

## Template for future entries

```
## YYYY-MM-DD — [What changed]

### [Decision title]
- **Why:**
- **Alternative considered:**
- **Evidence / outcome:**
- **Eval score before:** x.xx
- **Eval score after:**  x.xx
```
