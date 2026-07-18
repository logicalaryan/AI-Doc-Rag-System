# 📚 RAG Evaluation Metrics — Doubts & Definitions

> **Purpose:** A personal reference sheet for all evaluation metric concepts used in this RAG project.
> Designed to be read on GitHub. All sections are collapsible — click to expand.

---

## Table of Contents

| # | Concept | One-Line Summary |
|---|---|---|
| 1 | [What is `k`?](#1-what-is-k) | Number of chunks retrieved per query |
| 2 | [Hit Rate](#2-hit-rate) | Did we find the right chunk at all? |
| 3 | [Hit@k](#3-hitk) | Did we find the right chunk within top-k? |
| 4 | [Precision@k](#4-precisionk) | How clean were ALL retrieved chunks? |
| 5 | [Recall](#5-recall) | How many of ALL relevant chunks did we find? |
| 6 | [MRR](#6-mrr-mean-reciprocal-rank) | How high up was the correct chunk ranked? |
| 7 | [F1 Score](#7-f1-score) | Balance between Precision and Recall |
| 8 | [Latency](#8-latency) | How fast does the system respond? |
| 9 | [Why Recall & Latency were skipped](#9-why-recall--latency-were-skipped-in-this-project) | Project-specific decision |

---

## 1. What is `k`?

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
`k` is the **number of document chunks** the retriever fetches from ChromaDB for each user question.

### In this project
```
k = 3   (default, set via TOP_K in .env)
```

This means: for every question, the system pulls the **top 3 most similar chunks** from the vectorstore and passes them as context to the LLM.

### Why does k matter?

| k value | Effect |
|---|---|
| Too small (k=1) | Might miss the right chunk entirely |
| Good range (k=3–5) | Balanced — enough context, not too noisy |
| Too large (k=20) | Sends too much irrelevant text to the LLM → worse answers |

### Visual
```
User Question: "When was the company founded?"
                        │
                        ▼
              ChromaDB similarity search
                        │
              ┌─────────┴──────────┐
              │   Top k=3 chunks   │
              ├────────────────────┤
              │ Chunk 1 (score 0.91)│  ← "founded in 2015 by Alice..."
              │ Chunk 2 (score 0.74)│  ← "revenue reached 500 crore..."
              │ Chunk 3 (score 0.61)│  ← "headquarters in Bangalore..."
              └────────────────────┘
                        │
                   Sent to LLM
```

</details>

---

## 2. Hit Rate

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
Hit Rate measures: **for what fraction of questions did the retriever find at least one relevant chunk** in its top-k results?

### Formula
```
Hit Rate = Questions with at least 1 relevant chunk in top-k
           ─────────────────────────────────────────────────
                        Total questions
```

### It's a binary check per question
- ✅ Hit = the correct chunk appeared **anywhere** in the top-k results
- ❌ Miss = the correct chunk did NOT appear in the top-k results

### Example (5 questions, k=3)

| Question | Correct chunk in top-3? | Hit? |
|---|---|---|
| When was the company founded? | Yes (rank 1) | ✅ |
| What is the annual revenue? | Yes (rank 2) | ✅ |
| Where is the headquarters? | Yes (rank 1) | ✅ |
| What AI features are planned? | No | ❌ |
| Who founded the company? | Yes (rank 3) | ✅ |

```
Hit Rate = 4 / 5 = 0.80
```

### This project's actual result
```
Hit Rate = 0.70  (21 out of 30 test cases)
```

### What a good Hit Rate looks like
| Score | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent retrieval |
| 0.70 – 0.89 | Good — acceptable for most use cases |
| 0.50 – 0.69 | Needs improvement |
| < 0.50 | Poor — retriever is largely failing |

</details>

---

## 3. Hit@k

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
Hit@k is essentially the same as Hit Rate **but explicitly tied to a specific value of k**.

When someone says **Hit@3**, they mean:
> *"Did the correct chunk appear in the top-3 retrieved results?"*

When someone says **Hit@5**, they mean:
> *"Did the correct chunk appear in the top-5 retrieved results?"*

### Relationship to Hit Rate
```
Hit Rate (as used in this project) = Hit@k  where k = 3
```
They are the same thing — the "Hit Rate" label just omits the k for brevity.

### Why Hit@k is useful
It lets you compare retriever performance at different k values:

| Metric | Score |
|---|---|
| Hit@1 | 0.60 — correct chunk was rank 1 for 60% of questions |
| Hit@3 | 0.75 — expands to top 3, catches more |
| Hit@5 | 0.85 — expands to top 5, catches even more |
| Hit@10 | 0.95 — almost always finds it if you fetch 10 |

A rising Hit@k curve means the retriever **does find the right chunk**, just not always at the top — suggesting a **re-ranker** would help.

### Formula
```
Hit@k = Questions where correct chunk rank ≤ k
        ──────────────────────────────────────
                    Total questions
```

</details>

---

## 4. Precision@k

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
Precision@k measures: **of the k chunks retrieved, what fraction were actually relevant?**

It's about the **quality/cleanliness** of ALL returned results — not just whether one hit was found.

### Formula
```
Precision@k = Relevant chunks in top-k
              ─────────────────────────   (per question, then averaged)
                          k
```

### Example (k=3)

| Question | Retrieved chunks | Relevant? | Precision@3 |
|---|---|---|---|
| Q1 | [✅ chunk A, ✅ chunk B, ❌ chunk C] | 2 of 3 | 2/3 = 0.67 |
| Q2 | [✅ chunk X, ❌ chunk Y, ❌ chunk Z] | 1 of 3 | 1/3 = 0.33 |
| Q3 | [✅ chunk P, ✅ chunk Q, ✅ chunk R] | 3 of 3 | 3/3 = 1.00 |

```
Average Precision@3 = (0.67 + 0.33 + 1.00) / 3 = 0.67
```

### Difference from Hit Rate

| Metric | Question it answers |
|---|---|
| Hit Rate | Did we find **any** relevant chunk? (yes/no) |
| Precision@k | How many of the **k chunks** were relevant? |

Hit Rate can be 1.0 while Precision@3 is 0.33 — you found the answer, but 2 of 3 chunks were noise.

### This project's actual result
```
Precision@3 = 0.70
```
Since each test case has only **one** relevant chunk in the entire DB, the max possible Precision@3 is `1/3 = 0.33` when the correct chunk is found. The score of `0.70` here is because our matching is binary (hit = 1 chunk counted as relevant), so it effectively mirrors Hit Rate in this single-source setup.

### Why Precision matters for RAG
Lower precision = more irrelevant chunks sent to the LLM = higher chance of confused or hallucinated answers.

</details>

---

## 5. Recall

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
Recall measures: **of ALL relevant chunks that exist in the database, how many did the retriever actually find?**

### Formula
```
Recall@k = Relevant chunks retrieved in top-k
           ────────────────────────────────────
           Total relevant chunks in the database
```

### Example
Suppose for the question *"When was the company founded?"*, there are **3 relevant chunks** in the entire database:
- Chunk 5: *"founded in 2015 by Alice and Bob"*
- Chunk 12: *"established in 2015, the company..."*
- Chunk 19: *"since its founding year 2015..."*

If the retriever (k=3) returns chunks [5, 12, 7]:
- Found relevant: 2 (chunks 5 and 12)
- Total relevant: 3

```
Recall@3 = 2 / 3 = 0.67
```

### Precision vs Recall — The Trade-off

```
                        Precision
                        (Quality of results)
                             ▲
                             │      ●  Perfect system
                             │   ●
                             │●
                             └──────────────▶ Recall
                                             (Coverage of results)

More k → Higher Recall, Lower Precision (more results, more noise)
Less k → Lower Recall, Higher Precision (fewer results, cleaner)
```

### Why Recall was NOT implemented here
> See [Section 8](#8-why-recall--latency-were-skipped-in-this-project)

</details>

---

## 6. MRR (Mean Reciprocal Rank)

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
MRR measures: **how high up in the ranked list does the first relevant chunk appear?**

It doesn't just ask "did we find it?" — it asks "did we find it first?"

### Formula
```
Reciprocal Rank (per question) = 1 / rank of first relevant chunk

MRR = Average of Reciprocal Rank across all questions
```

### Example

| Question | Rank of first correct chunk | Reciprocal Rank |
|---|---|---|
| Q1 | Rank 1 | 1/1 = 1.00 |
| Q2 | Rank 2 | 1/2 = 0.50 |
| Q3 | Rank 3 | 1/3 = 0.33 |
| Q4 | Not found | 0.00 |

```
MRR = (1.00 + 0.50 + 0.33 + 0.00) / 4 = 0.46
```

### What MRR values mean

| MRR Score | Meaning |
|---|---|
| 1.00 | Perfect — correct chunk always at rank 1 |
| 0.50 | Correct chunk is at rank 2 on average |
| 0.33 | Correct chunk is at rank 3 on average |
| 0.00 | Never found |

### MRR vs Hit Rate

| Scenario | Hit Rate | MRR |
|---|---|---|
| Correct chunk always at rank 1 | 1.00 | 1.00 |
| Correct chunk always at rank 3 | 1.00 | 0.33 |
| Correct chunk never found | 0.00 | 0.00 |

**Key insight:** Hit Rate can be high while MRR is low — it means you're finding the answer, but burying it under irrelevant results. This signals a ranking/re-ranking problem.

### This project's actual result
```
MRR = 0.70
```
The `0.70` matches Hit Rate exactly — because every hit in this project was always at **rank 1** (never rank 2 or 3). When hits happened, the correct chunk was always the closest match.

</details>

---

## 7. F1 Score

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
F1 Score is the **harmonic mean of Precision and Recall**. It gives a single number that balances both — useful when you care equally about not missing relevant chunks (Recall) and not returning junk (Precision).

### Formula
```
         2 × Precision × Recall
F1  =  ─────────────────────────
           Precision + Recall
```

> The **harmonic mean** punishes extreme imbalances. If either Precision or Recall is 0, F1 = 0 — even if the other is perfect.

### Example

| Scenario | Precision | Recall | F1 Score |
|---|---|---|---|
| Both perfect | 1.00 | 1.00 | **1.00** |
| High P, Low R | 0.90 | 0.20 | **0.33** ← poor |
| Low P, High R | 0.20 | 0.90 | **0.33** ← poor |
| Balanced | 0.70 | 0.70 | **0.70** |
| One is zero | 0.80 | 0.00 | **0.00** |

### Why the harmonic mean and not a simple average?

```
Simple average of (0.90 + 0.10) / 2  = 0.50  ← looks okay
Harmonic mean  of (0.90 + 0.10)      = 0.18  ← reveals the imbalance
```

A system that returns only 1 chunk (high precision, low recall) would fool a simple average. F1 catches it.

### Precision vs Recall vs F1 — The Triangle

```
          You want ALL relevant chunks?  →  Optimize Recall
          You want ONLY relevant chunks? →  Optimize Precision
          You want BOTH balanced?        →  Optimize F1
```

### F1 in RAG context

| What F1 measures in RAG | Why it matters |
|---|---|
| Retrieval F1 | Are we fetching enough relevant chunks without flooding the LLM with noise? |
| Generation F1 (token-level) | Used in QA benchmarks like SQuAD — overlap between predicted and expected answer tokens |

### Token-level F1 (used in QA benchmarks like SQuAD)

This variant compares individual **words/tokens** between the predicted answer and the ground truth:

```
Ground truth: "The company was founded in 2015"
Predicted:    "It was founded in 2015 by Alice"

Common tokens: [was, founded, in, 2015]  → 4 tokens

Precision = 4 / 7  = 0.57   (4 of 7 predicted tokens are correct)
Recall    = 4 / 6  = 0.67   (4 of 6 ground truth tokens were found)
F1        = 2×0.57×0.67 / (0.57+0.67) = 0.62
```

### Why F1 is NOT implemented in this project
- **Retrieval F1** needs full Recall annotation (same problem as Recall — see Section 9)
- **Token-level F1** needs exact ground truth answer strings, not just keywords
- Our test cases only store one `expected_answer_contains` keyword — not a full answer string
- For a portfolio demo, Hit Rate + MRR + Precision@k already cover the retrieval story well

**In production:** Use SQuAD-style F1 via HuggingFace `evaluate` library:
```python
import evaluate
squad_metric = evaluate.load("squad")
result = squad_metric.compute(predictions=[...], references=[...])
print(result["f1"])  # token-level F1
```

</details>

---

## 8. Latency

<details>
<summary><strong>Click to expand</strong></summary>

### Definition
Latency is the **time taken** for the system to process a question and return an answer — measured in milliseconds (ms) or seconds (s).

### Components in a RAG system

```
Total Latency = Embedding time + Retrieval time + LLM time

┌──────────────────────────────────────────────────────┐
│  User sends question                                 │
│         │                                            │
│  ┌──────▼──────┐                                     │
│  │  Embed Q    │  ~10–50 ms   (HuggingFace, local)   │
│  └──────┬──────┘                                     │
│         │                                            │
│  ┌──────▼──────┐                                     │
│  │ ChromaDB    │  ~10–50 ms   (local disk search)    │
│  │  search     │                                     │
│  └──────┬──────┘                                     │
│         │                                            │
│  ┌──────▼──────┐                                     │
│  │ Gemini API  │  ~1000–3000 ms  ← BIGGEST factor    │
│  │  (LLM call) │  (network + model inference)        │
│  └──────┬──────┘                                     │
│         │                                            │
│  Answer returned                                     │
└──────────────────────────────────────────────────────┘
```

### Common latency metrics in production

| Metric | Meaning |
|---|---|
| **P50** | 50% of requests are faster than this (median) |
| **P95** | 95% of requests are faster than this |
| **P99** | 99% of requests are faster than this (worst cases) |

```
Example production targets:
  P50 < 1.5 sec
  P95 < 3.0 sec
  P99 < 5.0 sec
```

### Why Latency matters
- **Too slow** → users leave / bad UX
- **Streaming** → our `stream()` function solves perceived latency by showing tokens as they arrive
- **Retrieval latency** is negligible here (~20ms); the bottleneck is always the LLM API

### Why Latency was NOT measured here
> See [Section 8](#8-why-recall--latency-were-skipped-in-this-project)

</details>

---

## 9. Why Recall & Latency Were Skipped in This Project

<details>
<summary><strong>Click to expand</strong></summary>

### Why Recall was skipped

**Recall requires knowing ALL relevant chunks** in the database for every question.

```
test_cases.json stores:
  "expected_chunk_contains": "founded in 2015"   ← only ONE phrase

Recall needs:
  "relevant_chunk_ids": [3, 7, 12, 19]           ← ALL relevant chunks
```

Building that full annotation requires:
1. Manually reviewing every chunk in the vectorstore
2. Labelling which ones are relevant per question
3. Updating labels every time new documents are ingested

This is expensive human effort — not justified for a demo with a 4-line document.

**In production:** Use tools like [LlamaIndex evaluation](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) or [RAGAS](https://docs.ragas.io/) with pre-labelled benchmark datasets.

---

### Why Latency was skipped

**1. It's environment-dependent**
```
Same code, different results:
  Local CPU laptop       →  3.2 seconds
  Cloud GPU server       →  0.4 seconds
  Slow network / India   →  5+ seconds (Gemini API round-trip)
```
A number from one machine means nothing to someone else.

**2. The bottleneck is Gemini's servers, not our code**
```
ChromaDB search   =  ~20 ms    (our code)
HuggingFace embed =  ~40 ms    (our code)
Gemini API call   =  ~2000 ms  (Google's infrastructure)
```
Measuring total latency mostly measures Google's API performance.

**3. No SLA defined**
Latency only becomes a metric when you have a target to compare against (e.g., *"must respond in < 2 seconds"*). This portfolio project has no such requirement.

**4. Streaming mitigates perceived latency**
The `stream()` function in `app/chain.py` streams tokens as they arrive — so users see output immediately even if full generation takes 3 seconds.

**How to add it in production:**
```python
import time

start = time.perf_counter()
result = ask(question, vectorstore=vs)
latency_ms = (time.perf_counter() - start) * 1000

print(f"Latency: {latency_ms:.0f} ms")
```
Track as P50/P95 using OpenTelemetry or Prometheus.

</details>

---

## Quick Reference Card

```
┌─────────────────┬──────────────────────────────┬────────────────┐
│ Metric          │ Question Answered             │ Project Score  │
├─────────────────┼──────────────────────────────┼────────────────┤
│ k               │ How many chunks to fetch?     │ k = 3          │
│ Hit Rate / Hit@k│ Found it at all? (yes/no)     │ 0.70           │
│ Precision@k     │ How clean were ALL results?   │ 0.70           │
│ MRR             │ How high up was the hit?      │ 0.70           │
│ Recall          │ Found ALL relevant chunks?    │ Not measured*  │
│ Latency         │ How fast?                     │ Not measured*  │
└─────────────────┴──────────────────────────────┴────────────────┘

* See Section 8 for reasons why.
```

---

*Last updated: July 2026 | Project: RAG Document Q&A*
