"""
app/prompts.py — All prompt templates (single source of truth).

Every instruction sent to the LLM lives here so you can audit,
compare, and A/B-test prompts from one place.
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Main QA prompt
# ---------------------------------------------------------------------------
QA_TEMPLATE = """You are a helpful assistant that answers questions based ONLY
on the provided context excerpts from documents.

Rules:
- Answer ONLY from the context below. Do NOT use outside knowledge.
- If the answer is not in the context, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Be concise and factual.
- Where relevant, mention which part of the context supports your answer.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Condense / follow-up question prompt
# ---------------------------------------------------------------------------
CONDENSE_TEMPLATE = """Given the conversation history and a follow-up question,
rephrase the follow-up into a standalone question that contains all necessary
context to be understood without the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_TEMPLATE,
)
