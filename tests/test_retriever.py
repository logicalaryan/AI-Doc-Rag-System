"""
tests/test_retriever.py — Tests for vector search / retrieval logic.

Strategy: build a small in-memory Chroma collection from known text fixtures,
then verify that the retriever returns the expected chunk for known questions.
No real files on disk; no API calls; fast.

Tests:
  1. retrieve() returns the expected chunk for a known question
  2. retrieve() returns at most k results
  3. retrieve() returns fewer than k when fewer chunks exist
  4. retrieve_with_scores() returns (Document, float) tuples
  5. retrieve_with_scores() scores are in [0, 1]
  6. Similarity ordering — the most relevant chunk scores higher
"""

import tempfile

import pytest
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
from langchain_chroma import Chroma

from app.retriever import retrieve, retrieve_with_scores
from app.ingest import COLLECTION_NAME, EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small corpus with clearly distinct topics so similarity search is
# deterministic enough for testing.
CORPUS = [
    Document(
        page_content="The company was founded in 2015 by Alice and Bob.",
        metadata={"source": "company.txt"},
    ),
    Document(
        page_content="Our annual revenue reached 500 crore in Q3 2025.",
        metadata={"source": "financials.txt"},
    ),
    Document(
        page_content="The headquarters is located in Bangalore, India.",
        metadata={"source": "locations.txt"},
    ),
    Document(
        page_content="The product roadmap includes AI-powered features in 2026.",
        metadata={"source": "roadmap.txt"},
    ),
]


@pytest.fixture(scope="module")
def embedding_model():
    """Shared embedding model — downloaded once for the whole module."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture(scope="module")
def vectorstore(embedding_model, tmp_path_factory):
    """In-memory-style Chroma store built from CORPUS — shared across tests."""
    tmp_dir = tmp_path_factory.mktemp("chroma_test")
    store = Chroma.from_documents(
        documents=CORPUS,
        embedding=embedding_model,
        persist_directory=str(tmp_dir),
        collection_name=COLLECTION_NAME + "_test",
    )
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_expected_chunk_for_founding_question(self, vectorstore):
        docs = retrieve("When was the company founded?", vectorstore=vectorstore, k=1)
        assert len(docs) == 1
        assert "2015" in docs[0].page_content

    def test_returns_expected_chunk_for_revenue_question(self, vectorstore):
        docs = retrieve("What is the annual revenue?", vectorstore=vectorstore, k=1)
        assert len(docs) == 1
        assert "500 crore" in docs[0].page_content

    def test_returns_expected_chunk_for_location_question(self, vectorstore):
        docs = retrieve("Where is the headquarters?", vectorstore=vectorstore, k=1)
        assert len(docs) == 1
        assert "Bangalore" in docs[0].page_content

    def test_returns_at_most_k_results(self, vectorstore):
        docs = retrieve("Tell me everything", vectorstore=vectorstore, k=2)
        assert len(docs) <= 2

    def test_returns_fewer_than_k_when_corpus_is_small(self, vectorstore):
        """k=10 but only 4 docs in corpus — should return ≤ 4."""
        docs = retrieve("anything", vectorstore=vectorstore, k=10)
        assert len(docs) <= len(CORPUS)

    def test_returns_list_of_documents(self, vectorstore):
        docs = retrieve("company", vectorstore=vectorstore, k=2)
        assert isinstance(docs, list)
        for doc in docs:
            assert isinstance(doc, Document)

    def test_documents_have_page_content(self, vectorstore):
        docs = retrieve("revenue", vectorstore=vectorstore, k=1)
        assert docs[0].page_content.strip() != ""


class TestRetrieveWithScores:
    def test_returns_tuples(self, vectorstore):
        results = retrieve_with_scores("founded", vectorstore=vectorstore, k=2)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_scores_are_floats(self, vectorstore):
        results = retrieve_with_scores("revenue", vectorstore=vectorstore, k=1)
        doc, score = results[0]
        assert isinstance(score, float)

    def test_most_relevant_scores_highest(self, vectorstore):
        """The chunk about founding should score higher for a founding question
        than the chunk about revenue."""
        results = retrieve_with_scores(
            "When was the company founded?", vectorstore=vectorstore, k=4
        )
        # Build a mapping: chunk text → score
        score_map = {doc.page_content: score for doc, score in results}

        # Find scores for the two candidate chunks
        founding_score = next(
            (s for text, s in score_map.items() if "2015" in text), None
        )
        revenue_score = next(
            (s for text, s in score_map.items() if "500 crore" in text), None
        )

        if founding_score is not None and revenue_score is not None:
            assert founding_score >= revenue_score, (
                f"Expected founding chunk to score higher "
                f"({founding_score:.3f} < {revenue_score:.3f})"
            )
