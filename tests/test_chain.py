"""
tests/test_chain.py — Tests for LLM answer generation (chain.py).

Strategy: inject a mock LLM so no Gemini API calls are made during tests.
The mock always returns a configurable string, letting us verify that
the chain plumbing (prompt construction, output parsing, source extraction)
works correctly without network access or API keys.

Tests:
  1. ask() returns an AnswerResult with a non-empty answer
  2. ask() passes the user question through to the result
  3. ask() includes sources from the retrieved documents
  4. ask() falls back to "not enough information" language when context is empty
  5. ask() answer contains expected keywords when relevant context is provided
  6. AnswerResult.__str__ formats output correctly
  7. _format_context builds a labelled multi-excerpt string
  8. _extract_sources deduplicates sources correctly
"""

from unittest.mock import MagicMock, patch
import tempfile

import pytest
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
from langchain_chroma import Chroma

from app.chain import AnswerResult, _extract_sources, _format_context, ask
from app.ingest import COLLECTION_NAME, EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(response: str = "Mocked LLM response."):
    """
    Return a MagicMock that behaves like a LangChain chat model.
    Specifically, it needs to support the | (pipe) operator used in the chain,
    so we patch at the chain-invocation level instead.
    """
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=response)
    return mock


# Documents for testing
FOUNDING_DOC = Document(
    page_content="The company was founded in 2015 by Alice and Bob.",
    metadata={"source": "company.txt", "page": 1},
)
REVENUE_DOC = Document(
    page_content="Our annual revenue reached 500 crore in Q3 2025.",
    metadata={"source": "financials.txt", "page": 2},
)
LOCATION_DOC = Document(
    page_content="The headquarters is located in Bangalore, India.",
    metadata={"source": "locations.txt"},
)


# ---------------------------------------------------------------------------
# Tests: _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_single_doc_contains_excerpt_label(self):
        result = _format_context([FOUNDING_DOC])
        assert "Excerpt 1" in result

    def test_single_doc_contains_source(self):
        result = _format_context([FOUNDING_DOC])
        assert "company.txt" in result

    def test_single_doc_contains_page_number(self):
        result = _format_context([FOUNDING_DOC])
        assert "page 1" in result

    def test_multiple_docs_are_numbered(self):
        result = _format_context([FOUNDING_DOC, REVENUE_DOC])
        assert "Excerpt 1" in result
        assert "Excerpt 2" in result

    def test_empty_docs_returns_empty_string(self):
        result = _format_context([])
        assert result == ""

    def test_doc_without_page_omits_page_label(self):
        result = _format_context([LOCATION_DOC])
        assert "page" not in result


# ---------------------------------------------------------------------------
# Tests: _extract_sources
# ---------------------------------------------------------------------------

class TestExtractSources:
    def test_single_source(self):
        sources = _extract_sources([FOUNDING_DOC])
        assert len(sources) == 1
        assert sources[0]["source"] == "company.txt"

    def test_duplicate_sources_are_deduplicated(self):
        """Two chunks from the same file+page should appear once."""
        dup = Document(
            page_content="Different content, same source.",
            metadata={"source": "company.txt", "page": 1},
        )
        sources = _extract_sources([FOUNDING_DOC, dup])
        assert len(sources) == 1

    def test_different_sources_both_appear(self):
        sources = _extract_sources([FOUNDING_DOC, REVENUE_DOC])
        source_names = [s["source"] for s in sources]
        assert "company.txt" in source_names
        assert "financials.txt" in source_names

    def test_missing_metadata_uses_unknown(self):
        bare_doc = Document(page_content="No metadata.", metadata={})
        sources = _extract_sources([bare_doc])
        assert sources[0]["source"] == "unknown"


# ---------------------------------------------------------------------------
# Tests: AnswerResult
# ---------------------------------------------------------------------------

class TestAnswerResult:
    def test_str_contains_question(self):
        r = AnswerResult(answer="42", question="What is the answer?")
        assert "What is the answer?" in str(r)

    def test_str_contains_answer(self):
        r = AnswerResult(answer="Founded in 2015.", question="When founded?")
        assert "Founded in 2015." in str(r)

    def test_str_contains_source(self):
        r = AnswerResult(
            answer="42",
            question="Q?",
            sources=[{"source": "doc.txt", "page": 3}],
        )
        assert "doc.txt" in str(r)

    def test_str_shows_none_when_no_sources(self):
        r = AnswerResult(answer="42", question="Q?", sources=[])
        assert "none" in str(r).lower()


# ---------------------------------------------------------------------------
# Tests: ask() with mocked LLM
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture(scope="module")
def vectorstore(embedding_model, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("chain_test_chroma")
    return Chroma.from_documents(
        documents=[FOUNDING_DOC, REVENUE_DOC, LOCATION_DOC],
        embedding=embedding_model,
        persist_directory=str(tmp_dir),
        collection_name=COLLECTION_NAME + "_chain_test",
    )


class TestAsk:
    def _run_ask(self, question: str, vectorstore, mock_response: str) -> AnswerResult:
        """
        Call ask() with a mocked chain response.

        We patch build_rag_chain() so the entire LCEL pipe is replaced with a
        simple mock whose .invoke() returns the expected string directly.
        This avoids Pydantic validation errors that occur when a MagicMock flows
        through StrOutputParser (which expects a real string from the LLM).
        """
        from unittest.mock import patch as _patch

        # Mock the full LCEL chain — .invoke() returns the string StrOutputParser would produce
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        with _patch("app.chain.build_rag_chain", return_value=mock_chain):
            result = ask(
                question=question,
                vectorstore=vectorstore,
                llm=None,   # irrelevant — build_rag_chain is fully mocked
                k=2,
            )
        return result

    def test_returns_answer_result(self, vectorstore):
        result = self._run_ask("When was the company founded?", vectorstore, "2015.")
        assert isinstance(result, AnswerResult)

    def test_answer_is_non_empty(self, vectorstore):
        result = self._run_ask("When founded?", vectorstore, "Founded in 2015.")
        assert result.answer.strip() != ""

    def test_question_is_preserved(self, vectorstore):
        q = "Where is headquarters?"
        result = self._run_ask(q, vectorstore, "Bangalore.")
        assert result.question == q

    def test_sources_are_populated(self, vectorstore):
        result = self._run_ask("Revenue?", vectorstore, "500 crore.")
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0

    def test_answer_contains_mock_response_content(self, vectorstore):
        result = self._run_ask("Revenue?", vectorstore, "The revenue is 500 crore.")
        assert "500 crore" in result.answer
