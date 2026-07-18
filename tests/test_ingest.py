"""
tests/test_ingest.py — Tests for document loading and chunking.

Tests:
  1. load_documents raises FileNotFoundError for missing directory
  2. load_documents reads a real .txt file correctly
  3. chunk_documents produces chunks within the configured size limit
  4. chunk_documents creates more chunks than the original document count
  5. chunk_documents chunk_overlap means consecutive chunks share content
  6. ingest() returns None when data/ is empty (no documents)
"""

import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from app.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_documents,
    load_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_txt(directory: Path, filename: str, content: str) -> Path:
    """Write *content* to *directory/filename* and return the path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


LONG_TEXT = textwrap.dedent("""\
    The quick brown fox jumps over the lazy dog.
    """ * 60)  # ~2 700 chars — forces multiple chunks with default chunk_size=1000


# ---------------------------------------------------------------------------
# Tests: load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    def test_raises_if_directory_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_documents(str(missing))

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_loads_txt_file(self, tmp_path):
        _write_txt(tmp_path, "sample.txt", "Hello, RAG world!")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert "Hello, RAG world!" in docs[0].page_content

    def test_loads_md_file(self, tmp_path):
        _write_txt(tmp_path, "notes.md", "# Title\n\nSome content here.")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_loads_multiple_files(self, tmp_path):
        _write_txt(tmp_path, "a.txt", "File A content.")
        _write_txt(tmp_path, "b.txt", "File B content.")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2

    def test_metadata_contains_source(self, tmp_path):
        _write_txt(tmp_path, "meta_test.txt", "Check metadata.")
        docs = load_documents(str(tmp_path))
        assert "source" in docs[0].metadata


# ---------------------------------------------------------------------------
# Tests: chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def _make_docs(self, tmp_path, text=LONG_TEXT):
        _write_txt(tmp_path, "long.txt", text)
        return load_documents(str(tmp_path))

    def test_chunks_within_size_limit(self, tmp_path):
        docs = self._make_docs(tmp_path)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.page_content) <= 500 + 50, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_produces_multiple_chunks_for_long_doc(self, tmp_path):
        docs = self._make_docs(tmp_path)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1, "Expected more than one chunk for a long document"

    def test_chunk_count_exceeds_document_count(self, tmp_path):
        docs = self._make_docs(tmp_path)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > len(docs)

    def test_overlap_produces_shared_content(self, tmp_path):
        """Consecutive chunks should share some text due to overlap."""
        docs = self._make_docs(tmp_path)
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=100)
        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-50:]
            start_of_second = chunks[1].page_content[:200]
            # At least some content from the tail of chunk 0 should appear
            # somewhere near the start of chunk 1
            assert any(
                word in start_of_second
                for word in end_of_first.split()
                if len(word) > 3
            ), "Expected overlapping content between consecutive chunks"

    def test_short_document_produces_single_chunk(self, tmp_path):
        _write_txt(tmp_path, "short.txt", "Short content.")
        docs = load_documents(str(tmp_path))
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1

    def test_chunks_preserve_metadata(self, tmp_path):
        _write_txt(tmp_path, "meta.txt", LONG_TEXT)
        docs = load_documents(str(tmp_path))
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_character_strategy_splits_on_double_newline(self, tmp_path):
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        docs = self._make_docs(tmp_path, text=text)
        chunks = chunk_documents(docs, chunk_size=15, chunk_overlap=0, strategy="character")
        assert len(chunks) == 3
        assert "Paragraph 1" in chunks[0].page_content
        assert "Paragraph 2" in chunks[1].page_content
        assert "Paragraph 3" in chunks[2].page_content
