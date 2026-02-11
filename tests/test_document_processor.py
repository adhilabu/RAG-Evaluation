"""Basic tests for document processor."""
import pytest
from pathlib import Path
from document_processor import extract_pdf_text, clean_text
from document_processor.cleaner import clean_pages
from document_processor.chunker import create_rag_chunks, create_summary_chunks


def test_clean_text():
    """Test text cleaning function."""
    raw_text = """Page 1 of 100
    
    This is    some text with    extra spaces.
    
    
    
    And multiple blank lines."""
    
    cleaned = clean_text(raw_text)
    
    assert "Page 1 of 100" not in cleaned
    assert "extra  spaces" not in cleaned  # Multiple spaces removed
    assert "\n\n\n" not in cleaned  # Max 2 blank lines


def test_chunker_rag():
    """Test RAG chunking."""
    sample_pages = [
        {"page_number": 1, "text": "This is page one with some content. " * 100, "char_count": 4000},
        {"page_number": 2, "text": "This is page two with different content. " * 100, "char_count": 4000},
    ]
    
    chunks = create_rag_chunks(sample_pages, chunk_size=500, chunk_overlap=50, document_id="test")
    
    assert len(chunks) > 0
    assert all(chunk["document_id"] == "test" for chunk in chunks)
    assert all(chunk["chunk_type"] == "rag" for chunk in chunks)


def test_chunker_summary():
    """Test summary chunking."""
    sample_pages = [
        {"page_number": i, "text": f"Page {i} content. " * 1000, "char_count": 15000}
        for i in range(1, 11)
    ]
    
    chunks = create_summary_chunks(sample_pages, chunk_size=15000, chunk_overlap=500, document_id="test")
    
    assert len(chunks) > 0
    assert all(chunk["document_id"] == "test" for chunk in chunks)
    assert all(chunk["chunk_type"] == "summary" for chunk in chunks)
    assert all("page_range" in chunk for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
