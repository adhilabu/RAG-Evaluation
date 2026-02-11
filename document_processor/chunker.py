"""Dual chunking strategy for RAG storage and summarization."""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def create_rag_chunks(
    pages_data: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    document_id: str = None,
) -> List[Dict[str, any]]:
    """
    Create small chunks for RAG storage (512-1024 tokens).
    
    Args:
        pages_data: List of page dictionaries with cleaned text
        chunk_size: Target chunk size in characters (approximately tokens)
        chunk_overlap: Overlap between chunks
        document_id: Unique document identifier
        
    Returns:
        List of chunk dictionaries with metadata
    """
    # Combine all pages with page markers
    full_text = ""
    page_markers = []  # Track where each page starts
    
    for page in pages_data:
        page_start = len(full_text)
        full_text += page["text"] + "\n\n"
        page_markers.append({
            "page_number": page["page_number"],
            "start_pos": page_start,
            "end_pos": len(full_text),
        })
    
    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_text(full_text)
    
    # Add metadata to each chunk
    chunk_dicts = []
    current_position = 0
    
    for idx, chunk_text in enumerate(chunks):
        # Find which page this chunk belongs to
        chunk_start = full_text.find(chunk_text, current_position)
        chunk_end = chunk_start + len(chunk_text)
        
        # Determine page number(s) this chunk spans
        page_numbers = []
        for marker in page_markers:
            if chunk_start < marker["end_pos"] and chunk_end > marker["start_pos"]:
                page_numbers.append(marker["page_number"])
        
        chunk_dicts.append({
            "chunk_index": idx,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "token_count": count_tokens(chunk_text),
            "page_numbers": page_numbers,
            "page_number": page_numbers[0] if page_numbers else None,
            "document_id": document_id,
            "chunk_type": "rag",
        })
        
        current_position = chunk_start + 1
    
    return chunk_dicts


def create_summary_chunks(
    pages_data: List[Dict],
    chunk_size: int = 15000,
    chunk_overlap: int = 500,
    document_id: str = None,
) -> List[Dict[str, any]]:
    """
    Create large chunks for map-reduce summarization (10k-20k tokens).
    
    Args:
        pages_data: List of page dictionaries with cleaned text
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        document_id: Unique document identifier
        
    Returns:
        List of chunk dictionaries with metadata
    """
    # Combine all pages
    full_text = ""
    page_markers = []
    
    for page in pages_data:
        page_start = len(full_text)
        full_text += page["text"] + "\n\n"
        page_markers.append({
            "page_number": page["page_number"],
            "start_pos": page_start,
            "end_pos": len(full_text),
        })
    
    # Create text splitter for larger chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_text(full_text)
    
    # Add metadata
    chunk_dicts = []
    current_position = 0
    
    for idx, chunk_text in enumerate(chunks):
        chunk_start = full_text.find(chunk_text, current_position)
        chunk_end = chunk_start + len(chunk_text)
        
        # Determine page range
        page_numbers = []
        for marker in page_markers:
            if chunk_start < marker["end_pos"] and chunk_end > marker["start_pos"]:
                page_numbers.append(marker["page_number"])
        
        page_range = f"{min(page_numbers)}-{max(page_numbers)}" if page_numbers else "Unknown"
        
        chunk_dicts.append({
            "chunk_index": idx,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "token_count": count_tokens(chunk_text),
            "page_numbers": page_numbers,
            "page_range": page_range,
            "document_id": document_id,
            "chunk_type": "summary",
        })
        
        current_position = chunk_start + 1
    
    return chunk_dicts
