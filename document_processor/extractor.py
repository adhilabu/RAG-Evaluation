"""PDF text extraction using pypdf."""
from pypdf import PdfReader
from typing import List, Dict
from pathlib import Path


def extract_pdf_text(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from PDF with page-level metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing page data:
        - page_number: 1-indexed page number
        - text: Extracted text content
        - char_count: Character count for the page
        - word_count: Word count for the page
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        
        # Calculate metrics
        char_count = len(text)
        word_count = len(text.split())
        
        pages_data.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": text,
            "char_count": char_count,
            "word_count": word_count,
        })
    
    return pages_data


def get_pdf_metadata(pdf_path: str) -> Dict[str, any]:
    """
    Extract PDF metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with metadata (title, author, page count, etc.)
    """
    reader = PdfReader(pdf_path)
    
    # Get metadata from PDF info
    info = reader.metadata if reader.metadata else {}
    
    metadata = {
        "title": info.get("/Title", "Unknown") if isinstance(info.get("/Title"), str) else "Unknown",
        "author": info.get("/Author", "Unknown") if isinstance(info.get("/Author"), str) else "Unknown",
        "subject": info.get("/Subject", "") if isinstance(info.get("/Subject"), str) else "",
        "page_count": len(reader.pages),
        "file_path": str(pdf_path),
    }
    
    return metadata
