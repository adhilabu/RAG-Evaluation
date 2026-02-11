"""Text cleaning utilities to remove noise from extracted text."""
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing common noise patterns.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove common header/footer patterns
    # Pattern: "Page X of Y" or "Page X"
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
    
    # Remove multiple consecutive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove form feed characters
    text = text.replace('\f', '')
    
    return text.strip()


def clean_pages(pages_data: List[dict]) -> List[dict]:
    """
    Clean text for all pages.
    
    Args:
        pages_data: List of page dictionaries from extractor
        
    Returns:
        List of page dictionaries with cleaned text
    """
    cleaned_pages = []
    
    for page in pages_data:
        cleaned_page = page.copy()
        cleaned_page["text"] = clean_text(page["text"])
        cleaned_page["char_count"] = len(cleaned_page["text"])
        cleaned_page["word_count"] = len(cleaned_page["text"].split())
        cleaned_pages.append(cleaned_page)
    
    return cleaned_pages
