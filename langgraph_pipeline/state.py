"""State schema for LangGraph document summarization."""
from typing import TypedDict, List, Dict, Optional


class DocumentState(TypedDict):
    """State schema for document summarization workflow."""
    
    # Input
    document_id: str
    document_metadata: Dict[str, any]
    
    # Processing
    large_chunks: List[Dict[str, any]]  # From chunker
    total_chunks: int
    
    # Map phase outputs
    chunk_summaries: List[str]
    summaries_completed: int
    
    # Reduce phase output
    final_summary: str
    
    # Status tracking
    status: str  # "distributing", "mapping", "reducing", "complete", "error"
    error_message: Optional[str]
