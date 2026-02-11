"""Query endpoint for semantic search."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from rag_storage import search_documents
from rag_storage.qdrant_client import QdrantManager
from ..config import get_settings

router = APIRouter()
settings = get_settings()

# Initialize Qdrant
qdrant_manager = QdrantManager(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)


class QueryRequest(BaseModel):
    """Request model for search queries."""
    query: str
    limit: int = 5
    document_id: Optional[str] = None
    score_threshold: float = 0.5


@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Perform semantic search across document chunks.
    
    Args:
        query: Search query text
        limit: Maximum number of results
        document_id: Optional filter by specific document
        score_threshold: Minimum similarity score (0-1)
    """
    try:
        results = search_documents(
            query=request.query,
            qdrant_manager=qdrant_manager,
            api_key=settings.openai_api_key,
            limit=request.limit,
            document_id=request.document_id,
            score_threshold=request.score_threshold,
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")


@router.get("/collection/info")
async def get_collection_info():
    """Get Qdrant collection information."""
    try:
        info = qdrant_manager.get_collection_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")
