"""Summarization endpoint using LangGraph."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph_pipeline import create_summarization_graph, DocumentState
from .upload import documents_db
from ..config import get_settings

router = APIRouter()
settings = get_settings()


class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    document_id: str


@router.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    """
    Trigger summarization for a document using LangGraph map-reduce.
    
    This runs the LangGraph workflow:
    1. Distribute chunks
    2. Map: Parallel summarization of each chunk
    3. Reduce: Synthesize final summary
    """
    doc_id = request.document_id
    
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[doc_id]
    
    # Check if already summarized
    if doc_data.get("summary"):
        return JSONResponse(
            status_code=200,
            content={
                "document_id": doc_id,
                "status": "already_summarized",
                "summary": doc_data["summary"],
            }
        )
    
    try:
        # Create initial state
        initial_state: DocumentState = {
            "document_id": doc_id,
            "document_metadata": doc_data["metadata"],
            "large_chunks": doc_data["summary_chunks"],
            "total_chunks": 0,
            "chunk_summaries": [],
            "summaries_completed": 0,
            "final_summary": "",
            "status": "distributing",
            "error_message": None,
        }
        
        # Create and run graph
        graph = create_summarization_graph()
        
        # Run the workflow asynchronously
        config = {"configurable": {"thread_id": doc_id}}
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # Store summary
        documents_db[doc_id]["summary"] = final_state["final_summary"]
        documents_db[doc_id]["status"] = "summarized"
        
        return JSONResponse(
            status_code=200,
            content={
                "document_id": doc_id,
                "status": "complete",
                "summary": final_state["final_summary"],
                "chunks_processed": final_state["summaries_completed"],
            }
        )
        
    except Exception as e:
        documents_db[doc_id]["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")


@router.get("/summarize/{document_id}/status")
async def get_summarization_status(document_id: str):
    """Get summarization status for a document."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[document_id]
    
    return {
        "document_id": document_id,
        "status": doc_data["status"],
        "has_summary": doc_data["summary"] is not None,
        "summary": doc_data.get("summary"),
    }
