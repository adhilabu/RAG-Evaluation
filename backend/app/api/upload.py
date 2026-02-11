"""Upload endpoint for document processing."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
from datetime import datetime

from document_processor import extract_pdf_text, clean_text
from document_processor.extractor import get_pdf_metadata
from document_processor.cleaner import clean_pages
from document_processor.chunker import create_rag_chunks, create_summary_chunks
from rag_storage import QdrantManager, generate_embeddings
from ..config import get_settings

router = APIRouter()
settings = get_settings()

# In-memory storage for document metadata (in production, use a database)
documents_db = {}

# Initialize Qdrant
qdrant_manager = QdrantManager(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    This endpoint:
    1. Saves the PDF file
    2. Extracts and cleans text
    3. Creates dual chunks (RAG + Summarization)
    4. Generates embeddings and stores in Qdrant
    5. Returns document ID for further processing
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}.pdf"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract text from PDF
        pages_data = extract_pdf_text(str(file_path))
        
        # Get PDF metadata
        pdf_metadata = get_pdf_metadata(str(file_path))
        
        # Clean text
        cleaned_pages = clean_pages(pages_data)
        
        # Create RAG chunks (small)
        rag_chunks = create_rag_chunks(
            cleaned_pages,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            document_id=doc_id,
        )
        
        # Create summary chunks (large)
        summary_chunks = create_summary_chunks(
            cleaned_pages,
            chunk_size=settings.summary_chunk_size,
            chunk_overlap=settings.summary_chunk_overlap,
            document_id=doc_id,
        )
        
        # Generate embeddings for RAG chunks
        rag_texts = [chunk["text"] for chunk in rag_chunks]
        embeddings = generate_embeddings(
            rag_texts,
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        
        # Ensure collection exists
        qdrant_manager.create_collection(vector_size=1536)
        
        # Store in Qdrant
        point_ids = qdrant_manager.add_documents(rag_chunks, embeddings)
        
        # Store metadata in database
        documents_db[doc_id] = {
            "document_id": doc_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
            "metadata": pdf_metadata,
            "rag_chunks_count": len(rag_chunks),
            "summary_chunks_count": len(summary_chunks),
            "summary_chunks": summary_chunks,  # Store for summarization
            "status": "uploaded",
            "summary": None,
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "document_id": doc_id,
                "filename": file.filename,
                "page_count": pdf_metadata["page_count"],
                "rag_chunks": len(rag_chunks),
                "summary_chunks": len(summary_chunks),
                "status": "uploaded",
                "message": "Document uploaded and indexed successfully",
            }
        )
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    documents = []
    for doc_id, doc_data in documents_db.items():
        documents.append({
            "document_id": doc_id,
            "filename": doc_data["filename"],
            "page_count": doc_data["metadata"]["page_count"],
            "uploaded_at": doc_data["uploaded_at"],
            "status": doc_data["status"],
            "has_summary": doc_data["summary"] is not None,
        })
    
    return {"documents": documents, "total": len(documents)}


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return documents_db[document_id]
