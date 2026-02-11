"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import upload, summarize, query
from .config import get_settings

settings = get_settings()

app = FastAPI(
    title="Document Processing API",
    description="Map-Reduce Summarization and RAG Storage System",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(summarize.router, prefix="/api/v1", tags=["Summarization"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
