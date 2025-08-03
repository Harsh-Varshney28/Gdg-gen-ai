"""
FastAPI backend for RAG Question-Answering Assistant
"""
import logging
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import Config
from rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG QA Assistant API",
    description="A Retrieval-Augmented Generation system for document-based question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized for API")
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1)
    n_results: Optional[int] = Field(5, description="Number of similar chunks to retrieve", ge=1, le=20)
    use_groq: Optional[bool] = Field(True, description="Whether to use Groq API (True) or HuggingFace (False)")

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    context: str
    retrieved_chunks: int
    sources: List[str]
    processing_time: float

class IngestionRequest(BaseModel):
    clear_existing: Optional[bool] = Field(False, description="Whether to clear existing documents before ingesting")

class IngestionResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_added: int
    processing_time: Optional[float] = None

class SystemStatus(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    llm_services: Dict[str, bool]
    available_sources: List[str]
    config: Dict[str, Any]

# API Routes

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Question-Answering Assistant API",
        "version": "1.0.0",
        "status": "operational" if rag_pipeline else "initialization_failed",
        "docs": "/docs"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        status = rag_pipeline.get_system_status()
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "system_status": status
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        result = rag_pipeline.query(
            question=request.question,
            n_results=request.n_results,
            use_groq=request.use_groq
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("answer", "Query failed"))
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/upload", tags=["Documents"])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload PDF files to the documents directory
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        documents_path = Path(Config.DOCUMENTS_PATH)
        documents_path.mkdir(exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            # Validate file extension
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Got: {file.filename}"
                )
            
            # Save file
            file_path = documents_path / file.filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "path": str(file_path),
                "size": len(content)
            })
        
        return {
            "success": True,
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/process", response_model=IngestionResponse, tags=["Documents"])
async def process_documents(request: IngestionRequest = None):
    """
    Process uploaded documents and add them to the vector store
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        clear_existing = request.clear_existing if request else False
        
        result = rag_pipeline.ingest_documents(clear_existing=clear_existing)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return IngestionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """
    Get current system status and statistics
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        status = rag_pipeline.get_system_status()
        
        if status.get("status") == "error":
            raise HTTPException(status_code=500, detail=status.get("error", "Unknown error"))
        
        return SystemStatus(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/sources", tags=["Documents"])
async def list_sources():
    """
    Get list of available document sources
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        sources = rag_pipeline.vector_store.get_sources()
        return {
            "success": True,
            "sources": sources,
            "count": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Sources endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")

@app.delete("/documents/{source_name}", tags=["Documents"])
async def remove_document(source_name: str):
    """
    Remove a specific document from the vector store
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        result = rag_pipeline.remove_document(source_name)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove document endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")

@app.delete("/documents", tags=["Documents"])
async def clear_all_documents():
    """
    Clear all documents from the vector store
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        result = rag_pipeline.clear_all_documents()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear documents endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.get("/documents/{source_name}/summary", tags=["Documents"])
async def get_document_summary(source_name: str):
    """
    Get a summary of a specific document
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    try:
        summary = rag_pipeline.get_document_summary(source_name)
        
        if summary.startswith("No document found") or summary.startswith("Failed to generate"):
            raise HTTPException(status_code=404, detail=summary)
        
        return {
            "success": True,
            "source": source_name,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document summary: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# Run the application
if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
