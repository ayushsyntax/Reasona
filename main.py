# Flow:
# 1. Initializes the FastAPI backend and core RAG components.
# 2. Exposes endpoints for querying, uploading, and health checks.
# 3. Handles interaction between the frontend and the RAG engine.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.rag_engine import HyDE_SEAL_Engine
from core.vectorstore import VectorStoreManager
from core.models import QueryRequest, QueryResponse
from core.config import settings
from core.ingest import process_file

app = FastAPI(
    title="Reasona API",
    description="Self-correcting RAG system with HyDE",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_manager = VectorStoreManager(settings.chroma_path)
engine = HyDE_SEAL_Engine(vector_manager)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main endpoint for document-based queries"""
    try:
        result = engine.process_query(
            question=request.question,
            provider=request.provider,
            model=request.model
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index PDF, DOCX, or TXT documents"""
    try:
        content = await file.read()
        text = process_file(content, file.filename)
        
        vector_manager.add_documents(
            texts=[text],
            metadatas=[{"filename": file.filename}]
        )
        
        return {
            "message": f"Successfully uploaded {file.filename}",
            "chunks": len(text) // 500  
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "provider": settings.llm_provider}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
