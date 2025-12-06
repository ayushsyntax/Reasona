# Flow:
# 1. Launches a highly optimized FastAPI backend with concurrent RAG components.
# 2. Uses async I/O and background executors for faster document processing.
# 3. Minimizes latency with lightweight endpoints and efficient resource usage.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from core.rag_engine import HyDE_SEAL_Engine
from core.vectorstore import VectorStoreManager
from core.models import QueryRequest, QueryResponse
from core.config import settings
from core.ingest import process_file
import asyncio

app = FastAPI(
    title="Reasona API",
    description="Fast self-correcting RAG system (HyDE + SEAL)",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

vector_manager = VectorStoreManager(settings.chroma_path)
engine = HyDE_SEAL_Engine(vector_manager)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Optimized concurrent query processing"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: engine.process_query(
                question=request.question,
                provider=request.provider,
                model=request.model,
            ),
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Asynchronous document upload and indexing"""
    try:
        content = await file.read()
        filename = file.filename or "unnamed.txt"

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, process_file, content, filename)

        await loop.run_in_executor(
            executor,
            lambda: vector_manager.add_documents(
                [text],
                [{"filename": filename}],
            ),
        )

        return {"message": f"✅ Indexed: {filename}", "chunks": max(1, len(text) // 500)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Quick health endpoint"""
    return {"status": "healthy", "provider": settings.llm_provider}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        loop="uvloop",   
        timeout_keep_alive=5,
    )
