# Flow:
# 1. Defines Pydantic models for request and response data.
# 2. Structures queries, corrections, and responses for consistency.
# 3. Ensures type safety and clear communication between backend and frontend.

from pydantic import BaseModel
from typing import List, Dict, Optional

class QueryRequest(BaseModel):
    question: str
    provider: str = "ollama"
    model: str = "qwen3:1.7b"

class SelfEdit(BaseModel):
    original_chunk: str
    improved_chunk: str
    qa_pairs: List[Dict[str, str]]

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]
    was_corrected: bool = False
    self_edit_performed: bool = False
    meta: Optional[Dict] = None
