# Flow:
# 1. Defines Pydantic models for request and response data.
# 2. Structures queries, self-edits, and responses for type safety.
# 3. Supports consistent data handling across the API.

from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    """User question with provider selection"""
    question: str
    provider: str = "ollama"  
    model: str = "llama3.2"

class SelfEdit(BaseModel):
    """Stores self-correction results"""
    original_chunk: str
    improved_chunk: str
    qa_pairs: List[dict]

class QueryResponse(BaseModel):
    """Complete response with metadata"""
    answer: str
    retrieved_docs: List[str]
    was_corrected: bool = False
    self_edit_performed: bool = False
