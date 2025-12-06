# Flow:
# 1. Initializes a fast persistent Chroma vector database.
# 2. Uses lightweight HuggingFace embeddings for rapid encoding.
# 3. Enables quick add/retrieve operations for RAG pipelines.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from typing import List, Optional

class VectorStoreManager:
    """Fast and efficient local vector database manager."""

    def __init__(self, path: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def get_retriever(self, collection_name: str = "docs"):
        """Fast retriever for top-k relevant documents."""
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        ).as_retriever(search_kwargs={"k": 3})

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None, collection_name: str = "docs"):
        """Quickly add text chunks into vectorstore."""
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        vectorstore.add_texts(texts=texts, metadatas=metadatas or [{}] * len(texts))
