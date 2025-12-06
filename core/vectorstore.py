# Flow:
# 1. Initializes a persistent Chroma vector database.
# 2. Embeds documents using HuggingFace embeddings.
# 3. Supports adding documents and retrieving relevant ones for RAG tasks.

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    """Handles all vector database operations using local embeddings."""
    
    def __init__(self, path: str = "./data/chroma"):
        """Initialize persistent vector storage"""
        self.client = chromadb.PersistentClient(path=path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  
        )
    
    def get_retriever(self, collection_name: str = "docs"):
        """Return a retriever for finding relevant documents"""
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})  
    
    def add_documents(self, texts: list, metadatas: list = None, collection_name: str = "docs"):
        """Add new documents to the collection"""
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas or [{}] * len(texts)
        )
