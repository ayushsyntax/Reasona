# Flow:
# 1. Initializes a persistent Chroma vector database for long-term storage.
# 2. Uses HuggingFace MiniLM embeddings for efficient semantic encoding.
# 3. Applies RecursiveCharacterTextSplitter (~1000 tokens, 150 overlap) for structured chunking.
# 4. Prepares and cleans text before embedding to ensure consistency.
# 5. Supports fast add and retrieval operations for RAG-based pipelines.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from typing import List, Optional, Union

class VectorStoreManager:
    def __init__(self, path: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    def get_retriever(self, collection_name: str = "docs"):
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        ).as_retriever(search_kwargs={"k": 3})

    def _prepare_texts(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [d.page_content for d in self._splitter.create_documents([texts])]
        ready = []
        for t in texts:
            if len(t) > 1500:
                ready.extend([d.page_content for d in self._splitter.create_documents([t])])
            else:
                ready.append(t)
        return ready

    def add_documents(
        self,
        texts: Union[str, List[str]],
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "docs",
    ):
        clean_texts = self._prepare_texts(texts)
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        metadatas = metadatas or [{}] * len(clean_texts)
        vectorstore.add_texts(texts=clean_texts, metadatas=metadatas)
