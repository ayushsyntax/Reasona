# Flow:
# 1. Quickly detects file type and extracts text content.
# 2. Supports PDF, DOCX, and TXT formats using lightweight loaders.
# 3. Returns clean text ready for vectorization or retrieval.

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
import tempfile, os

def process_file(content: bytes, filename: str) -> str:
    """Fast text extraction from supported files."""
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        loader = (
            PyMuPDFLoader(path) if ext == ".pdf"
            else UnstructuredWordDocumentLoader(path) if ext == ".docx"
            else TextLoader(path) if ext == ".txt"
            else None
        )
        if not loader:
            raise ValueError(f"Unsupported file type: {filename}")
        return "\n".join(doc.page_content.strip() for doc in loader.load())
    except Exception as e:
        raise ValueError(f"File processing failed: {e}")
    finally:
        os.remove(path)
