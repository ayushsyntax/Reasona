# Flow:
# 1. Detects file type and uses loaders to extract text.
# 2. Supports PDF, DOCX, and TXT file formats.
# 3. Returns extracted text for further processing or indexing.

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
import tempfile
import os

def process_file(content: bytes, filename: str) -> str:
    """Load and extract text from supported document types using LangChain loaders."""
    # Save file temporarily for LangChain loaders
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        if filename.endswith('.pdf'):
            loader = PyMuPDFLoader(tmp_path)
        elif filename.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(tmp_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        # Load and combine document text
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text

    except Exception as e:
        raise ValueError(f"File processing failed: {str(e)}")

    finally:
        # Clean up temporary file
        os.remove(tmp_path)
