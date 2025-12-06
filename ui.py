# Flow:
# 1. Provides a Streamlit interface for document upload and querying.
# 2. Connects to the FastAPI backend to process and retrieve answers.
# 3. Displays results, retrieved context, and learning feedback to the user.

import streamlit as st
import requests
import time

PROVIDERS = {
    "Local (Ollama)": {"provider": "ollama", "model": "llama3.2"},
    "Cloud (OpenAI)": {"provider": "openai", "model": "gpt-4o"},
    "Cloud (Google)": {"provider": "google", "model": "gemini-2.5-flash"}
}


st.set_page_config(
    page_title="Reasona RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Reasona: Self-Correcting RAG")


with st.sidebar:
    st.header("Configuration")
    
    selected = st.selectbox(
        "Select LLM Provider",
        list(PROVIDERS.keys()),
        help="Local runs offline; Cloud requires an API key."
    )
    config = PROVIDERS[selected]
    
    api_base = st.text_input(
        "API Base (Ollama only)",
        "http://localhost:11434",
        help="Enter the Ollama server address."
    )
    
    st.info(f"Current provider: {selected}")

st.header("Document Upload")
uploaded_file = st.file_uploader(
    "Choose a document (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    help="File will be processed and indexed for retrieval."
)

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
            backend_url = api_base.replace(":11434", ":8000")
            response = requests.post(f"{backend_url}/upload", files=files)
            response.raise_for_status()
            
            result = response.json()
            st.success(f"Document indexed: {result['message']}")
            
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

st.header("Ask Questions")
question = st.text_input(
    "Enter your question",
    placeholder="What information are you looking for?",
    help="The system will search your uploaded documents and self-correct if needed."
)

if question:
    with st.spinner("Searching and generating answer..."):
        try:
            backend_url = api_base.replace(":11434", ":8000")
            response = requests.post(
                f"{backend_url}/query",
                json={
                    "question": question,
                    "provider": config["provider"],
                    "model": config["model"]
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            st.subheader("Answer")
            st.write(result["answer"])
            
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(result["retrieved_docs"][:3]):
                    st.markdown(f"**Document {i+1}**")
                    st.text(doc[:300] + "...")
            
            if result["self_edit_performed"]:
                st.success("✅ The system learned from this interaction.")
            elif result["was_corrected"]:
                st.warning("⚠️ The answer was flagged for review.")
            
        except Exception as e:
            st.error(f"Query failed: {str(e)}")

st.markdown("---")
st.caption("Reasona v1.0 | Self-correcting RAG system")
