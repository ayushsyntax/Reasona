# Flow:
# 1. Provides a fast and clean Streamlit UI for Reasona RAG.
# 2. Enables model selection, document upload, and contextual querying.
# 3. Waits for backend responses without timeout interruptions.

import streamlit as st
import requests
import re
import time
from requests.exceptions import RequestException

API_BASE = "http://localhost:8000"
PROVIDERS = {
    "Local (Ollama)": {"provider": "ollama", "models": ["qwen3:1.7b", "qwen3:4b", "llama3.2"]},
    "Cloud (OpenAI)": {"provider": "openai", "models": ["gpt-4o", "gpt-4-turbo"]},
    "Cloud (Google)": {"provider": "google", "models": ["gemini-2.5-flash", "gemini-2.0-pro"]},
}

st.set_page_config(page_title="Reasona", page_icon="🧠", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = "ollama"
if "model" not in st.session_state:
    st.session_state.model = "qwen3:1.7b"

st.title("🧠 Reasona")
st.caption("Self-correcting RAG • HyDE + SEAL Engine")

with st.sidebar:
    st.header("⚙️ Model Configuration")
    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
    st.session_state.provider = PROVIDERS[provider_name]["provider"]
    model_name = st.selectbox("Model", PROVIDERS[provider_name]["models"], index=0)
    st.session_state.model = model_name
    st.info(f"Active Model: **{st.session_state.model}**")

with st.expander("📂 Upload Documents", expanded=False):
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post(f"{API_BASE}/upload", files=files)  # No timeout
                res.raise_for_status()
                st.success(f"✅ Indexed: {uploaded_file.name}")
            except RequestException as e:
                st.error(f"Upload failed: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        with st.spinner("Generating..."):
            try:
                res = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "provider": st.session_state.provider,
                        "model": st.session_state.model,
                    },  
                )
                res.raise_for_status()
                data = res.json()
                answer = re.sub(r"<think>.*?</think>", "", data.get("answer", ""), flags=re.DOTALL).strip()
                answer = answer if answer else "_No clear answer found._"
                st.markdown(f"### 🧩 Answer\n{answer}")
                if data.get("self_edit_performed"):
                    st.info("🧠 Learned and updated knowledge base.")
                elif data.get("was_corrected"):
                    st.warning("⚠️ Answer was corrected.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.caption(f"⏱️ {round(time.time() - start_time, 2)}s")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Reasona  • Optimized RAG Interface ")
