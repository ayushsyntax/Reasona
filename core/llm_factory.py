# 1. Provides a factory function to select and initialize LLMs.
# 2. Chooses the model based on provider (Ollama, OpenAI, or Google).
# 3. Returns a ready-to-use language model instance for the app.

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from core.config import settings
import os


def get_llm(provider: str, model: str, temperature: float = 0.1) -> BaseLanguageModel:
    """Return the appropriate language model based on provider."""
    
    provider = provider.lower().strip() if provider else settings.llm_provider
    model = model or settings.model_name

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        if not api_key or api_key == "your_openai_key_here":
            raise ValueError("OpenAI API key missing in environment or .env file.")
        return ChatOpenAI(model=model, temperature=temperature)
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY", settings.google_api_key)
        if not api_key:
            raise ValueError("Google API key missing in environment or .env file.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    
    else:  
        if not model or "llama" in model.lower():
            model = "qwen3:1.7b"
        return ChatOllama(model=model, temperature=0.1, num_ctx=2048)

