# Flow:
# 1. Provides a factory function to select and initialize LLMs.
# 2. Chooses the model based on provider (OpenAI, Google, or Ollama).
# 3. Returns a ready-to-use language model instance for the app.

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

def get_llm(provider: str, model: str, temperature: float = 0.1) -> BaseLanguageModel:
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    
    elif provider == "google":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    
    else:  
        return ChatOllama(model=model, temperature=temperature)
