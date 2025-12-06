# Flow:
# 1. Defines and loads configuration settings using Pydantic.
# 2. Reads environment variables from a .env file.
# 3. Makes settings available throughout the backend and frontend.

from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    llm_provider: Literal["openai", "google", "ollama"] = "ollama"
    model_name: str = "llama3.2"
    
    openai_api_key: str = ""
    google_api_key: str = ""
    
    ollama_host: str = "http://localhost:11434"
    
    chroma_path: str = "./data/chroma"
    upload_path: str = "./data/uploads"

    class Config:
        env_file = ".env"

settings = Settings()
