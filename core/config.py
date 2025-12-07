# 1. Defines and loads configuration settings using Pydantic.
# 2. Reads environment variables from a .env file.
# 3. Makes settings available throughout the backend and frontend

from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Centralized configuration for Reasona."""

    llm_provider: Literal["openai", "google", "ollama"] = "ollama"
    model_name: str = "qwen3:1.7b"

    openai_api_key: str | None = None
    google_api_key: str | None = None

    ollama_host: str = "http://localhost:11434"

    chroma_path: str = "./data/chroma"
    upload_path: str = "./data/uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
