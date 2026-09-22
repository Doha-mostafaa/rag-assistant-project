from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    vector_store_path: str = str(PROJECT_ROOT / "data" / "vector_store")
    collection_name: str = "pytorch_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"
    retrieval_n_results: int = 4
    ollama_host: str = "http://localhost:11434"

settings = Settings()