"""
app/core/config.py
------------------
Centralised settings loaded from .env via pydantic-settings.
Every module imports `settings` — never os.getenv() directly.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/text-embedding-004"

    # Neo4j
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # Agent
    max_search_results: int = 5
    max_graph_hops: int = 3

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
