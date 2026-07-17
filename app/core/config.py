"""Central configuration, loaded from environment / .env (pydantic-settings)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- OpenRouter (the only paid dependency) ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    reasoning_model: str = "deepseek/deepseek-chat-v3-0324:free"
    utility_model: str = "deepseek/deepseek-chat-v3-0324:free"

    # --- Storage ---
    database_url: str = "postgresql://eldostor:eldostor@db:5432/eldostor"

    # --- Local embeddings / rerank (free, open-source) ---
    embedding_model: str = "intfloat/multilingual-e5-base"
    enable_rerank: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Agent loop ---
    max_agent_iterations: int = 6

    # --- Contract generation (M5) ---
    # A whole bilingual contract is emitted as one submit_contract tool call, so the
    # completion needs plenty of room.
    generation_max_tokens: int = 8000
    max_generation_iterations: int = 8


settings = Settings()
