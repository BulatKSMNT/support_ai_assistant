from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    api_key: SecretStr
    data_dir: Path = Path("data/local")
    search_mode: Literal["lexical", "vector", "hybrid"] = "hybrid"
    generation_mode: Literal["ollama", "extractive"] = "ollama"
    ollama_url: str = "http://127.0.0.1:11435"
    chat_model: str = "qwen2.5:1.5b"
    embedding_model: str = "embeddinggemma"
    qdrant_url: str = ""
    qdrant_api_key: SecretStr | None = None
    collection: str = "support_knowledge"
    chunk_size: int = Field(default=900, ge=200, le=1800)
    chunk_overlap: int = Field(default=120, ge=0, le=400)
    top_k: int = Field(default=4, ge=1, le=8)
    candidate_k: int = Field(default=16, ge=4, le=50)
    min_vector_score: float = Field(default=0.45, ge=0, le=1)
    context_score_ratio: float = Field(default=0.6, ge=0, le=1)
    max_document_chars: int = Field(default=100_000, ge=1000, le=500_000)
    max_upload_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)
    max_total_chunks: int = Field(default=5000, ge=100, le=20_000)
    max_parallel_generations: int = Field(default=2, ge=1, le=8)
    upstream_timeout: float = Field(default=180, ge=1, le=600)
    upstream_attempts: int = Field(default=2, ge=1, le=3)

    @field_validator("api_key")
    @classmethod
    def secure_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if len(key) < 24 or key.startswith(("replace-", "change-")):
            raise ValueError("Generate an API key first: python3 scripts/setup.py")
        return value

    @field_validator("ollama_url", "qdrant_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        if not value:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("Expected an http(s) service URL without credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_limits(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self

    @property
    def uses_vectors(self) -> bool:
        return self.search_mode != "lexical"

    @property
    def index_signature(self) -> str:
        # Prevent silent reuse of incompatible vectors/chunking rules.
        return (
            f"v1:{self.uses_vectors}:{self.embedding_model}:{self.chunk_size}:{self.chunk_overlap}"
        )
