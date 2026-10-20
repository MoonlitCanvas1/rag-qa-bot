"""Application settings, loaded from environment variables and an optional .env file."""

import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Credentials (optional at load time so `ingest`/`reset` work without a key)
    anthropic_api_key: SecretStr | None = None

    # Storage
    chroma_dir: Path = Path("./chroma_db")
    data_dir: Path = Path("./data")
    collection_name: str = "documents"

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4-20250514"

    # Chunking
    chunk_size: int = Field(default=1000, ge=50)
    chunk_overlap: int = Field(default=200, ge=0)

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=50)
    min_score: float = Field(default=0.30, ge=-1.0, le=1.0)

    # Generation
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    """Send logs to stderr so stdout stays clean for answers."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # Quiet noisy third-party loggers unless the user asks for DEBUG.
    if level.upper() != "DEBUG":
        for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
