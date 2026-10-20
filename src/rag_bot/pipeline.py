"""Glue: ingest -> retrieve -> generate."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_bot.config import Settings
from rag_bot.generate import Answer, Generator
from rag_bot.ingest import IngestStats, ingest_path
from rag_bot.retrieve import Retriever, get_vectorstore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the stages. Heavy resources (embeddings, Chroma, API client) are lazy."""

    def __init__(
        self,
        settings: Settings,
        *,
        store_factory: Callable[[], Any] | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings
        self._store_factory = store_factory or (lambda: get_vectorstore(settings))
        self._store: Any | None = None
        self._generator = generator or Generator(settings)

    @property
    def store(self) -> Any:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def ingest(self, path: Path | None = None) -> IngestStats:
        target = path or self.settings.data_dir
        return ingest_path(
            target,
            self.store,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            base_dir=self.settings.data_dir,
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> Answer:
        retriever = Retriever(self.store, self.settings.top_k, self.settings.min_score)
        chunks = retriever.retrieve(question, top_k=top_k, min_score=min_score)
        logger.info("Answering with %d context chunks", len(chunks))
        return self._generator.generate(question, chunks)

    def reset(self) -> None:
        """Delete the collection. The next access recreates an empty one."""
        self.store.delete_collection()
        self._store = None
        logger.info("Collection %r deleted", self.settings.collection_name)
