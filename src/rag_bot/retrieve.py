"""Embeddings, vector store construction, and top-k retrieval with a score filter."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from rag_bot.config import Settings

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    score: float  # cosine similarity; higher is better
    page: int | None = None
    chunk_index: int | None = None


@lru_cache(maxsize=1)
def get_embeddings(model_name: str) -> "Embeddings":
    """Load the sentence-transformers model once per process."""
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(settings: Settings) -> "Chroma":
    """Open (or create) the persistent Chroma collection using cosine distance."""
    from langchain_chroma import Chroma

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def distance_to_score(distance: float) -> float:
    """Convert Chroma cosine distance (1 - cosine similarity) to a similarity score."""
    return 1.0 - distance


def filter_by_score(chunks: list[RetrievedChunk], min_score: float) -> list[RetrievedChunk]:
    """Keep chunks whose score is at least ``min_score`` (inclusive)."""
    return [c for c in chunks if c.score >= min_score]


def _to_chunk(doc: Document, score: float) -> RetrievedChunk:
    meta = doc.metadata
    page = meta.get("page")
    index = meta.get("chunk_index")
    return RetrievedChunk(
        content=doc.page_content,
        source=str(meta.get("source", "unknown")),
        score=score,
        page=int(page) if page is not None else None,
        chunk_index=int(index) if index is not None else None,
    )


class Retriever:
    """Top-k similarity search followed by a minimum-score filter."""

    def __init__(self, store: Any, top_k: int, min_score: float) -> None:
        self._store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score

        results: list[tuple[Document, float]] = self._store.similarity_search_with_score(
            query, k=k
        )
        scored = [_to_chunk(doc, distance_to_score(dist)) for doc, dist in results]
        kept = filter_by_score(scored, threshold)
        kept.sort(key=lambda c: c.score, reverse=True)

        logger.debug(
            "Retrieved %d chunks, kept %d with score >= %.2f", len(scored), len(kept), threshold
        )
        return kept
