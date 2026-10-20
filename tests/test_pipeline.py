from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_bot.config import Settings
from rag_bot.generate import (
    NO_CONTEXT_ANSWER,
    GenerationError,
    Generator,
    extract_citations,
)
from rag_bot.pipeline import RAGPipeline
from rag_bot.retrieve import RetrievedChunk


class FakeStore:
    def __init__(self, results: list[tuple[Document, float]] | None = None) -> None:
        self.results = results or []
        self.docs: dict[str, Document] = {}
        self.deleted = False

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.results[:k]

    def get(self, where: dict[str, Any]) -> dict[str, list[str]]:
        (key, value), = where.items()
        return {"ids": [i for i, d in self.docs.items() if d.metadata.get(key) == value]}

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.docs.pop(i, None)

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.docs.update(zip(ids, documents, strict=True))

    def delete_collection(self) -> None:
        self.deleted = True


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "anthropic_api_key": "test-key",
        "chroma_dir": tmp_path / "chroma",
        "data_dir": tmp_path / "data",
        "min_score": 0.3,
        "top_k": 4,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def mock_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)]
    )
    return client


def hit(text: str, distance: float, **meta: Any) -> tuple[Document, float]:
    return Document(page_content=text, metadata={"source": "guide.pdf", **meta}), distance


def test_ask_calls_llm_with_context_and_returns_cited_sources(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = FakeStore(
        [
            hit("The warranty lasts two years.", 0.1, page=4),
            hit("Shipping takes five days.", 0.2, page=7),
            hit("Irrelevant text.", 0.95),  # below min_score, must not reach the LLM
        ]
    )
    client = mock_client("The warranty is two years [1].")
    pipeline = RAGPipeline(
        settings,
        store_factory=lambda: store,
        generator=Generator(settings, client=client),
    )

    answer = pipeline.ask("How long is the warranty?")

    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    prompt = kwargs["messages"][0]["content"]
    assert "The warranty lasts two years." in prompt
    assert "Shipping takes five days." in prompt
    assert "Irrelevant text." not in prompt
    assert "How long is the warranty?" in prompt

    assert answer.text == "The warranty is two years [1]."
    assert [(c.index, c.source, c.page) for c in answer.citations] == [(1, "guide.pdf", 4)]


def test_ask_without_relevant_chunks_skips_llm(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = mock_client("should not be used")
    pipeline = RAGPipeline(
        settings,
        store_factory=lambda: FakeStore([hit("far away", 0.99)]),
        generator=Generator(settings, client=client),
    )

    answer = pipeline.ask("anything?")

    client.messages.create.assert_not_called()
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.citations == []


def test_min_score_override_is_applied(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = mock_client("ok [1]")
    pipeline = RAGPipeline(
        settings,
        store_factory=lambda: FakeStore([hit("medium match", 0.5)]),  # score 0.5
        generator=Generator(settings, client=client),
    )

    pipeline.ask("q", min_score=0.9)
    client.messages.create.assert_not_called()

    pipeline.ask("q", min_score=0.4)
    client.messages.create.assert_called_once()


def test_missing_api_key_raises_only_when_llm_needed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, anthropic_api_key=None)
    pipeline = RAGPipeline(settings, store_factory=lambda: FakeStore([hit("match", 0.1)]))
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        pipeline.ask("q")

    empty = RAGPipeline(settings, store_factory=lambda: FakeStore())
    assert empty.ask("q").text == NO_CONTEXT_ANSWER  # no key needed when nothing is retrieved


def test_ingest_uses_data_dir_by_default(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir()
    (settings.data_dir / "note.txt").write_text("hello " * 50, encoding="utf-8")
    store = FakeStore()
    pipeline = RAGPipeline(settings, store_factory=lambda: store)

    stats = pipeline.ingest()

    assert stats.files == 1
    assert len(store.docs) == stats.chunks
    assert {d.metadata["source"] for d in store.docs.values()} == {"note.txt"}


def test_reset_deletes_collection(tmp_path: Path) -> None:
    store = FakeStore()
    pipeline = RAGPipeline(make_settings(tmp_path), store_factory=lambda: store)
    pipeline.reset()
    assert store.deleted


def test_extract_citations_handles_groups_and_out_of_range() -> None:
    chunks = [
        RetrievedChunk(content=f"text {i}", source=f"s{i}.txt", score=0.5) for i in range(1, 4)
    ]
    citations = extract_citations("A [2][1]. B [1, 3]. C [9]. D [0].", chunks)
    assert [c.index for c in citations] == [1, 2, 3]
