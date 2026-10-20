import pytest
from langchain_core.documents import Document

from rag_bot.retrieve import Retriever, RetrievedChunk, distance_to_score, filter_by_score


class FakeStore:
    """Returns canned (Document, cosine_distance) pairs and records calls."""

    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results[:k]


def doc(text: str, **meta: object) -> Document:
    return Document(page_content=text, metadata={"source": "a.txt", **meta})


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(content="x", source="a.txt", score=score)


def test_distance_to_score() -> None:
    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(0.25) == pytest.approx(0.75)


def test_filter_by_score_drops_low_and_keeps_boundary() -> None:
    chunks = [chunk(0.9), chunk(0.3), chunk(0.29), chunk(-0.1)]
    kept = filter_by_score(chunks, 0.3)
    assert [c.score for c in kept] == [0.9, 0.3]


def test_filter_by_score_can_return_empty() -> None:
    assert filter_by_score([chunk(0.1), chunk(0.2)], 0.5) == []


def test_retriever_filters_sorts_and_maps_metadata() -> None:
    store = FakeStore(
        [
            (doc("weak", chunk_index=0), 0.9),  # score 0.1 -> dropped
            (doc("ok", page=3, chunk_index=1), 0.5),  # score 0.5
            (doc("best", chunk_index=2), 0.1),  # score 0.9
        ]
    )
    results = Retriever(store, top_k=5, min_score=0.3).retrieve("question")

    assert [r.content for r in results] == ["best", "ok"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].page == 3
    assert results[1].chunk_index == 1
    assert results[0].page is None


def test_retriever_passes_top_k_and_supports_overrides() -> None:
    store = FakeStore([(doc(str(i)), 0.1) for i in range(10)])
    retriever = Retriever(store, top_k=3, min_score=0.0)

    assert len(retriever.retrieve("q")) == 3
    assert len(retriever.retrieve("q", top_k=5)) == 5
    assert store.calls == [("q", 3), ("q", 5)]

    # a stricter override threshold filters everything out (score 0.9 < 0.95)
    assert retriever.retrieve("q", min_score=0.95) == []


def test_blank_query_skips_search() -> None:
    store = FakeStore([(doc("x"), 0.1)])
    assert Retriever(store, top_k=3, min_score=0.0).retrieve("   ") == []
    assert store.calls == []
