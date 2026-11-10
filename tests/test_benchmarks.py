"""Step 5 QA suite: Hit Rate@K and MRR@K on a deterministic mock ranking with known ground truth.

Each query below is answered by a scripted ranking [R1, R2, ...]. The relevant section is
"Gold", so the position of the first Gold chunk is the query's first relevant rank:

    query        ranking                               first relevant rank
    first        Gold at R1                            1
    third        Gold at R3                            3
    seventh      Gold at R7 (outside K=5)              7
    missed       Gold never retrieved                  None
    duplicate    Gold chunks at R2 and R4              2   (only the first counts)
    tied         every score 0.0, Gold returned 2nd    2   (ties keep the retriever's order)

Expected values, worked by hand (6 queries):

    Hit Rate@1  = 1/6                       (first)
    Hit Rate@3  = 4/6                       (first, third, duplicate, tied)
    Hit Rate@5  = 4/6
    Hit Rate@10 = 5/6                       (+ seventh)
    MRR@1  = 1 / 6                                   = 1/6
    MRR@3  = (1 + 1/3 + 1/2 + 1/2) / 6               = 7/18
    MRR@5  = (1 + 1/3 + 1/2 + 1/2) / 6               = 7/18
    MRR@10 = (1 + 1/3 + 1/7 + 1/2 + 1/2) / 6         = 26/63
"""

from collections.abc import Sequence

import pytest

from semantic_search_benchmarker.benchmarker import (
    BenchmarkConfig,
    BuiltIndex,
    EvalDataset,
    EvalQuery,
    evaluate_index,
    hit_rate_at_k,
    mrr_at_k,
)
from semantic_search_benchmarker.chunking import DocumentChunk, StructureAwareChunker
from semantic_search_benchmarker.indexer import Retriever, SearchResult

KS = [1, 3, 5, 10]
FIRST_RANKS: list[int | None] = [1, 3, 7, None, 2, 2]
EXPECTED_HIT_RATE = {1: 1 / 6, 3: 4 / 6, 5: 4 / 6, 10: 5 / 6}
EXPECTED_MRR = {1: 1 / 6, 3: 7 / 18, 5: 7 / 18, 10: 26 / 63}


# --- the metric functions on their own ---------------------------------------------------


@pytest.mark.parametrize("k", KS)
def test_5_hit_rate_and_mrr_functions_match_hand_computed_values(k: int) -> None:
    """Test 5: metric math on known first-relevant ranks."""
    assert hit_rate_at_k(FIRST_RANKS, k) == pytest.approx(EXPECTED_HIT_RATE[k])
    assert mrr_at_k(FIRST_RANKS, k) == pytest.approx(EXPECTED_MRR[k])


def test_5_missed_ranks_score_zero_without_division_by_zero() -> None:
    assert mrr_at_k([None, None], 10) == 0.0
    assert hit_rate_at_k([None, None], 10) == 0.0
    assert mrr_at_k([7], 5) == 0.0  # retrieved, but outside the top K: no credit
    assert hit_rate_at_k([7], 5) == 0.0
    assert mrr_at_k([], 5) == 0.0
    assert hit_rate_at_k([], 5) == 0.0


def test_5_mrr_is_bounded_and_rewards_rank_1_fully() -> None:
    assert mrr_at_k([1, 1, 1], 1) == 1.0
    assert 0.0 <= mrr_at_k(FIRST_RANKS, 10) <= hit_rate_at_k(FIRST_RANKS, 10) <= 1.0


# --- the same ranking pushed through the real benchmark engine ---------------------------


def _chunk(chunk_id: str, section: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        content=chunk_id,
        parent_doc="mock.md",
        metadata={"section_paths": [[section]], "start_char": 0, "end_char": 0},
    )


CHUNKS = [_chunk("gold", "Gold"), _chunk("gold-2", "Gold")] + [_chunk(f"x{i}", f"Other {i}") for i in range(1, 10)]

# Scripted rankings [R1, R2, ...] per query. Every score is tied at 0.0, so ranks come
# purely from list order, exactly as they would from a real retriever's output order.
RANKINGS: dict[str, list[str]] = {
    "first": ["gold", "x1", "x2"],
    "third": ["x1", "x2", "gold", "x3"],
    "seventh": ["x1", "x2", "x3", "x4", "x5", "x6", "gold", "x7"],
    "missed": ["x1", "x2", "x3"],
    "duplicate": ["x1", "gold", "x2", "gold-2", "x3"],
    "tied": ["x1", "gold", "x2"],
}


class MockRetriever(Retriever):
    """Returns the scripted ranking for each query, with every score tied at 0.0."""

    @property
    def name(self) -> str:
        return "mock"

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._store_chunks(chunks)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        return [
            SearchResult.from_chunk(self._chunks[cid], rank=r, score=0.0, retriever=self.name)
            for r, cid in enumerate(RANKINGS[query][:k], start=1)
        ]


def test_5_benchmark_engine_computes_the_same_values_end_to_end() -> None:
    retriever = MockRetriever()
    retriever.index(CHUNKS)
    config = BenchmarkConfig("mock", StructureAwareChunker(), MockRetriever)
    built = BuiltIndex(config, CHUNKS, retriever, 0.0)
    dataset = EvalDataset(
        queries=[EvalQuery(query_id=q, query=q, relevant_headers=["Gold"]) for q in RANKINGS]
    )

    result = evaluate_index(built, dataset, KS, warmup=False)

    assert [e.first_relevant_rank for e in result.per_query] == FIRST_RANKS
    for k in KS:
        assert result.hit_rate[k] == pytest.approx(EXPECTED_HIT_RATE[k]), k
        assert result.mrr[k] == pytest.approx(EXPECTED_MRR[k]), k
