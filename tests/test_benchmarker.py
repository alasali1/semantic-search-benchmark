"""Tests for relevance judging, metrics and the BM25 index (no embedding model needed)."""

from pathlib import Path

import numpy as np
import pytest

from semantic_search_benchmarker.benchmarker import (
    BenchmarkConfig,
    EvalDataset,
    EvalQuery,
    ResolvedSpan,
    evaluate_config,
    evaluate_ranking,
    header_target_matches,
    load_corpus,
    load_eval_dataset,
    matched_targets,
    resolve_spans,
    usable_targets,
)
from semantic_search_benchmarker.chunking import DocumentChunk, FixedSizeChunker, StructureAwareChunker
from semantic_search_benchmarker.indexer import BM25Index, SearchResult, bm25_tokenize, l2_normalize


def _result(chunk_id: str, rank: int, paths: list[list[str]], doc: str = "a.md") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        rank=rank,
        score=1.0,
        content="",
        parent_doc=doc,
        metadata={"section_paths": paths},
        retriever="test",
    )


def test_header_matching_by_title_and_path_suffix() -> None:
    paths = [["Ops", "Backups", "Snapshots"]]
    assert header_target_matches("snapshots", paths)
    assert header_target_matches("Backups > Snapshots", paths)
    assert not header_target_matches("Restore > Snapshots", paths)
    assert not header_target_matches("Backups", paths)  # parent title is not the section


def test_evaluate_ranking_metrics() -> None:
    query = EvalQuery(query_id="q", query="x", relevant_headers=["A", "B"])
    results = [
        _result("c1", 1, [["Other"]]),
        _result("c2", 2, [["A"]]),
        _result("c3", 3, [["A"]]),
        _result("c4", 4, [["B"]]),
    ]
    ev = evaluate_ranking(query, results, {"header:A", "header:B"}, ks=[1, 2, 5])
    assert ev.first_relevant_rank == 2
    assert ev.recall_at_k == {1: 0.0, 2: 0.5, 5: 1.0}


def test_source_doc_restricts_header_matches() -> None:
    query = EvalQuery(query_id="q", query="x", relevant_headers=["Install"], source_doc="b.md")
    results = [_result("c1", 1, [["Install"]], doc="a.md"), _result("c2", 2, [["Install"]], doc="b.md")]
    ev = evaluate_ranking(query, results, {"header:Install"}, ks=[1, 2])
    assert ev.first_relevant_rank == 2


def test_query_requires_targets() -> None:
    with pytest.raises(ValueError):
        EvalQuery(query_id="q", query="x")


def test_bm25_tokenizer_strips_punctuation() -> None:
    assert bm25_tokenize("Set `max_tokens`, e.g. MAX-Tokens!") == ["set", "max", "tokens", "e", "g", "max", "tokens"]


def test_bm25_ranks_keyword_match_first_and_drops_non_matches() -> None:
    chunks = [
        DocumentChunk(chunk_id=f"c{i}", content=t, parent_doc="d.md")
        for i, t in enumerate(["snapshot restore guide", "rate limiting headers", "kubernetes helm chart"])
    ]
    index = BM25Index()
    index.index(chunks)
    results = index.search("How do I restore a snapshot?", k=3)
    assert [r.chunk_id for r in results] == ["c0"]
    assert results[0].rank == 1 and results[0].retriever == "bm25"
    assert index.search("", k=3) == []


def test_l2_normalize_unit_rows_and_zero_safe() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(out[1], [0.0, 0.0])


WRAPPED_CORPUS = {
    "ops.md": (
        "# Ops\n\n## Snapshots\n\nTake a snapshot backup of the collection\nwithout stopping writes.\n\n"
        "## Scaling\n\nAdd replicas for throughput.\n"
    )
}
SNAPSHOT_SPAN = "snapshot backup of the collection without stopping writes"


def test_evaluate_config_end_to_end_with_header_and_span_targets() -> None:
    dataset = EvalDataset(
        queries=[
            EvalQuery(query_id="h", query="snapshot backup", relevant_headers=["Snapshots"]),
            EvalQuery(query_id="s", query="replicas", relevant_spans=["Add replicas for throughput."]),
        ]
    )
    config = BenchmarkConfig("bm25", StructureAwareChunker(), BM25Index)
    result = evaluate_config(config, WRAPPED_CORPUS, dataset, ks=[1, 3])
    assert result.num_chunks == 2
    assert result.num_queries_evaluated == 2 and result.num_queries_skipped == 0
    assert result.hit_rate[1] == 1.0 and result.mrr[1] == 1.0 and result.recall[3] == 1.0
    assert result.latency_p95_ms >= result.latency_p50_ms >= 0


@pytest.mark.parametrize("key", ["relevant_chunk_ids", "target_chunk_ids"])
def test_chunk_id_targets_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="chunk-ID targets are no longer supported"):
        EvalQuery.model_validate({"query_id": "q", "query": "x", key: ["ops.md::structure::0001"]})


def test_span_resolves_across_hard_wrapped_lines() -> None:
    dataset = EvalDataset(queries=[EvalQuery(query_id="q", query="x", relevant_spans=[SNAPSHOT_SPAN])])
    resolve_spans(dataset, WRAPPED_CORPUS)
    (span,) = dataset.queries[0].resolved_spans
    assert span.doc == "ops.md"
    assert WRAPPED_CORPUS["ops.md"][span.start_char : span.end_char] == (
        "snapshot backup of the collection\nwithout stopping writes"
    )


def test_missing_span_raises_instead_of_silently_matching_nothing() -> None:
    dataset = EvalDataset(queries=[EvalQuery(query_id="q", query="x", relevant_spans=["not in any doc"])])
    with pytest.raises(ValueError, match="span not found"):
        resolve_spans(dataset, WRAPPED_CORPUS)


def test_ambiguous_span_raises_unless_source_doc_disambiguates() -> None:
    corpus = {"a.md": "# A\n\nRun the installer.\n", "b.md": "# B\n\nRun the installer.\n"}
    ambiguous = EvalDataset(queries=[EvalQuery(query_id="q", query="x", relevant_spans=["Run the installer."])])
    with pytest.raises(ValueError, match="matches 2 places"):
        resolve_spans(ambiguous, corpus)
    scoped = EvalDataset(
        queries=[EvalQuery(query_id="q", query="x", relevant_spans=["Run the installer."], source_doc="b.md")]
    )
    resolve_spans(scoped, corpus)
    assert scoped.queries[0].resolved_spans[0].doc == "b.md"


def test_unresolved_spans_are_an_error_not_a_silent_skip() -> None:
    query = EvalQuery(query_id="q", query="x", relevant_spans=["anything"])
    with pytest.raises(ValueError, match="unresolved"):
        usable_targets(query)


def test_span_needs_half_its_characters_inside_the_chunk() -> None:
    query = EvalQuery(
        query_id="q",
        query="x",
        relevant_spans=["whatever"],
        resolved_spans=[ResolvedSpan(doc="a.md", start_char=100, end_char=200)],
    )

    def hit(start: int, end: int, doc: str = "a.md") -> bool:
        result = SearchResult(
            chunk_id="c", rank=1, score=1.0, content="", parent_doc=doc,
            metadata={"start_char": start, "end_char": end}, retriever="test",
        )  # fmt: skip
        return "span:0" in matched_targets(result, query)

    assert hit(150, 400)  # exactly half of the span (50 of 100 characters)
    assert not hit(151, 400)  # one character short of half
    assert hit(0, 300)  # contains the whole span
    assert not hit(0, 100) and not hit(200, 300)  # touching but not overlapping
    assert not hit(100, 200, doc="b.md")  # right offsets, wrong document


def test_span_matching_without_chunk_offsets_is_an_error() -> None:
    query = EvalQuery(
        query_id="q",
        query="x",
        relevant_spans=["whatever"],
        resolved_spans=[ResolvedSpan(doc="a.md", start_char=0, end_char=10)],
    )
    with pytest.raises(ValueError, match="start_char/end_char"):
        matched_targets(_result("c1", 1, [], doc="a.md"), query)


def test_span_ground_truth_survives_any_chunking() -> None:
    """The same answer key must be satisfiable by every chunking strategy."""
    dataset = EvalDataset(queries=[EvalQuery(query_id="q", query="x", relevant_spans=[SNAPSHOT_SPAN])])
    resolve_spans(dataset, WRAPPED_CORPUS)
    query = dataset.queries[0]
    for chunker in (StructureAwareChunker(), FixedSizeChunker(8, 2), FixedSizeChunker(16, 4), FixedSizeChunker(64, 8)):
        chunks = chunker.chunk_documents(WRAPPED_CORPUS)
        hits = [c for c in chunks if matched_targets(SearchResult.from_chunk(c, 1, 0.0, "t"), query)]
        assert hits, chunker.strategy


def test_every_query_is_scored_by_every_chunking_strategy() -> None:
    dataset = EvalDataset(
        queries=[
            EvalQuery(query_id="a", query="snapshot backup", relevant_spans=[SNAPSHOT_SPAN]),
            EvalQuery(query_id="b", query="replicas", relevant_headers=["Scaling"]),
        ]
    )
    for chunker in (StructureAwareChunker(), FixedSizeChunker(16, 4), FixedSizeChunker(256, 32)):
        result = evaluate_config(BenchmarkConfig("bm25", chunker, BM25Index), WRAPPED_CORPUS, dataset, ks=[3])
        assert result.num_queries_evaluated == 2 and result.num_queries_skipped == 0


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(not (DATA_DIR / "eval_queries.json").exists(), reason="sample data not present")
def test_sample_eval_set_is_answerable_by_the_structure_chunker() -> None:
    """Every sample query's ground truth must exist in the sample corpus (no typos)."""
    corpus = load_corpus(DATA_DIR / "docs")
    dataset = load_eval_dataset(DATA_DIR / "eval_queries.json", corpus)
    chunks = StructureAwareChunker().chunk_documents(corpus)
    for query in dataset.queries:
        results = [SearchResult.from_chunk(c, 1, 0.0, "t") for c in chunks]
        satisfied = set().union(*(matched_targets(r, query) for r in results))
        assert satisfied == usable_targets(query), query.query_id
