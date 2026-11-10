"""Retrieval benchmark: chunking strategy x retriever, scored against a labelled query set.

Ground truth
------------
Each query in ``eval_queries.json`` lists one or more *targets*, and both kinds are
defined on the original documents, never on chunk positions, so every chunking strategy
is judged against the same answer key:

* ``relevant_headers`` - section titles (``"Snapshots"``) or header paths
  (``"Backups > Snapshots"``). A retrieved chunk satisfies a header target if it is
  credited with a matching section (see ``section_paths`` in :mod:`.chunking`): it must
  hold at least half of the section body, or about 100 tokens of it.
* ``relevant_spans`` - short verbatim snippets of the source text (whitespace-insensitive,
  so a quote may span a hard-wrapped line). Each span is located in its document when the
  dataset is loaded with a corpus (:func:`resolve_spans`); a span that cannot be found, or
  matches more than once, raises an error instead of silently scoring against nothing. A
  retrieved chunk satisfies a span target if it contains at least half of the span's
  characters.

Chunk IDs (``doc.md::structure::0003``) are positions in one particular chunk list, so
they are *not* accepted as targets: they go stale silently when documents or chunk sizes
change. ``source_doc`` optionally restricts matching to one document, for corpora where
several documents share a header name such as "Installation" or a snippet of text.

Metrics (for each K)
--------------------
* Hit Rate@K: fraction of queries with at least one relevant chunk in the top K.
* Recall@K: mean over queries of (targets satisfied by the top K) / (targets). Recall is
  measured over *targets* rather than relevant chunks; otherwise a strategy that cuts a
  section into many small chunks would be penalised just for producing more chunks.
* MRR@K: mean of 1 / rank of the first relevant chunk (0 if none in the top K).
* Latency: wall-clock time of ``Retriever.search`` per query, including query
  embedding, reported as p50 / p95 / mean in milliseconds after one warm-up query.

Run ``semantic-search-benchmarker --help`` (or ``python -m semantic_search_benchmarker.benchmarker``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from semantic_search_benchmarker.chunking import (
    BaseChunker,
    DocumentChunk,
    FixedSizeChunker,
    StructureAwareChunker,
)
from semantic_search_benchmarker.indexer import (
    BM25Index,
    DenseVectorIndex,
    Embedder,
    OpenAIEmbedder,
    Retriever,
    SearchResult,
    SentenceTransformerEmbedder,
)

logger = logging.getLogger(__name__)

HEADER_PATH_SEP = ">"

SPAN_MIN_COVERAGE = 0.5
"""Share of a ground-truth span's characters a chunk must contain to satisfy the span."""

_CHUNK_ID_KEYS = frozenset({"relevant_chunk_ids", "target_chunk_ids"})


# --------------------------------------------------------------------------------------
# Evaluation dataset
# --------------------------------------------------------------------------------------


class ResolvedSpan(BaseModel):
    """A ground-truth text span located in a source document (character offsets)."""

    model_config = ConfigDict(frozen=True)

    doc: str
    start_char: int
    end_char: int  # exclusive


class EvalQuery(BaseModel):
    """A labelled query. At least one of ``relevant_headers`` / ``relevant_spans`` is required."""

    query_id: str = Field(validation_alias=AliasChoices("query_id", "id"))
    query: str
    relevant_headers: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("relevant_headers", "target_headers"),
    )
    relevant_spans: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("relevant_spans", "target_spans"),
        description="Verbatim snippets of the source text that answer the query.",
    )
    source_doc: str | None = None
    resolved_spans: list[ResolvedSpan] = Field(
        default_factory=list,
        exclude=True,
        description="Filled in by resolve_spans(); one entry per relevant_spans item.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_chunk_id_targets(cls, data: object) -> object:
        if isinstance(data, dict) and _CHUNK_ID_KEYS & data.keys():
            raise ValueError(
                "chunk-ID targets are no longer supported: a chunk ID is a position in one "
                "chunk list and goes stale when documents or chunk sizes change. "
                "Use relevant_headers (section titles) or relevant_spans (text snippets)."
            )
        return data

    @model_validator(mode="after")
    def _require_targets(self) -> EvalQuery:
        if not self.relevant_headers and not self.relevant_spans:
            raise ValueError(f"query {self.query_id!r} has no relevant_headers or relevant_spans")
        if any(not span.strip() for span in self.relevant_spans):
            raise ValueError(f"query {self.query_id!r} has an empty relevant_spans entry")
        return self


class EvalDataset(BaseModel):
    """A collection of labelled queries with unique IDs."""

    name: str = "eval"
    queries: list[EvalQuery]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        ids = [q.query_id for q in self.queries]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate query_id values: {sorted(dupes)}")
        return self


def load_eval_dataset(path: str | Path, corpus: Mapping[str, str] | None = None) -> EvalDataset:
    """Load ``eval_queries.json``: either ``{"queries": [...]}`` or a bare list.

    Pass ``corpus`` (``doc_id -> text``, as from :func:`load_corpus`) to locate every
    ``relevant_spans`` snippet at load time; a span that is missing or ambiguous raises
    ``ValueError`` immediately. Datasets with spans must be resolved before scoring.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"queries": data}
    data.setdefault("name", path.stem)
    dataset = EvalDataset.model_validate(data)
    if corpus is not None:
        resolve_spans(dataset, corpus)
    return dataset


def _doc_matches(parent_doc: str, source_doc: str | None) -> bool:
    return source_doc is None or parent_doc == source_doc or Path(parent_doc).name == source_doc


def resolve_spans(dataset: EvalDataset, corpus: Mapping[str, str]) -> EvalDataset:
    """Locate every ``relevant_spans`` snippet in the corpus (in place) and return the dataset.

    Matching is case-sensitive but ignores differences in whitespace, so a snippet written
    on one line still matches text that is hard-wrapped in the Markdown source.

    Raises:
        ValueError: if a snippet is not found, or matches more than one place (lengthen
            the snippet or set ``source_doc``). A span that silently matched nothing would
            make its query unanswerable for every retriever.
    """
    for query in dataset.queries:
        docs = [d for d in sorted(corpus) if _doc_matches(d, query.source_doc)]
        if query.source_doc and not docs:
            raise ValueError(f"query {query.query_id!r}: source_doc {query.source_doc!r} not in corpus")
        resolved: list[ResolvedSpan] = []
        for span in query.relevant_spans:
            pattern = re.compile(r"\s+".join(re.escape(word) for word in span.split()))
            matches = [(d, m.start(), m.end()) for d in docs for m in pattern.finditer(corpus[d])]
            if not matches:
                raise ValueError(f"query {query.query_id!r}: span not found in corpus: {span!r}")
            if len(matches) > 1:
                where = ", ".join(f"{d}@{s}" for d, s, _ in matches)
                raise ValueError(
                    f"query {query.query_id!r}: span matches {len(matches)} places ({where}); "
                    f"lengthen the snippet or set source_doc: {span!r}"
                )
            doc, start, end = matches[0]
            resolved.append(ResolvedSpan(doc=doc, start_char=start, end_char=end))
        query.resolved_spans = resolved
    return dataset


def load_corpus(docs_dir: str | Path, pattern: str = "**/*.md") -> dict[str, str]:
    """Read Markdown files under ``docs_dir``, keyed by POSIX path relative to it."""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"docs directory not found: {root}")
    corpus = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.glob(pattern))
        if p.is_file()
    }
    if not corpus:
        raise ValueError(f"no files matching {pattern!r} under {root}")
    return corpus


# --------------------------------------------------------------------------------------
# Relevance judging and metrics
# --------------------------------------------------------------------------------------


def _normalize_header(text: str) -> str:
    return " ".join(text.strip().strip("#").lower().split())


def _parse_header_target(target: str) -> tuple[str, ...]:
    return tuple(_normalize_header(part) for part in target.split(HEADER_PATH_SEP))


def header_target_matches(target: str, section_paths: Sequence[Sequence[str]]) -> bool:
    """True if a header target matches any section path.

    ``"Snapshots"`` matches a section whose own title is "Snapshots".
    ``"Backups > Snapshots"`` must match the trailing part of the path, so it does not
    match "Restore > Snapshots". Comparison ignores case and extra whitespace.
    """
    wanted = _parse_header_target(target)
    for path in section_paths:
        normalized = tuple(_normalize_header(p) for p in path)
        if len(normalized) >= len(wanted) and normalized[-len(wanted) :] == wanted:
            return True
    return False


def usable_targets(query: EvalQuery) -> set[str]:
    """The query's targets as ``"header:<title>"`` / ``"span:<index>"`` keys.

    Targets are defined on the source documents, so every chunking strategy is scored
    against the same set; no configuration can drop a query the others keep.

    Raises:
        ValueError: if the query has spans that were never resolved against the corpus.
    """
    if len(query.resolved_spans) != len(query.relevant_spans):
        raise ValueError(
            f"query {query.query_id!r} has unresolved relevant_spans; "
            "call resolve_spans(dataset, corpus) or pass the corpus to load_eval_dataset()"
        )
    targets = {f"header:{h}" for h in query.relevant_headers}
    targets |= {f"span:{i}" for i in range(len(query.relevant_spans))}
    return targets


def matched_targets(result: SearchResult, query: EvalQuery) -> set[str]:
    """Target keys (as in :func:`usable_targets`) that one search result satisfies."""
    hits: set[str] = set()
    meta = result.metadata
    if query.relevant_headers and _doc_matches(result.parent_doc, query.source_doc):
        paths = meta.get("section_paths", [])
        hits |= {f"header:{h}" for h in query.relevant_headers if header_target_matches(h, paths)}
    for i, span in enumerate(query.resolved_spans):
        if span.doc != result.parent_doc:
            continue
        if "start_char" not in meta or "end_char" not in meta:
            raise ValueError(
                f"chunk {result.chunk_id!r} has no start_char/end_char metadata, "
                f"so it cannot be matched against the text spans of query {query.query_id!r}"
            )
        overlap = min(meta["end_char"], span.end_char) - max(meta["start_char"], span.start_char)
        if overlap >= SPAN_MIN_COVERAGE * (span.end_char - span.start_char):
            hits.add(f"span:{i}")
    return hits


def hit_rate_at_k(first_relevant_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of queries whose first relevant result is at rank <= ``k``.

    ``first_relevant_ranks`` holds one 1-indexed rank per query, or ``None`` for a query
    where nothing relevant was retrieved. An empty input gives 0.0.
    """
    if not first_relevant_ranks:
        return 0.0
    return sum(r is not None and r <= k for r in first_relevant_ranks) / len(first_relevant_ranks)


def mrr_at_k(first_relevant_ranks: Sequence[int | None], k: int) -> float:
    """Mean Reciprocal Rank at ``k``: mean of ``1 / rank`` of the first relevant result.

    A query scores 0.0 if it missed (``None``) or its first relevant result is ranked
    beyond ``k``, so ``1 / rank`` is never evaluated for a miss. An empty input gives 0.0.
    """
    if not first_relevant_ranks:
        return 0.0
    return sum(1.0 / r for r in first_relevant_ranks if r is not None and r <= k) / len(first_relevant_ranks)


class QueryEvaluation(BaseModel):
    """Per-query outcome for one configuration."""

    query_id: str
    first_relevant_rank: int | None
    recall_at_k: dict[int, float]
    latency_ms: float
    retrieved_chunk_ids: list[str]


def evaluate_ranking(
    query: EvalQuery,
    results: Sequence[SearchResult],
    targets: set[str],
    ks: Sequence[int],
    latency_ms: float = 0.0,
) -> QueryEvaluation:
    """Score one ranked result list against the query's usable ``targets``."""
    if not targets:
        raise ValueError(f"query {query.query_id!r} has no usable targets")

    first_rank: int | None = None
    covered: set[str] = set()
    covered_after: list[int] = []  # number of targets covered after each rank
    for rank, result in enumerate(results, start=1):
        hits = matched_targets(result, query) & targets
        if hits and first_rank is None:
            first_rank = rank
        covered |= hits
        covered_after.append(len(covered))

    recall: dict[int, float] = {}
    for k in ks:
        n_covered = covered_after[min(k, len(covered_after)) - 1] if covered_after else 0
        recall[k] = n_covered / len(targets)

    return QueryEvaluation(
        query_id=query.query_id,
        first_relevant_rank=first_rank,
        recall_at_k=recall,
        latency_ms=latency_ms,
        retrieved_chunk_ids=[r.chunk_id for r in results],
    )


# --------------------------------------------------------------------------------------
# Benchmark configuration and results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """A named (chunker, retriever) pair. The factory builds a fresh, empty index."""

    name: str
    chunker: BaseChunker
    retriever_factory: Callable[[], Retriever]


class ConfigResult(BaseModel):
    """Aggregate metrics for one configuration."""

    name: str
    retriever: str
    chunk_strategy: str
    num_chunks: int
    mean_chunk_tokens: float
    index_time_s: float
    num_queries_evaluated: int
    num_queries_skipped: int  # always 0 now (targets don't depend on the chunker); kept so old reports load
    hit_rate: dict[int, float]
    recall: dict[int, float]
    mrr: dict[int, float]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    per_query: list[QueryEvaluation]


class BenchmarkReport(BaseModel):
    """Full benchmark output; serialise with ``model_dump_json``."""

    created_at: datetime
    dataset: str
    num_docs: int
    num_queries: int
    ks: list[int]
    results: list[ConfigResult]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per configuration with metric columns, e.g. ``HR@5``, ``MRR@10``."""
        rows = []
        for r in self.results:
            row: dict[str, object] = {"config": r.name, "chunks": r.num_chunks}
            for k in self.ks:
                row[f"HR@{k}"] = r.hit_rate[k]
            for k in self.ks:
                row[f"R@{k}"] = r.recall[k]
            for k in self.ks:
                row[f"MRR@{k}"] = r.mrr[k]
            row["p50 ms"] = r.latency_p50_ms
            row["p95 ms"] = r.latency_p95_ms
            rows.append(row)
        return pd.DataFrame(rows).set_index("config")


@dataclass(frozen=True)
class BuiltIndex:
    """A configuration whose index has been built and is ready to query."""

    config: BenchmarkConfig
    chunks: list[DocumentChunk]
    retriever: Retriever
    index_time_s: float


def build_index(config: BenchmarkConfig, corpus: Mapping[str, str]) -> BuiltIndex:
    """Chunk the corpus with the config's chunker and build a fresh index."""
    chunks = config.chunker.chunk_documents(corpus)
    retriever = config.retriever_factory()
    t0 = time.perf_counter()
    retriever.index(chunks)
    index_time = time.perf_counter() - t0
    logger.info("[%s] %d chunks indexed in %.2fs", config.name, len(chunks), index_time)
    return BuiltIndex(config=config, chunks=chunks, retriever=retriever, index_time_s=index_time)


def evaluate_config(
    config: BenchmarkConfig,
    corpus: Mapping[str, str],
    dataset: EvalDataset,
    ks: Sequence[int],
    warmup: bool = True,
) -> ConfigResult:
    """Chunk the corpus, build the index, run every query and aggregate metrics."""
    resolve_spans(dataset, corpus)
    return evaluate_index(build_index(config, corpus), dataset, ks, warmup)


def evaluate_index(
    built: BuiltIndex,
    dataset: EvalDataset,
    ks: Sequence[int],
    warmup: bool = True,
) -> ConfigResult:
    """Run every query against an already-built index and aggregate metrics.

    Separated from :func:`build_index` so callers (e.g. the dashboard) can build once
    and evaluate many times with different K values.
    """
    ks = sorted(set(ks))
    max_k = ks[-1]
    config, chunks, retriever, index_time = (
        built.config,
        built.chunks,
        built.retriever,
        built.index_time_s,
    )

    if warmup and dataset.queries:
        # The first call can pay one-off costs (lazy init, kernel compilation, caches).
        retriever.search(dataset.queries[0].query, max_k)

    # Every query is scored by every configuration: targets are defined on the source
    # documents, so none can be unusable for a particular chunking strategy.
    evaluations: list[QueryEvaluation] = []
    for query in dataset.queries:
        targets = usable_targets(query)
        t0 = time.perf_counter()
        results = retriever.search(query.query, max_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        evaluations.append(evaluate_ranking(query, results, targets, ks, latency_ms))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    ranks = [e.first_relevant_rank for e in evaluations]
    latencies = np.array([e.latency_ms for e in evaluations]) if evaluations else np.zeros(1)
    return ConfigResult(
        name=config.name,
        retriever=retriever.name,
        chunk_strategy=config.chunker.strategy,
        num_chunks=len(chunks),
        mean_chunk_tokens=mean([c.metadata.get("token_count", 0) for c in chunks]),
        index_time_s=index_time,
        num_queries_evaluated=len(evaluations),
        num_queries_skipped=0,
        hit_rate={k: hit_rate_at_k(ranks, k) for k in ks},
        recall={k: mean([e.recall_at_k[k] for e in evaluations]) for k in ks},
        mrr={k: mrr_at_k(ranks, k) for k in ks},
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        latency_mean_ms=float(latencies.mean()),
        per_query=evaluations,
    )


def run_benchmark(
    corpus: Mapping[str, str],
    dataset: EvalDataset,
    configs: Sequence[BenchmarkConfig],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> BenchmarkReport:
    """Build and evaluate every configuration on the same corpus and query set."""
    resolve_spans(dataset, corpus)
    built = [build_index(c, corpus) for c in configs]
    return evaluate_indexes(built, dataset, ks, num_docs=len(corpus))


def evaluate_indexes(
    built: Sequence[BuiltIndex],
    dataset: EvalDataset,
    ks: Sequence[int] = (1, 3, 5, 10),
    num_docs: int = 0,
) -> BenchmarkReport:
    """Evaluate already-built indexes and assemble a report."""
    if not ks or min(ks) < 1:
        raise ValueError("ks must contain positive integers")
    return BenchmarkReport(
        created_at=datetime.now(UTC),
        dataset=dataset.name,
        num_docs=num_docs,
        num_queries=len(dataset.queries),
        ks=sorted(set(ks)),
        results=[evaluate_index(b, dataset, ks) for b in built],
    )


def build_default_configs(
    embedder: Embedder,
    small_chunk: tuple[int, int] = (256, 32),
    large_chunk: tuple[int, int] = (1024, 128),
    chroma_client: object | None = None,
    collection_prefix: str = "dense",
) -> list[BenchmarkConfig]:
    """The four standard configurations.

    BM25 uses the same small fixed-size chunks as the "Dense (Small)" configuration, so
    that pair isolates the effect of the retriever with chunking held constant.

    Args:
        embedder: Shared embedding model for all dense configurations.
        small_chunk: ``(chunk_size, chunk_overlap)`` in tokens for small chunks.
        large_chunk: ``(chunk_size, chunk_overlap)`` in tokens for large chunks.
        chroma_client: Optional Chroma client (e.g. ``PersistentClient``); defaults to in-memory.
        collection_prefix: Prefix for Chroma collection names; use distinct prefixes to keep
            several sets of indexes alive in one client.
    """
    import chromadb

    client = chroma_client if chroma_client is not None else chromadb.EphemeralClient()
    small = FixedSizeChunker(*small_chunk)
    large = FixedSizeChunker(*large_chunk)
    structure = StructureAwareChunker()

    def dense(collection: str) -> Callable[[], Retriever]:
        return lambda: DenseVectorIndex(embedder, collection_name=collection, client=client)

    return [
        BenchmarkConfig(f"BM25 (Baseline, {small_chunk[0]} tok)", small, BM25Index),
        BenchmarkConfig(f"Dense (Small, {small_chunk[0]} tok)", small, dense(f"{collection_prefix}-{small.strategy}")),
        BenchmarkConfig(f"Dense (Large, {large_chunk[0]} tok)", large, dense(f"{collection_prefix}-{large.strategy}")),
        BenchmarkConfig("Dense (Structure-Aware)", structure, dense(f"{collection_prefix}-{structure.strategy}")),
    ]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BM25 vs. dense retrieval across chunking strategies."
    )
    parser.add_argument("--docs", type=Path, default=Path("data/docs"), help="Markdown corpus directory")
    parser.add_argument("--queries", type=Path, default=Path("data/eval_queries.json"), help="Eval dataset")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="Cut-offs for @K metrics")
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "openai"],
        default="sentence-transformers",
        help="Embedding backend for dense configurations",
    )
    parser.add_argument("--model", default=None, help="Override the embedding model name")
    parser.add_argument("--small-chunk", type=int, nargs=2, default=[256, 32], metavar=("SIZE", "OVERLAP"))
    parser.add_argument("--large-chunk", type=int, nargs=2, default=[1024, 128], metavar=("SIZE", "OVERLAP"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/benchmark_report.json"), help="JSON report path"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging")
    return parser.parse_args(argv)


def _make_embedder(backend: str, model: str | None) -> Embedder:
    if backend == "openai":
        return OpenAIEmbedder(model) if model else OpenAIEmbedder()
    return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: run the four default configurations and write a JSON report."""
    from dotenv import load_dotenv

    load_dotenv()  # picks up OPENAI_API_KEY from .env if present
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    corpus = load_corpus(args.docs)
    dataset = load_eval_dataset(args.queries, corpus)
    embedder = _make_embedder(args.backend, args.model)
    configs = build_default_configs(
        embedder, small_chunk=tuple(args.small_chunk), large_chunk=tuple(args.large_chunk)
    )

    report = run_benchmark(corpus, dataset, configs, ks=args.k)

    print(
        f"\nDataset: {report.dataset} | docs: {report.num_docs} | queries: {report.num_queries} "
        f"| embedder: {embedder.name}\n"
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.to_dataframe().to_string(float_format=lambda x: f"{x:.3f}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull report (incl. per-query results) written to {args.output}")


if __name__ == "__main__":
    main()
