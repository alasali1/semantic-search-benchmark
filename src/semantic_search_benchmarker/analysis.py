"""Head-to-head failure analysis: where did BM25 beat dense retrieval, and why?

For each evaluated query, the reciprocal rank of the first relevant chunk is compared
between a BM25 run and a dense run. For queries where one side wins, rule-based
diagnostics explain the gap using signals that can be measured directly:

* **Lexical overlap** - share of the query's content words (stop words removed) that
  appear in the relevant chunks. High overlap favours BM25; low overlap (a paraphrase)
  favours dense retrieval.
* **Exact identifiers** - tokens such as ``dimension_mismatch``, ``409`` or
  ``X-RateLimit`` that occur verbatim in the relevant text. BM25 weights these rare
  tokens heavily; small embedding models split them into generic sub-word pieces.
* **Distractors** - the loser's top-ranked, non-relevant hit and the query terms it
  matched (high-IDF terms sorted first).
* **Truncation** - relevant chunks longer than the embedding model's input window are
  only partly embedded.

These are heuristics, so they explain likely causes rather than prove them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from semantic_search_benchmarker.benchmarker import (
    BuiltIndex,
    ConfigResult,
    EvalDataset,
    EvalQuery,
    matched_targets,
    usable_targets,
)
from semantic_search_benchmarker.chunking import DocumentChunk
from semantic_search_benchmarker.indexer import BM25Index, SearchResult, bm25_tokenize

# Small English stop-word list: enough to keep "how do I ..." from counting as overlap.
STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom
    why will with would you your yours e g also many much rather via""".split()
)

# Query tokens that look like code/identifiers: snake_case, digits, dotted/hyphenated
# names, ALLCAPS, camelCase, or anything in backticks.
_IDENTIFIER_RE = re.compile(
    r"`([^`]+)`"
    r"|(\b[A-Za-z0-9]+(?:[_.\-][A-Za-z0-9]+)+\b)"
    r"|(\b\w*\d\w*\b)"
    r"|(\b[A-Z]{2,}\b)"
    r"|(\b[a-z]+[A-Z]\w*\b)"
)

# Lexical-overlap thresholds used to phrase explanations.
HIGH_OVERLAP = 0.6
LOW_OVERLAP = 0.34

Winner = Literal["bm25", "dense", "tie"]


class HeadToHead(BaseModel):
    """Comparison of BM25 and dense retrieval for one query."""

    query_id: str
    query: str
    targets: list[str]
    winner: Winner
    bm25_rank: int | None
    dense_rank: int | None
    term_overlap: float | None
    matched_terms: list[str]
    missing_terms: list[str]
    bm25_top: str | None
    dense_top: str | None
    reasons: list[str]


def query_terms(text: str) -> list[str]:
    """BM25 tokens of ``text`` without stop words, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for tok in bm25_tokenize(text):
        if tok not in STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


def identifier_tokens(text: str) -> list[str]:
    """Identifier-like substrings of a query (see module docstring)."""
    found: dict[str, None] = {}
    for m in _IDENTIFIER_RE.finditer(text):
        token = next(g for g in m.groups() if g)
        found.setdefault(token, None)
    return list(found)


def chunk_label(chunk: DocumentChunk | SearchResult) -> str:
    """Readable location of a chunk.

    ``doc.md › Section › Subsection`` for a single section; ``doc.md › [A | B | C]``
    (leaf titles) when a fixed-size chunk spans several sections.
    """
    meta = chunk.metadata
    paths = [p for p in meta.get("section_paths") or [] if p] or [meta.get("header_path") or []]
    if len(paths) > 1:
        return f"{chunk.parent_doc} › [{' | '.join(p[-1] for p in paths)}]"
    return " › ".join([chunk.parent_doc, *paths[0]]) if paths[0] else chunk.parent_doc


def _is_relevant(chunk: DocumentChunk, query: EvalQuery, targets: set[str]) -> bool:
    return bool(matched_targets(SearchResult.from_chunk(chunk, 1, 0.0, ""), query) & targets)


def _relevant_chunks(built: BuiltIndex, query: EvalQuery) -> list[DocumentChunk]:
    targets = usable_targets(query)
    return [c for c in built.chunks if _is_relevant(c, query, targets)]


def _reciprocal(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _fmt_rank(rank: int | None) -> str:
    return f"rank {rank}" if rank else "not found"


def head_to_head(
    dataset: EvalDataset,
    bm25_result: ConfigResult,
    dense_result: ConfigResult,
    bm25_index: BuiltIndex,
    dense_index: BuiltIndex,
    dense_window_tokens: int | None = 256,
    model_token_counts: Mapping[str, int] | None = None,
) -> list[HeadToHead]:
    """Compare BM25 and dense results query-by-query and explain each win.

    Args:
        dataset: The evaluation queries both results were computed on.
        bm25_result / dense_result: Benchmark results to compare.
        bm25_index / dense_index: The built indexes behind those results (used to
            inspect chunk text, IDF and the top hits).
        dense_window_tokens: Embedding model input window (in model tokens), used for
            the truncation check; ``None`` disables it.
        model_token_counts: ``chunk_id -> length in the embedding model's own tokens``
            (e.g. from ``SentenceTransformerEmbedder.count_tokens``). Chunks missing from
            it fall back to the tiktoken ``token_count``, which underestimates word pieces.
    """
    if not isinstance(bm25_index.retriever, BM25Index):
        raise TypeError("bm25_index must wrap a BM25Index")
    bm25: BM25Index = bm25_index.retriever
    bm25_eval = {e.query_id: e for e in bm25_result.per_query}
    dense_eval = {e.query_id: e for e in dense_result.per_query}
    bm25_chunks = {c.chunk_id: c for c in bm25_index.chunks}
    dense_chunks = {c.chunk_id: c for c in dense_index.chunks}

    comparisons: list[HeadToHead] = []
    for query in dataset.queries:
        b, d = bm25_eval.get(query.query_id), dense_eval.get(query.query_id)
        if b is None or d is None:
            continue  # skipped by at least one configuration

        rr_b, rr_d = _reciprocal(b.first_relevant_rank), _reciprocal(d.first_relevant_rank)
        winner: Winner = "bm25" if rr_b > rr_d else "dense" if rr_d > rr_b else "tie"

        # Lexical overlap is measured against what BM25 indexes: its relevant chunks.
        relevant_bm25 = _relevant_chunks(bm25_index, query)
        relevant_tokens = {t for c in relevant_bm25 for t in bm25_tokenize(c.content)}
        terms = query_terms(query.query)
        matched = [t for t in terms if t in relevant_tokens]
        missing = [t for t in terms if t not in relevant_tokens]
        overlap = len(matched) / len(terms) if terms and relevant_bm25 else None

        bm25_top = bm25_chunks.get(b.retrieved_chunk_ids[0]) if b.retrieved_chunk_ids else None
        dense_top = dense_chunks.get(d.retrieved_chunk_ids[0]) if d.retrieved_chunk_ids else None

        reasons: list[str] = []
        if winner != "tie":
            reasons = _explain(
                winner=winner,
                query=query,
                terms=terms,
                matched=matched,
                missing=missing,
                overlap=overlap,
                relevant_text=" ".join(c.content for c in relevant_bm25),
                relevant_dense=_relevant_chunks(dense_index, query),
                loser_top=dense_top if winner == "bm25" else bm25_top,
                loser_rank=d.first_relevant_rank if winner == "bm25" else b.first_relevant_rank,
                winner_rank=b.first_relevant_rank if winner == "bm25" else d.first_relevant_rank,
                bm25=bm25,
                dense_window_tokens=dense_window_tokens,
                model_token_counts=model_token_counts or {},
            )

        comparisons.append(
            HeadToHead(
                query_id=query.query_id,
                query=query.query,
                targets=[*query.relevant_headers, *query.relevant_spans],
                winner=winner,
                bm25_rank=b.first_relevant_rank,
                dense_rank=d.first_relevant_rank,
                term_overlap=overlap,
                matched_terms=matched,
                missing_terms=missing,
                bm25_top=chunk_label(bm25_top) if bm25_top else None,
                dense_top=chunk_label(dense_top) if dense_top else None,
                reasons=reasons,
            )
        )
    return comparisons


def _explain(
    *,
    winner: Winner,
    query: EvalQuery,
    terms: Sequence[str],
    matched: Sequence[str],
    missing: Sequence[str],
    overlap: float | None,
    relevant_text: str,
    relevant_dense: Sequence[DocumentChunk],
    loser_top: DocumentChunk | None,
    loser_rank: int | None,
    winner_rank: int | None,
    bm25: BM25Index,
    dense_window_tokens: int | None,
    model_token_counts: Mapping[str, int],
) -> list[str]:
    """Rule-based explanation of why ``winner`` ranked a relevant chunk higher."""
    reasons: list[str] = []
    by_idf = sorted(matched, key=bm25.term_idf, reverse=True)
    n_terms = len(terms)
    lowered_relevant = relevant_text.lower()

    if winner == "bm25":
        exact = [t for t in identifier_tokens(query.query) if t.lower() in lowered_relevant]
        if exact:
            reasons.append(
                f"Exact-match identifiers {_quote(exact)} appear verbatim in the relevant section. "
                "BM25 weights rare tokens like these heavily, while the embedding model splits "
                "them into generic sub-word pieces that carry little meaning."
            )
        if overlap is not None and overlap >= HIGH_OVERLAP:
            reasons.append(
                f"High lexical overlap: {len(matched)}/{n_terms} query terms occur in the relevant "
                f"section (most distinctive by IDF: {_quote(by_idf[:3])}), so keyword matching "
                "alone pins down the answer."
            )
        if dense_window_tokens and relevant_dense:
            lengths = [
                model_token_counts.get(c.chunk_id, int(c.metadata.get("token_count", 0)))
                for c in relevant_dense
            ]
            if min(lengths) > dense_window_tokens:
                reasons.append(
                    f"Every relevant chunk exceeds the embedding model's {dense_window_tokens}-token "
                    f"input window (shortest: {min(lengths)} tokens), so part of the relevant text "
                    "was never embedded. If the answer sits late in the chunk, its vector cannot "
                    "reflect it."
                )
        if loser_top is not None and not _top_is_relevant(loser_rank):
            reasons.append(
                f"Dense retrieval ranked “{chunk_label(loser_top)}” first: its embedding was closer "
                "to the query than the target's, a common near-miss when sections share a topic "
                "or a chunk's vector is dominated by unrelated text."
            )
    else:
        if overlap is not None and overlap <= LOW_OVERLAP:
            missing_txt = f" (missing: {_quote(missing[:4])})" if missing else ""
            reasons.append(
                f"Vocabulary mismatch: only {len(matched)}/{n_terms} query terms appear in the "
                f"relevant section{missing_txt}. The query paraphrases the docs, which embeddings "
                "match by meaning but BM25 cannot."
            )
        if loser_rank is None:
            reasons.append("BM25 did not retrieve any relevant chunk within the top K.")
        if loser_top is not None and not _top_is_relevant(loser_rank):
            hit_terms = set(bm25_tokenize(loser_top.content))
            shared = sorted((t for t in terms if t in hit_terms), key=bm25.term_idf, reverse=True)
            if shared:
                reasons.append(
                    f"BM25's top hit “{chunk_label(loser_top)}” matched on {_quote(shared[:4])}. "
                    "Those words also appear outside the target section, so keyword scoring was "
                    "drawn to the wrong place."
                )
            else:
                reasons.append(f"BM25's top hit was “{chunk_label(loser_top)}”, not the target section.")

    if not reasons:
        reasons.append(
            f"Both systems found the target ({'BM25' if winner == 'bm25' else 'dense'} "
            f"at {_fmt_rank(winner_rank)}, the other at {_fmt_rank(loser_rank)}); the gap comes "
            "from how close competing chunks scored, not from a clear lexical or semantic failure."
        )
    return reasons


def _top_is_relevant(first_relevant_rank: int | None) -> bool:
    return first_relevant_rank == 1


def _quote(items: Sequence[str]) -> str:
    return ", ".join(f"`{i}`" for i in items)


def summarize(comparisons: Sequence[HeadToHead]) -> dict[str, int]:
    """Count wins, ties and queries both systems missed."""
    return {
        "bm25": sum(c.winner == "bm25" for c in comparisons),
        "dense": sum(c.winner == "dense" for c in comparisons),
        "tie": sum(c.winner == "tie" and c.bm25_rank is not None for c in comparisons),
        "both_missed": sum(c.winner == "tie" and c.bm25_rank is None for c in comparisons),
    }
