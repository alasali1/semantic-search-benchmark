"""Step 5 QA suite: retrieval behaviour of BM25 (exact match) and dense search (paraphrase)."""

import uuid
from pathlib import Path

import pytest

from semantic_search_benchmarker.benchmarker import load_corpus
from semantic_search_benchmarker.chunking import (
    BaseChunker,
    DocumentChunk,
    FixedSizeChunker,
    StructureAwareChunker,
)
from semantic_search_benchmarker.indexer import BM25Index, DenseVectorIndex, Embedder, bm25_tokenize

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
IDENTIFIER = "CUSTOM_CONFIG_FLAG_XYZ"

# A doc where the identifier's component words ("custom", "config", "flag") are common,
# so BM25 must rely on the rare token "xyz" (and the combination) to rank the right chunk.
FLAGS_DOC = f"""# Feature Flags

## Overview

Feature flags turn behaviour on and off. Every flag lives in the config file, and a
custom flag can override any default config value. Config changes reload automatically.

## Custom Flags

Custom flags are declared in the custom section of the config. Each custom flag needs a
name, a default and a description. Flag names are upper case.

## Experimental Search Mode

Set `{IDENTIFIER}=true` to enable the experimental search mode. It is off by default.

## Removing Flags

Delete the flag from the config and restart. Removed custom flags are ignored.
"""


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    """The sample docs plus the feature-flag doc, so the identifier competes with real text."""
    docs = load_corpus(DOCS_DIR)
    docs["feature-flags.md"] = FLAGS_DOC
    return docs


@pytest.mark.parametrize(
    "chunker", [StructureAwareChunker(), FixedSizeChunker(64, 8)], ids=["structure", "fixed-64"]
)
@pytest.mark.parametrize("query", [IDENTIFIER, f"What does {IDENTIFIER} do?", IDENTIFIER.lower()])
def test_3_bm25_exact_match_identifier_is_rank_1(
    corpus: dict[str, str], chunker: BaseChunker, query: str
) -> None:
    """Test 3: a rare technical identifier is retrieved at Rank 1 by BM25."""
    chunks = chunker.chunk_documents(corpus)
    # Premise: other chunks share the identifier's component words but not the identifier.
    distractors = [c for c in chunks if IDENTIFIER not in c.content and "custom" in c.content.lower()]
    assert distractors, "test needs distractor chunks containing the identifier's component words"

    index = BM25Index()
    index.index(chunks)
    results = index.search(query, k=5)

    assert results, "BM25 returned nothing"
    assert results[0].rank == 1
    assert IDENTIFIER in results[0].content, results[0].content


# --------------------------------------------------------------------------------------
# Dense paraphrase retrieval (needs the all-MiniLM-L6-v2 model)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    try:
        from semantic_search_benchmarker.indexer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    except Exception as exc:  # model not downloaded and no network, etc.
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.fixture(scope="module")
def structure_chunks() -> list[DocumentChunk]:
    return StructureAwareChunker().chunk_documents(load_corpus(DOCS_DIR))


@pytest.fixture(scope="module")
def dense_index(embedder: Embedder, structure_chunks: list[DocumentChunk]) -> DenseVectorIndex:
    index = DenseVectorIndex(embedder, collection_name=f"qa-{uuid.uuid4().hex[:12]}")
    index.index(structure_chunks)
    return index


PARAPHRASES = [
    ("Key Rotation", "swap secrets while service stays up"),
    ("Key Rotation", "replace passwords with zero outage"),
    ("Autoscaling", "grow machine count automatically under heavy load"),
    ("Verifying Signatures", "confirm a callback truly came from us"),
    ("Verifying Signatures", "prove incoming notifications are authentic"),
]
TOP_K = 5


@pytest.mark.parametrize("section,query", PARAPHRASES)
def test_4_dense_retrieves_paraphrase_with_zero_shared_keywords(
    dense_index: DenseVectorIndex, structure_chunks: list[DocumentChunk], section: str, query: str
) -> None:
    """Test 4: a conceptual query sharing no words with the source is found in top-K by dense search."""
    target = next(c for c in structure_chunks if c.header_path[-1] == section)

    # Premise: the query has zero exact keywords in common with the source text.
    shared = set(bm25_tokenize(query)) & set(bm25_tokenize(target.content))
    assert shared == set(), f"query is not keyword-free: {shared}"

    results = dense_index.search(query, k=TOP_K)
    assert target.chunk_id in [r.chunk_id for r in results], [r.metadata["header_path"] for r in results]

    # Contrast: keyword search cannot find it, because there is nothing to match.
    bm25 = BM25Index()
    bm25.index(structure_chunks)
    assert target.chunk_id not in [r.chunk_id for r in bm25.search(query, k=TOP_K)]
