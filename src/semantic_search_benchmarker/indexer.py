"""Dense (ChromaDB) and sparse (BM25) indexes behind one retrieval interface.

Every index implements :class:`Retriever`: ``index(chunks)`` followed by
``search(query, k) -> list[SearchResult]``. The benchmarker relies only on that
interface, so new backends (hybrid, re-rankers, other vector stores) plug in without
touching evaluation code.

Embedders are separate from the vector store (:class:`Embedder`) so one loaded model
can be shared by several dense indexes, e.g. one per chunking strategy.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from semantic_search_benchmarker.chunking import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


# --------------------------------------------------------------------------------------
# Result model and retriever interface
# --------------------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One ranked hit. All retrievers return this type, so callers can compare
    backends interchangeably."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    rank: int = Field(ge=1, description="1-indexed rank in the result list.")
    score: float = Field(
        description="Backend-specific relevance score, higher is better "
        "(cosine similarity for dense, BM25 score for sparse). Not comparable across backends."
    )
    content: str
    parent_doc: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    retriever: str = Field(description="Name of the retriever that produced the hit.")

    @classmethod
    def from_chunk(
        cls, chunk: DocumentChunk, rank: int, score: float, retriever: str
    ) -> SearchResult:
        """Build a result from the original chunk, keeping its full metadata."""
        return cls(
            chunk_id=chunk.chunk_id,
            rank=rank,
            score=float(score),
            content=chunk.content,
            parent_doc=chunk.parent_doc,
            metadata=dict(chunk.metadata),
            retriever=retriever,
        )


class Retriever(ABC):
    """Abstract index: build once with :meth:`index`, then call :meth:`search`."""

    def __init__(self) -> None:
        # Original chunks by ID. Results are hydrated from here, so full metadata
        # (nested lists etc.) is returned even if the backend can only store scalars.
        self._chunks: dict[str, DocumentChunk] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable retriever name, e.g. ``"bm25"`` or ``"dense:all-MiniLM-L6-v2"``."""

    @abstractmethod
    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """(Re)build the index from ``chunks``, replacing any previous contents."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to ``k`` results ordered by descending relevance."""

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def _store_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk_id values in input chunks")
        self._chunks = {c.chunk_id: c for c in chunks}


# --------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------


def l2_normalize(vectors: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    """Scale each row to unit L2 norm (zero vectors are left as zeros)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-normalised float32 vectors."""

    @property
    def name(self) -> str:
        """Model identifier used in reports."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed corpus texts; returns an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query; returns an array of shape ``(dim,)``."""
        ...


class SentenceTransformerEmbedder:
    """Local HuggingFace embedder (default ``all-MiniLM-L6-v2``, 384-d)."""

    def __init__(
        self,
        model_name: str = DEFAULT_ST_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        # Imported lazily: torch is heavy and is not needed by the OpenAI/BM25 paths.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model_name.split("/")[-1]

    @property
    def max_seq_length(self) -> int:
        """Maximum input length in model word pieces; longer inputs are truncated."""
        return int(self._model.max_seq_length)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._warn_on_truncation(texts)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(vector)[0]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Input length of each text in *model* tokens (word pieces incl. special tokens).

        This is what the truncation limit applies to; it is usually higher than the
        tiktoken count, especially for code.
        """
        if not texts:
            return []
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True)["input_ids"]
        return [len(ids) for ids in encoded]

    def _warn_on_truncation(self, texts: Sequence[str]) -> None:
        """Log how many inputs exceed the model window: their tail is never embedded."""
        truncated = sum(n > self.max_seq_length for n in self.count_tokens(texts))
        if truncated:
            logger.warning(
                "%s: %d/%d inputs exceed max_seq_length=%d word pieces and will be truncated",
                self.name,
                truncated,
                len(texts),
                self.max_seq_length,
            )


class OpenAIEmbedder:
    """OpenAI embeddings API (default ``text-embedding-3-small``, 1536-d).

    Reads ``OPENAI_API_KEY`` from the environment unless a client is supplied.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._model = model
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = self._client.embeddings.create(model=self._model, input=batch)
            # The API returns items with an explicit index; don't rely on ordering.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return l2_normalize(np.array(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._embed(texts)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._embed([text])[0]


# --------------------------------------------------------------------------------------
# Dense vector index (ChromaDB)
# --------------------------------------------------------------------------------------


def _chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    """Flatten chunk metadata to the scalar values Chroma accepts (used for filtering)."""
    meta = chunk.metadata
    return {
        "parent_doc": chunk.parent_doc,
        "strategy": str(meta.get("strategy", "")),
        "start_line": int(meta.get("start_line", 0)),
        "end_line": int(meta.get("end_line", 0)),
        "header_path": " > ".join(meta.get("header_path", [])),
    }


class DenseVectorIndex(Retriever):
    """Cosine-similarity vector search backed by a ChromaDB HNSW collection.

    Embeddings are computed by the injected :class:`Embedder` and L2-normalised before
    insertion, so cosine distance equals ``1 - dot product`` and ``score = 1 - distance``.
    Note that HNSW is an approximate search; on small corpora it is effectively exact.
    """

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        import chromadb

        self._embedder = embedder
        self._client = client if client is not None else chromadb.EphemeralClient()
        self._collection_name = collection_name
        self._collection: Any | None = None

    @property
    def name(self) -> str:
        return f"dense:{self._embedder.name}"

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._store_chunks(chunks)

        from chromadb.errors import NotFoundError

        # Rebuild from scratch so repeated runs never mix old and new chunks.
        try:
            self._client.delete_collection(self._collection_name)
        except NotFoundError:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we always supply embeddings ourselves
        )
        if not chunks:
            return

        embeddings = self._embedder.embed_documents([c.content for c in chunks])
        batch = self._client.get_max_batch_size()
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + batch],
                documents=[c.content for c in part],
                metadatas=[_chroma_metadata(c) for c in part],
            )
        logger.info("Indexed %d chunks into Chroma collection %r", len(chunks), self._collection_name)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self._collection is None:
            raise RuntimeError("index() must be called before search()")
        n = min(k, len(self._chunks))
        if n <= 0:
            return []
        query_vec = self._embedder.embed_query(query)
        response = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n,
            include=["distances"],
        )
        ids: list[str] = response["ids"][0]
        distances: list[float] = response["distances"][0]
        return [
            SearchResult.from_chunk(self._chunks[cid], rank=r, score=1.0 - dist, retriever=self.name)
            for r, (cid, dist) in enumerate(zip(ids, distances, strict=True), start=1)
        ]


# --------------------------------------------------------------------------------------
# Sparse keyword index (BM25)
# --------------------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace.

    Punctuation *and underscores* become separators, so ``max_tokens`` and
    ``max-tokens`` both index as ``["max", "tokens"]`` and match the query "max tokens".
    """
    return [tok for tok in _NON_ALNUM.split(text.lower()) if tok]


class BM25Index(Retriever):
    """Okapi BM25 keyword retrieval via ``rank_bm25.BM25Okapi``.

    Chunks sharing no terms with the query are not returned, so a query may yield fewer
    than ``k`` results. (Filtering on term overlap rather than ``score > 0`` matters:
    BM25Okapi gives IDF = 0 to a term found in exactly half the chunks, so a genuine
    match can score 0.)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = bm25_tokenize,
    ) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b
        self._tokenize = tokenizer
        self._bm25: Any | None = None
        self._ids: list[str] = []
        self._token_sets: list[frozenset[str]] = []

    @property
    def name(self) -> str:
        return "bm25"

    def term_idf(self, term: str) -> float:
        """BM25Okapi IDF of an (already tokenized) term; 0.0 if unseen or not indexed."""
        if self._bm25 is None:
            return 0.0
        return float(self._bm25.idf.get(term, 0.0))

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._store_chunks(chunks)
        self._ids = [c.chunk_id for c in chunks]
        tokenized = [self._tokenize(c.content) for c in chunks]
        self._token_sets = [frozenset(toks) for toks in tokenized]
        # BM25Okapi divides by corpus size, so an empty corpus is handled explicitly.
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b) if chunks else None

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self._bm25 is None:
            return []
        tokens = self._tokenize(query)
        if not tokens or k <= 0:
            return []
        scores: NDArray[np.float64] = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

        # Exclude chunks with no query term in common (see class docstring).
        query_terms = frozenset(tokens)
        matches = np.fromiter((not query_terms.isdisjoint(s) for s in self._token_sets), dtype=bool)
        scores = np.where(matches, scores, -np.inf)

        # argpartition gives the top-k in O(n); then sort just those k.
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]

        results: list[SearchResult] = []
        for idx in top:
            if not matches[idx]:
                break
            chunk = self._chunks[self._ids[idx]]
            results.append(
                SearchResult.from_chunk(chunk, rank=len(results) + 1, score=scores[idx], retriever=self.name)
            )
        return results
