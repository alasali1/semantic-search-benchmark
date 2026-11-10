# Semantic Search Benchmarker

Indexes technical Markdown documentation using several chunking strategies and compares
dense vector retrieval with BM25 keyword search on a labelled query set.

## Layout

```
src/semantic_search_benchmarker/
  chunking.py     # DocumentChunk, FixedSizeChunker (tiktoken), StructureAwareChunker (headers + code fences)
  indexer.py      # SearchResult, Retriever ABC, DenseVectorIndex (ChromaDB, cosine), BM25Index, embedders
  benchmarker.py  # eval dataset, relevance judging, Hit Rate / Recall / MRR @K, latency, CLI
  analysis.py     # BM25-vs-dense head-to-head failure analysis with rule-based explanations
  app.py          # Streamlit dashboard
data/docs/              # sample corpus
data/eval_queries.json  # sample labelled queries
tests/
```

## Usage

```bash
uv sync
uv run semantic-search-benchmarker                       # local all-MiniLM-L6-v2
uv run semantic-search-benchmarker --backend openai      # text-embedding-3-small (needs OPENAI_API_KEY / .env)
uv run semantic-search-benchmarker --docs path/to/docs --queries path/to/eval_queries.json --k 1 5 10
uv run semantic-search-dashboard                         # Streamlit dashboard on http://localhost:8501
uv run pytest
```

The benchmark CLI prints a summary table and writes `results/benchmark_report.json`, which
includes the retrieved chunk IDs and metrics for every query.

## Docker

```bash
docker compose up --build                     # dashboard on http://localhost:8501
docker compose run --rm benchmark             # CLI benchmark -> ./results/benchmark_report.json
docker compose run --rm benchmark semantic-search-benchmarker --backend openai --k 1 5 10
docker compose run --rm benchmark pytest      # test suite
```

* `./data` is mounted read-only, so you can edit docs and queries without rebuilding.
* `OPENAI_API_KEY` is read from `.env` if it exists. The file is never copied into the image.
* The image uses CPU-only PyTorch and includes `all-MiniLM-L6-v2`, so the local backend
  works without a network connection.

## Dashboard

* **Live Search Comparator:** enter a query and K, pick a chunking strategy, and see BM25
  and dense results side by side. Each result shows its score (BM25 or cosine), its
  location and line numbers, how many query terms it contains, and the text with
  matching terms highlighted. Choosing a query from the eval set marks relevant hits with ✓.
* **Benchmark & Evaluation:** pick K values and click **Run Benchmark**. You get a
  comparison table, Hit Rate@K / MRR@K and latency charts, a chunking-strategy table (with
  model-window truncation and trivial-K warnings), and a failure-analysis panel listing the
  queries each retriever won and the likely reasons.

Indexes are built once per corpus and cached, so both tabs reuse them. Edit a doc and
they rebuild automatically, or use **Rebuild indexes** in the sidebar.

## Configurations

| Name | Chunking | Retriever |
|---|---|---|
| BM25 (Baseline) | fixed 256 tokens / 32 overlap | `rank_bm25.BM25Okapi` |
| Dense (Small) | fixed 256 / 32 | Chroma, cosine |
| Dense (Large) | fixed 1024 / 128 | Chroma, cosine |
| Dense (Structure-Aware) | one chunk per `#`/`##`/`###` section | Chroma, cosine |

BM25 uses the same chunks as Dense (Small), so comparing those two isolates the retriever.

## Eval dataset format

```json
{
  "name": "my-eval",
  "queries": [
    {"query_id": "q1", "query": "How do I rotate API keys?",
     "relevant_headers": ["Key Rotation"], "source_doc": "api-reference.md"},
    {"query_id": "q2", "query": "...", "relevant_headers": ["Authentication > API Keys"]},
    {"query_id": "q3", "query": "...", "relevant_chunk_ids": ["api-reference.md::structure::0002"]}
  ]
}
```

* `relevant_headers` work with every chunking strategy. A chunk counts as relevant when it
  overlaps the body of a section with that title. Use `A > B` to match a path suffix.
* `relevant_chunk_ids` only apply to the configuration whose chunker produced those IDs.
  Other configurations ignore them, and a query left with no targets is counted as skipped.
* Recall@K is computed over *targets*, not chunks, so finer chunking is not penalised.
