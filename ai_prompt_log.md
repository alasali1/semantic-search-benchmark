# AI Prompt Log & Code Audit

## Prompts used

| Step | Prompt (summary) | Output |
|---|---|---|
| 1-2 | Build a modular pipeline: tiktoken fixed-size + structure-aware Markdown chunkers, ChromaDB dense index (normalised, cosine) + BM25Okapi index, benchmark engine (Hit Rate@K, Recall@K, MRR@K, p50/p95 latency) over 4 configurations | `chunking.py`, `indexer.py`, `benchmarker.py`, sample corpus, eval set, tests |
| 4 | Streamlit dashboard: live BM25-vs-dense comparator with scores and highlighting; benchmark tab with table, charts, chunking comparison and failure analysis | `app.py`, `analysis.py` |
| 3 | Audit the generated code for the five failure modes below | this document |

## Step 3 audit: verdicts

Each point was checked by reading the code **and** running an experiment against the real
implementation and sample corpus (scripts are reproduced in the appendix).

| # | Point | Verdict |
|---|---|---|
| 1 | Ground-truth mapping / leakage | ⚠️ **Compromise:** core design is correct, but two weaknesses |
| 2 | Cosine distance vs. similarity | ✅ **No bug** |
| 3 | BM25 tokenization asymmetry | ✅ **No bug** |
| 4 | Code-fence slicing in the fixed chunker | ❌ **Confirmed defect** → ✅ **fixed** (see below) |
| 5 | MRR math | ✅ **No bug** |

---

### 1. Information / data leakage in ground-truth mapping: ⚠️ compromise

**What is correct.** Ground truth is primarily defined by *section titles*
(`relevant_headers`), not chunk indices. All 20 sample queries use header targets and 0 use
chunk IDs. A chunk counts as relevant by checking which source sections it overlaps
(`section_paths` metadata), so results stay valid when chunk sizes change. Recall is
computed over targets rather than chunks, so strategies that produce more chunks are not
penalised.

**Compromise 1a: index-based chunk IDs are still accepted as targets.**
`chunk_id` = `"{doc}::{strategy}::{index:04d}"` (`chunking.py`, `_make_id`). This is exactly
the "arbitrary chunk array index" the audit warns about. It has two consequences:

* *Stale IDs pass silently.* Adding one section to `api-reference.md` made
  `api-reference.md::structure::0003` point to a different section ("Changelog" instead
  of "Rate Limiting"). The ID still exists, so `usable_targets` accepts it and the query
  is scored against the wrong text.
* *Unequal denominators.* A query with only chunk-ID targets is skipped by every other
  chunking strategy, so configurations can end up averaged over different sets of queries.

**Compromise 1b: very lenient overlap credit for fixed-size chunks.** A chunk is credited
with a section if it overlaps *any single line* of that section's body. Measured:

| Strategy | Chunks | Sections credited per chunk | Credits covering < 25% of the section |
|---|---|---|---|
| fixed-256 | 13 | 3.08 | 4 / 40 |
| fixed-1024 | 4 | 7.50 | 0 / 30 |
| structure | 28 | 1.00 | 0 / 28 |

A 1024-token chunk is "relevant" to ~7.5 different questions at once, and some 256-token
chunks get credit for a section from a few trailing lines. This inflates Hit Rate for
large chunks: it measures "the answer is somewhere inside", not "the chunk is about the answer".

**Resolution.**
1. Replace `relevant_chunk_ids` with **`relevant_spans`**: short verbatim text snippets from
   the source (e.g. `"Revoked keys stop working within 60 seconds"`). At load time each span
   is located in its document and stored as a character range. A chunk is relevant if it
   contains at least 50% of the span's characters. This is independent of chunking, never
   goes stale silently (a span that can't be found raises an error), and removes the
   skipped-query problem.
2. Add character offsets (`start_char`, `end_char`) to chunk metadata so spans and sections
   can be matched precisely.
3. Require a **minimum coverage** for header credit: the chunk must contain at least half of
   the section body, or at least ~100 tokens of it (whichever is smaller). One stray line
   no longer counts.

### 2. Cosine distance vs. cosine similarity inversion: ✅ no bug

* The collection is created with `{"hnsw": {"space": "cosine"}}`. Chroma returns *distances*
  sorted ascending (closest first), and the code converts with `score = 1.0 - distance`
  (`indexer.py`, `DenseVectorIndex.search`), so higher score = more relevant and rank 1 = closest.
* Embeddings are L2-normalised before insertion and at query time (norms measured = 1.000000).

Evidence:
* Chroma's ranking equals a brute-force NumPy dot-product ranking for **20/20** queries.
* Scores are monotonically descending in **20/20** result lists.
* Every chunk retrieves *itself* at rank 1 (**28/28**).

### 3. BM25 tokenization asymmetry: ✅ no bug

* One function, `bm25_tokenize` (lower-case, split on every non-alphanumeric character
  including `_`), is stored as `self._tokenize` and used for **both** the corpus
  (`BM25Index.index`) and queries (`BM25Index.search`).
* Evidence: `"error 409 dimension_mismatch"`, `"ERROR 409 Dimension-Mismatch!!"` and
  ``"Error: 409 (`dimension.mismatch`)?"`` return identical chunks with identical scores.

*Not a bug, but a limitation:* there is no stemming, so `snapshots` ≠ `snapshot` (on both
sides, symmetrically). An optional improvement is a light stemmer (e.g. Snowball) applied
inside `bm25_tokenize`.

### 4. Code-block slicing in the naive chunker: ❌ confirmed defect

`FixedSizeChunker` cuts purely by token count and ignores Markdown structure. Measured:

| Strategy | Chunks with an unclosed or unopened fence |
|---|---|
| fixed-256 | **3 / 13** |
| fixed-1024 | 0 / 4 (no splits by luck of the small corpus) |
| structure | 0 / 28 |

Example: a 256-token chunk ends with `` ```yaml\napiVersion: autoscaling/v ``, an opened
fence cut mid-word. The next chunk starts inside YAML with no opening fence. This damages
the code snippet for embedding, BM25 and anyone reading the result.

**Resolution: fence-aware windowing in `FixedSizeChunker`:**
1. Compute the token ranges of all fenced blocks (re-using `parse_markdown_sections`' fence logic).
2. If a window would end inside a fence, move the end **back** to just before the fence opens,
   provided the chunk keeps at least ~50% of `chunk_size`. Otherwise move it **forward** to the
   closing fence if that stays within a hard cap (e.g. 1.5 × `chunk_size`).
3. If a single code block is larger than the cap, split it but **repair** both halves: append
   a closing ```` ``` ```` to the first chunk and prepend the original opening line
   (e.g. ```` ```yaml ````) to the next.
4. Add a test asserting every fixed-size chunk has balanced fences.

The structure-aware chunker already keeps fences intact. Its own trade-off is unbounded
section size, but no sample section currently exceeds the 256-word-piece model window.

### 5. MRR calculation math: ✅ no bug

`mrr = mean(1/r if r is not None and r <= k else 0.0)` (`benchmarker.py`, `evaluate_index`).
`first_relevant_rank` is 1-indexed (`enumerate(results, start=1)`) and `None` when nothing
relevant was retrieved.

Evidence, running scripted rankings through the real `evaluate_index`:

| Query outcome | MRR@5 | MRR@10 |
|---|---|---|
| relevant at rank 3 | 0.333 | 0.333 |
| relevant at rank 7 (outside K=5) | **0.000** | 0.143 |
| nothing relevant | **0.000** | **0.000** |
| empty result list | **0.000** | **0.000** |

Aggregates: MRR@5 = 0.0833 and MRR@10 = 0.1190 (expected (1/3)/4 and (1/3 + 1/7)/4). There is
no division by zero: misses never reach `1/r`, and `mean([])` is guarded to return 0.0.

*Related note:* queries skipped because of unusable chunk-ID targets (see 1a) are left out of
the denominator rather than counted as 0. Removing chunk-ID targets (resolution 1.1) eliminates this.

---

## Fix applied for item 4: fence-aware fixed-size chunking

**Where:** `src/semantic_search_benchmarker/chunking.py`
* `_fenced_blocks()` (new): finds every fenced code block by byte offset, using the same
  fence rules as `parse_markdown_sections`.
* `FixedSizeChunker.chunk()`: the window loop now checks each cut point:
  1. cut lands inside a block → end the chunk **before** the opening fence (if the chunk
     keeps ≥ 50% of `chunk_size`);
  2. otherwise **extend** to the closing fence (if the chunk stays ≤ 150% of `chunk_size`);
  3. otherwise (block too big for any chunk) **split at a line break and repair**: the first
     half gets a closing fence, the next chunk starts with the original opener (e.g. ```` ```yaml ````).
  Overlap is also prevented from starting a chunk mid-block.
* New metadata `fence_reopened` / `fence_closed` flags repaired chunks. `start_char`/`end_char`
  still point at the source text, excluding the added fence lines.
* `respect_code_blocks=False` keeps the old naive behaviour (strategy name `fixed-…-naive`)
  for comparison.

**Result:** broken fences in the sample corpus 3/13 → **0/13**. There were also 0 in 300 random
documents with code blocks, and every section remains creditable to some chunk (0/300 failures).
Chunk sizes now vary (sample: 134-256 tokens). Benchmark effect: Dense (Small) HR@1
0.85 → 0.90, since whole code examples now embed together.

**Tests added** (`tests/test_chunking.py`): fences are balanced across 5 window sizes for small
and oversized blocks; small blocks are kept whole; oversized blocks are repaired and split only
at line breaks; naive mode still breaks fences; the sample corpus has no broken fences.
`test_char_offsets_locate_chunks_in_the_source` was updated to ignore the added fence lines.

### Related fix: chunking was O(windows × sections)

The post-fix review measured 8.4 s to chunk a 1.8 MB document. The review blamed per-chunk
prefix decoding, **which was wrong**: profiling showed 97% of the time in `_credited_sections`,
which re-derived every section of the document (5.4 million `has_body` calls) for each window.

**Fix:** `_SectionRanges` computes section token ranges once per document; crediting a window
is now a binary search plus a short walk. Section lookup by line also uses binary search, and
character offsets come from a single prefix-count pass. Time: **8.37 s → 0.27 s**.
`test_fast_section_credit_matches_brute_force` checks that the results are identical to the
original per-section scan.

## Other bugs found and fixed during development

| Bug | Impact | Fix |
|---|---|---|
| BM25 dropped results with `score <= 0` | BM25Okapi gives IDF = 0 to a term in exactly half the chunks, so true matches were discarded (caught by a unit test) | Filter on "shares ≥ 1 query term" instead of score |
| Truncation check used tiktoken counts | Dashboard reported 0% of 256-token chunks over MiniLM's window; the real figure (word pieces) is 69% | Count with the embedding model's own tokenizer |
| Chunk labels showed only the first overlapped section | A relevant BM25 hit was labelled "Pagination" while actually covering "Errors" | Label lists every section a chunk spans |
| Chart restyling in the render script replaced the Altair config | Offline renders lost the theme (app itself unaffected) | Set only `config.background` |

## Appendix: reproducing the audit

The audit used two throwaway scripts run with `.venv/Scripts/python`. They:
* count the sections credited per chunk and the coverage fraction (point 1);
* edit a document and compare the content behind the same chunk ID (point 1);
* compare Chroma rankings to brute-force cosine and run the self-retrieval check (point 2);
* compare BM25 results across casing/punctuation variants (point 3);
* count chunks with an odd number of fence lines (point 4);
* push scripted rankings through `evaluate_index` (point 5).
