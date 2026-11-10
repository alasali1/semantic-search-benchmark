"""Streamlit dashboard: live BM25-vs-dense search comparator and benchmark explorer.

Run with ``uv run semantic-search-dashboard`` (or ``uv run streamlit run
src/semantic_search_benchmarker/app.py``).

Indexes are built once per (corpus contents, embedding backend) and cached with
``st.cache_resource``. Both tabs share those indexes, so running the benchmark never
re-embeds the corpus.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from semantic_search_benchmarker.analysis import HeadToHead, chunk_label, head_to_head, query_terms, summarize
from semantic_search_benchmarker.benchmarker import (
    BenchmarkConfig,
    BenchmarkReport,
    BuiltIndex,
    EvalDataset,
    EvalQuery,
    build_default_configs,
    build_index,
    evaluate_indexes,
    load_corpus,
    load_eval_dataset,
    matched_targets,
    usable_targets,
)
from semantic_search_benchmarker.indexer import BM25Index, Embedder, SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = PROJECT_ROOT / "data" / "docs"
DEFAULT_QUERIES = PROJECT_ROOT / "data" / "eval_queries.json"
K_OPTIONS = [1, 3, 5, 10, 20]

# --------------------------------------------------------------------------------------
# Visual tokens: reference data-viz palette, categorical slots 1-4, light and dark steps.
# Colour follows the configuration (entity), in fixed order, across every chart.
# --------------------------------------------------------------------------------------
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
CHROME = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

CSS = """
<style>
.ssb-card { border: 1px solid rgba(128,128,128,.28); border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
.ssb-head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; font-size: .9rem; }
.ssb-rank { font-weight: 700; }
.ssb-score { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-variant-numeric: tabular-nums; }
.ssb-terms { opacity: .75; font-size: .8rem; }
.ssb-rel { font-size: .78rem; font-weight: 600; color: #0ca30c; border: 1px solid #0ca30c; border-radius: 10px; padding: 0 7px; }
.ssb-loc { opacity: .7; font-size: .8rem; margin: 2px 0 6px; }
.ssb-body { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .78rem; white-space: pre-wrap;
            max-height: 230px; overflow-y: auto; line-height: 1.45; }
.ssb-body mark { background: rgba(237,161,0,.38); color: inherit; border-radius: 2px; padding: 0 1px; }
</style>
"""


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSuite:
    """Every index the dashboard needs, built from one corpus snapshot."""

    benchmark: list[BuiltIndex]  # the four standard configurations, in report order
    live: dict[str, tuple[BuiltIndex, BuiltIndex]]  # chunking label -> (bm25, dense)
    embedder_name: str
    model_window: int | None  # embedding input limit in model tokens, if known
    model_token_counts: dict[str, int]  # chunk_id -> length in the embedding model's tokens


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(backend: str, model: str) -> Embedder:
    """Load (once per process) the embedding model shared by all dense indexes."""
    from semantic_search_benchmarker.indexer import OpenAIEmbedder, SentenceTransformerEmbedder

    if backend == "openai":
        return OpenAIEmbedder(model) if model else OpenAIEmbedder()
    return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()


@st.cache_resource(show_spinner="Chunking and indexing the corpus…", max_entries=3)
def build_suite(fingerprint: str, backend: str, model: str, _corpus: dict[str, str]) -> IndexSuite:
    """Build all indexes. Cached by corpus fingerprint, so editing a doc triggers a rebuild."""
    import chromadb

    embedder = get_embedder(backend, model)
    configs = build_default_configs(
        embedder,
        chroma_client=chromadb.EphemeralClient(),
        collection_prefix=f"ssb-{backend[:6]}-{fingerprint[:10]}",  # unique per cache entry
    )
    bm25_small, dense_small, dense_large, dense_structure = (build_index(c, _corpus) for c in configs)

    def bm25_for(dense: BuiltIndex) -> BuiltIndex:
        chunker = dense.config.chunker
        return build_index(BenchmarkConfig(f"BM25 ({chunker.strategy})", chunker, BM25Index), _corpus)

    window = getattr(embedder, "max_seq_length", None)
    counter = getattr(embedder, "count_tokens", None)
    dense_chunks = {c.chunk_id: c.content for b in (dense_small, dense_large, dense_structure) for c in b.chunks}
    model_counts = dict(zip(dense_chunks, counter(list(dense_chunks.values())), strict=True)) if counter else {}
    return IndexSuite(
        benchmark=[bm25_small, dense_small, dense_large, dense_structure],
        live={
            "256-token chunks": (bm25_small, dense_small),
            "1024-token chunks": (bm25_for(dense_large), dense_large),
            "Structure-aware": (bm25_for(dense_structure), dense_structure),
        },
        embedder_name=embedder.name,
        model_window=int(window) if window else (8191 if backend == "openai" else None),
        model_token_counts=model_counts,
    )


def corpus_fingerprint(corpus: dict[str, str]) -> str:
    """Content hash of the corpus, used as the index cache key."""
    digest = hashlib.sha256()
    for doc_id in sorted(corpus):
        digest.update(doc_id.encode())
        digest.update(b"\0")
        digest.update(corpus[doc_id].encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def theme_mode() -> str:
    """'dark' or 'light' to match Streamlit's active theme (defaults to light)."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escape ``text`` and wrap whole-word, case-insensitive matches of ``terms`` in <mark>.

    Word boundaries mirror the BM25 tokenizer (alphanumerics only), so ``max`` is
    highlighted inside ``max_tokens``, consistent with how BM25 saw the text.
    """
    if not terms:
        return html.escape(text)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    out: list[str] = []
    pos = 0
    for m in pattern.finditer(text):
        out.append(html.escape(text[pos : m.start()]))
        out.append(f"<mark>{html.escape(m.group(0))}</mark>")
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def result_card(result: SearchResult, terms: list[str], score_label: str, relevant: bool | None) -> str:
    """HTML card for one search hit: rank, score, term matches, location, highlighted text."""
    content_tokens = set(re.split(r"[\W_]+", result.content.lower()))
    n_matched = sum(t in content_tokens for t in terms)
    meta = result.metadata
    lines = f"lines {meta.get('start_line', '?')}–{meta.get('end_line', '?')}"
    badge = '<span class="ssb-rel">✓ relevant</span>' if relevant else ""
    return (
        '<div class="ssb-card">'
        f'<div class="ssb-head"><span class="ssb-rank">#{result.rank}</span>'
        f'<span class="ssb-score">{score_label} {result.score:.3f}</span>'
        f'<span class="ssb-terms">{n_matched}/{len(terms)} query terms</span>{badge}</div>'
        f'<div class="ssb-loc">{html.escape(chunk_label(result))} · {lines} · '
        f"{meta.get('token_count', '?')} tok</div>"
        f'<div class="ssb-body">{highlight(result.content, terms)}</div>'
        "</div>"
    )


def _styled(chart: alt.TopLevelMixin, mode: str) -> alt.TopLevelMixin:
    """Recessive chrome: hairline solid grid, muted axis labels, system sans."""
    c = CHROME[mode]
    return (
        chart.configure(background="transparent", font=FONT)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=c["grid"], gridWidth=1, domainColor=c["axis"], tickColor=c["axis"],
            labelColor=c["muted"], titleColor=c["secondary"], labelFontSize=11,
            titleFontSize=11, titleFontWeight="normal",
        )
        .configure_header(labelColor=c["primary"], labelFontSize=13, labelFontWeight=600, title=None)
        .configure_legend(
            labelColor=c["secondary"], titleColor=c["secondary"], orient="top",
            symbolType="square", labelFontSize=12, title=None, labelLimit=260,
        )
    )


def quality_chart(report: BenchmarkReport, mode: str) -> alt.TopLevelMixin:
    """Grouped bars of Hit Rate@K and MRR@K (small multiples on one shared 0-1 axis)."""
    names = [r.name for r in report.results]
    rows = [
        {"config": r.name, "K": f"@{k}", "metric": metric, "value": values[k]}
        for r in report.results
        for metric, values in (("Hit Rate@K", r.hit_rate), ("MRR@K", r.mrr))
        for k in report.ks
    ]
    k_order = [f"@{k}" for k in report.ks]
    base = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("K:N", sort=k_order, title=None, scale=alt.Scale(paddingInner=0.3), axis=alt.Axis(labelAngle=0, domain=True, ticks=False)),
        xOffset=alt.XOffset("config:N", sort=names, scale=alt.Scale(paddingInner=0.12)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5, domain=False, ticks=False)),
        color=alt.Color("config:N", sort=names, scale=alt.Scale(domain=names, range=SERIES[mode])),
        tooltip=[
            alt.Tooltip("config:N", title="Configuration"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("K:N"),
            alt.Tooltip("value:Q", title="Value", format=".3f"),
        ],
    ).properties(width=250, height=240)
    return _styled(base.facet(column=alt.Column("metric:N", title=None, sort=["Hit Rate@K", "MRR@K"])), mode)


def latency_chart(report: BenchmarkReport, mode: str) -> alt.TopLevelMixin:
    """Horizontal bars of mean query latency; each bar labelled since BM25 is ~100x faster."""
    names = [r.name for r in report.results]
    df = pd.DataFrame(
        {
            "config": r.name,
            "mean": r.latency_mean_ms,
            "p50": r.latency_p50_ms,
            "p95": r.latency_p95_ms,
            "label": f"{r.latency_mean_ms:.2f} ms",
        }
        for r in report.results
    )
    enc_y = alt.Y("config:N", sort=names, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=240))
    tooltip = [
        alt.Tooltip("config:N", title="Configuration"),
        alt.Tooltip("mean:Q", title="Mean (ms)", format=".2f"),
        alt.Tooltip("p50:Q", title="p50 (ms)", format=".2f"),
        alt.Tooltip("p95:Q", title="p95 (ms)", format=".2f"),
    ]
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=enc_y,
        x=alt.X("mean:Q", title="Mean query latency (ms)", axis=alt.Axis(tickCount=5, domain=False, ticks=False)),
        color=alt.Color("config:N", sort=names, scale=alt.Scale(domain=names, range=SERIES[mode]), legend=None),
        tooltip=tooltip,
    )
    labels = alt.Chart(df).mark_text(align="left", dx=5, fontSize=11, color=CHROME[mode]["secondary"]).encode(
        y=enc_y, x="mean:Q", text="label:N", tooltip=tooltip
    )
    return _styled((bars + labels).properties(height=4 * 34, width="container"), mode)


def report_table(report: BenchmarkReport, suite: IndexSuite) -> pd.DataFrame:
    """Full comparison table (also the accessible table view of the charts)."""
    strategies = {b.config.name: b.config.chunker.strategy for b in suite.benchmark}
    rows = []
    for r in report.results:
        row: dict[str, object] = {
            "Configuration": r.name,
            "Chunking": strategies.get(r.name, r.chunk_strategy),
            "Chunks": r.num_chunks,
            "Mean chunk tokens": round(r.mean_chunk_tokens),
        }
        row |= {f"HR@{k}": r.hit_rate[k] for k in report.ks}
        row |= {f"MRR@{k}": r.mrr[k] for k in report.ks}
        row |= {f"Recall@{k}": r.recall[k] for k in report.ks}
        row |= {
            "Mean latency (ms)": r.latency_mean_ms,
            "p50 (ms)": r.latency_p50_ms,
            "p95 (ms)": r.latency_p95_ms,
            "Index time (s)": r.index_time_s,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Configuration")


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------


def live_search_tab(suite: IndexSuite, dataset: EvalDataset | None) -> None:
    """Side-by-side BM25 vs dense results for an ad-hoc query."""
    queries_by_text: dict[str, EvalQuery] = {q.query: q for q in dataset.queries} if dataset else {}

    def use_example() -> None:
        if st.session_state.get("example"):
            st.session_state["live_query"] = st.session_state["example"]

    if queries_by_text:
        st.selectbox(
            "Try a query from the eval set (relevant hits get a ✓ badge)",
            [""] + list(queries_by_text),
            key="example",
            on_change=use_example,
            format_func=lambda q: q or "—",
        )

    c_query, c_k, c_chunk = st.columns([6, 1.2, 3])
    query = c_query.text_input("Search query", key="live_query", placeholder="e.g. how do I rotate API keys?")
    k = c_k.number_input("Top K", min_value=1, max_value=25, value=5, step=1)
    chunking = c_chunk.segmented_control(
        "Chunking (same chunks for both sides)", list(suite.live), default="256-token chunks"
    ) or "256-token chunks"

    if not query.strip():
        st.info("Enter a query to compare keyword and semantic retrieval side by side.")
        return

    bm25_built, dense_built = suite.live[chunking]
    terms = query_terms(query)
    eval_query = queries_by_text.get(query.strip())

    col_bm25, col_dense = st.columns(2, gap="large")
    for col, built, title, score_label in (
        (col_bm25, bm25_built, "BM25 keyword search", "BM25"),
        (col_dense, dense_built, f"Dense vector search · {suite.embedder_name}", "cosine"),
    ):
        t0 = time.perf_counter()
        results = built.retriever.search(query, int(k))
        latency = (time.perf_counter() - t0) * 1000
        with col:
            st.subheader(title)
            st.caption(f"{len(results)} results · {latency:.1f} ms · {len(built.chunks)} chunks indexed")
            if not results:
                st.warning("No results. BM25 only returns chunks that share at least one term with the query.")
            targets = usable_targets(eval_query) if eval_query else set()
            cards = [
                result_card(
                    r,
                    terms,
                    score_label,
                    bool(matched_targets(r, eval_query) & targets) if eval_query else None,
                )
                for r in results
            ]
            st.markdown("".join(cards), unsafe_allow_html=True)

    st.caption(
        "Scores are not comparable across columns: BM25 is an unbounded sum of term weights, "
        "cosine similarity of unit vectors lies in [-1, 1]. Highlighted words are query terms "
        "(stop words removed). Dense retrieval does not use them; they show lexical overlap only."
    )


def benchmark_tab(suite: IndexSuite, dataset: EvalDataset | None, num_docs: int, mode: str) -> None:
    """Run the evaluation suite and explore the results."""
    if dataset is None:
        st.error("No evaluation dataset loaded. Check the queries path in the sidebar.")
        return

    c_k, c_btn = st.columns([4, 1], vertical_alignment="bottom")
    ks = c_k.pills("Cut-offs (K)", K_OPTIONS, selection_mode="multi", default=[3, 5, 10])
    run = c_btn.button("Run Benchmark", type="primary", width="stretch", disabled=not ks)

    if run:
        with st.spinner(f"Evaluating {len(dataset.queries)} queries × {len(suite.benchmark)} configurations…"):
            st.session_state["report"] = evaluate_indexes(suite.benchmark, dataset, sorted(ks), num_docs=num_docs)
            st.session_state["report_key"] = st.session_state.get("suite_key")

    report: BenchmarkReport | None = st.session_state.get("report")
    if report is None or st.session_state.get("report_key") != st.session_state.get("suite_key"):
        st.info(f"Click **Run Benchmark** to evaluate {len(dataset.queries)} queries from `{dataset.name}`.")
        return

    st.caption(
        f"{report.num_queries} queries · {report.num_docs} documents · embedder `{suite.embedder_name}` · "
        f"run at {report.created_at:%H:%M:%S} UTC"
    )

    # --- Comparison table (also the table view behind the charts) ---------------------
    st.subheader("Comparison")
    table = report_table(report, suite)
    pct_cols = [c for c in table.columns if c.startswith(("HR@", "MRR@", "Recall@"))]
    st.dataframe(
        table,
        width="stretch",
        column_config={
            **{c: st.column_config.NumberColumn(c, format="%.3f") for c in pct_cols},
            **{c: st.column_config.NumberColumn(c, format="%.2f") for c in ("Mean latency (ms)", "p50 (ms)", "p95 (ms)", "Index time (s)")},
        },
    )

    c_quality, c_latency = st.columns([3, 2], gap="large")
    with c_quality:
        st.markdown("**Retrieval quality**")
        st.altair_chart(quality_chart(report, mode), theme=None)
    with c_latency:
        st.markdown("**Average query latency**")
        st.altair_chart(latency_chart(report, mode), theme=None, width="stretch")
        st.caption("Dense latency includes embedding the query. Hover for p50 / p95.")

    chunking_section(report, suite)
    failure_analysis(report, suite, dataset)


def chunking_section(report: BenchmarkReport, suite: IndexSuite) -> None:
    """Compare the three dense chunking strategies with the caveats that matter."""
    st.subheader("Chunking strategies (dense retrieval)")
    k_max = report.ks[-1]
    built = {b.config.name: b for b in suite.benchmark}
    rows = []
    notes: list[str] = []
    for r in report.results:
        b = built.get(r.name)
        if b is None or isinstance(b.retriever, BM25Index):
            continue
        tokens = [c.metadata.get("token_count", 0) for c in b.chunks]
        # Truncation is decided by the model's tokenizer, not tiktoken; fall back if unknown.
        model_tokens = [suite.model_token_counts.get(c.chunk_id, t) for c, t in zip(b.chunks, tokens, strict=True)]
        over = (
            sum(t > suite.model_window for t in model_tokens) / len(model_tokens)
            if suite.model_window and tokens
            else 0.0
        )
        rows.append(
            {
                "Strategy": r.name,
                "Chunks": r.num_chunks,
                "Mean tokens": round(r.mean_chunk_tokens),
                "Max tokens": max(tokens, default=0),
                "Over model window": over,
                f"HR@{k_max}": r.hit_rate[k_max],
                f"MRR@{k_max}": r.mrr[k_max],
            }
        )
        trivial = [k for k in report.ks if k >= r.num_chunks]
        if trivial:
            notes.append(
                f"**{r.name}** has only {r.num_chunks} chunks, so at K ≥ {r.num_chunks} every chunk is "
                f"returned and @{'/@'.join(map(str, trivial))} metrics are trivially perfect."
            )
        if over > 0:
            notes.append(
                f"**{r.name}**: {over:.0%} of chunks exceed the model's {suite.model_window}-token input "
                "window (counted with the model's own tokenizer). Text past the window is silently "
                "truncated and never embedded."
            )
    st.dataframe(
        pd.DataFrame(rows).set_index("Strategy"),
        width="stretch",
        column_config={
            "Over model window": st.column_config.NumberColumn(format="percent"),
            f"HR@{k_max}": st.column_config.NumberColumn(format="%.3f"),
            f"MRR@{k_max}": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    for note in notes:
        st.markdown(f"⚠️ {note}")


def failure_analysis(report: BenchmarkReport, suite: IndexSuite, dataset: EvalDataset) -> None:
    """Queries where BM25 beat dense retrieval and vice versa, with diagnostics."""
    st.subheader("Failure analysis: BM25 vs. dense")
    bm25_built = suite.benchmark[0]
    dense_names = [b.config.name for b in suite.benchmark[1:]]
    dense_name = st.selectbox(
        "Compare BM25 against",
        dense_names,
        help="The first option uses the same 256-token chunks as BM25, so only the retriever differs.",
    )
    dense_built = next(b for b in suite.benchmark if b.config.name == dense_name)
    results = {r.name: r for r in report.results}

    comparisons = head_to_head(
        dataset,
        results[bm25_built.config.name],
        results[dense_name],
        bm25_built,
        dense_built,
        dense_window_tokens=suite.model_window,
        model_token_counts=suite.model_token_counts,
    )
    counts = summarize(comparisons)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BM25 wins", counts["bm25"])
    m2.metric("Dense wins", counts["dense"])
    m3.metric("Ties", counts["tie"])
    m4.metric("Both missed", counts["both_missed"])
    st.caption(
        f"A win means a higher reciprocal rank for the first relevant chunk within the top {report.ks[-1]}. "
        "Explanations are heuristics based on term overlap, IDF, identifiers and chunk length."
    )

    col_b, col_d = st.columns(2, gap="large")
    for col, side, title in ((col_b, "bm25", "Where BM25 beat dense"), (col_d, "dense", "Where dense beat BM25")):
        with col:
            st.markdown(f"**{title}**")
            wins = [c for c in comparisons if c.winner == side]
            if not wins:
                st.caption("No queries.")
            for c in wins:
                _comparison_expander(c)

    missed = [c for c in comparisons if c.winner == "tie" and c.bm25_rank is None]
    if missed:
        with st.expander(f"Queries both systems missed ({len(missed)})"):
            for c in missed:
                st.markdown(f"- **{c.query_id}** {c.query} · target: {', '.join(c.targets)}")


def _comparison_expander(c: HeadToHead) -> None:
    def rank(r: int | None) -> str:
        return f"#{r}" if r else "miss"

    with st.expander(f"{c.query_id} · BM25 {rank(c.bm25_rank)} vs dense {rank(c.dense_rank)} · {c.query}"):
        st.markdown(f"**Target:** {', '.join(c.targets)}")
        if c.term_overlap is not None:
            st.markdown(
                f"**Query terms found in the target section:** {c.term_overlap:.0%} "
                f"({', '.join(c.matched_terms) or 'none'}"
                + (f"; missing: {', '.join(c.missing_terms)}" if c.missing_terms else "")
                + ")"
            )
        st.markdown(f"**BM25 top hit:** {c.bm25_top or '—'}  \n**Dense top hit:** {c.dense_top or '—'}")
        st.markdown("**Why:**\n" + "\n".join(f"- {reason}" for reason in c.reasons))


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Semantic Search Benchmarker", page_icon="🔎", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Semantic Search Benchmarker")
    st.caption("BM25 keyword search vs. dense vector retrieval over technical Markdown docs")

    with st.sidebar:
        st.header("Data")
        docs_dir = st.text_input("Docs directory", str(DEFAULT_DOCS))
        queries_path = st.text_input("Eval queries", str(DEFAULT_QUERIES))
        st.header("Embeddings")
        backend = st.radio("Backend", ["sentence-transformers", "openai"], horizontal=True)
        model = st.text_input(
            "Model (blank = default)",
            "",
            help="Defaults: all-MiniLM-L6-v2 (local) or text-embedding-3-small (needs OPENAI_API_KEY).",
        )

    try:
        corpus = load_corpus(docs_dir)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load corpus: {exc}")
        st.stop()

    dataset: EvalDataset | None
    try:
        dataset = load_eval_dataset(queries_path, corpus)
    except (OSError, ValueError) as exc:
        dataset = None
        st.sidebar.warning(f"Eval dataset not loaded: {exc}")

    if backend == "openai":
        from dotenv import load_dotenv

        load_dotenv()

    fingerprint = corpus_fingerprint(corpus)
    try:
        suite = build_suite(fingerprint, backend, model.strip(), corpus)
    except Exception as exc:  # surface model/API failures in the UI instead of a stack trace
        st.error(f"Failed to build indexes with backend `{backend}`: {exc}")
        st.stop()

    # The benchmark report is only valid for the indexes it was computed on.
    st.session_state["suite_key"] = (fingerprint, backend, model.strip(), queries_path)

    with st.sidebar:
        st.header("Corpus")
        st.markdown(
            f"{len(corpus)} documents · {len(dataset.queries) if dataset else 0} eval queries  \n"
            f"Embedder: `{suite.embedder_name}`"
        )
        if st.button("Rebuild indexes"):
            build_suite.clear()
            st.rerun()

    live, bench = st.tabs(["🔎 Live Search Comparator", "📊 Benchmark & Evaluation"])
    with live:
        live_search_tab(suite, dataset)
    with bench:
        benchmark_tab(suite, dataset, len(corpus), theme_mode())


if __name__ == "__main__":  # Streamlit executes the script as __main__
    main()
