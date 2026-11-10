"""Tests for Markdown parsing and both chunking strategies."""

import re
from pathlib import Path

import pytest

from semantic_search_benchmarker.chunking import (
    DocumentChunk,
    FixedSizeChunker,
    StructureAwareChunker,
    _credited_sections,
    parse_markdown_sections,
)

DOC = """Intro paragraph before any header.

# Guide

## Install

Run this:

```bash
# this is a shell comment, not a header
pip install lumen
## also not a header
```

#### Deep header stays inside Install

## Configure ##

Set `m = 16`.

### Advanced

~~~
# tilde fence comment
~~~
Tuning notes.
"""


def test_sections_respect_code_fences_and_levels() -> None:
    sections = parse_markdown_sections(DOC)
    paths = [s.header_path for s in sections]
    assert paths == [
        (),
        ("Guide",),
        ("Guide", "Install"),
        ("Guide", "Configure"),
        ("Guide", "Configure", "Advanced"),
    ]
    install = sections[2]
    assert "# this is a shell comment" in install.text
    assert "## also not a header" in install.text
    assert "#### Deep header" in install.text
    assert not sections[1].has_body  # "# Guide" is immediately followed by "## Install"


def test_unterminated_fence_runs_to_eof() -> None:
    sections = parse_markdown_sections("# A\n```\n# not header\n")
    assert [s.header_path for s in sections] == [("A",)]


def test_structure_chunker_one_chunk_per_section_with_lines() -> None:
    chunks = StructureAwareChunker().chunk(DOC, "doc.md")
    assert [c.header_path for c in chunks] == [
        [],
        ["Guide", "Install"],
        ["Guide", "Configure"],
        ["Guide", "Configure", "Advanced"],
    ]
    install = chunks[1]
    assert install.content.startswith("## Install")
    assert install.content.count("```") == 2  # code block kept whole
    lines = DOC.splitlines()
    assert lines[install.metadata["start_line"] - 1] == "## Install"
    assert lines[install.metadata["end_line"] - 1] == "#### Deep header stays inside Install"
    assert install.chunk_id == "doc.md::structure::0001"
    assert install.parent_doc == "doc.md"


def test_fixed_chunker_respects_size_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(text, "d.md")
    assert len(chunks) > 1
    assert all(c.metadata["token_count"] <= 50 for c in chunks)
    # Consecutive windows share their 10-token overlap.
    first_tail = chunker._encoding.encode(chunks[0].content)[-5:]
    assert chunker._encoding.decode(first_tail).strip() in chunks[1].content
    # The windows together cover the whole document.
    assert chunks[0].content.startswith("word0") and chunks[-1].content.endswith("word499")


def test_fixed_chunker_line_numbers_and_sections() -> None:
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=0).chunk(DOC, "doc.md")
    lines = DOC.splitlines()
    for c in chunks:
        start, end = c.metadata["start_line"], c.metadata["end_line"]
        assert 1 <= start <= end <= len(lines)
        # Chunk text begins on its reported start line.
        assert c.content.splitlines()[0].strip() in lines[start - 1]
    covered = {tuple(p) for c in chunks for p in c.section_paths}
    assert ("Guide", "Install") in covered and ("Guide", "Configure", "Advanced") in covered
    assert ("Guide",) not in covered  # header-only sections are never attributed


def test_fixed_chunker_handles_multibyte_text() -> None:
    text = "héllo wörld — ünïcödé ✓ " * 50
    chunks = FixedSizeChunker(chunk_size=16, chunk_overlap=4).chunk(text, "u.md")
    assert chunks and all(c.content for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_fixed_chunker_rejects_bad_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size, overlap)


def test_empty_document_yields_no_chunks() -> None:
    assert FixedSizeChunker(10, 2).chunk("   \n", "e.md") == []
    assert StructureAwareChunker().chunk("   \n", "e.md") == []


# --------------------------------------------------------------------------------------
# Section credit (minimum coverage) and character offsets
# --------------------------------------------------------------------------------------

LONG_BODY = "x" * 400
CREDIT_DOC = f"# T\n\n## Long\n{LONG_BODY}\n\n## Short\nyyyyyyyyyy\n"


def _credit(text: str, win_start: int, win_end: int, fraction: float = 0.5, floor: int = 100) -> set[str]:
    """Run the credit rule with one token per byte, so thresholds are exact."""
    raw = text.encode()
    line_starts = [0] + [i + 1 for i, b in enumerate(raw) if b == ord("\n")]
    offsets = list(range(len(raw) + 1))
    paths = _credited_sections(
        parse_markdown_sections(text), line_starts, offsets, len(raw), win_start, win_end, fraction, floor
    )
    return {p[-1] for p in paths}


def test_stray_lines_do_not_earn_section_credit() -> None:
    short_start = CREDIT_DOC.index("yyyyyyyyyy")
    # The Short body is "yyyyyyyyyy\n" = 11 tokens, so half of it is 5.5 -> 6 tokens needed.
    assert "Short" not in _credit(CREDIT_DOC, short_start - 50, short_start + 5)
    assert "Short" in _credit(CREDIT_DOC, short_start - 50, short_start + 6)


def test_long_sections_need_floor_tokens_not_half() -> None:
    long_start = CREDIT_DOC.index(LONG_BODY)
    # Half of the Long body is ~200 tokens, but 100 tokens is enough (the floor).
    assert "Long" in _credit(CREDIT_DOC, long_start, long_start + 100)
    assert "Long" not in _credit(CREDIT_DOC, long_start, long_start + 99)


def test_credit_ignores_headers_and_preamble() -> None:
    long_header = CREDIT_DOC.index("## Long")
    # A window covering only header lines earns no body credit.
    assert _credit(CREDIT_DOC, 0, long_header + len("## Long")) == set()


def test_every_section_is_credited_to_some_chunk() -> None:
    for size, overlap in [(20, 0), (16, 4), (64, 8), (400, 50)]:
        chunks = FixedSizeChunker(size, overlap).chunk(DOC, "doc.md")
        credited = {tuple(p) for c in chunks for p in c.section_paths}
        wanted = {s.header_path for s in parse_markdown_sections(DOC) if s.header_path and s.has_body}
        assert wanted <= credited, (size, overlap, wanted - credited)


def test_lenient_settings_reproduce_the_old_any_overlap_credit() -> None:
    short_start = CREDIT_DOC.index("yyyyyyyyyy")
    window = (short_start - 50, short_start + 1)  # a single stray character of Short
    assert "Short" in _credit(CREDIT_DOC, *window, fraction=1e-9, floor=1)
    assert "Short" not in _credit(CREDIT_DOC, *window)


@pytest.mark.parametrize("kwargs", [{"min_section_fraction": 0}, {"min_section_fraction": 1.5}, {"min_section_tokens": 0}])
def test_fixed_chunker_rejects_bad_credit_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(64, 8, **kwargs)


def _source_text(c: DocumentChunk) -> str:
    """Chunk content minus fence lines added when a code block was split and repaired."""
    lines = c.content.split("\n")
    if c.metadata.get("fence_reopened"):
        lines = lines[1:]
    if c.metadata.get("fence_closed"):
        lines = lines[:-1]
    return "\n".join(lines)


def test_char_offsets_locate_chunks_in_the_source() -> None:
    for chunker in (FixedSizeChunker(20, 4), FixedSizeChunker(64, 0), StructureAwareChunker()):
        for c in chunker.chunk(DOC, "doc.md"):
            assert DOC[c.metadata["start_char"] : c.metadata["end_char"]] == _source_text(c)


def test_char_offsets_with_multibyte_text_and_crlf() -> None:
    text = "héllo wörld — ünïcödé ✓ " * 50
    for c in FixedSizeChunker(16, 4).chunk(text, "u.md"):
        assert text[c.metadata["start_char"] : c.metadata["end_char"]] == c.content
    crlf = DOC.replace("\n", "\r\n")
    for c in StructureAwareChunker().chunk(crlf, "doc.md"):
        sliced = crlf[c.metadata["start_char"] : c.metadata["end_char"]]
        assert sliced.replace("\r\n", "\n") == c.content


# --------------------------------------------------------------------------------------
# Code fences in the fixed-size chunker (audit item #4) and fast section credit
# --------------------------------------------------------------------------------------

_FENCE_LINE = re.compile(r"^ {0,3}(```|~~~)", re.MULTILINE)
SMALL_BLOCK = "```bash\n# install\npip install lumen\nlumen serve --port 7700\n```"
BIG_BLOCK = "```yaml\n" + "".join(f"key_{i}: value_{i}\n" for i in range(120)) + "```"


def _doc_with(block: str, prose_words: int = 120) -> str:
    prose = " ".join(f"word{i}" for i in range(prose_words))
    return f"# Guide\n\n## Setup\n\n{prose}\n\n{block}\n\n## After\n\n{prose}\n"


def _balanced(c: DocumentChunk) -> bool:
    return len(_FENCE_LINE.findall(c.content)) % 2 == 0


@pytest.mark.parametrize("size,overlap", [(16, 4), (32, 8), (64, 16), (100, 0), (256, 32)])
def test_fixed_chunks_never_leave_a_fence_open(size: int, overlap: int) -> None:
    for block in (SMALL_BLOCK, BIG_BLOCK):
        for c in FixedSizeChunker(size, overlap).chunk(_doc_with(block), "d.md"):
            assert _balanced(c), (size, overlap, c.content)


def test_small_block_is_kept_whole_in_one_chunk() -> None:
    chunks = FixedSizeChunker(64, 8).chunk(_doc_with(SMALL_BLOCK), "d.md")
    holding = [c for c in chunks if "pip install lumen" in c.content]
    assert holding and all(SMALL_BLOCK in c.content for c in holding)
    assert not any(c.metadata["fence_reopened"] or c.metadata["fence_closed"] for c in chunks)


def test_oversized_block_is_split_and_repaired() -> None:
    chunks = FixedSizeChunker(64, 8).chunk(_doc_with(BIG_BLOCK), "d.md")
    closed = [c for c in chunks if c.metadata["fence_closed"]]
    reopened = [c for c in chunks if c.metadata["fence_reopened"]]
    assert closed and reopened
    assert all(c.content.endswith("\n```") for c in closed)
    assert all(c.content.startswith("```yaml\n") for c in reopened)
    # Splits fall on line breaks: no code line is cut in half.
    code_lines = {f"key_{i}: value_{i}" for i in range(120)}
    for c in chunks:
        for line in c.content.splitlines():
            if line.startswith("key_"):
                assert line in code_lines, line
    # Every line of the block survives somewhere, in order across chunks.
    assert code_lines <= {ln for c in chunks for ln in c.content.splitlines()}


def test_naive_mode_keeps_the_old_behaviour() -> None:
    naive = FixedSizeChunker(64, 8, respect_code_blocks=False)
    assert naive.strategy == "fixed-64-8-naive"
    assert any(not _balanced(c) for c in naive.chunk(_doc_with(BIG_BLOCK), "d.md"))


def test_sample_corpus_has_no_broken_fences() -> None:
    from semantic_search_benchmarker.benchmarker import load_corpus

    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "docs")
    for size, overlap in [(256, 32), (1024, 128), (64, 8)]:
        for c in FixedSizeChunker(size, overlap).chunk_documents(corpus):
            assert _balanced(c), c.chunk_id


def _brute_force_credit(text: str, chunker: FixedSizeChunker, c: DocumentChunk) -> list[list[str]]:
    """The original per-window section scan, kept as an independent reference."""
    enc = chunker._encoding
    tokens = enc.encode(text, disallowed_special=())
    offsets = [0]
    for tb in enc.decode_tokens_bytes(tokens):
        offsets.append(offsets[-1] + len(tb))
    raw = text.encode()
    line_starts = [0] + [m.end() for m in re.finditer(rb"\n", raw)]
    # Recover the chunk's token window from its byte span.
    b0 = len(text[: c.metadata["start_char"]].encode())
    b1 = len(text[: c.metadata["end_char"]].encode())
    win_start = max(i for i, o in enumerate(offsets) if o <= b0)
    win_end = min(i for i, o in enumerate(offsets) if o >= b1)
    min_tokens = max(1, min(chunker.min_section_tokens, chunker.chunk_size // 2))
    out = []
    for s in parse_markdown_sections(text):
        if not s.header_path or not s.has_body:
            continue
        body_b0 = line_starts[s.body_start_line - 1]
        body_b1 = line_starts[s.end_line] if s.end_line < len(line_starts) else len(raw)
        t0 = max(i for i, o in enumerate(offsets) if o <= body_b0)
        t1 = min(i for i, o in enumerate(offsets) if o >= body_b1)
        overlap = min(win_end, t1) - max(win_start, t0)
        if t1 > t0 and overlap > 0 and overlap >= min(chunker.min_section_fraction * (t1 - t0), min_tokens):
            out.append(list(s.header_path))
    return out


def test_fast_section_credit_matches_brute_force() -> None:
    text = "# T\n\n" + "".join(
        f"## S{i}\n\n" + " ".join(f"w{i}x{j}" for j in range(3 + (i * 7) % 40)) + "\n\n" for i in range(25)
    )
    for size, overlap in [(16, 4), (40, 0), (64, 16)]:
        chunker = FixedSizeChunker(size, overlap)
        for c in chunker.chunk(text, "d.md"):
            assert c.section_paths == _brute_force_credit(text, chunker, c), (size, overlap, c.chunk_id)


# --------------------------------------------------------------------------------------
# Step 5 QA suite: Tests 1 and 2
# --------------------------------------------------------------------------------------

OVERLAP_TEXTS = {
    "ascii": " ".join(f"token{i}" for i in range(400)),
    # CJK and emoji: tiktoken splits some of these characters across two tokens.
    "multibyte": "部署 向量 数据库 🚀🔥 " * 120,
}


@pytest.mark.parametrize("kind", list(OVERLAP_TEXTS))
@pytest.mark.parametrize("size,overlap", [(16, 4), (32, 8), (50, 10), (16, 0)])
def test_1_chunk_overlap_integrity(kind: str, size: int, overlap: int) -> None:
    """Test 1: consecutive fixed chunks share exactly `overlap` tokens and drop no characters."""
    text = OVERLAP_TEXTS[kind]
    chunker = FixedSizeChunker(size, overlap)
    token_bytes = chunker._encoding.decode_tokens_bytes(chunker._encoding.encode(text))
    chunks = chunker.chunk(text, "d.md")
    assert len(chunks) > 2

    # The token windows tile the whole document: the first starts at 0, the last ends at the end.
    assert chunks[0].metadata["start_token"] == 0
    assert chunks[-1].metadata["end_token"] == len(token_bytes)

    for a, b in zip(chunks, chunks[1:], strict=False):
        ma, mb = a.metadata, b.metadata
        # Exactly the configured overlap, and windows advance by chunk_size - overlap.
        assert ma["end_token"] - mb["start_token"] == overlap
        assert mb["start_token"] - ma["start_token"] == size - overlap
        if overlap:
            # The overlapping source text appears at the end of A and the start of B ...
            shared = text[mb["start_char"] : ma["end_char"]]
            assert shared and a.content.endswith(shared) and b.content.startswith(shared)
            # ... and it contains the text of the shared tokens.
            shared_tokens = b"".join(token_bytes[mb["start_token"] : ma["end_token"]])
            assert shared_tokens.decode("utf-8", errors="ignore").strip() in shared

    # No character drops: each chunk is an exact slice of the source, and together the
    # chunks cover every non-whitespace character.
    covered = [False] * len(text)
    for c in chunks:
        assert text[c.metadata["start_char"] : c.metadata["end_char"]] == c.content
        for i in range(c.metadata["start_char"], c.metadata["end_char"]):
            covered[i] = True
    dropped = [ch for ch, ok in zip(text, covered, strict=True) if not ok and not ch.isspace()]
    assert dropped == []


FENCED_DOC = """# Runbook

## Install

Run the installer:

```bash
# this comment looks like an H1 header
## and this like an H2
pip install lumen
```

Then verify.

## Configure

~~~toml
# tilde fence with a header-like line
[server]
port = 7700
~~~

````markdown
### an H3 inside a four-backtick fence
```python
print("a nested fence that must not close the outer one")
```
````

## Upgrade

Plain text section.
"""

FENCED_BLOCKS = [
    "```bash\n# this comment looks like an H1 header\n## and this like an H2\npip install lumen\n```",
    "~~~toml\n# tilde fence with a header-like line\n[server]\nport = 7700\n~~~",
    "````markdown\n### an H3 inside a four-backtick fence\n```python\n"
    'print("a nested fence that must not close the outer one")\n```\n````',
]


def test_2_structure_chunker_preserves_code_fences() -> None:
    """Test 2: code blocks in ``` (or ~~~) are never severed by the structure-aware chunker."""
    chunks = StructureAwareChunker().chunk(FENCED_DOC, "runbook.md")

    # Header-like lines inside code blocks must not create sections.
    assert [c.header_path for c in chunks] == [
        ["Runbook", "Install"],
        ["Runbook", "Configure"],
        ["Runbook", "Upgrade"],
    ]
    # Every block appears whole, in exactly one chunk.
    for block in FENCED_BLOCKS:
        assert sum(block in c.content for c in chunks) == 1, block
    # Every chunk starts at a real header and contains only complete (balanced) fences.
    for c in chunks:
        assert c.content.startswith("## ")
        assert len(_FENCE_LINE.findall(c.content)) % 2 == 0, c.content
