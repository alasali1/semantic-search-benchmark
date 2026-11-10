"""Markdown ingestion and chunking strategies.

Two strategies are provided, both producing :class:`DocumentChunk` objects with a
uniform metadata schema so downstream indexing and evaluation code never needs to
know which chunker produced a chunk:

* :class:`FixedSizeChunker` - sliding token windows (``tiktoken``) with overlap.
* :class:`StructureAwareChunker` - one chunk per Markdown section (``#``/``##``/``###``),
  never splitting fenced code blocks.

Chunk metadata schema (all values JSON-serialisable)::

    {
        "strategy":      str,              # e.g. "fixed-256-32" or "structure"
        "chunk_index":   int,              # position of the chunk within its document
        "start_line":    int,              # 1-indexed, inclusive
        "end_line":      int,              # 1-indexed, inclusive
        "start_char":    int,              # 0-indexed character offset into the source text
        "end_char":      int,              # exclusive; source[start_char:end_char] ~ content
        "token_count":   int,              # tiktoken token count of ``content``
        "header_path":   list[str],        # header hierarchy at the chunk's first line
        "section_paths": list[list[str]],  # sections this chunk is credited with (see below)
        "start_token":    int,             # fixed-size only: token window [start_token, end_token)
        "end_token":      int,             #   in the document's tiktoken encoding
        "fence_reopened": bool,            # fixed-size only: chunk starts with a re-added fence opener
        "fence_closed":   bool,            # fixed-size only: chunk ends with an added closing fence
    }

``section_paths`` is what makes header-based ground truth comparable across chunking
strategies. A fixed-size chunk is credited with a section only if it holds a meaningful
share of that section's body: at least ``min_section_fraction`` of it (default half), or
``min_section_tokens`` tokens of it (default 100), whichever is smaller. One stray line
of a section no longer counts, so large chunks are not rewarded for merely *containing*
many sections. A structure-aware chunk is exactly one section, so it is always credited
with that section alone.

``start_char`` / ``end_char`` locate the chunk in the original document independently of
how it was cut, which is what lets text-span ground truth (``relevant_spans``) be
compared across chunking strategies.
"""

from __future__ import annotations

import bisect
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_ENCODING = "cl100k_base"
"""tiktoken encoding used for token counting (matches OpenAI embedding models)."""

# ATX header: up to 3 spaces of indent, 1-6 '#', then whitespace or end of line.
# An optional closing sequence of '#' characters is stripped from the title.
_HEADER_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)(?:[ \t]+#+)?[ \t]*$")
# Fenced code block opener/closer: ``` or ~~~ (3+ chars), up to 3 spaces of indent.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A retrievable unit of text cut from a source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Globally unique, deterministic chunk identifier.")
    content: str = Field(description="Chunk text that gets embedded / BM25-indexed.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Header hierarchy, line numbers, strategy info (see module docstring).",
    )
    parent_doc: str = Field(description="Identifier of the source document (relative path).")

    @property
    def header_path(self) -> list[str]:
        """Header hierarchy active at the first line of the chunk."""
        return list(self.metadata.get("header_path", []))

    @property
    def section_paths(self) -> list[list[str]]:
        """Header paths of every section whose body this chunk overlaps."""
        return [list(p) for p in self.metadata.get("section_paths", [])]


@dataclass(frozen=True)
class MarkdownSection:
    """A contiguous region of a Markdown document owned by a single header.

    ``level`` is 0 for the preamble (text before the first header).
    Line numbers are 1-indexed and inclusive; ``start_line`` is the header line itself.
    """

    header_path: tuple[str, ...]
    level: int
    start_line: int
    end_line: int
    text: str

    @property
    def body_start_line(self) -> int:
        """First line after the header (the preamble has no header line)."""
        return self.start_line + (1 if self.level else 0)

    @property
    def has_body(self) -> bool:
        """True if the section contains any non-whitespace text beyond its header."""
        lines = self.text.splitlines()
        body = lines[1:] if self.level else lines
        return any(line.strip() for line in body)


# --------------------------------------------------------------------------------------
# Markdown parsing
# --------------------------------------------------------------------------------------


def _is_closing_fence(line: str, fence_char: str, fence_len: int) -> bool:
    """Return True if ``line`` closes a fence opened with ``fence_len`` x ``fence_char``."""
    m = _FENCE_RE.match(line)
    if not m:
        return False
    marker, rest = m.group(1), m.group(2)
    return marker[0] == fence_char and len(marker) >= fence_len and not rest.strip()


def parse_markdown_sections(text: str, split_levels: int = 3) -> list[MarkdownSection]:
    """Split Markdown into header-delimited sections, respecting fenced code blocks.

    Headers deeper than ``split_levels`` (e.g. ``####``) stay inside their parent section.
    Lines inside ``` / ~~~ fences are never interpreted as headers, so a ``# comment``
    in a shell snippet does not start a new section. An unterminated fence runs to EOF.
    Setext-style headers (underlined with ``===``) are not recognised.

    Args:
        text: Raw Markdown source.
        split_levels: Deepest header level that starts a new section (1-6).

    Returns:
        Sections in document order. A whitespace-only preamble is dropped.
    """
    if not 1 <= split_levels <= 6:
        raise ValueError(f"split_levels must be in [1, 6], got {split_levels}")

    lines = text.splitlines(keepends=True)
    sections: list[MarkdownSection] = []
    stack: list[tuple[int, str]] = []  # (level, title) of currently open headers

    cur_path: tuple[str, ...] = ()
    cur_level = 0
    cur_start = 1
    cur_lines: list[str] = []
    fence: tuple[str, int] | None = None  # (char, length) of the open fence, if any

    def flush(end_line: int) -> None:
        body = "".join(cur_lines)
        if not cur_lines or (cur_level == 0 and not body.strip()):
            return
        sections.append(
            MarkdownSection(
                header_path=cur_path,
                level=cur_level,
                start_line=cur_start,
                end_line=end_line,
                text=body,
            )
        )

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")

        if fence is not None:
            if _is_closing_fence(line, *fence):
                fence = None
            cur_lines.append(raw_line)
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence = (marker[0], len(marker))
            cur_lines.append(raw_line)
            continue

        header_match = _HEADER_RE.match(line)
        if header_match and len(header_match.group(1)) <= split_levels:
            flush(lineno - 1)
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = tuple(t for _, t in stack)
            cur_level = level
            cur_start = lineno
            cur_lines = [raw_line]
            continue

        cur_lines.append(raw_line)

    flush(len(lines))
    return sections


def _credited_sections(
    sections: list[MarkdownSection],
    line_starts: list[int],
    token_offsets: list[int],
    total_bytes: int,
    win_start: int,
    win_end: int,
    min_fraction: float,
    min_tokens: int,
) -> list[list[str]]:
    """Header paths of the sections a token window ``[win_start, win_end)`` is credited with.

    Only a section's *body* counts (a window that merely ends on the next section's header
    line is not credited), and the header-less preamble is never reported. A window must
    also hold a meaningful share of the body: at least ``min_fraction`` of the section's
    tokens, or ``min_tokens`` tokens, whichever is smaller. The token floor keeps long
    sections answerable when they are spread over several windows (no single window may
    hold half of them), and the ``min_fraction`` cap keeps short sections strict.

    Args:
        sections: Parsed sections of the document.
        line_starts: Byte offset at which each line starts (``line_starts[n - 1]`` for line n).
        token_offsets: Byte offset at which each token starts, plus the document length.
        total_bytes: Length of the document in bytes.
        win_start: Index of the window's first token.
        win_end: Index one past the window's last token.
        min_fraction: Share of the section body that earns credit (0 < x <= 1).
        min_tokens: Absolute number of body tokens that earns credit regardless of share.
    """
    ranges = _SectionRanges.build(sections, line_starts, token_offsets, total_bytes)
    return ranges.credit(win_start, win_end, min_fraction, min_tokens)


@dataclass(frozen=True)
class _SectionRanges:
    """Body token ranges ``[t0, t1)`` of every creditable section, in document order.

    Built once per document so that crediting a window is a binary search plus a short
    walk. Re-deriving every section for every window made chunking O(windows x sections):
    8 s for a 1.8 MB document.
    """

    starts: list[int]  # t0 of each section body
    ends: list[int]  # t1 of each section body (sections don't overlap, so also sorted)
    paths: list[tuple[str, ...]]

    @classmethod
    def build(
        cls,
        sections: list[MarkdownSection],
        line_starts: list[int],
        token_offsets: list[int],
        total_bytes: int,
    ) -> _SectionRanges:
        starts: list[int] = []
        ends: list[int] = []
        paths: list[tuple[str, ...]] = []
        for s in sections:
            if not s.header_path or not s.has_body:
                continue
            body_b0 = line_starts[s.body_start_line - 1]
            body_b1 = line_starts[s.end_line] if s.end_line < len(line_starts) else total_bytes
            t0 = bisect.bisect_right(token_offsets, body_b0) - 1
            t1 = bisect.bisect_left(token_offsets, body_b1)
            if t1 > t0:
                starts.append(t0)
                ends.append(t1)
                paths.append(s.header_path)
        return cls(starts, ends, paths)

    def credit(
        self, win_start: int, win_end: int, min_fraction: float, min_tokens: int
    ) -> list[list[str]]:
        """Sections the window ``[win_start, win_end)`` is credited with (see :func:`_credited_sections`)."""
        paths: list[list[str]] = []
        # First section whose body ends after the window starts; stop once one starts after it ends.
        for i in range(bisect.bisect_right(self.ends, win_start), len(self.starts)):
            t0, t1 = self.starts[i], self.ends[i]
            if t0 >= win_end:
                break
            overlap = min(win_end, t1) - max(win_start, t0)
            if overlap > 0 and overlap >= min(min_fraction * (t1 - t0), min_tokens):
                paths.append(list(self.paths[i]))
        return paths


@dataclass(frozen=True)
class _FencedBlock:
    """A fenced code block located by UTF-8 byte offsets in the source."""

    start: int  # byte offset where the opening fence line begins
    end: int  # byte offset where the closing fence line ends (EOF if never closed)
    opener: str  # opening fence line without indentation, e.g. "```yaml"
    closer: str  # matching closing fence, e.g. "```"


def _fenced_blocks(raw: bytes) -> list[_FencedBlock]:
    """Find every fenced code block, using the same fence rules as the section parser."""
    blocks: list[_FencedBlock] = []
    open_at: int | None = None
    fence_char, fence_len, opener = "", 0, ""
    pos = 0
    for raw_line in raw.split(b"\n"):
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
        if open_at is None:
            m = _FENCE_RE.match(line)
            if m:
                open_at, fence_char, fence_len = pos, m.group(1)[0], len(m.group(1))
                opener = line.strip()
        elif _is_closing_fence(line, fence_char, fence_len):
            blocks.append(_FencedBlock(open_at, pos + len(raw_line), opener, fence_char * fence_len))
            open_at = None
        pos += len(raw_line) + 1
    if open_at is not None:  # unterminated fence runs to EOF, as in parse_markdown_sections
        blocks.append(_FencedBlock(open_at, len(raw), opener, fence_char * fence_len))
    return blocks


def _line_char_ranges(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character offsets of every line, end excluding the line terminator.

    Uses ``str.splitlines`` so line numbers agree with :func:`parse_markdown_sections`.
    """
    ranges: list[tuple[int, int]] = []
    pos = 0
    for with_eol, plain in zip(text.splitlines(keepends=True), text.splitlines(), strict=True):
        ranges.append((pos, pos + len(plain)))
        pos += len(with_eol)
    return ranges


# --------------------------------------------------------------------------------------
# Chunkers
# --------------------------------------------------------------------------------------


class BaseChunker(ABC):
    """Common interface for all chunking strategies."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    @abstractmethod
    def strategy(self) -> str:
        """Short, filesystem/ID-safe name of the strategy and its parameters."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """Split a single document into chunks.

        Args:
            text: Raw Markdown text of the document.
            doc_id: Stable identifier of the document; becomes ``parent_doc``.
        """

    def chunk_documents(self, corpus: Mapping[str, str]) -> list[DocumentChunk]:
        """Chunk every document in ``corpus`` (``doc_id -> text``) in sorted order."""
        chunks: list[DocumentChunk] = []
        for doc_id in sorted(corpus):
            chunks.extend(self.chunk(corpus[doc_id], doc_id))
        return chunks

    def count_tokens(self, text: str) -> int:
        """Number of tiktoken tokens in ``text``."""
        return len(self._encoding.encode(text, disallowed_special=()))

    def _make_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}::{self.strategy}::{index:04d}"


class FixedSizeChunker(BaseChunker):
    """Sliding-window chunker over ``tiktoken`` tokens.

    Windows of ``chunk_size`` tokens advance by ``chunk_size - chunk_overlap`` tokens.
    BPE tokens in ``cl100k_base`` normally carry their leading space (``" word"``), so
    window boundaries fall on word boundaries in practice. Slicing is done on the UTF-8
    byte offsets of the tokens, so the chunk text is an exact substring of the source
    and line numbers can be recovered precisely.

    A window is credited with a section (``section_paths`` metadata, used by header-based
    ground truth) only if it holds at least ``min_section_fraction`` of the section's body
    or ``min_section_tokens`` tokens of it, whichever is smaller. The token floor is also
    capped at half the window so tiny windows can still earn credit for long sections.

    Code blocks (``respect_code_blocks=True``, the default). A window edge is never left
    inside a fenced code block, because half a snippet with an unclosed fence garbles
    the code for embedding, BM25 and display. When a cut would land inside a block:

    1. **Cut before it** - end the chunk just before the opening fence, if the chunk keeps
       at least half of ``chunk_size``. The whole block starts the next chunk.
    2. **Otherwise include it** - extend the chunk to the closing fence, if the chunk stays
       within 1.5 x ``chunk_size``.
    3. **Otherwise split and repair** - the block is too big to fit anywhere, so it is cut
       at a line break and both halves are made valid: the first gets a closing fence, the
       next starts with the original opening line (e.g. ```` ```yaml ````).

    Overlap is likewise not allowed to start a chunk mid-block. Chunks therefore vary in
    length (roughly 0.5x to 1.5x ``chunk_size``). Repaired chunks are flagged with
    ``fence_reopened`` / ``fence_closed`` metadata; their ``start_char``/``end_char`` still
    point at the source text, excluding the added fence lines. Pass
    ``respect_code_blocks=False`` for the purely naive token windows.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        encoding_name: str = DEFAULT_ENCODING,
        min_section_fraction: float = 0.5,
        min_section_tokens: int = 100,
        respect_code_blocks: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} "
                f"for chunk_size={chunk_size}"
            )
        if not 0 < min_section_fraction <= 1:
            raise ValueError(f"min_section_fraction must be in (0, 1], got {min_section_fraction}")
        if min_section_tokens < 1:
            raise ValueError(f"min_section_tokens must be positive, got {min_section_tokens}")
        super().__init__(encoding_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_fraction = min_section_fraction
        self.min_section_tokens = min_section_tokens
        self.respect_code_blocks = respect_code_blocks

    @property
    def strategy(self) -> str:
        base = f"fixed-{self.chunk_size}-{self.chunk_overlap}"
        return base if self.respect_code_blocks else f"{base}-naive"

    def chunk(self, text: str, doc_id: str) -> list[DocumentChunk]:
        if not text.strip():
            return []

        raw = text.encode("utf-8")
        tokens = self._encoding.encode(text, disallowed_special=())
        # Cumulative byte offset of each token boundary: offsets[i] is where token i starts.
        offsets = [0]
        for token_bytes in self._encoding.decode_tokens_bytes(tokens):
            offsets.append(offsets[-1] + len(token_bytes))
        if offsets[-1] != len(raw):  # tiktoken round-trips losslessly; guard anyway.
            raise RuntimeError(f"Token offsets do not cover document {doc_id!r}")

        # Byte offset at which each line starts, for byte-offset -> line-number lookup.
        line_starts = [0] + [m.end() for m in re.finditer(rb"\n", raw)]
        sections = parse_markdown_sections(text)

        # chars_before[b] = number of characters in raw[:b]. Counting the bytes that start a
        # UTF-8 character once is O(n); decoding the prefix for every chunk was O(n^2).
        chars_before = [0] * (len(raw) + 1)
        for i, byte in enumerate(raw):
            chars_before[i + 1] = chars_before[i] + ((byte & 0xC0) != 0x80)

        # Per-document lookups, built once so per-chunk work stays O(log n).
        section_ranges = _SectionRanges.build(sections, line_starts, offsets, len(raw))
        section_start_lines = [sec.start_line for sec in sections]

        def section_at_line(line: int) -> list[str]:
            i = bisect.bisect_right(section_start_lines, line) - 1
            return list(sections[i].header_path) if i >= 0 and line <= sections[i].end_line else []

        blocks = _fenced_blocks(raw) if self.respect_code_blocks else []
        block_starts = [blk.start for blk in blocks]

        def block_around(byte_pos: int) -> _FencedBlock | None:
            """The code block that ``byte_pos`` falls strictly inside, if any."""
            i = bisect.bisect_left(block_starts, byte_pos) - 1
            return blocks[i] if i >= 0 and byte_pos < blocks[i].end else None

        n = len(tokens)
        min_keep = max(1, self.chunk_size // 2)  # shortest chunk when cutting before a block
        max_len = self.chunk_size + self.chunk_size // 2  # longest chunk when including one
        min_tokens = max(1, min(self.min_section_tokens, self.chunk_size // 2))

        chunks: list[DocumentChunk] = []
        start = 0
        reopen: _FencedBlock | None = None  # block this chunk continues (needs its opener)
        while start < n:
            end = min(start + self.chunk_size, n)
            close: _FencedBlock | None = None  # block this chunk splits (needs a closer)

            block = block_around(offsets[end]) if end < n else None
            if block is not None:
                before = bisect.bisect_right(offsets, block.start) - 1  # cut before opener
                after = bisect.bisect_left(offsets, block.end)  # cut after closer
                if before - start >= min_keep:
                    end = before
                elif after - start <= max_len:
                    end = after
                else:
                    # Block too big to keep whole: split at the last line break that still
                    # leaves a reasonable chunk, so code lines are not cut mid-token.
                    for e in range(end, start + min_keep, -1):
                        if raw[offsets[e] - 1] == 0x0A:
                            end = e
                            break
                    close = block

            b0, b1 = offsets[start], offsets[end]
            # Trim surrounding whitespace so line numbers point at real content. A chunk
            # that continues a code block keeps its first line's indentation.
            segment = raw[b0:b1]
            b0 += len(segment) - len(segment.lstrip(b"\r\n" if reopen else None))
            b1 -= len(segment) - len(segment.rstrip())

            # tiktoken can split one multi-byte character across two tokens. If a window
            # edge falls inside such a character, the *later* chunk owns it: the start moves
            # back to the character's first byte and the end moves back before it. Moving
            # both forward dropped the character from both chunks when overlap was 0.
            while b0 > 0 and (raw[b0] & 0xC0) == 0x80:
                b0 -= 1
            while b1 > b0 and b1 < len(raw) and (raw[b1] & 0xC0) == 0x80:
                b1 -= 1

            if b1 > b0:
                content = raw[b0:b1].decode("utf-8")
                if reopen is not None:
                    content = f"{reopen.opener}\n{content}"
                if close is not None:
                    content = f"{content}\n{close.closer}"
                start_line = bisect.bisect_right(line_starts, b0)
                end_line = bisect.bisect_right(line_starts, b1 - 1)
                index = len(chunks)
                repaired = reopen is not None or close is not None
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._make_id(doc_id, index),
                        content=content,
                        parent_doc=doc_id,
                        metadata={
                            "strategy": self.strategy,
                            "chunk_index": index,
                            "start_line": start_line,
                            "end_line": end_line,
                            "start_char": chars_before[b0],
                            "end_char": chars_before[b1],
                            "start_token": start,
                            "end_token": end,
                            "token_count": self.count_tokens(content) if repaired else end - start,
                            "header_path": section_at_line(start_line),
                            "section_paths": section_ranges.credit(
                                start, end, self.min_section_fraction, min_tokens
                            ),
                            "fence_reopened": reopen is not None,
                            "fence_closed": close is not None,
                        },
                    )
                )

            if end >= n:
                break
            if close is not None:
                # Continue the split block right where this chunk stopped (no overlap, so
                # the repaired halves don't duplicate code lines).
                start, reopen = end, close
            else:
                next_start = max(end - self.chunk_overlap, start + 1)
                inside = block_around(offsets[next_start])
                if inside is not None:
                    # Overlap would begin mid-block: skip past the block (already complete
                    # in this chunk) instead of starting with a fragment.
                    next_start = max(next_start, min(bisect.bisect_left(offsets, inside.end), end))
                start, reopen = next_start, None
        return chunks


class StructureAwareChunker(BaseChunker):
    """One chunk per Markdown section, split on ``#`` / ``##`` / ``###`` headers.

    Fenced code blocks are kept intact inside the section that contains them, and
    ``#`` lines inside fences are not treated as headers. Each chunk's content starts
    with its own header line. Sections with a header but no body (e.g. a ``#`` title
    immediately followed by a ``##`` subsection) are skipped: their title survives in
    the ``header_path`` of every child chunk.

    Sections are not size-capped, so very long sections may exceed an embedding
    model's context window (``all-MiniLM-L6-v2`` truncates at 256 word pieces).
    """

    def __init__(self, split_levels: int = 3, encoding_name: str = DEFAULT_ENCODING) -> None:
        if not 1 <= split_levels <= 6:
            raise ValueError(f"split_levels must be in [1, 6], got {split_levels}")
        super().__init__(encoding_name)
        self.split_levels = split_levels

    @property
    def strategy(self) -> str:
        return "structure" if self.split_levels == 3 else f"structure-h{self.split_levels}"

    def chunk(self, text: str, doc_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        line_chars = _line_char_ranges(text)
        for section in parse_markdown_sections(text, self.split_levels):
            if not section.has_body:
                continue

            # Trim blank lines at either end and shift the line range accordingly.
            lines = section.text.splitlines()
            lead = next(i for i, ln in enumerate(lines) if ln.strip())
            trail = next(i for i, ln in enumerate(reversed(lines)) if ln.strip())
            content = "\n".join(lines[lead : len(lines) - trail])

            # Character range of the trimmed section in the source text.
            first_line = section.start_line + lead
            last_line = section.end_line - trail
            start_char = line_chars[first_line - 1][0]
            end_char = line_chars[last_line - 1][1]

            index = len(chunks)
            path = list(section.header_path)
            chunks.append(
                DocumentChunk(
                    chunk_id=self._make_id(doc_id, index),
                    content=content,
                    parent_doc=doc_id,
                    metadata={
                        "strategy": self.strategy,
                        "chunk_index": index,
                        "start_line": first_line,
                        "end_line": last_line,
                        "start_char": start_char,
                        "end_char": end_char,
                        "token_count": self.count_tokens(content),
                        "header_path": path,
                        "section_paths": [path] if path else [],
                    },
                )
            )
        return chunks
