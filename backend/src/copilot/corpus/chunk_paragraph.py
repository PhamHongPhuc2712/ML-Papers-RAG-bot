"""Paragraph-aware packing: an alternative to the fixed-window chunker.

The fixed window cuts wherever the token count runs out, which routinely lands
mid-sentence and mid-argument. This policy instead packs whole paragraphs up to
a budget, so a chunk ends where the author ended a thought. Only a paragraph too
large to fit at all is split, and then on sentence boundaries.

Reconstructing paragraphs from a PDF is the hard part. pypdf's extracted text
carries **no blank lines at all** — every line is a wrapped column line — so the
usual delimiter is unavailable. The signal that survives is typographic: the
last line of a paragraph both ends a sentence and falls short of the column
width, while a sentence ending mid-paragraph runs to the margin. Blank lines are
still honoured when an adapter preserves them.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import Any

from .chunk import EVIDENCE_EXCLUDED_KINDS, Section, TokenSpans

# A line must be at least this fraction shorter than the column width before a
# sentence ending on it is read as the end of a paragraph.
SHORT_LINE_RATIO = 0.85
_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = (".", "!", "?")
# Terminators that do not end a sentence: initials, abbreviations, decimals.
_ABBREVIATIONS = frozenset(
    {
        "al", "e.g", "i.e", "eg", "ie", "fig", "figs", "eq", "eqs", "ref", "refs",
        "sec", "secs", "tab", "tabs", "cf", "vs", "etc", "approx", "resp", "no",
        "dr", "prof", "mr", "mrs", "ms", "st", "inc", "ltd", "jr", "sr", "vol",
    }
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def _is_sentence_end(candidate: str) -> bool:
    """Whether text ending here is a real sentence end rather than an abbreviation."""

    stripped = candidate.rstrip("\"')]").rstrip()
    if not stripped or stripped[-1] not in _SENTENCE_END:
        return False
    trailing = re.search(r"([\w.]+)\.$", stripped)
    if trailing is None:
        return True
    token = trailing.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return False
    # "0.5" or a single initial such as "J." continues the sentence.
    return not (token.isdigit() or len(token) == 1)


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation, keeping abbreviations and decimals intact."""

    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(collapsed):
        candidate = collapsed[start : match.start()]
        if _is_sentence_end(candidate):
            sentences.append(candidate.strip())
            start = match.end()
    tail = collapsed[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_paragraphs(text: str, *, short_line_ratio: float = SHORT_LINE_RATIO) -> list[str]:
    """Recover paragraphs from section text.

    Blank lines win when an adapter preserves them. Otherwise a paragraph ends at
    a line that both closes a sentence and is short of the column width, which is
    how a paragraph's last line looks once the layout is gone.
    """

    if _BLANK_LINE.search(text):
        blocks = [" ".join(block.split()) for block in _BLANK_LINE.split(text)]
        return [block for block in blocks if block]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) < 3:
        return [" ".join(lines)]
    # The column width is what a full line looks like in this section.
    width = statistics.median(len(line) for line in lines)
    limit = width * short_line_ratio

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if len(line) < limit and _is_sentence_end(line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _chunk(
    section: Section, text: str, token_count: int, ordinal: int, *, fragment: str | None = None
) -> dict[str, Any]:
    return {
        "section": section.name,
        "text": text,
        "token_count": token_count,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "ordinal": ordinal,
        "kind": section.kind,
        "section_ordinal": section.ordinal,
        "evidence_default": section.kind not in EVIDENCE_EXCLUDED_KINDS,
        "fragment": fragment,
    }


def _hard_split(text: str, spans: TokenSpans, limit: int) -> list[str]:
    """Last resort for text with no sentence boundaries at all.

    Reference lists and table dumps can run for thousands of tokens without a
    single sentence terminator. Nothing structural is left to cut on, so the
    ceiling is enforced on the token window instead — the one place this policy
    behaves like the fixed one.
    """

    tokens = list(spans(text))
    if len(tokens) <= limit:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + limit, len(tokens))
        # A slice re-encodes a token or two higher than the window it was cut
        # from, because a boundary piece tokenizes differently in isolation.
        # Shrink until the slice itself measures within the limit, so the
        # ceiling holds for the text actually stored.
        while end > start + 1:
            piece = text[tokens[start][0] : tokens[end - 1][1]]
            if len(spans(piece)) <= limit:
                break
            end -= 1
        parts.append(text[tokens[start][0] : tokens[end - 1][1]])
        start = end
    return parts


def _split_oversized(
    paragraph: str, spans: TokenSpans, max_tokens: int, ceiling: int
) -> list[str]:
    """Break one outsized paragraph on sentence boundaries, never mid-sentence."""

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in split_sentences(paragraph):
        size = len(spans(sentence))
        if size > ceiling:
            # A single "sentence" larger than the ceiling has no usable
            # boundary; flush what we have and cut it on tokens.
            if current:
                parts.append(" ".join(current))
                current, current_tokens = [], 0
            parts.extend(_hard_split(sentence, spans, max_tokens))
            continue
        if current and current_tokens + size > max_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += size
    if current:
        parts.append(" ".join(current))
    return parts


def _carry(text: str, spans: TokenSpans, sentences: int, budget: int) -> list[str]:
    """The trailing sentences to repeat, bounded by a token budget.

    Two sentences of prose are tens of tokens. Two "sentences" of a reference
    list can be well over a thousand, and carrying those unbounded starts the
    next chunk already past its budget — which is how a 3,052-token chunk
    appeared in a section whose ceiling was 1,200.
    """

    if sentences <= 0:
        return []
    carried: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)[-sentences:]):
        size = len(spans(sentence))
        if carried and total + size > budget:
            break
        carried.insert(0, sentence)
        total += size
        if total >= budget:
            break
    return carried


def _pack_blocks(
    blocks: Sequence[str],
    spans: TokenSpans,
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_sentences: int,
    ceiling: int,
) -> list[str]:
    """Pack one section's blocks into chunk texts.

    A chunk closes once it has reached ``target_tokens`` or the next block would
    carry it past ``max_tokens``, so chunks settle into the band between the two
    rather than at one fixed size.
    """

    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0
    # True while `current` holds only sentences carried from the chunk just
    # emitted, which must not be published again on their own.
    carried_only = False
    for block in blocks:
        size = len(spans(block))
        if current and (current_tokens >= target_tokens or current_tokens + size > max_tokens):
            text = " ".join(current)
            packed.extend(_hard_split(text, spans, ceiling))
            carry = _carry(text, spans, overlap_sentences, max(1, max_tokens // 4))
            current = list(carry)
            current_tokens = len(spans(" ".join(carry))) if carry else 0
            carried_only = True
        current.append(block)
        current_tokens += size
        carried_only = False
    if current and not carried_only:
        packed.extend(_hard_split(" ".join(current), spans, ceiling))
    return packed


def pack_paragraphs(
    sections: Sequence[Section],
    spans: TokenSpans,
    *,
    target_tokens: int = 800,
    max_tokens: int = 900,
    paragraph_max_tokens: int = 1200,
    overlap_sentences: int = 2,
) -> list[dict[str, Any]]:
    """Pack whole paragraphs into chunks of at most ``max_tokens``.

    A chunk ends where the author ended a paragraph, so it closes on a complete
    thought rather than wherever a token budget ran out. A paragraph above
    ``paragraph_max_tokens`` is split on sentence boundaries, never mid-sentence.
    Each chunk after the first may repeat the previous one's last
    ``overlap_sentences`` sentences, so a claim spanning a boundary stays
    retrievable whole. Windows never cross a section, exactly as the
    fixed-window policy guarantees.

    No chunk exceeds ``paragraph_max_tokens``. That is the real ceiling:
    ``max_tokens`` is the packing budget, while a single paragraph is allowed to
    stand alone up to the ceiling rather than be broken for the sake of a
    rounder number.
    """

    if target_tokens <= 0 or max_tokens < target_tokens or paragraph_max_tokens < max_tokens:
        raise ValueError("invalid_paragraph_window")
    if overlap_sentences < 0:
        raise ValueError("invalid_paragraph_window")

    chunks: list[dict[str, Any]] = []
    for section in sections:
        blocks: list[str] = []
        for paragraph in split_paragraphs(section.text):
            # A paragraph is kept whole up to the ceiling, which is what buys the
            # policy its intact-argument property; beyond that it must be cut.
            if len(spans(paragraph)) > paragraph_max_tokens:
                blocks.extend(
                    _split_oversized(paragraph, spans, max_tokens, paragraph_max_tokens)
                )
            else:
                blocks.append(paragraph)
        for text in _pack_blocks(
            blocks,
            spans,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            ceiling=paragraph_max_tokens,
        ):
            chunks.append(_chunk(section, text, len(spans(text)), len(chunks)))
    return chunks
