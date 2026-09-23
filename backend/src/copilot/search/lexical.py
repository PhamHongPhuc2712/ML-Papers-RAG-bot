"""Exact BM25 over paper text, and its sparse decomposition.

This module is two things at once, and the second is the reason it exists.

It is a scorer: ``search`` ranks documents by the BM25 formula in spec §7 with
k1=1.2 and b=0.75, over ``title + '\\n' + abstract`` tokenized by Unicode
normalization, lowercasing and alphanumeric splitting.

It is also the **oracle** that P2.2's Qdrant sparse vectors must reproduce.
Spec §7 splits BM25 into a document weight ``tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))``
and a query weight ``log(1+(N-df+0.5)/(df+0.5))`` whose dot product is the
score. Qdrant stores the document half already weighted and applies **no server
IDF modifier**, so any disagreement between ``search`` and
``encode_query``·``encode_document`` here is a disagreement that would ship.
``test_sparse_dot_product_reproduces_the_oracle`` is that contract.

Scale note: this index is in-memory and sized for paper-level retrieval —
85,729 papers of title and abstract. It is not sized for 3.4 M chunks; chunk
retrieval is served by Qdrant in P2.2 and validated against this oracle on a
sample rather than reimplemented here.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

# Spec §7. Changing either creates a new release: the document weights bake
# them in, so stored sparse vectors stop matching the oracle.
K1 = 1.2
B = 0.75

_TERM = re.compile(r"[0-9a-z]+")


class LexicalError(RuntimeError):
    """The index cannot answer this request."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True, slots=True)
class CorpusStatistics:
    """What a release pins, so a later run can prove it used the same numbers."""

    documents: int
    vocabulary_size: int
    average_length: float
    k1: float
    b: float


def tokenize(text: str) -> list[str]:
    """Unicode-normalize, lowercase, keep alphanumeric runs.

    NFKC before lowercasing so composed and decomposed forms of the same word
    become one term. Splitting on non-alphanumerics keeps acronyms intact as
    single terms ("rag", "gpt2") while dropping punctuation.
    """

    return _TERM.findall(unicodedata.normalize("NFKC", text).lower())


def paper_text(title: str, abstract: str | None) -> str:
    """The canonical lexical document for a paper (spec §7)."""

    return f"{title}\n{abstract}" if abstract else title


class BM25:
    """An exact BM25 index over a fixed document set."""

    def __init__(self, *, k1: float = K1, b: float = B) -> None:
        self._k1 = k1
        self._b = b
        self._fitted = False
        self._lengths: dict[str, int] = {}
        self._frequencies: dict[str, Counter[str]] = {}
        self._postings: dict[str, list[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._vocabulary: tuple[str, ...] = ()
        self._term_ids: dict[str, int] = {}
        self._average_length = 0.0

    def fit(self, documents: Mapping[str, str]) -> None:
        """Build the index. Term IDs come from the sorted vocabulary (spec §7)."""

        self._lengths = {}
        self._frequencies = {}
        self._postings = {}
        self._document_frequency = Counter()
        for doc_id, text in documents.items():
            terms = tokenize(text)
            counts = Counter(terms)
            self._frequencies[doc_id] = counts
            self._lengths[doc_id] = len(terms)
            self._document_frequency.update(counts.keys())
            # The postings list holds references to the same id strings, so a
            # term costs one pointer per document rather than a second copy.
            for term in counts:
                self._postings.setdefault(term, []).append(doc_id)
        for ids in self._postings.values():
            ids.sort()
        total = sum(self._lengths.values())
        # A corpus of empty documents has no average length to normalize by.
        self._average_length = total / len(self._lengths) if self._lengths and total else 0.0
        self._vocabulary = tuple(sorted(self._document_frequency))
        self._term_ids = {term: index for index, term in enumerate(self._vocabulary)}
        self._fitted = True

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def statistics(self) -> CorpusStatistics:
        self._require_fitted()
        return CorpusStatistics(
            documents=len(self._lengths),
            vocabulary_size=len(self._vocabulary),
            average_length=self._average_length,
            k1=self._k1,
            b=self._b,
        )

    def postings(self, term: str) -> tuple[str, ...]:
        """The documents containing a term, sorted by ID."""

        self._require_fitted()
        return tuple(self._postings.get(term, ()))

    def term_id(self, term: str) -> int:
        """The release-stable integer for a term, for sparse vector indices."""

        self._require_fitted()
        try:
            return self._term_ids[term]
        except KeyError as error:
            raise LexicalError("unknown_term", term) from error

    def query_weight(self, term: str) -> float:
        """``log(1+(N-df+0.5)/(df+0.5))`` — the unique query-term weight."""

        self._require_fitted()
        df = self._document_frequency.get(term, 0)
        if not df:
            return 0.0
        total = len(self._lengths)
        return math.log(1 + (total - df + 0.5) / (df + 0.5))

    def document_weight(self, doc_id: str, term: str) -> float:
        """``tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`` — the stored half."""

        self._require_fitted()
        frequency = self._frequencies.get(doc_id, Counter()).get(term, 0)
        if not frequency:
            return 0.0
        length = self._lengths.get(doc_id, 0)
        # avgdl is 0 only when every document is empty, and then no term has a
        # frequency, so this never divides by zero.
        normalized = length / self._average_length if self._average_length else 0.0
        saturation = self._k1 * (1 - self._b + self._b * normalized)
        return frequency * (self._k1 + 1) / (frequency + saturation)

    def encode_query(self, query: str) -> dict[str, float]:
        """Query terms to weights. Terms outside the vocabulary are omitted."""

        self._require_fitted()
        return {
            term: self.query_weight(term)
            for term in dict.fromkeys(tokenize(query))
            if term in self._term_ids
        }

    def encode_document(self, doc_id: str) -> dict[str, float]:
        """One document's already-weighted sparse vector."""

        self._require_fitted()
        return {
            term: self.document_weight(doc_id, term)
            for term in self._frequencies.get(doc_id, Counter())
        }

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Rank documents by BM25. Ties break on document ID (spec §7)."""

        self._require_fitted()
        if limit < 1:
            raise LexicalError("invalid_limit", str(limit))
        weights = self.encode_query(query)
        if not weights:
            return []
        scores: dict[str, float] = {}
        for term, query_weight in weights.items():
            for doc_id in self._postings.get(term, ()):
                contribution = query_weight * self.document_weight(doc_id, term)
                if contribution:
                    scores[doc_id] = scores.get(doc_id, 0.0) + contribution
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise LexicalError("not_fitted")
