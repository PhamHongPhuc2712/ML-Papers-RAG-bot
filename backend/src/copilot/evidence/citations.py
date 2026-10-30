"""Structural validation of citations, before any answer text is shown (spec §9, P4.1).

A claim is accepted only when every evidence id it cites names evidence that was
actually sent, each factual claim cites at least one, no claim cites one piece twice,
and any span it quotes appears in the evidence it cites. These checks prove an id is
real; whether it is the *right* id is the judge's question (E6), not this module's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import Claim, Evidence

# A quoted span this long or longer is a quotation to check; shorter ones are terms.
MIN_QUOTE_CHARS = 12
_QUOTED = re.compile(r'"([^"]{' + str(MIN_QUOTE_CHARS) + r',})"')
_SPACE = re.compile(r"\s+")

UNKNOWN_EVIDENCE = "unknown_evidence"
UNCITED_CLAIM = "uncited_claim"
DUPLICATE_REFERENCE = "duplicate_reference"
QUOTE_NOT_IN_SOURCE = "quote_not_in_source"


@dataclass(frozen=True)
class CitationProblem:
    """One rule one claim broke; ``claim`` is its index in the answer."""

    claim: int
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}:{self.detail}" if self.detail else self.code


def _ids(claim: Claim | Mapping[str, Any]) -> list[str]:
    ids = claim.evidence_ids if isinstance(claim, Claim) else claim["evidence_ids"]
    return [str(item) for item in ids]


def _text(claim: Claim | Mapping[str, Any]) -> str:
    return claim.text if isinstance(claim, Claim) else str(claim.get("text", ""))


def validate_citations(
    claims: Sequence[Claim | Mapping[str, Any]], allowed_ids: Iterable[str]
) -> list[str]:
    """The ids the claims cite that were never sent, sorted and unique."""

    allowed = set(allowed_ids)
    referenced = {source for claim in claims for source in _ids(claim)}
    return sorted(referenced - allowed)


def normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def check_claims(
    claims: Sequence[Claim | Mapping[str, Any]], evidence: Mapping[str, Evidence]
) -> list[CitationProblem]:
    """Every rule, per claim, in claim order; an empty list is a structurally valid answer."""

    problems: list[CitationProblem] = []
    for index, claim in enumerate(claims):
        ids = _ids(claim)
        if not ids:
            problems.append(CitationProblem(index, UNCITED_CLAIM))
            continue
        unknown = sorted(set(ids) - set(evidence))
        if unknown:
            problems.append(CitationProblem(index, UNKNOWN_EVIDENCE, ",".join(unknown)))
        if len(set(ids)) != len(ids):
            problems.append(CitationProblem(index, DUPLICATE_REFERENCE))
        cited = " ".join(normalize(evidence[item].text) for item in ids if item in evidence)
        for quote in _QUOTED.findall(_text(claim)):
            if normalize(quote) not in cited:
                problems.append(CitationProblem(index, QUOTE_NOT_IN_SOURCE, quote[:40]))
    return problems


def provenance_valid(
    claims: Sequence[Claim | Mapping[str, Any]], evidence: Mapping[str, Evidence]
) -> bool:
    """True when every claim passes every rule: the G4 provenance-validity unit."""

    return not check_claims(claims, evidence)
