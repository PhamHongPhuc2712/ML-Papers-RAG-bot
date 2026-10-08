"""Structural validation of citations: ids, references and quotes, before any answer counts."""

from __future__ import annotations

from uuid import uuid4

from copilot.contracts import Claim, Evidence
from copilot.evidence.citations import check_claims, provenance_valid, validate_citations


def _evidence(label: str, text: str) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id="v1",
        chunk_id=uuid4(),
        text=text,
        section="Method",
    )


EVIDENCE = {
    "E1": _evidence("E1", "We train the encoder with a contrastive objective over 1,000 pairs."),
    "E2": _evidence("E2", "Results improve by 3.2 points on the held-out split."),
}


def test_unknown_evidence_is_rejected():
    claims = [
        {"text": "A supported claim", "evidence_ids": ["E1"]},
        {"text": "Invented source", "evidence_ids": ["E999"]},
    ]
    assert validate_citations(claims, {"E1", "E2"}) == ["E999"]
    assert validate_citations(claims[:1], {"E1"}) == []


def test_unknown_ids_are_sorted_and_unique_across_claims():
    claims = [
        {"text": "a", "evidence_ids": ["E9", "E3"]},
        {"text": "b", "evidence_ids": ["E3", "E1"]},
    ]
    assert validate_citations(claims, {"E1"}) == ["E3", "E9"]


def test_an_id_from_another_request_is_unknown_here():
    """Evidence ids are only meaningful within the evidence actually sent."""

    other = _evidence("E1", "text from another release")
    claims = [Claim(text="cites the other release", evidence_ids=[other.id])]
    assert provenance_valid(claims, {"E2": EVIDENCE["E2"]}) is False
    assert provenance_valid(claims, EVIDENCE) is True


def test_every_factual_claim_needs_at_least_one_evidence_id():
    problems = check_claims([Claim(text="No source at all", evidence_ids=[])], EVIDENCE)
    assert [p.code for p in problems] == ["uncited_claim"]
    assert problems[0].claim == 0


def test_a_claim_may_not_cite_the_same_evidence_twice():
    problems = check_claims([Claim(text="twice", evidence_ids=["E1", "E1"])], EVIDENCE)
    assert [p.code for p in problems] == ["duplicate_reference"]


def test_a_quoted_span_must_appear_in_the_cited_evidence():
    ok = Claim(text='The paper says "contrastive objective over 1,000 pairs".', evidence_ids=["E1"])
    assert check_claims([ok], EVIDENCE) == []
    # Whitespace and case differences are not a mismatch.
    loose = Claim(text='It uses a "Contrastive  objective over 1,000 pairs"', evidence_ids=["E1"])
    assert check_claims([loose], EVIDENCE) == []
    wrong = Claim(text='The paper says "improve by 3.2 points" here.', evidence_ids=["E1"])
    problems = check_claims([wrong], EVIDENCE)
    assert [p.code for p in problems] == ["quote_not_in_source"]
    # Cited together, the span may come from either piece of evidence.
    assert check_claims([Claim(text=wrong.text, evidence_ids=["E1", "E2"])], EVIDENCE) == []


def test_short_quoted_fragments_are_not_checked_as_quotes():
    short = Claim(text='The "encoder" is trained.', evidence_ids=["E2"])
    assert check_claims([short], EVIDENCE) == []


def test_problems_are_reported_per_claim_in_order():
    claims = [
        Claim(text="fine", evidence_ids=["E1"]),
        Claim(text="bad id", evidence_ids=["E7"]),
        Claim(text="none", evidence_ids=[]),
    ]
    problems = check_claims(claims, EVIDENCE)
    assert [(p.claim, p.code, p.detail) for p in problems] == [
        (1, "unknown_evidence", "E7"),
        (2, "uncited_claim", ""),
    ]
    assert provenance_valid(claims, EVIDENCE) is False
