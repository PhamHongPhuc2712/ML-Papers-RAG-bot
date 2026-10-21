"""The rule that repairs bibliography chunks stored as body text (2026-10-20)."""

from __future__ import annotations

import pytest

from copilot.corpus.relabel import misread_reference, proceedings_volume


@pytest.mark.parametrize(
    "name",
    [
        "References",
        "REFERENCES",
        "Bibliography",
        "Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf",
        "Wu,T.,Yang,G.,Li,Z.,Zhang,K.,Liu,Z.,Guibas,L.,Lin,D.: Gpt-4v(ision) is a",
        "Fridovich-Keil, S., Meanti, G., Warburg,",
        "Müller, T., Evans, A.",
    ],
)
def test_references_and_citation_lines_are_misread_references(name):
    assert misread_reference(name)


@pytest.mark.parametrize(
    "name",
    [
        "Introduction",
        "Related Work",
        "Appendix C, I.",
        "Acknowledgments",
        "Technical Details",
        "Correction with Both Acoustic and Confidence References",
        "Theory Assumptions and Proofs",
    ],
)
def test_real_sections_are_left_alone(name):
    assert not misread_reference(name)


@pytest.mark.parametrize(
    ("title", "abstract"),
    [
        (
            "Proceedings of the 61st Annual Meeting of the Association for Computational"
            " Linguistics (Volume 1: Long Papers)",
            None,
        ),
        (
            "Proceedings of the 2024 Conference on Empirical Methods in Natural Language"
            " Processing",
            "",
        ),
    ],
)
def test_a_whole_proceedings_volume_is_recognized(title, abstract):
    assert proceedings_volume(title, abstract)


@pytest.mark.parametrize(
    ("title", "abstract"),
    [
        (
            "Disentangling Societal Inequality from Model Biases: Gender Inequality in Divorce"
            " Court Proceedings",
            None,
        ),
        ("Proceedings of the Workshop Replicated", "We replicate twelve workshop papers."),
        ("Contrastive Sentence Embeddings", None),
    ],
)
def test_a_paper_is_not_a_proceedings_volume(title, abstract):
    assert not proceedings_volume(title, abstract)
