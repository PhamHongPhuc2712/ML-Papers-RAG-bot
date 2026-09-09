"""Pure normalization and input-shape acceptance tests for corpus identity."""

from __future__ import annotations

import pytest

from copilot.corpus.normalize import normalize_arxiv, normalize_doi, normalize_title


def test_external_id_normalization() -> None:
    assert normalize_doi(" https://doi.org/10.1000/ABC ") == "10.1000/abc"
    assert normalize_arxiv("2401.01234v2") == ("2401.01234", "v2")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "invalid_doi"),
        ("doi: nope", "invalid_doi"),
        ("https://doi.org/10/short", "invalid_doi"),
    ],
)
def test_invalid_doi_is_rejected(value: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        normalize_doi(value)


@pytest.mark.parametrize("value", ["", "2401", "https://arxiv.org/abs/not-an-id"])
def test_invalid_arxiv_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="invalid_arxiv_id"):
        normalize_arxiv(value)


def test_title_matching_uses_nfkc_and_collapsed_unicode_whitespace() -> None:
    assert normalize_title("  Ｍodel\u00a0  Cards  ") == "model cards"
