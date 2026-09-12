"""Export eligibility policy: pure rules that need no services."""

from __future__ import annotations

import pytest

from copilot.corpus.export import (
    PRIVATE_FIELDS,
    SCHEMA_VERSION,
    ExportSchemaError,
    coverage_percentage,
    public_export_rows,
    reject_private_fields,
    require_schema_version,
)


def test_private_or_unlicensed_text_is_not_exported():
    rows = [
        {"id": "a", "kind": "corpus", "redistribution": "allowed", "text": "public"},
        {"id": "b", "kind": "upload", "redistribution": "allowed", "text": "private"},
        {"id": "c", "kind": "corpus", "redistribution": "unknown", "text": "unknown"},
    ]
    assert [r["id"] for r in public_export_rows(rows)] == ["a"]


def test_database_spelling_of_eligible_is_publishable():
    """Spec §5 names the enum value ``eligible``; the plan example says ``allowed``.

    Both denote the same decision, so the filter accepts either rather than
    silently dropping every real row the corpus pipeline writes.
    """

    rows = [
        {"id": "a", "kind": "corpus", "redistribution": "eligible"},
        {"id": "b", "kind": "corpus", "redistribution": "restricted"},
    ]
    assert [r["id"] for r in public_export_rows(rows)] == ["a"]


def test_missing_or_unknown_eligibility_never_defaults_to_publishable():
    rows = [
        {"id": "a", "kind": "corpus"},
        {"id": "b", "kind": "corpus", "redistribution": None},
        {"id": "c", "kind": "corpus", "redistribution": "ALLOWED"},
    ]
    assert public_export_rows(rows) == []


def test_private_fields_are_rejected_before_any_artifact_is_written():
    with pytest.raises(ExportSchemaError, match="private_field"):
        reject_private_fields([{"id": "a", "user_id": "u1"}])
    for field in PRIVATE_FIELDS:
        with pytest.raises(ExportSchemaError):
            reject_private_fields([{"id": "a", field: "x"}])
    reject_private_fields([{"id": "a", "title": "fine"}])


def test_unknown_schema_version_is_refused():
    require_schema_version({"schema_version": SCHEMA_VERSION})
    with pytest.raises(ExportSchemaError, match="schema_version"):
        require_schema_version({"schema_version": SCHEMA_VERSION + 1})
    with pytest.raises(ExportSchemaError, match="schema_version"):
        require_schema_version({})


def test_unknown_denominator_yields_null_coverage_never_one_hundred():
    assert coverage_percentage(100, None) is None
    assert coverage_percentage(0, None) is None
    assert coverage_percentage(50, 200) == 25.0
    assert coverage_percentage(100, 100) == 100.0
    # A denominator smaller than the count is inconsistent, not 100% coverage.
    assert coverage_percentage(120, 100) is None
