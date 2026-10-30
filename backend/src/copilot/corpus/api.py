"""Public corpus coverage endpoint (spec §10).

Coverage is reported per venue-year with separate counts for metadata, abstracts
and full text, because they fail independently: a paper can be indexed with an
abstract and no full text, and saying so is the difference between a coverage
claim and a guess. A venue-year whose published total has not been verified
reports a null percentage.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import APIRouter, Request
from sqlalchemy import Engine, text

from .export import coverage_rows

_COVERAGE_SQL = """
select coalesce(v.name, '') as venue,
       coalesce(p.publication_year, 0) as year,
       p.abstract as abstract,
       coalesce(pv.parse_status, '') as parse_status
from papers p
left join venues v on v.id = p.venue_id
left join paper_versions pv on pv.paper_id = p.id
where p.merged_into is null
"""


def expected_totals(path: str | Path) -> dict[tuple[str, int], int]:
    """Published venue-year totals that have actually been verified."""

    config = Path(path)
    if not config.is_file():
        return {}
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    coverage = (raw or {}).get("coverage", {}) if isinstance(raw, Mapping) else {}
    expected = coverage.get("expected", {}) if isinstance(coverage, Mapping) else {}
    totals: dict[tuple[str, int], int] = {}
    if isinstance(expected, Mapping):
        for venue, years in expected.items():
            if isinstance(years, Mapping):
                for year, count in years.items():
                    totals[(str(venue), int(year))] = int(count)
    return totals


def corpus_coverage(engine: Engine, expected: Mapping[tuple[str, int], int]) -> dict[str, Any]:
    """Counts as of now, straight from the database."""

    with engine.connect() as connection:
        papers = [dict(row) for row in connection.execute(text(_COVERAGE_SQL)).mappings()]
    rows = coverage_rows(papers, dict(expected))
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "venue_years": rows,
        "totals": {
            "papers": sum(row["papers"] for row in rows),
            "with_abstract": sum(row["with_abstract"] for row in rows),
            "with_fulltext": sum(row["with_fulltext"] for row in rows),
            "failed_parse": sum(row["failed_parse"] for row in rows),
            "venue_years": len(rows),
        },
    }


def router(engine: Engine, artifacts_config: str | Path) -> APIRouter:
    """Public, unauthenticated coverage route."""

    api = APIRouter(prefix="/v1/corpus", tags=["corpus"])
    expected = expected_totals(artifacts_config)

    @api.get("/coverage")
    def coverage(request: Request) -> dict[str, Any]:
        return {"request_id": str(uuid4()), **corpus_coverage(engine, expected)}

    return api
