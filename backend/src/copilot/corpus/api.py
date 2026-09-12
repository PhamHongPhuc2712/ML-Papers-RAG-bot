"""Public corpus endpoints: venue-year coverage with honest denominators."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from .export import snapshot_coverage

router = APIRouter(prefix="/v1/corpus", tags=["corpus"])


@router.get("/coverage")
def coverage(request: Request) -> dict[str, Any]:
    """Report ingested counts per venue-year.

    Percentages are null wherever no authoritative denominator has been
    confirmed for that venue-year: an ingested count says nothing on its own
    about what fraction of the venue it covers (spec §4).
    """

    engine: Engine = request.app.state.db_engine
    with Session(bind=engine, autoflush=False, expire_on_commit=False) as session:
        venues = snapshot_coverage(session)
    return {
        "request_id": str(uuid4()),
        "as_of": datetime.now(UTC).isoformat(),
        "venues": venues,
    }
