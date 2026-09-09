# Backend development

Python 3.12 backend managed by uv 0.12.10 with `backend/.venv` as the only
project environment. `uv` may not be on `PATH` on this machine; every command
below also works as `py -3.12 -m uv ...`.

## Setup

1. Copy `.env.example` to `.env` and set `DATA_DIR`, the single root for all
   persistent local data (documented default `C:\ml-copilot-data`). Deleting
   that directory deletes the whole dataset, including Docker service state.
   Startup fails in every environment when it is unset.
2. Create the locked environment:

   ```powershell
   uv sync --project backend --frozen --group dev
   ```

3. Start the isolated test services with Docker Desktop (Linux containers).
   They bind-mount `${DATA_DIR}/test/postgres` and `${DATA_DIR}/test/qdrant`:

   ```powershell
   docker compose --profile test up -d postgres-test qdrant-test
   ```

## Commands (from the repository root)

```powershell
uv run --project backend ruff check backend
uv run --project backend mypy backend/src
uv run --project backend pytest backend/tests -m "not integration" -q   # offline
uv run --project backend pytest backend/tests -q                        # needs services
uv run --project backend alembic -c backend/alembic.ini upgrade head    # needs DATABASE_URL
```

## Integration tests

The fixture reads `TEST_DATABASE_URL`, `TEST_QDRANT_URL`,
`TEST_QDRANT_COLLECTION_PREFIX` and `DATA_DIR` explicitly and fails — never
skips — when one is missing. Destructive cleanup refuses a database not named
`test_*`, a collection prefix not starting with `test_`, and any data root
other than `${DATA_DIR}/test`. With the Compose defaults:

```powershell
$env:DATA_DIR = 'C:\ml-copilot-data'
$env:TEST_DATABASE_URL = 'postgresql+psycopg://copilot_test:copilot_test_password@127.0.0.1:55432/test_copilot'
$env:TEST_QDRANT_URL = 'http://127.0.0.1:56333'
$env:TEST_QDRANT_COLLECTION_PREFIX = 'test_'
uv run --project backend pytest backend/tests -q
```

## Fixture replay

`copilot fixtures load` replays `data/fixtures/metadata.jsonl` through the
identity resolver and reports loaded/quarantined/conflict counts. Raw source
JSON is retained by checksum under `${DATA_DIR}/sources` unless
`--staging-dir` overrides it:

```powershell
uv run --project backend python -m copilot.cli fixtures load --database-url $env:TEST_DATABASE_URL
```

## Pilot ingestion

`corpus ingest` lists a venue-year through its membership sources, keeps
eligible records, sorts them by source item ID, takes the sample limit and
enqueues one durable job per paper; `worker run` processes the resolve →
download → parse chain until nothing is due. Both read `DATABASE_URL` and
`DATA_DIR` from `.env`; PDFs and raw source JSON land under
`${DATA_DIR}/sources`. OpenReview's API challenges guest requests, so set
`OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` (a free account) in `.env`
first.

```powershell
uv run --project backend python -m copilot.cli corpus ingest --manifest configs/corpus.yaml --limit 100
uv run --project backend python -m copilot.cli worker run --worker-id pilot-1
```

Re-running `corpus ingest` is idempotent: jobs are keyed by source item and
revision, and completed work is never repeated.
