# Setup

## Requirements

- Linux (developed on WSL2), Docker, [uv](https://docs.astral.sh/uv/), Node.js 22.
- An NVIDIA GPU for the embedding model and reranker (developed on an RTX 3080, 16 GB).
- Disk: the corpus database, indexes and models take a few hundred GB under `DATA_DIR`; building
  the corpus streams one venue-year of PDFs at a time.
- An OpenAI API key for answers (the only paid resource), under a daily spend cap.

## Configure

```bash
cp .env.example .env
```

Set at least:

| Variable | What it is |
|---|---|
| `DATA_DIR` | The single root of every persistent byte: Postgres and Qdrant volumes, models, benchmarks, exports, backups. Deleting it deletes the dataset. Startup fails when it is unset. |
| `DATABASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION_PREFIX` | The local services below |
| `SECRET_KEY` | Signs access tokens |
| `SMTP_HOST`, `SMTP_PORT`, `MAIL_FROM` | Where sign-in codes are sent (Mailpit locally) |
| `OPENAI_API_KEY`, `LLM_DAILY_SPEND_CAP_USD` | Answer generation and its daily cap |
| `HF_TOKEN` | Optional: downloads the corpus mirror without anonymous rate limits |

`.env.test` holds only `DATA_DIR`, `TEST_DATABASE_URL`, `TEST_QDRANT_URL` and
`TEST_QDRANT_COLLECTION_PREFIX` for the integration tests. Never pass `.env` to pytest: its
`DATABASE_URL` would point the suite at the development database.

## Install and start the services

```bash
docker compose --profile core up -d postgres qdrant mailpit     # Mailpit inbox: http://127.0.0.1:8025
uv sync --project backend --frozen --group dev --group models   # models: PyTorch for real models
uv run --env-file .env --project backend alembic -c backend/alembic.ini upgrade head
uv run --env-file .env --project backend python -m copilot.cli search fetch-model
```

## Build the corpus and its indexes

```bash
CLI="uv run --env-file .env --project backend python -m copilot.cli"
$CLI corpus run                       # every venue-year in configs/venues.yaml; resumable
$CLI corpus run --only ICLR:2024      # or one venue-year
$CLI corpus export --run SNAP --out SNAP
$CLI search build-index --manifest $DATA_DIR/exports/SNAP/manifest.json --collections papers,chunks
$CLI search validate-index --release SNAP
$CLI corpus activate --release SNAP
```

`corpus run` mirrors a venue-year, ingests it, parses with several workers, verifies, then deletes
that venue-year's PDFs (except ones that failed to parse). Progress is kept under `DATA_DIR`, so
a restart continues where it stopped. `corpus coverage` reports what is loaded.

## Run

```bash
uv run --env-file .env --project backend uvicorn --factory copilot.app:create_default_app --port 8000
cd frontend && npm ci && npm run build && BACKEND_URL=http://127.0.0.1:8000 npm run start
```

Open http://localhost:3000, sign in with any email address and paste the code from the Mailpit
inbox. The API warms its search before it accepts requests (about 40 s, longer right after
Qdrant restarts).

## Checks

```bash
uv run --project backend ruff check backend
uv run --project backend mypy --config-file backend/pyproject.toml backend/src
uv run --project backend pytest backend/tests -m "not integration" -q       # offline, as in CI
docker compose --profile test up -d postgres-test qdrant-test mailpit-test
uv run --env-file .env.test --project backend pytest backend/tests -q        # full suite
cd frontend && npm run lint && npm run typecheck && npm test && npm run e2e  # e2e: scripted backend
```

`mypy` must be given `--config-file`; from the repository root it otherwise finds no
configuration and drops strict mode. The integration fixtures refuse any database or Qdrant prefix
not starting with `test_`. A walk through the real stack (it signs in through Mailpit and spends a
few cents): `REAL_STACK=1 REAL_EMAIL=you@example.org npx playwright test real-stack.spec.ts`.

## Operations

```bash
$CLI llm spend --day 2026-10-30        # calls, tokens and cost by model and purpose
$CLI chat stats --day 2026-10-30       # turns, errors, stage timings, cost; no text
$CLI backup create                     # user tables to $DATA_DIR/backups/<UTC stamp>
$CLI backup restore --from DIR --database-url URL [--replace]
```
