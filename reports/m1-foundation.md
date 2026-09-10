# M1 foundation evidence

Date: 2026-09-10
Task: P1.1, runnable API and isolated persistence harness
Commits: `c9ec8f4` (task), `f652b9a` (fixture fix carrying the integration evidence)
Host: Windows 11 Home 10.0.26200, 13.7 GB RAM

Adapted from the pre-reset implementation at `04f6e3e` for the 2026-09-09
design revision: `DATA_DIR` required in every environment, Compose bind
mounts under it instead of named volumes, and a `<DATA_DIR>/test` guard on
destructive fixtures. This report is local-only; `reports/` is gitignored by
the owner's choice.

## Selected runtime and dependency versions

Python 3.12.5 (`py -3.12`). uv 0.12.10 installed into the user site and
invoked as `py -3.12 -m uv`; `backend/.venv` is the only project environment,
created with `uv sync --project backend --frozen --group dev`. The lockfile
pins FastAPI 0.116.1, Uvicorn 0.35.0, pydantic-settings 2.10.1, SQLAlchemy
2.0.43, psycopg[binary] 3.2.9, Alembic 1.16.5, qdrant-client 1.19.0, httpx
0.28.1; development tools pytest 8.4.1, pytest-cov 6.2.1, Ruff 0.12.10,
mypy 1.17.1.

Services: Docker Desktop 4.90.0 (Docker 29.7.2, Compose v5.5.1) on WSL
2.7.13 (kernel 6.18.33.2-2), both installed via winget on 2026-09-09 with one
reboot. Compose pins `postgres:17.11-bookworm` and `qdrant/qdrant:v1.19.1`.
Test profile bind mounts: `C:\ml-copilot-data\test\postgres` and
`C:\ml-copilot-data\test\qdrant`, ports 55432 and 56333.

## RED evidence

Before the source package existed (tests and lockfile restored first, venv
synced with `--no-install-project`):

```text
backend/.venv/Scripts/python -m pytest backend/tests/integration/test_health.py -q
ImportError while loading conftest '...backend\tests\conftest.py'.
backend\tests\conftest.py:12: in <module>
    from copilot.app import create_app
E   ModuleNotFoundError: No module named 'copilot'
```

## Interim failure, then GREEN

First full run against the containers: 22 passed, 45 integration errors, all
at session-fixture setup with
`test_data_dir_must_be_the_test_subdirectory`. Cause: pydantic-settings
resolves the `DATA_DIR` environment alias ahead of the fixture's explicit
`data_dir=<root>/test` keyword, so the guard correctly refused the root. Fixed
in `f652b9a` by pointing `DATA_DIR` at the test subdirectory for the session
with a `MonkeyPatch`. Second run:

```text
py -3.12 -m uv run --project backend --frozen pytest backend/tests -q
67 passed in 56.12s
py -3.12 -m uv run --project backend --frozen ruff check backend
All checks passed!
py -3.12 -m uv run --project backend --frozen mypy backend/src
Success: no issues found in 11 source files
```

Offline CI set (`-m "not integration"`): 22 passed. Environment for the full
run: `DATA_DIR=C:\ml-copilot-data`, `TEST_DATABASE_URL=postgresql+psycopg://copilot_test:...@127.0.0.1:55432/test_copilot`,
`TEST_QDRANT_URL=http://127.0.0.1:56333`, `TEST_QDRANT_COLLECTION_PREFIX=test_`.

## Acceptance cases

| Case | Test |
|---|---|
| Liveness does not imply readiness; missing DB is a typed 503 without credentials | `test_liveness_does_not_imply_readiness_when_database_is_unavailable` |
| Empty database is not ready (`corpus_not_ready`) | `test_liveness_and_missing_release` |
| Readiness requires the active release's vector collections | `test_readiness_requires_vector_collections_after_active_release`, `test_readiness_is_ok_when_active_release_vectors_exist` |
| Production rejects mock mode and default secrets | two `test_production_rejects_*` cases |
| Startup rejects an unset `DATA_DIR` in every environment | `test_unset_data_dir_is_rejected_in_every_environment[development|test|production]` |
| `DATA_DIR` read from environment, trailing separator normalized, relative path rejected, test env needs the `test` subdirectory | `backend/tests/unit/test_config.py` |
| Migrations start cleanly | `test_migration_creates_foundation_tables` |
| Fixture cleanup refuses non-`test_` database, non-`test_` prefix, and a data root outside `<DATA_DIR>/test` | three `test_cleanup_guard_rejects_*` cases |
| Postgres and Qdrant run on Windows bind mounts under `DATA_DIR` | `test_services_run_on_bind_mounts_under_the_test_data_root`; see below |

## Bind-mount verification and recorded warnings

`docker compose --profile test up -d --wait` reported both containers
Healthy. After the run `C:\ml-copilot-data\test\postgres` contains
`PG_VERSION`, `base`, `global`, `pg_commit_ts`, ... — the cluster was
initialised on the bind mount — and `C:\ml-copilot-data\test\qdrant` exists.

Container logs:

- postgres-test: `fixing permissions on existing directory /var/lib/postgresql/data ... ok`; `initdb: warning: enabling "trust" authentication for local connections` (standard image behaviour, loopback-only test service).
- qdrant-test: `WARN qdrant: There is a potential issue with the filesystem for storage path ./storage. Details: Unrecognized filesystem - cannot guarantee data safety`. The Windows bind mount is exposed to the container through Docker Desktop's file sharing, which Qdrant does not recognise. Acceptable for disposable test collections; it is a durability caveat for the `core` profile's real corpus vectors and must be weighed when P2.2 builds the first serving index (options: accept and rely on reproducible rebuilds from the PostgreSQL source of truth, or keep Qdrant storage in a named volume and document the exception to the delete-one-directory guarantee).

## Limits and follow-up

- `infra/Dockerfile.backend` was not built; the CI workflow was pushed but its first GitHub Actions run had not completed when this report was written.
- `.env` was created from `.env.example` for Compose (gitignored).
- Raw source artifacts default to `<DATA_DIR>/sources`, a subdirectory not yet listed in the specification §3 table.
- `docs/` is now untracked in git (`60953e6`); the design documents exist only on this machine.
