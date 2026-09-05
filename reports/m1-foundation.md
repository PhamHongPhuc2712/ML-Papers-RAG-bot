# M1 foundation evidence

Date: 2026-09-05
Task: P1.1, runnable API and isolated persistence harness
Worktree: `codex/p1-foundation`

## Selected runtime and dependency versions

The provisioned `backend/.venv` is Python 3.12.5. The reproducible package
resolver is uv 0.12.10, invoked from
`.superpowers/runtime/uv-0.12.10/uv.exe`; `backend/uv.lock` resolves 53
packages. The application pins FastAPI 0.116.1, Uvicorn 0.35.0,
pydantic-settings 2.10.1, SQLAlchemy 2.0.43, psycopg[binary] 3.2.9, Alembic
1.16.5, Qdrant client 1.19.0, and httpx 0.28.1. Development tools are pytest
8.4.1, pytest-cov 6.2.1, Ruff 0.12.10, and mypy 1.17.1. These versions were
resolved together under Python 3.12 and installed into the required project
venv with `uv sync --project backend --active`.

Compose pins PostgreSQL `17.11-bookworm` and Qdrant `v1.19.1`. The native
runtime used for integration evidence is PostgreSQL 17.11 on
`127.0.0.1:55432/test_copilot` and Qdrant 1.19.1 on
`http://127.0.0.1:56333`; both services are loopback-only and disposable.
The integration fixture uses the `test_` database and Qdrant collection
namespace and refuses destructive cleanup otherwise.

## RED evidence

Before production modules existed:

```text
& '.superpowers/runtime/uv-0.12.10/uv.exe' run --project backend pytest backend/tests/integration/test_health.py -q
ImportError while loading conftest '...backend/tests/conftest.py'.
backend/tests/conftest.py:11: in <module>
    from copilot.app import create_app
E   ModuleNotFoundError: No module named 'copilot'
```

The failure was collection-time module absence, which is the expected RED
condition for the new backend package.

## GREEN and verification evidence

The complete suite ran against the native PostgreSQL and Qdrant services with
no skipped integration checks:

```text
uv run --project backend pytest backend/tests -q
............                                                             [100%]
12 passed in 3.72s
```

The focused checks also passed:

```text
uv run --project backend pytest backend/tests/unit -q
3 passed in 0.05s

uv run --project backend pytest backend/tests/integration/test_health.py -q
9 passed in 3.68s

uv run --project backend ruff check backend
All checks passed!

uv run --project backend mypy backend/src
Success: no issues found in 7 source files
```

The acceptance cases cover independent liveness and readiness, missing active
release, missing database with typed 503 and no credentials in the response,
missing vector collections after an active release, ready vectors, production
mock/default-secret rejection, migration table creation, and database/Qdrant
cleanup namespace guards.

Migration replay against the isolated database:

```text
uv run --project backend alembic -c backend/alembic.ini downgrade base
INFO  [alembic.runtime.migration] Running downgrade 0000_foundation -> , Create corpus release and active release pointer tables.

uv run --project backend alembic -c backend/alembic.ini upgrade head
INFO  [alembic.runtime.migration] Running upgrade  -> 0000_foundation, Create corpus release and active release pointer tables.
```

Programmatic inspection after the replay reported `['alembic_version']` after
downgrade and `['active_release', 'alembic_version', 'corpus_releases']` after
upgrade. The active pointer has a foreign key to `corpus_releases` and a check
constraint limiting its singleton key to `active`.

## Delivered files

The implementation adds the `copilot` package, validated settings, section 6
Pydantic contracts and adapter protocols, SQLAlchemy foundation models, the
Alembic environment and `0000_foundation` migration, the FastAPI health API,
the real-service pytest fixtures/tests, locked project configuration, Compose
services, backend image definition, environment example, CI checks, LF line
ending policy, and a backend setup README. The task report is maintained in
`.superpowers/sdd/2026-09-05-01-corpus-foundation/task-1-report.md`.

## Limits and follow-up

Docker is not installed on this Windows host, so `docker compose config` and an
image build were not run; the Compose YAML was parsed successfully with the
locked Python environment. Native Qdrant logs
`Filesystem type check is not supported on this platform` for its Windows
storage path; HTTP readiness and collection create/upsert/delete smoke checks
passed. Linux container validation remains a CI concern. Authentication,
corpus ingestion, jobs, and application routes beyond health stay outside P1.1.
