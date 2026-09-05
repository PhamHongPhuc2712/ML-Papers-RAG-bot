# Backend development

The backend targets Python 3.12 and keeps its dependencies in `backend/.venv`.
From the repository root, use the repository-pinned uv executable when it is not
already on `PATH`:

```powershell
$uv = '.superpowers/runtime/uv-0.12.10/uv.exe'
& $uv sync --project backend --active
& $uv run --project backend pytest backend/tests/unit -q
```

The API uses PostgreSQL through psycopg 3 and Qdrant for vector readiness. The
integration fixture requires explicit test endpoints and refuses a database or
collection namespace without the `test_` prefix. For the portable native test
services prepared for this worktree, set:

```powershell
$env:TEST_DATABASE_URL = 'postgresql+psycopg://copilot_test:copilot_test_pw_2026@127.0.0.1:55432/test_copilot'
$env:TEST_QDRANT_URL = 'http://127.0.0.1:56333'
$env:TEST_QDRANT_COLLECTION_PREFIX = 'test_'
& $uv run --project backend pytest backend/tests/integration/test_health.py -q
```

With Docker available, start the pinned test services with
`docker compose --profile test up -d postgres-test qdrant-test`. The Compose
defaults use the same `test_copilot` database and `test_` collection namespace;
override the URLs if ports are changed. Apply the foundation migration with
`alembic -c backend/alembic.ini upgrade head` after setting `DATABASE_URL`.

The native-service paths, verified versions, startup commands, and the Windows
Qdrant filesystem limitation are recorded in the runtime report under
`.superpowers/runtime/report.md` and in `reports/m1-foundation.md`.
