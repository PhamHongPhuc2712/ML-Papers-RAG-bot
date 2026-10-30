# Backend

The `copilot` Python package: the FastAPI application, the durable job worker, the corpus
pipeline, search, the chat agent and the evaluation CLI. Python 3.12, managed by
[uv](https://docs.astral.sh/uv/) with `backend/.venv` as its only environment.

```bash
uv sync --project backend --frozen --group dev                  # offline development and CI
uv sync --project backend --frozen --group dev --group models   # + PyTorch, for the real models
uv run --project backend python -m copilot.cli --help           # every offline command
```

- `src/copilot/` — the package; [architecture](../docs/architecture.md) maps its modules.
- `migrations/` — Alembic migrations (`alembic -c backend/alembic.ini upgrade head`).
- `tests/unit/` — offline tests, run in CI (`pytest -m "not integration"`).
- `tests/integration/` — need the Compose `test` profile (`postgres-test`, `qdrant-test`,
  `mailpit-test`) and `.env.test`; missing services fail the suite rather than skip it.
- `tests/fixtures/` — recorded LLM responses and small corpora; unit and integration tests use a
  deterministic two-dimensional embedder, so CI never downloads a model.

Setup, services and the full command list: [docs/setup.md](../docs/setup.md).
