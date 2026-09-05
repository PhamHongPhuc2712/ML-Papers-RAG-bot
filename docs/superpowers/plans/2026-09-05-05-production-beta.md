# Production Beta and Portfolio Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the product reliable enough for external researchers and substantiate the ML engineering story.

**Architecture:** Existing modules gain consistent tracing, quotas, CI, reproducible deployment and recovery. Public claims are tied to evidence, with personal and external-user validation kept explicit.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL, Qdrant, Next.js/TypeScript, pytest, Playwright; subsystem additions are specified below.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md). Read the relevant sections and global contracts before implementing.

**Status:** Draft for review. No implementation tasks completed.

## Global Constraints

- Python 3.12; Node.js 22; TypeScript strict mode; UTF-8 files; UTC timestamps.
- Lock Python and JavaScript dependencies; pin container images and model revisions before benchmark or deployment runs.
- Hugging Face is an offline corpus/artifact destination and is never queried during user requests.
- Public corpus data and private user documents use separate Qdrant collections and storage namespaces.
- Derive user identity from a verified token; never trust a client-supplied user_id.
- All benchmark runs record corpus, split, model, configuration, code, hardware, and seed versions.
- No paid cloud resources, public publishing, or application implementation occur while preparing these documents.

## Entry checkpoint and working conventions

Entry: M1–M4 technical gates pass. Read spec §§10–12 and 14. CI/observability basics already started earlier; this milestone closes system-level gaps. Provisioning requires concrete hardware/budget/provider settings at execution time.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P5.1: Enforce deterministic CI and reproducible builds

**Depends on:** P1–P4 technical tasks

**Files:**

- Modify `.github/workflows/checks.yml`; create `.github/workflows/model-evaluation.yml`.
- Create `infra/Dockerfile.frontend`, `infra/ci_validate.py`; update Compose and backend Dockerfile.
- Create `backend/tests/unit/test_run_manifest.py`; modify evaluation manifest validation.

**Interfaces:** `validate_run_manifest(manifest: dict) -> None`; workflow entry points execute existing `eval smoke`, pytest and frontend scripts. Expensive workflow accepts config/split/budget and produces immutable report artifacts, never mutating production state.

- [ ] Write the regression test below in `backend/tests/unit/test_run_manifest.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.evaluation.report import validate_run_manifest

def test_unversioned_benchmark_is_rejected():
    with pytest.raises(ValueError, match="missing_manifest_fields"):
        validate_run_manifest({"run_id": "demo", "seed": 42})
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_run_manifest.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
REQUIRED_FIELDS = {
    "run_id", "git_sha", "corpus_release", "dataset_revision", "split_revision",
    "model_revisions", "config_sha256", "hardware", "seed", "timing_method"
}

def validate_run_manifest(manifest: dict) -> None:
    missing = REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise ValueError("missing_manifest_fields:" + ",".join(sorted(missing)))
```

CI jobs: Python locked sync→Ruff→mypy→unit; isolated Postgres/Qdrant→migrations→integration→eval smoke; Node locked install→typecheck→lint→build→Playwright; Docker image build using exact dependency locks. Cache by lockfile/model digest. Pull requests from forks do not receive secrets; deterministic suite requires none. Upload test/benchmark reports even on failure. Nightly/manual workflow uses budgeted configured models and a fixed dataset revision, recording provider IDs and cost. Environment promotion uses the exact tested image digest and blocks incompatible manifests. Add `infra/ci_validate.py` to validate locally that all required check commands and exit statuses are collected before marking a release candidate.

- [ ] Verify the additional acceptance cases: CI fails on fixture ranking regression >0.03 absolute; invalid citations fail; missing model/split revision rejected; integration cannot silently skip on missing Docker; frontend API type drift fails; model API unavailable doesn't break deterministic checks; lockfile changes invalidate caches; no production secret in build artifacts.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_run_manifest.py -q` again, then run the full local CI command sequence and a clean Docker build; verify artifact manifest contains all required check exits. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m5-ci.md` with job names, commands, exits and tested image digests; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `ci: enforce reproducible builds and evaluation regression checks`.

## P5.2: Add operational tracing, quotas and spend controls

**Depends on:** P5.1

**Files:**

- Create `backend/src/copilot/operations/{tracing,limits,costs}.py`, `backend/migrations/versions/0005_operations.py`, `configs/limits.yaml`.
- Create `backend/tests/unit/test_trace_redaction.py`, `backend/tests/integration/test_limits.py`.
- Instrument search/chat/jobs/recommendations; create `docs/runbooks/observability.md`.

**Interfaces:** `safe_trace(fields: dict) -> dict`; `reserve_usage(principal_or_ip: str, kind: str, units: int, request_id: UUID) -> bool`; `record_cost(request_id: UUID, usage: dict) -> None`. Rate counters and generation leases use atomic PostgreSQL transactions and server time.

- [ ] Write the regression test below in `backend/tests/unit/test_trace_redaction.py` and add the additional acceptance cases listed for this task.

```python
from copilot.operations.tracing import safe_trace

def test_private_content_does_not_enter_default_trace():
    result = safe_trace({"request_id": "r", "latency_ms": 20, "query": "private",
                         "document_text": "secret", "authorization": "Bearer secret"})
    assert result == {"request_id": "r", "latency_ms": 20}
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_trace_redaction.py backend/tests/integration/test_limits.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
TRACE_FIELDS = {"request_id", "latency_ms", "stage", "status", "error_code",
                "corpus_release", "model_revision", "candidate_count",
                "input_tokens", "output_tokens", "cost", "job_id"}

def safe_trace(fields: dict) -> dict:
    return {key: value for key, value in fields.items() if key in TRACE_FIELDS}
```

Use allowlisted structured logs and optional OpenTelemetry exporter; one request ID spans stages. Candidate IDs may be added only under an explicit safe field schema, never wholesale provider responses. Implement spec quotas across API processes, authenticated user keys and trusted-proxy-aware anonymous keys. Reserve generation concurrency and conservative token-cost budget before provider calls; reconcile actual usage afterward, retain unresolved reserves until expiry/reconciliation to avoid overspend on timeout. Missing cost configuration disables priced calls with a clear typed error. Daily cap is runtime operator configuration, not a guessed monetary amount. Retention jobs enforce seven-day diagnostic/30-day operational/180-day event defaults. Dashboard exposes aggregate latency/errors/costs, not private query text.

- [ ] Verify the additional acceptance cases: two processes cannot bypass quotas; idempotent retry doesn't reserve twice; lease expires after crashed request; spend cap race rejected; partial provider usage reconciled; missing price config blocks paid call; forwarded headers spoofing fails; private text never exported in default telemetry; retention respects deletion state.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_trace_redaction.py backend/tests/integration/test_limits.py -q` again, then run concurrency/rate-limit integration tests and induce model, vector and worker failures while checking trace correlation. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m5-operations.md` with quota concurrency cases, failure traces and cost reservation behavior; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: enforce operational quotas and trace request performance`.

## P5.3: Deploy a staging release and prove recovery and account deletion

**Depends on:** P5.2

**Files:**

- Create `infra/deploy.py`, `infra/backup.py`, `infra/restore.py`, `infra/release-manifest.schema.json`.
- Create `backend/src/copilot/identity/deletion.py`, `backend/tests/integration/test_account_deletion.py`.
- Create `docs/runbooks/{deployment,recovery,deletion}.md`; extend deletion worker/CLI.

**Interfaces:** `delete_account(principal: Principal) -> UUID` queues deletion; deployment CLI consumes a release manifest naming image digests, migration head, corpus/model releases and environment. `infra/restore.py --manifest PATH --target staging-restore` refuses production targets without an explicit separately handled operator workflow.

- [ ] Write the regression test below in `backend/tests/integration/test_account_deletion.py` and add the additional acceptance cases listed for this task.

```python
def test_account_delete_immediately_hides_private_state(api_client, user_headers):
    headers = user_headers("11111111-1111-4111-8111-111111111111")
    response = api_client.delete("/v1/me", headers=headers)
    assert response.status_code == 202
    assert response.json()["job_id"]
    assert api_client.get("/v1/library", headers=headers).status_code in (401, 404)
```

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_account_deletion.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```json
{
  "schema_version": 1,
  "environment": "staging",
  "release_policy": {
    "require_image_digest": true,
    "require_migration_head": true,
    "require_corpus_release": true,
    "require_ready_check": true,
    "retain_previous_release": true
  },
  "recovery_targets": {"rpo_hours": 24, "rto_hours": 4}
}
```

At execution time choose an actual container host/region based on M2 capacity, GPU need and operator budget, then record its deployment adapter/configuration. Do not invent deployed IDs in this plan. The deployed frontend/API/worker use exact tested artifacts; configure TLS, origins, non-owner DB role, private upload storage, server-only vector credentials and health checks. Deploy to staging first, migrate using expand/contract, run known-answer search and authenticated user journey. Back up DB and upload manifest; retain corpus/vector artifacts separately with version checks. Restore to a clean staging namespace and check canonical counts, saved papers, chats, private uploads and search canaries. Account deletion tombstones immediately, revokes usable access in the app, and removes owned rows, vectors, blobs, caches and diagnostic captures; retry physical deletion to 24-hour SLA. Document backup retention and handling of deletions after restoration.

- [ ] Verify the additional acceptance cases: broken readiness blocks deployment; incompatible DB migration blocks release; rollback switches app and matching corpus pointer; restored system serves known-answer query; private uploads restored only to private storage; other users survive account deletion; failed vector cleanup retries; deleted account cannot resurrect through cached token; tombstones reapplied after restore.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_account_deletion.py -q` again, then run actual staging backup/restore and rollback drills, and execute deletion lifecycle through worker completion. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m5-recovery.md` with actual timestamps, measured RPO/RTO, restore counts and deletion completion; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `ops: add staged deployment recovery and account deletion`.

## P5.4: Measure full-system latency, isolation and failure behavior

**Depends on:** P5.3

**Files:**

- Create `infra/load_test.py`, `backend/tests/integration/test_tenant_adversarial.py`, `frontend/tests/beta-journeys.spec.ts`.
- Create `configs/load-test.yaml`, `reports/performance/README.md`.

**Interfaces:** Load CLI `python infra/load_test.py --config configs/load-test.yaml --out reports/performance/staging.json`; output includes every attempted request, status, latency, scenario, cold/warm label and hardware/environment versions. Tests use authorized staging accounts and synthetic fixture uploads.

- [ ] Write the regression test below in `backend/tests/integration/test_tenant_adversarial.py` and add the additional acceptance cases listed for this task.

```python
def test_job_status_does_not_leak_owner(api_client, user_headers, alice_upload_job_id):
    bob = user_headers("22222222-2222-4222-8222-222222222222")
    response = api_client.get(f"/v1/jobs/{alice_upload_job_id}", headers=bob)
    assert response.status_code == 404
    assert "progress_done" not in response.json()
```

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_tenant_adversarial.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```yaml
concurrency: 5
warmup_requests: 20
measured_requests: 200
scenarios:
  search: 0.5
  feed: 0.3
  library: 0.2
budgets:
  search_p95_ms: 3000
  cached_feed_p95_ms: 500
  research_answer_p95_ms: 20000
  pdf_20_pages_p95_ms: 120000
record_failures: true
record_cold_starts_separately: true
```

Begin fixture-only load, then bounded real-model research requests under a recorded budget. `alice_upload_job_id` fixture creates an owned job using the actual upload API and leaves it queued for the ownership test. Exercise guessed IDs for all private routes and payload-injection attempts against document retrieval; test raw SQL under runtime role and Qdrant query filters. Measure 1k and then 10k paper memory/storage/build throughput before full-scale ingestion; report exact corpus, point count and model hardware. Stop Qdrant, model process and a worker independently; confirm typed degradation and intact library. Check cold-start measurements separately and count timeouts/errors in aggregate reports. Repeat only if new changes or failures justify it.

- [ ] Verify the additional acceptance cases: no cross-user access across chat/jobs/uploads/events/feed; protected storage URLs expire; slow model doesn't exhaust ASGI event loop; cancelled streams release concurrency; Qdrant failure doesn't corrupt library; worker crash recovers; all three workflow journeys pass; measured budgets meet targets or documented remediation narrows beta capacity.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_tenant_adversarial.py -q` again, then run `npm --prefix frontend run test:e2e -- beta-journeys.spec.ts` and the staged load/failure test configuration. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/performance/staging.json` and `reports/m5-beta-readiness.md` with actual pass/fail against each budget; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `test: verify beta isolation load and dependency failures`.

## P5.5: Prepare an honest launch and ML engineering case study

**Depends on:** P5.4; P3.5 longitudinal observations may remain pending

**Files:**

- Create `docs/portfolio-case-study.md`, `docs/beta-pilot.md`, `docs/experiment-log.md`, root `README.md`.
- Create `backend/src/copilot/evaluation/release_gate.py`, `backend/tests/unit/test_release_gate.py`; extend CLI.

**Interfaces:** `missing_release_evidence(evidence: dict[str,str]) -> list[str]`; `release-check --evidence reports/release-evidence.json` checks actual report paths/checksums and recorded statuses; it does not infer production readiness from file existence alone.

- [ ] Write the regression test below in `backend/tests/unit/test_release_gate.py` and add the additional acceptance cases listed for this task.

```python
from copilot.evaluation.release_gate import missing_release_evidence

def test_passing_unit_tests_do_not_substitute_for_restore():
    evidence = {"corpus": "pass", "retrieval": "pass", "recommendations": "pass",
                "rag": "pass", "isolation": "pass", "load": "pass",
                "recovery": "pending"}
    assert missing_release_evidence(evidence) == ["recovery"]
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_release_gate.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
REQUIRED_GATES = ("corpus", "retrieval", "recommendations", "rag",
                  "isolation", "load", "recovery")

def missing_release_evidence(evidence: dict[str, str]) -> list[str]:
    return [gate for gate in REQUIRED_GATES if evidence.get(gate) != "pass"]
```

Write a reproducible README with actual setup/evaluation/deployment commands, known limitations, verified corpus size and demo route. Case study includes retrieval ablations, recommendation features and temporal evaluation, failures, cost/latency trade-offs, contribution attribution and measured outcomes. Use Open WebUI only as a referenced UX study; do not claim its implementation as this project's work. Prepare a five-minute demo: topic search→related papers→save→feed change→grounded answer→evidence. Plan a first pilot with 3–5 researchers recruited manually by the user, ask them to complete the three workflows with their own topics, record consented task completion/useful papers/failure notes and a second visit after one week. No outreach is sent by this task without user instruction. One-person longitudinal results are labeled personal; pending external-user validation stays pending. Public availability starts only after release gates and the actual deployment choice are reviewed.

- [ ] Verify the additional acceptance cases: all public claims link to measured reports; no target corpus count presented as achieved; missing recovery/isolation evidence blocks ready status; no private queries in screenshots; pilot tasks aren't scripted to guarantee relevance; recommendation feedback limitations explicit; case study explains at least one failed experiment and its decision.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_release_gate.py -q` again, then run `uv run --project backend python -m copilot.cli release-check --evidence reports/release-evidence.json` and rehearse the demo against the tested release. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `docs/portfolio-case-study.md`, `reports/release-evidence.json` and the dated beta pilot protocol/results as observations become available; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `docs: publish reproducible ml engineering case study and beta protocol`.

## Exit checkpoint

G5 requires staging journeys, isolation, quotas, cost controls, load results, rollback and restore. A public beta can start with single-person evaluation honestly disclosed; do not call it validated for researchers until pilot evidence exists. Next: optional Plan 6; M1–M5 form the portfolio MVP.

