# Production Beta and Portfolio Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the product reliable on the local self-hosted stack and substantiate the ML engineering story; reliability for external researchers follows a funded deployment, which is deferred.

**Architecture:** Existing modules gain consistent tracing, quotas, CI, and — deferred until a deployment exists — reproducible deployment and recovery. Public claims are tied to evidence, with personal and external-user validation kept explicit.

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
- Postgres and Qdrant are self-hosted via Docker on the developer's own machine by default; no managed database, vector, or storage subscription is used unless a future, explicitly-recorded decision changes this.
- All persistent local data lives under one configurable root directory, default `C:\ml-copilot-data\` on the developer's machine, so the entire dataset can be deleted by removing that directory.
- The hosted LLM/multimodal generation API is the one paid resource in the project, funded personally by the developer; its cost tracking and daily spend cap from P5.2 remain required because real personal money is involved.

## Entry checkpoint and working conventions

Entry: M1–M4 technical gates pass. Read spec §§1, 10–12 and 14. CI/observability basics already started earlier; this milestone closes system-level gaps.

Portfolio scope. Per spec §1 the portfolio MVP is M1–M4 plus the free-tier-compatible subset of this milestone: P5.1, the cost-tracking half of P5.2, the account-deletion and local backup/restore half of P5.3, P5.4 and P5.5. The deployment half of P5.3 and the public-traffic half of P5.2 are deferred and marked in place below rather than removed, so the reasoning stays auditable and a future funded deployment can pick them up unchanged. Everything still required here runs on the developer's own machine against `DATA_DIR`; no provider account, region or infrastructure spend is needed at execution time.

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

**Scope: required, with one deferred part.** Tracing, per-request LLM cost logging and the operator-configured daily spend cap stay fully in force: the hosted generation API is this project's single paid line item, funded personally by the developer, so the cap protects real personal money. Deferred: operating the rate limits as public multi-tenant protection for a paid deployment. The quota code, its atomic counters and its tests are still built and exercised against the developer's own traffic — they are good portfolio evidence and a prerequisite for any later public phase — but enforcing them against strangers' traffic is not a portfolio requirement and is revisited only if a recorded decision funds a public deployment.

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

Use allowlisted structured logs and optional OpenTelemetry exporter; one request ID spans stages. Candidate IDs may be added only under an explicit safe field schema, never wholesale provider responses. Implement spec quotas across API processes, authenticated user keys and trusted-proxy-aware anonymous keys; per the scope note, the anonymous/public-traffic path is built and tested but its operational enforcement is deferred with the public deployment. Reserve generation concurrency and conservative token-cost budget before provider calls; reconcile actual usage afterward, retain unresolved reserves until expiry/reconciliation to avoid overspend on timeout. Missing cost configuration disables priced calls with a clear typed error. Daily cap is runtime operator configuration, not a guessed monetary amount. Retention jobs enforce seven-day diagnostic/30-day operational/180-day event defaults. Dashboard exposes aggregate latency/errors/costs, not private query text.

- [ ] Verify the additional acceptance cases: two processes cannot bypass quotas; idempotent retry doesn't reserve twice; lease expires after crashed request; spend cap race rejected; partial provider usage reconciled; missing price config blocks paid call; forwarded headers spoofing fails; private text never exported in default telemetry; retention respects deletion state.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_trace_redaction.py backend/tests/integration/test_limits.py -q` again, then run concurrency/rate-limit integration tests and induce model, vector and worker failures while checking trace correlation. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m5-operations.md` with quota concurrency cases, failure traces and cost reservation behavior; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: enforce operational quotas and trace request performance`.

## P5.3: Deploy a staging release and prove recovery and account deletion

**Status: Deferred.** Not required for the portfolio deliverable; revisit only if the project moves toward a funded or public deployment. Nothing below is removed — the staged release, measured RPO/RTO drill and managed backup target described here assume rented, always-on infrastructure that the zero-budget constraint in §2 does not fund, so they are recorded as deferred rather than deleted.

Two parts of this task are **not** deferred, because they run against the local self-hosted stack and remain user-facing guarantees: the account-deletion lifecycle (`delete_account`, tombstoning, physical cleanup to the 24-hour SLA) and a local backup/restore into a clean local namespace with count verification. Do those; skip the staging host, the deployment adapter and the measured recovery-target drill until a deployment exists.

**Depends on:** P5.2

**Files:**

- Create `infra/backup.py`, `infra/restore.py` (required, local). Create `infra/deploy.py`, `infra/release-manifest.schema.json` (deferred with the deployment).
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

Deferred until a funded deployment exists: at execution time choose an actual container host/region based on M2 capacity, GPU need and operator budget, then record its deployment adapter/configuration. Do not invent deployed IDs in this plan. The deployed frontend/API/worker use exact tested artifacts; configure TLS, origins, non-owner DB role, private upload storage, server-only vector credentials and health checks. Deploy to staging first, migrate using expand/contract, run known-answer search and authenticated user journey.

Still required on the local stack: back up DB and upload manifest under `${DATA_DIR}/backups/`; retain corpus/vector artifacts separately with version checks. Restore into a clean local namespace and check canonical counts, saved papers, chats, private uploads and search canaries. Account deletion tombstones immediately, revokes usable access in the app, and removes owned rows, vectors, blobs, caches and diagnostic captures; retry physical deletion to 24-hour SLA. Document backup retention and handling of deletions after restoration.

- [ ] Verify the additional acceptance cases. Required on the local stack: restored system serves known-answer query; private uploads restored only to private storage; other users survive account deletion; failed vector cleanup retries; deleted account cannot resurrect through cached token; tombstones reapplied after restore. Deferred with the deployment: broken readiness blocks deployment; incompatible DB migration blocks release; rollback switches app and matching corpus pointer.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_account_deletion.py -q` again, then run the local backup/restore drill and execute the deletion lifecycle through worker completion; the staging and rollback drills are deferred with the deployment. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m5-recovery.md` with actual timestamps, local restore counts and deletion completion, and record the staged-deployment and measured RPO/RTO sections as deferred rather than pending; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `ops: add staged deployment recovery and account deletion`.

## P5.4: Measure full-system latency, isolation and failure behavior

**Scope: required, with load results reframed as local benchmarks.** The tenant-isolation and adversarial-access tests are required and unchanged — they run fine against the local self-hosted stack and are strong portfolio evidence. The latency and load numbers stay measured and reported, but they are informational benchmarks on the developer's own hardware: they characterize the system and catch regressions, and they do not gate a public beta's capacity, because no public beta is in scope. Record the machine's actual CPU, RAM, free disk and GPU state alongside them so a result can be reproduced or superseded.

**Depends on:** P5.2. P5.3's deferred deployment work is not a prerequisite; the local stack is the test target.

**Files:**

- Create `infra/load_test.py`, `backend/tests/integration/test_tenant_adversarial.py`, `frontend/tests/beta-journeys.spec.ts`.
- Create `configs/load-test.yaml`, `reports/performance/README.md`.

**Interfaces:** Load CLI `python infra/load_test.py --config configs/load-test.yaml --out reports/performance/staging.json`; output includes every attempted request, status, latency, scenario, cold/warm label and hardware/environment versions. Tests use authorized local test accounts and synthetic fixture uploads against the self-hosted stack.

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

- [ ] Verify the additional acceptance cases: no cross-user access across chat/jobs/uploads/events/feed; protected storage URLs expire; slow model doesn't exhaust ASGI event loop; cancelled streams release concurrency; Qdrant failure doesn't corrupt library; worker crash recovers; all three workflow journeys pass; measured budgets are reported against the targets as informational local benchmarks, with any shortfall documented and explained rather than treated as a release blocker.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_tenant_adversarial.py -q` again, then run `npm --prefix frontend run test:e2e -- beta-journeys.spec.ts` and the staged load/failure test configuration. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/performance/staging.json` and `reports/m5-beta-readiness.md` with measured results against each budget, labeled as informational benchmarks on named local hardware rather than release gates, and with the isolation results reported as pass/fail; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `test: verify beta isolation load and dependency failures`.

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

Write a reproducible README with actual setup and evaluation commands, known limitations, verified corpus size and demo route. Setup commands are the local self-hosted ones (Docker Compose against `DATA_DIR`); deployment commands are added only once a deployment exists. Because P5.3's staged drill is deferred, `release-check` will report `recovery` as unmet during the portfolio phase: record it as deferred against that decision rather than marking it passed, and do not describe the release as production-ready while it stands. Case study includes retrieval ablations, recommendation features and temporal evaluation, failures, cost/latency trade-offs, contribution attribution and measured outcomes. Use Open WebUI only as a referenced UX study; do not claim its implementation as this project's work. Prepare a five-minute demo: topic search→related papers→save→feed change→grounded answer→evidence. Plan a first pilot with 3–5 researchers recruited manually by the user, ask them to complete the three workflows with their own topics, record consented task completion/useful papers/failure notes and a second visit after one week. No outreach is sent by this task without user instruction. One-person longitudinal results are labeled personal; pending external-user validation stays pending. Public availability starts only after release gates and the actual deployment choice are reviewed.

- [ ] Verify the additional acceptance cases: all public claims link to measured reports; no target corpus count presented as achieved; missing recovery/isolation evidence blocks ready status; no private queries in screenshots; pilot tasks aren't scripted to guarantee relevance; recommendation feedback limitations explicit; case study explains at least one failed experiment and its decision.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_release_gate.py -q` again, then run `uv run --project backend python -m copilot.cli release-check --evidence reports/release-evidence.json` and rehearse the demo against the tested release. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `docs/portfolio-case-study.md`, `reports/release-evidence.json` and the dated beta pilot protocol/results as observations become available; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `docs: publish reproducible ml engineering case study and beta protocol`.

## Exit checkpoint

Portfolio exit (required): deterministic CI (P5.1); tracing, per-request LLM cost logging and the daily spend cap (P5.2); local tenant-isolation and adversarial-access tests, dependency-failure behavior and the three workflow journeys, with latency and load reported as local benchmarks (P5.4); account deletion and a local backup/restore with verified counts (the non-deferred part of P5.3); and the honest case study (P5.5). The portfolio MVP is M1–M4 plus this subset.

Deferred (recorded, not deleted): staged deployment onto rented infrastructure, the measured RPO/RTO recovery drill, managed backup targets, and rate limiting operated as public multi-tenant protection. G5 as written in spec §11 still names staging journeys, rollback and restore among its evidence; that gate is unchanged by this revision and is reachable only after a future recorded decision funds a deployment. Until then, report G5 as partially satisfied with the deferred items named — do not mark it passed on local evidence alone.

A public beta remains out of scope. If one later starts, it can begin with single-person evaluation honestly disclosed; do not call it validated for researchers until pilot evidence exists. Next: optional Plan 6.
