# ML Research Copilot — Progress Tracker

Last updated: 2026-09-06.

## Current state

- Planning documents: prepared for review.
- Implementation: **1 / 30 tasks complete**.
- MVP implementation: **1 / 25 tasks complete**.
- Extended experiments: **0 / 5 tasks complete**.
- Passing implementation gates: **0 / 6**.
- Corpus indexed: none yet.
- Benchmarks observed: none yet.
- Personal or external-user study observations: none yet.

Status vocabulary: **Not started**, **In progress**, **In review**, **Done**, **Blocked**. A blocked entry records the concrete missing dependency; it is not counted as complete. No placeholders in reports should be mistaken for collected evidence.

## Task board

| Task | Deliverable | Status | Depends on | Evidence / commit |
|---|---|---|---|---|
| P1.1 | Create a runnable API and isolated persistence test harness | Done | Design review | `e7859f0`; [foundation evidence](../../reports/m1-foundation.md); 12 native-service tests, Ruff, mypy and migration replay passed; Docker build unverified |
| P1.2 | Normalize publication identity and preserve version provenance | Done | P1.1 | [identity evidence](../../reports/m1-identity.md) |
| P1.3 | Parse documents into traceable sections and chunks | Not started | P1.2 | Not produced |
| P1.4 | Ingest an accepted-paper pilot through resumable jobs | Not started | P1.2, P1.3 | Not produced |
| P1.5 | Export immutable corpus snapshots and publish coverage | Not started | P1.4 | Not produced |
| P2.1 | Establish labeled retrieval data, metrics and a true BM25 baseline | Not started | P1.5 | Not produced |
| P2.2 | Index paper and chunk vectors with atomic release switching | Not started | P2.1 | Not produced |
| P2.3 | Implement reproducible fusion and bounded cross-encoder reranking | Not started | P2.2 | Not produced |
| P2.4 | Expose search, metadata, related papers and stable pagination | Not started | P2.3 | Not produced |
| P2.5 | Run ablations and choose the first search configuration | Not started | P2.4 | Not produced |
| P3.1 | Add verified identity, private library and preference storage | Not started | P2.4 | Not produced |
| P3.2 | Record impressions and idempotent recommendation feedback | Not started | P3.1 | Not produced |
| P3.3 | Build cold-start and personalized ranking with faithful reasons | Not started | P3.2, P2.3 | Not produced |
| P3.4 | Build the research workspace and collect real visibility events | Not started | P3.3, P2.4 | Not produced |
| P3.5 | Evaluate recommendations without temporal or exposure leakage | Not started | P3.4 | Not produced |
| P4.1 | Retrieve bounded evidence and validate citation provenance | Not started | P2.3, P3.1 | Not produced |
| P4.2 | Add controlled routing, generation and interruption-safe streams | Not started | P4.1, P3.1 | Not produced |
| P4.3 | Process private PDF uploads with ownership and deletion guarantees | Not started | P4.2, P1.3–P1.4 | Not produced |
| P4.4 | Expose chat, evidence and multimodal document interactions | Not started | P4.3, P3.4 | Not produced |
| P4.5 | Calibrate grounded-answer evaluation and regression gates | Not started | P4.4, P2.5 | Not produced |
| P5.1 | Enforce deterministic CI and reproducible builds | Not started | P1–P4 technical tasks | Not produced |
| P5.2 | Add operational tracing, quotas and spend controls | Not started | P5.1 | Not produced |
| P5.3 | Deploy a staging release and prove recovery and account deletion | Not started | P5.2 | Not produced |
| P5.4 | Measure full-system latency, isolation and failure behavior | Not started | P5.3 | Not produced |
| P5.5 | Prepare an honest launch and ML engineering case study | Not started | P5.4; P3.5 longitudinal observations may remain pending | Not produced |
| P6.1 | Expand coverage through independently verified venue adapters | Not started | G5; capacity measurements from P5.4 | Not produced |
| P6.2 | Add evidence-based comparison and a bounded research graph | Not started | P4.5, P5.2 | Not produced |
| P6.3 | Evaluate citation-graph recommendation candidates | Not started | P3.5, P6.1 | Not produced |
| P6.4 | Train and compare a learned ranker only after label readiness | Not started | P2.5, P3.5; data readiness audit | Not produced |
| P6.5 | Present research landscapes and experiment evidence | Not started | P6.2; P6.3/P6.4 reports as available | Not produced |

## Gate board

| Gate | Required evidence | State |
|---|---|---|
| G1 | 100-paper replay, identity checks, 20-paper parser audit, snapshot restore | Not started |
| G2 | Four retrieval baselines, stable serving release, validation decision | Not started |
| G3 | Auth and three workflows, cold start, offline recommendation report | Not started |
| G4 | Cited chat, human claim-support audit, private upload lifecycle | Not started |
| G5 | CI, staging, quotas, isolation, load, backup/restore, launch evidence | Not started |
| G6 | Actual graph/learned-ranking/synthesis experiments and decisions | Not started |

## Observation checkpoints

| Checkpoint | Start condition | Completion evidence | State |
|---|---|---|---|
| Personal recommendation study | M3 instrumentation works | Four dated weeks of personal feedback and temporal evaluation | Not started |
| External-researcher pilot | Beta ready and user chooses recruits | 3–5 researcher sessions and one-week follow-up observations | Not started |
| Learned ranker data readiness | Retrieval labels accumulated | Audit of distinct families, human labels and uncontaminated test split | Not started |
| Capacity expansion | 1k and 10k indexes measured | Storage/throughput budget supports next corpus release | Not started |

## Decisions awaiting review

| Decision | Proposed default | When it matters |
|---|---|---|
| Release order | Corpus→retrieval→recommendations→evidence→beta | Before implementation |
| Initial corpus | Accepted ICLR 2024 pilot, then ICLR/ICML/NeurIPS 2022–2026 as available | Before live ingestion |
| Initial sources of personal interests | Actual topics/seed papers supplied during M1/M2 labeling | Before meaningful recommendation judgments |
| Queue | Durable PostgreSQL leases | M1 |
| Upload retention | 30 days; user deletion supported; physical cleanup within 24 hours | M4 |
| Hardware and spend | Measure locally first; no assumed GPU or budget | Before model/provisioning decisions |
| Deployment provider/region | Choose after capacity and cost measurements | Before M5 staging |
| External recruiting | User-led first cohort of 3–5 researchers | After G5 |

## Session log

Record one dated row after each execution session.

| Date | Tasks changed | Checks and results | Evidence paths / commit | Next action |
|---|---|---|---|---|
| 2026-09-05 | Planning only | See planning-verification.md; no application tests run | Specification and six plans | Review proposed design and sequence |
| 2026-09-05 | P1.1/P1.2 execution setup | Python 3.12.5 venv isolation verified; both GPT-5.6 Luna/max workers stopped with workspace out-of-credits error | codex/p1-foundation; backend/.venv; portable uv in ignored runtime directory | Resume Luna workers after credits are available; no application tests have run |
| 2026-09-05 | P1.1 completed | 9 focused integration and 12 full tests passed against PostgreSQL 17.11/Qdrant 1.19.1; 5 CI unit tests passed without service variables; Ruff, mypy, migration down/up and diff check passed | `e7859f0`; [foundation evidence](../../reports/m1-foundation.md) | Review P1.1 branch or begin P1.2 |
| 2026-09-06 | P1.2 completed | 26 focused identity tests and 38 full backend tests passed against PostgreSQL 17.11/Qdrant 1.19.1; 13 offline CI tests, Ruff, mypy, migration upgrade/downgrade/upgrade and fixture replay passed | [identity evidence](../../reports/m1-identity.md); task commit recorded in the report | Review P1.2 branch or begin P1.3 |
| 2026-09-06 | P1.2 review fixes | 34 focused identity tests and 46 full backend tests passed against PostgreSQL 17.11/Qdrant 1.19.1; 13 offline CI tests, Ruff, mypy, migration replay and fixture replay passed with explicit `TEST_*` services | [identity evidence](../../reports/m1-identity.md); [task-2 report](../../.superpowers/sdd/2026-09-05-01-corpus-foundation/task-2-report.md); task commit recorded in the report | Review P1.2 fix commit or begin P1.3 |
| 2026-09-06 | P1.2 review fixes round 2 | 37 focused identity tests and 49 full backend tests passed against PostgreSQL 17.11/Qdrant 1.19.1; 13 offline CI tests, Ruff, mypy, migration duplicate reconciliation/down-up/no-op replay, and twice-replayed fixtures passed with explicit `TEST_*` services | [identity evidence](../../reports/m1-identity.md); [task-2 report](../../.superpowers/sdd/2026-09-05-01-corpus-foundation/task-2-report.md); commit recorded in the task report | Review P1.2 fix commit or begin P1.3 |

## Definition of a completed task

All named acceptance cases have evidence; required tests actually ran; code was reviewed; report contains actual versions/measurements; task commit is recorded. If a human-observation checkpoint takes weeks, record engineering completion separately and leave the observation checkpoint pending. Do not publish synthetic or planned results as measured outcomes.
