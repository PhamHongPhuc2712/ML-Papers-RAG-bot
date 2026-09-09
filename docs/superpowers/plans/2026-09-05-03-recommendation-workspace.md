# Research Workspace and Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver all three discovery workflows and a measurable content-based personalized feed.

**Architecture:** Verified identity protects library and feedback. A versioned profile/candidate/ranking pipeline powers a thin Next.js workspace and records real exposure for evaluation.

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

Entry: search API and G2 report are available. P3.1 may start once P2.4 passes; final milestone review includes P2.5. Read spec §§8, 10–12. Recommendation evaluation is initially single-person evidence.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P3.1: Add verified identity, private library and preference storage

**Depends on:** P2.4

**Files:**

- Create `backend/src/copilot/identity/auth.py`, `backend/src/copilot/library/{service,api}.py`.
- Create `backend/migrations/versions/0002_users_library.py`; modify app/contracts/db models.
- Create `backend/tests/integration/test_library_auth.py`; extend `backend/tests/conftest.py` with signed-token fixtures.

**Interfaces:** `verify_token(token: str) -> Principal`; `save_paper(principal: Principal, paper_id: UUID) -> None`; `list_library(principal: Principal) -> list[dict]`. HTTP library and /me/interests endpoints follow spec §10. `user_headers(user_id: str)` signs a short-lived JWT with a test-only key accepted by the injected test issuer; no production backdoor.

- [ ] Write the regression test below in `backend/tests/integration/test_library_auth.py` and add the additional acceptance cases listed for this task.

```python
def test_library_isolation(api_client, user_headers, sample_paper_id):
    alice = user_headers("11111111-1111-4111-8111-111111111111")
    bob = user_headers("22222222-2222-4222-8222-222222222222")
    path = f"/v1/library/{sample_paper_id}"
    assert api_client.put(path, headers=alice).status_code == 204
    assert len(api_client.get("/v1/library", headers=alice).json()["items"]) == 1
    assert api_client.get("/v1/library", headers=bob).json()["items"] == []
    assert api_client.put(path).status_code == 401
```

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_library_auth.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```sql
ALTER TABLE saved_papers ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_papers FORCE ROW LEVEL SECURITY;
CREATE POLICY saved_papers_owner ON saved_papers
USING (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid);
```

Create users/interests/preference_events/seed_papers/saved_papers tables and policies on all private tables. Append topic/seed add/remove events transactionally when current preferences change; preserve the history needed by P3.5. Verify JWT using configured algorithm allowlist, issuer/audience, expiry and JWKS; reject alg=none and unknown issuer. Resolve Principal once per request. Set transaction-local `app.user_id` with a parameterized set_config query on the request's checked-out DB connection. The runtime role is neither owner nor BYPASSRLS; migrations use a separate role. Library PUT is idempotent and DELETE tolerates absence. `/me/interests` atomically replaces a validated list of topics/seed IDs (at most 20 of each); return explicit membership errors for unknown seeds. Extend tests with `sample_paper_id` fixture that creates one public paper using the P1 importer.

- [ ] Verify the additional acceptance cases: forged/expired/wrong audience token; JWKS rotation; cross-user reads/writes; RLS on raw SQL with actual runtime role; pooled connection identity reset; save twice yields one row; rejected supplied user_id; seed removal preserved correctly; missing auth doesn't reach private DB query.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_library_auth.py -q` again, then run identity/library integration tests against the non-owner role and inspect generated OpenAPI contracts. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m3-auth.md` with role/policy assertions and token rotation results; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add authenticated library and research preferences`.

## P3.2: Record impressions and idempotent recommendation feedback

**Depends on:** P3.1

**Files:**

- Create `backend/src/copilot/recommendations/{events,api}.py`, `backend/migrations/versions/0003_recommendations.py`.
- Create `backend/tests/unit/test_event_weights.py`, `backend/tests/integration/test_events.py`; modify contracts/db models.

**Interfaces:** `event_weight(event_type: str) -> float`; `record_event(principal: Principal, event: FeedbackEvent) -> Literal['accepted','duplicate']`; `record_impressions(principal, request_id, surface, papers, model_version, release_id) -> list[UUID]`; visibility endpoint validates owned impression ID. FeedResponse.papers maps IDs to PaperSummary and FeedResponse.impressions maps each paper ID to an immutable impression ID for this response.

- [ ] Write the regression test below in `backend/tests/unit/test_event_weights.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.recommendations.events import event_weight

def test_feedback_meanings_are_distinct():
    assert event_weight("save") == 3.0
    assert event_weight("open") == 0.25
    assert event_weight("unsave") == 0.0
    with pytest.raises(ValueError, match="unsupported_event"):
        event_weight("hover")
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_event_weights.py backend/tests/integration/test_events.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
WEIGHTS = {"save": 3.0, "like": 4.0, "read": 2.0, "ask": 1.0,
           "open": 0.25, "unsave": 0.0, "dislike": 0.0}

def event_weight(event_type: str) -> float:
    if event_type not in WEIGHTS:
        raise ValueError("unsupported_event")
    return WEIGHTS[event_type]
```

Persist served and visible impressions separately, with surface, rank, model/profile/release versions and timestamps. Unique(user_id,event_id) deduplicates transport retries. Validate paper existence, impression ownership and matching paper; library-origin feedback can omit impression. Save/unsave updates library state and records the event in one transaction through a shared service, avoiding double recording from UI calls. Store dislike as a stateful exclusion/negative signal, not a positive weight. Coalesce repeated opens/asks by paper/day when building features, without losing raw event provenance. Reduce save/like/read into one stateful contribution per paper, rather than counting repeated messages as extra preference. Queue profile refresh idempotently with a ten-second debounce after meaningful changes.

- [ ] Verify the additional acceptance cases: retry same event returns duplicate; another user cannot reference an impression; impression-paper mismatch rejected; future time >5 minutes rejected; received_at stored; visible marking idempotent; no impression conversion invented for library-only save; save/event transaction rolls back together; events survive worker interruption.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_event_weights.py backend/tests/integration/test_events.py -q` again, then run event/library integration tests and inspect a complete served→visible→save sequence. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m3-feedback.md` with event schema, deduplication traces and exposure denominator definition; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: record recommendation exposure and idempotent feedback`.

## P3.3: Build cold-start and personalized ranking with faithful reasons

**Depends on:** P3.2, P2.3

**Files:**

- Create `backend/src/copilot/recommendations/{profile,rank,service}.py`, `configs/recommendations.yaml`.
- Create `backend/tests/unit/test_profile.py`, `backend/tests/unit/test_recommendation_rank.py`, `backend/tests/integration/test_feed.py`.

**Interfaces:** `weighted_profile(vectors: list[list[float]], weights: list[float]) -> list[float]|None`; `score_components(features: dict[str,float], has_seed: bool, has_negative: bool) -> float`; `diversify(candidates: list[dict], limit: int, lambda_: float=0.8) -> list[dict]`; `RecommendationService.feed(principal: Principal, request: RecommendationRequest) -> FeedResponse`.

- [ ] Write the regression test below in `backend/tests/unit/test_profile.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.recommendations.profile import weighted_profile

def test_profile_weights_and_no_signal_fallback():
    result = weighted_profile([[1.0, 0.0], [0.0, 1.0]], [3.0, 1.0])
    assert result == pytest.approx([3 / (10 ** 0.5), 1 / (10 ** 0.5)])
    assert weighted_profile([], []) is None
    assert weighted_profile([[0.0, 0.0]], [1.0]) is None
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_profile.py backend/tests/unit/test_recommendation_rank.py backend/tests/integration/test_feed.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
import math

def weighted_profile(vectors, weights):
    if len(vectors) != len(weights):
        raise ValueError("profile_alignment")
    if not vectors:
        return None
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("profile_dimension")
    if any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("profile_weight")
    mean = [sum(vector[i] * weight for vector, weight in zip(vectors, weights))
            for i in range(dim)]
    norm = math.sqrt(sum(value * value for value in mean))
    if not math.isfinite(norm):
        raise ValueError("profile_nonfinite")
    return [value / norm for value in mean] if norm >= 1e-8 else None
```

Implement every component formula in spec §8 with table-driven tests: decay, normalized cosine, available-seed weight redistribution, year-local popularity, negative affinity and MMR. Candidate union caps are 200/100/100/50; seed branch cap is combined, not per seed. Exclude saved/read/disliked IDs both before ranking and when reading a cached feed. Persist profile hash based on effective preferences/events and embedding revision. Reasons contain real contributing topic/seed ID/component values; UI text renders these facts. Cold start requires no LLM or fabricated profile. Feed recomputation uses P1 jobs, versioned run/item tables and a 24-hour expiry; stale previous feed remains available with flag. Model/index errors use recent eligible fallback and record degradation.

- [ ] Verify the additional acceptance cases: empty/zero profile; mismatched dimensions; repeated events don't inflate ranking; decayed interests; unsave isn't dislike; absent citation count neutral; negative affinity excludes explicit dislikes; MMR diversifies near-duplicate papers deterministically; no future papers; profile/release cache invalidation; fresh exclusions applied to stale feed; faithful reasons point to actual seeds.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_profile.py backend/tests/unit/test_recommendation_rank.py backend/tests/integration/test_feed.py -q` again, then run recommendation unit/integration suites and one end-to-end preference→feedback→refresh sequence. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m3-recommender.md` with score breakdowns, fallback cases and versioned ranking configuration; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add personalized content ranking and cold-start feed`.

## P3.4: Build the research workspace and collect real visibility events

**Depends on:** P3.3, P2.4

**Files:**

- Create `frontend/{package.json,package-lock.json,tsconfig.json,next.config.ts,playwright.config.ts}`.
- Create `frontend/src/app/{layout,page}.tsx`, search/feed/library/settings pages and `papers/[id]/page.tsx`.
- Create `frontend/src/features/{search,library,feed}/`, `frontend/src/lib/{api,auth}.ts`, `frontend/tests/research-workspace.spec.ts`.

**Interfaces:** Generate TypeScript API types from exported OpenAPI; `searchPapers(request: SearchRequest): Promise<SearchResponse>`, `getFeed(request: RecommendationRequest): Promise<FeedResponse>` in api.ts. `PaperCard` consumes PaperSummary, optional reason/impression_id and explicit callbacks. Visibility observer emits once per impression after ≥50% visibility for one continuous second.

- [ ] Write the regression test below in `frontend/tests/research-workspace.spec.ts` and add the additional acceptance cases listed for this task.

```typescript
import { test, expect } from "@playwright/test";

test("search, save and revisit the library", async ({ page }) => {
  await page.goto("/search");
  await page.getByRole("textbox", { name: "Search papers" }).fill("contrastive learning");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const first = page.getByRole("article").first();
  const title = await first.getByRole("heading").innerText();
  await first.getByRole("button", { name: "Save paper", exact: true }).click();
  await page.getByRole("link", { name: "Library", exact: true }).click();
  await expect(page.getByRole("heading", { name: title, exact: true })).toBeVisible();
});
```

- [ ] Run `npm --prefix frontend run test:e2e -- research-workspace.spec.ts`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```typescript
export async function apiJson<T>(path: string, token: string | null,
                                 init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", "Bearer " + token);
  const response = await fetch(process.env.NEXT_PUBLIC_API_URL + path,
                               { ...init, headers });
  if (!response.ok) throw await response.json();
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
```

Initialize Next.js App Router with strict TypeScript, accessible labeled components and responsive layout; lock dependencies. The browser obtains access tokens from the project's own self-hosted identity endpoints (spec §10); signing keys and server-side secrets never enter NEXT_PUBLIC variables. Playwright setup signs in the seeded test user through a test auth server/configuration and seeds a deterministic public corpus; production builds reject that server configuration. Add `test:e2e`, `typecheck`, `lint`, `build` scripts. Expose topic/seed onboarding, explicit search filters, related-work page, personalized feed, save/read/dislike and recommendation reasons. Cross-link the three workflows. Display actual coverage and as-of date. Keep card keys stable across refresh so exposure is not counted twice. Unit-test observer with a fake clock, and verify a real browser scroll/visibility event. Include empty/loading/error/degraded/stale/session-expiry UI states.

- [ ] Verify the additional acceptance cases: search→paper→related→save; topics/seeds→feed; dismiss doesn't return after refresh; library persists across sign-in; keyboard navigation and accessible status; invisible cards don't count; quick scroll doesn't count; one second visible does count once; expired session preserves user input; mobile layout usable.

- [ ] Run `npm --prefix frontend run test:e2e -- research-workspace.spec.ts` again, then run `npm --prefix frontend run typecheck`, `npm --prefix frontend run lint`, `npm --prefix frontend run build`, then all three workflow journeys. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m3-user-journeys.md` with local browser captures, accessibility findings and recorded event rows; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: build searchable research workspace and recommendation feed`.

## P3.5: Evaluate recommendations without temporal or exposure leakage

**Depends on:** P3.4

**Files:**

- Create `backend/src/copilot/evaluation/{recommendations,temporal}.py`, `configs/experiments/recommendations.yaml`.
- Create `backend/tests/unit/test_temporal_eval.py`, `data/fixtures/recommendations/profiles.jsonl`.
- Create `docs/personal-study.md`, `reports/recommendations/README.md`; extend evaluation CLI.

**Interfaces:** `available_before(rows: list[dict], cutoff: datetime) -> list[dict]`; `run_recommendations(config: Path, cutoff: datetime|None, out: Path) -> dict`. All feature rows carry available_at; interaction received_at, edge observed_at and paper first_seen_at determine it.

- [ ] Write the regression test below in `backend/tests/unit/test_temporal_eval.py` and add the additional acceptance cases listed for this task.

```python
from datetime import datetime, timezone
from copilot.evaluation.temporal import available_before

def test_future_features_are_excluded():
    cutoff = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = [
        {"id": "old", "available_at": datetime(2026, 8, 31, tzinfo=timezone.utc)},
        {"id": "future", "available_at": datetime(2026, 9, 2, tzinfo=timezone.utc)},
    ]
    assert [row["id"] for row in available_before(rows, cutoff)] == ["old"]
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_temporal_eval.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def available_before(rows, cutoff):
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_must_be_timezone_aware")
    return [row for row in rows if row["available_at"] < cutoff]
```

Prepare 20 developer-authored profiles and pooled, blinded judgments, keeping related profiles in one split. Evaluate popularity, embedding-only, heuristic personalized and MMR variants with the same candidates/labels. Include novelty as separate judgment, not a synonym for relevance. Temporal loader reconstructs state at cutoff (save/unsave, likes, seeds and topics) and rejects future feature snapshots; current profile cannot be replayed backward. Log candidate judged coverage and avoid treating unexposed items as negative. Start the four-week personal study and record useful new papers/session, visible-impression save/like rates and failure notes. Implementation can pass deterministic tests before four weeks elapse, but the longitudinal evidence gate stays pending; do not fabricate observations or keep a foreground agent waiting for weeks.

- [ ] Verify the additional acceptance cases: future interactions/citation counts/edges excluded; repeated work versions don't leak as new positives; family splits disjoint; zero visible impressions yields undefined rate not zero success; synthetic profiles labeled as such; benchmark rejects current-only citation snapshots for historical scoring; test split remains locked.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_temporal_eval.py -q` again, then run `uv run --project backend python -m copilot.cli eval recommendations --config configs/experiments/recommendations.yaml --out reports/recommendations/pilot` and verify report manifests. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/recommendations/pilot/report.md`, judgment revision and dated personal-study entries; keep the four-week checkpoint open until observed; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: evaluate recommendation quality and temporal correctness`.

## Exit checkpoint

G3 technical exit: auth isolation, three browser journeys, cold-start/feedback refresh and offline recommendation comparison pass. Personal longitudinal study is a separate dated evidence gate; it can overlap M4/M5 and cannot be marked observed early. Next: Plan 4.

