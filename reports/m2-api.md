# M2 API evidence — search, paper metadata, related papers and stable pagination

Date: 2026-09-30
Task: P2.4, expose search, metadata, related papers and stable pagination
Status: **done** — every plan acceptance case passes against the test services, and the
real API served the live corpus over HTTP, with the caveats on timing below.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB. `postgres:17.11-bookworm`,
`qdrant/qdrant:v1.19.1`; FastAPI 0.116.1, Starlette 0.47.3, Pydantic 2.13.5, uvicorn
0.35.0, SQLAlchemy 2.0.43, torch 2.14.0+cu130. Release `m2-20260924T095724Z`.
Code: branch `phuc` from `fb36459`, plus this task's commit.

## What exists

| Piece | Where |
|---|---|
| `POST /v1/search`, `GET /v1/papers/{id}`, `GET /v1/papers/{id}/related`, the error shape, the namespace guard, startup warm-up | `search/api.py` |
| `encode_cursor` / `decode_cursor`, the cache and cursor keys, `OrderingCache` in PostgreSQL | `search/cache.py` |
| `search_orderings` table: hashes, release, the ordering as JSONB, expiry | `migrations/versions/0004_search_orderings.py`, `db/models.py` |
| The service split into `rank` (a whole ordering) and `page` (one page with metadata), plus `identity`, `warm`, `close` | `search/service.py` |
| Router, error handlers and lifespan wiring; `openapi_schema()` | `app.py` |
| `api export-schema --out frontend/openapi.json` | `cli.py` |
| The published contract, sorted for review | `frontend/openapi.json` |

## The contract

All three endpoints are public and read-only. Every success carries `request_id`. Every
failure has one shape, `{"error": {"code", "message", "retryable"}, "request_id"}`, and
a validation message says where and why a request failed without echoing the submitted
value (tested with a 2,400-character query).

| Endpoint | Success | Errors |
|---|---|---|
| `POST /v1/search` — `SearchRequest` | `SearchResponse`: ranked items, a `papers` metadata map, `next_cursor`, `degraded`, `warnings` | 422 `validation_error` / `cursor_invalid` / `cursor_mismatch`; 410 `cursor_expired`; 503 `corpus_not_ready`, `candidates_unavailable`, `release_outside_namespace`, `database_unavailable` |
| `GET /v1/papers/{id}` | `PaperDetail`: canonical id, `redirected_from`, `PaperSummary`, acceptance type, identifiers, source links with `available`, parse status | 404 `paper_not_found`; 422 malformed id; 503 |
| `GET /v1/papers/{id}/related` — `limit`, `year_from`, `year_to`, `venues`, `fulltext_only` | `SearchResponse` of dense neighbours, seed excluded | 404; 422; 503 `vector_unavailable`, `release_outside_namespace` |

Validation follows spec §10 and runs before anything is served: query 1–2,000
characters and not blank, limit 1–50, `year_from ≤ year_to` (`invalid_year_interval`),
and one of the four modes. The contract previously allowed 100 results and any query
length; both are now the spec's bounds.

### Pagination

A search's ordering is kept for ten minutes (`configs/search.yaml`,
`cache.ttl_seconds: 600`): its top 200 for `bm25`, `dense` and `hybrid`, and only the
reranked head of at most 50 for `hybrid_rerank`, so reranker logits and RRF scores never
share a list. A cursor is `<payload>.<signature>`, both URL-safe base64. The payload is
compact JSON naming the ordering's id, an offset and the expiry, and the signature is
HMAC-SHA256 under a key derived from `SECRET_KEY` for cursors alone. The signature is
checked first and in constant time (`hmac.compare_digest`), so nothing in an unsigned
payload is trusted, not even a claim that it expired.

- **Pages come from one stored ordering.** Ranks are positions in that ordering (page
  two starts at 11), boundaries cannot move, and nothing repeats.
- **A cursor is bound to its question.** The ordering row keeps a hash of query, mode
  and filters, and a cursor sent with a different question is `cursor_mismatch`.
- **A release switch does not move open pages.** A cursor keeps paging the ordering it
  was issued for, and the response names that ordering's release. A new search reads
  the new release.
- **A repeated search reuses the ordering**, keyed by the hash of the question, the
  release, the embedder revision and the ranker identity: branches, candidate count,
  depth, RRF constant, reranker revision and pair budget. **A degraded ordering is never
  reused**: its own cursors page it, but the next identical search ranks again, because
  the reranker that timed out may be back.
- **No query text is stored**, only its hashes. A test searches for a distinctive
  phrase and scans every stored row for it.
- **Expiry is typed.** 410 `cursor_expired` tells the client to search again; a cursor
  that fails its signature, or is malformed, is 422 `cursor_invalid`. Expired rows are
  deleted as new orderings are written, and a dropped release takes its orderings with it
  (`ON DELETE CASCADE`).

A response's `request_id` identifies that HTTP call. A fresh ranking stores its ordering
under the same id, so the first page, its trace and its cache row share one id; later
pages and cache hits get their own.

### Paper metadata and merges

`GET /v1/papers/{id}` follows `merged_into` to the surviving paper, with a recursive
query bounded at 32 steps; merges refuse cycles, so a longer chain is reported as
corrupt rather than followed. The survivor's metadata comes back with the requested id in
`redirected_from`. Source links are the paper's `pdf_url` and the landing page of its
best version (parsed first, then newest). `source.available` is false when neither
exists — the plan's "source URL unavailable" state is said, not implied. On the live
corpus every paper has both links; 3,759 of them are the zero-byte PDFs of NeurIPS 2025
and ICLR 2026, which report `parse_status: "not_pdf"` and `fulltext_indexed: false`.

### Related papers

The seed's stored dense vector is read from the active release's paper collection and
queried with the caller's filters, the seed excluded by a Qdrant `must_not` on its id
and again in application code. Items carry their cosine as `scores.dense`. A seed with
no point in the release — a paper added after the build — returns an empty page with
`warnings: ["seed_not_indexed"]` rather than a guess. Related papers need no model at
request time, so they keep working when the embedder is down.

### Only this deployment's public collections

Before either search endpoint reads Qdrant, the active release's two collection names
must equal `collection_names(QDRANT_COLLECTION_PREFIX, release_id)` exactly. A test-mode
app, whose prefix is `test_`, therefore cannot serve a `dev_` release. A release naming
a private upload collection (`user_documents_*`) is refused the same way. Both cases are
tested and return 503 `release_outside_namespace`, not retryable, and the refusal is
logged.

### Startup

The lifespan builds the service once: fixture models in mock mode; otherwise the pinned
BGE-M3 and `bge-reranker-v2-m3`. A model that fails to load degrades search instead of
stopping the API — no embedder gives `dense_unavailable`, no reranker gives
`rerank_unavailable` (spec §12). It then **warms** the service against the active
release: each branch and the reranker run once, outside every deadline. The real-corpus
smoke below is why.

## Real-corpus smoke over HTTP

`copilot_v2` was migrated to `0004_search_orderings` (additive: one table), then
`uvicorn --factory copilot.app:create_default_app` was run against it. The queries were
written for the smoke; no LitSearch text was used. `/health/ready` returned 200 with the
release id.

**The GPU was shared throughout.** Another project on this host
(`esci-multimodel-ltr`) ran GPU smoke and capacity scripts during these runs, visible in
`nvidia-smi` beside the API process. That process is not this project's and was left
running. Timings that touch the GPU are **contended** and are not latency evidence; P2.3's
uncontended pilot remains the ranking-time measurement (request p95 631 ms).

| Run | Condition | First page (8 new queries) | Page two | Same search again |
|---|---|---|---|---|
| A: `hybrid_rerank`, before warm-up | GPU memory 3.4–9.9 GB, peak utilization 98% | **first request `dense_timeout`, 1,900 ms**; the other 7 undegraded, 644–975 ms | p50 11.6 ms, max 13.2 | p50 20.5 ms; one 531 ms (below) |
| B: `hybrid_rerank`, after warm-up | GPU utilization median and peak 100% | no `dense_timeout`; **4 of 8 `rerank_timeout`**, 1,456–1,775 ms | p50 10.6 ms, max 14.2 | the 4 undegraded 18–22 ms; the 4 degraded ranked again, 1.57–1.83 s |
| C: `bm25`, the same 8 queries | no GPU work | p50 41.2 ms, max 56.6 | p50 15.8 ms, max 19.1 | p50 23.3 ms |

What the runs establish:

- **Pages are right on the real corpus.** Every page two held ranks 11–20, shared no
  paper with page one, named the release, and offered a further cursor.
- **Warm-up fixed a real defect.** Unwarmed, the first request after a restart spent its
  1 s dense budget on first-call CUDA work and was served BM25-only with `dense_timeout`.
  Warmed, no request in run B lost its dense branch, even on a saturated GPU. A
  regression test now fails if startup stops warming each stage.
- **Degraded orderings are not reused, as designed.** In run A the repeat of the first,
  degraded query took 531 ms — it was ranked again, not served from the cache. In run
  B the four degraded orderings were ranked again on repeat and the four healthy ones came
  from the cache in about 20 ms.
- **What the API adds** is small, measured where the GPU plays no part (run C): a first
  page, including BM25, the ordering's write and metadata, at p50 41 ms. Serving a stored
  page costs 10–16 ms.
- The cache is in PostgreSQL, so it survives an API restart. A second smoke of run A's
  queries after a restart was all cache hits, p50 22 ms, so it measured nothing about
  warm-up and is excluded here; run B used eight new queries.
- Paper detail took 30–82 ms, related papers 76–392 ms; the higher figures are each
  process's first call.

Two responses, abridged, from the smoke:

```json
POST /v1/search  {"query": "contrastive learning for sentence embeddings", "mode": "hybrid_rerank", "filters": {}, "limit": 10}
{"items": [{"rank": 1, "scores": {"bm25": 21.1635, "dense": 0.7209, "rrf": 0.0318, "rerank": 6.9062}, ...},
           {"rank": 2, "scores": {"bm25": 16.3103, "dense": 0.7128, "rrf": 0.0271, "rerank": 6.7812}, ...}],
 "papers": {"…": {"title": "Contrastive Learning of Sentence Embeddings from Scratch", "venue": "EMNLP", "year": 2023, ...}},
 "next_cursor": "<148 characters>", "degraded": false, "warnings": [], "corpus_release_id": "m2-20260924T095724Z"}

GET /v1/papers/{id}   (SPARF: Neural Radiance Fields From Sparse and Noisy Poses)
{"source": {"pdf_url": "https://openaccess.thecvf.com/content/CVPR2023/papers/Truong_SPARF_…_paper.pdf",
            "landing_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Truong_SPARF_…_paper.html",
            "available": true},
 "identifiers": [{"namespace": "papercli", ...}], "parse_status": "parsed", "redirected_from": null, ...}
```

Related papers for SPARF with `year_from=2024&limit=5` returned five 2024 NeRF papers
(TrackNeRF, RS-NeRF, Pano-NeRF, ColNeRF and one on adaptive multi-exposure), cosine
0.828 down to 0.802, the seed absent.

## Tests

`backend/tests/integration/test_search_api.py` — 28 tests over HTTP against the seeded
release, which the service suite and this one now share through
`backend/tests/integration/conftest.py`. Time is a clock the tests move by hand; nothing
sleeps.

| Acceptance case | Test |
|---|---|
| The plan's regression example | `test_search_validation_and_metadata` |
| Invalid year interval, bounds, blank query, unknown mode | `test_invalid_requests_are_422_in_the_error_shape` (6 cases), `test_related_papers_honour_filters_and_limit` |
| Request ids on errors and successes; no echo of input | `test_successes_carry_a_fresh_request_id`, `test_a_validation_message_never_echoes_the_submitted_text`, every error assertion |
| Stable page boundaries, no duplicates | `test_pages_are_contiguous_stable_and_never_repeat`, `test_hybrid_rerank_pages_only_through_its_reranked_head` |
| Signed-cursor tampering | `test_a_tampered_or_foreign_cursor_is_refused` |
| Expiry | `test_an_expired_cursor_is_410_and_asks_for_a_fresh_search` (599 s serves, 600 s is 410) |
| Release change keeps the cached version until expiry | `test_a_release_switch_leaves_open_pages_on_their_release` |
| Cache key and reuse; degraded never reused; no query text stored | `test_a_repeated_search_reuses_its_cached_ordering`, `test_a_degraded_ordering_pages_but_is_never_reused`, `test_the_cache_stores_no_query_text` |
| Unknown and merged paper ids | `test_an_unknown_paper_is_404`, `test_a_merged_paper_resolves_to_its_survivor` (real `merge_papers`) |
| Source URL unavailable | `test_a_paper_without_any_source_link_says_so` |
| Related: seed excluded, filters, limit, unindexed seed | `test_related_papers_exclude_the_seed_and_rank_by_dense_score`, `test_related_papers_honour_filters_and_limit`, `test_a_seed_outside_the_release_is_reported_not_guessed` |
| Test mode cannot expose private or foreign indexes | `test_only_this_deployments_public_collections_are_served` (a `dev_` release; a `user_documents` collection) |
| OpenAPI structure, not generator ordering | `test_the_published_schema_matches_the_running_contract` |
| Startup warms each stage | `test_startup_warms_every_stage_on_the_active_release` — fails with the warm call removed, checked |

`backend/tests/unit/test_cursor.py` — 22 tests: round trip; a changed offset, signature
or secret; exclusive expiry; an unsigned payload not trusted even to say it expired; six
malformed tokens; eight correctly signed payloads of the wrong shape (missing id,
non-UUID, negative, string, boolean or float numbers, wrong version); no cursor for a
negative offset or a non-UUID; the query key ignoring venue order and page size; the
ranking key changing with release, embedder and reranker.

The regression example failed first for the stated reason: `assert 404 == 422`, since
the route did not exist.

## Deviations from the plan, and why

- **Files beyond the list.** P2.4 also changed:
  - `contracts.py` — the spec's limit of 50 and 2,000 characters, blank queries, and the
    year interval;
  - `search/service.py` — the `rank`/`page` split a cache needs, plus `identity`, `warm`
    and `close`;
  - `configs/search.yaml` — the cache depth and lifetime;
  - a migration and ORM model for the cache table ("cache server-side in PostgreSQL"
    needs one);
  - `cli.py` — the plan's own `api export-schema` command;
  - `tests/integration/conftest.py` — the seeded release, moved out of P2.3's service
    suite so both suites build the same one. P2.3's 20 tests pass unchanged on it.
- **The schema lives at `frontend/openapi.json`,** as the plan's command says, although
  the frontend (M3) does not exist yet; the file is the contract M3 will be built against.
  `api export-schema` builds the app from placeholder settings and never runs its
  lifespan, so exporting needs no database, model or secret.
- **Startup warm-up** is not in the plan; the real-corpus smoke showed the first request
  after a restart degraded without it.
- **Not done here:** public rate limiting (`public with rate limit` in spec §10) is
  P5.2's, deferred for the portfolio phase. Related papers return one page, without a
  cursor. Search results are not remapped through merges: a paper merged after a release
  was built can still appear under its old id until the next release, whereas
  `GET /v1/papers/{id}` does follow the merge. The live corpus has no merges.

## Commands and results

```text
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 50 files
uv run --env-file .env.test --project backend pytest \
    backend/tests/integration/test_search_api.py backend/tests/unit/test_cursor.py -q  -> 50 passed
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 406 passed in 2m24s
uv run --project backend pytest backend/tests -m "not integration" -q           -> 262 passed
uv run --project backend python -m copilot.cli api export-schema --out frontend/openapi.json
    -> 6 paths, version 0.1.0 (run with DATA_DIR unset)
uv run --env-file .env --project backend alembic -c backend/alembic.ini upgrade head
    -> copilot_v2: 0003_jobs -> 0004_search_orderings
git diff --check                                                                -> clean
```
