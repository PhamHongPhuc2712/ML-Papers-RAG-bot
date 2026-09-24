# M2 index evidence — vectors, release validation and switching

Date: 2026-09-24
Task: P2.2, index paper and chunk vectors with atomic release switching
Status: **in progress** — the paper collection of release `m2-20260924T095724Z` is
built and verified; its chunk collection is building. The release is staged and
**not active**: activation needs both collections, and refuses otherwise.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB (driver 610.47),
`DATA_DIR` on ext4 with 320 GB free. `postgres:17.11-bookworm`, `qdrant/qdrant:v1.19.1`,
qdrant-client 1.19.0, torch 2.14.0+cu130, transformers 5.17.0, tokenizers 0.23.2.
Code: branch `phuc` from `8ae11d4`, plus this task's commit.

## What exists

| Piece | Where |
|---|---|
| `validate_vectors`, the pinned BGE-M3 adapter, the 2-d fixture model, the embedding cache | `models/embeddings.py` |
| Release-pinned BM25 statistics fitted by streaming, sharing the oracle's weight functions | `search/lexical.py` (`BM25Vocabulary`) |
| Collection lifecycle, build, canaries, validation | `search/index.py` |
| Dense and exact-BM25 retrievers, addressed by release id | `search/dense.py`, `search/sparse.py` |
| Release records, capture-once, build records, supersede-on-switch, drop | `corpus/releases.py` |
| `search fetch-model / build-index / validate-index / drop-index / compare-precision / compare-oracle`, `corpus activate` | `cli.py` |
| Model pin and collection layout | `configs/models.yaml` |

## The snapshot it builds from

`corpus export` could not produce this snapshot as it stood: it loaded every chunk row,
text included, into Python before discarding all of them under the rights filter. It
now streams from a server-side cursor into bounded shards and never fetches withheld
text (commit `78f1704`).

```text
corpus export --run m2-20260924T095724Z --out m2-20260924T095724Z
  -> papers 85,729 | chunks_total 3,426,221 | chunks_exported 0 | withheld 3,426,221
  -> 2m52s wall, 456 MB peak RSS; 4 shards (papers 90 MB, chunk identities 360 + 233 MB)
corpus validate --manifest m2-20260924T095724Z/manifest.json   -> valid: true
```

Manifest sha256 `20c26513…12ad4`. Digests: papers `fc9bdc79…7deb`, chunks
`d4ee3262…4d25`. Chunk text is withheld (`redistribution: unknown` everywhere), so the
chunk collection is built from `copilot_v2` and bound to the snapshot by the chunks
digest — activation compares the build's digest with the manifest's.

## Model pin and the precision comparison

`BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, dense head (`[CLS]`, L2-normalized
in float32), identity `…#cls-l2-v1`. `pytorch_model.bin` matches the hub's LFS sha256
`b5e0ce34…ad38`; `tokenizer.json` is the same file `configs/parsing.yaml` pins for
chunking (`21106b6d…dd08`). The load report lists only `pooler.dense.{weight,bias}` as
unexpected — the pooling layer is deliberately not built — and nothing missing.

Spec §7 requires CPU float32 as the baseline before GPU reduced precision:

```text
search compare-precision --manifest m2-20260924T095724Z/manifest.json --sample 256
```

| Texts (256 each) | cosine mean | cosine min | top-10 neighbour overlap mean / min | CPU fp32 | GPU fp16 |
|---|---|---|---|---|---|
| papers | 0.9999984 | 0.99995 | 0.9996 / 0.9 | 1.1 /s | 74.6 /s |
| chunks | 0.9999976 | 0.99995 | 0.9957 / 0.9 | 0.7 /s | 61.7 /s |

float16 on the GPU is adopted: no vector moves by more than 5×10⁻⁵ in cosine, and the
occasional neighbour swap is between near-ties. CPU float32 at under 1 chunk/s would take
~57 days for the chunk collection.

## Build measurements

Measured on the pilots spec §12 asks for before committing to the full corpus. `--limit N`
builds the first N documents of a collection in the order the full build uses, so a
pilot's encoded batches are a prefix of the full build's and come back as cache hits
(observed: 4 hits in the 10k pilot, 40 in the full build).

| Build | Points | Encode rate | Wall | Allocated disk | Per point | Truncated |
|---|---|---|---|---|---|---|
| papers, pilot-1k | 1,024 | 97.5 /s | 14.8 s | 14.2 MB | 13.9 KB | 0 |
| papers, pilot-10k | 10,240 | 99.8 /s | 98 s | 121.6 MB | 11.9 KB | 1 |
| **papers, full** | **85,729** | 100.1 /s | 13m19s | **591.9 MB** | **6.9 KB** | 3 |
| chunks, pilot-1k (40,960 = 1k papers × 40) | 40,960 | 64.2 /s | 17m50s | 374.8 MB | 9.1 KB | 0 |
| chunks, pilot-10k (409,600 = 10k papers × 40) | 409,600 | 60.1 /s | 1h51m | 3,185.6 MB | 7.8 KB | 0 |

Allocated disk is `du`'s blocks, not apparent size: Qdrant preallocates sparsely, and the
1k paper collection is 847 MB apparent against 14 MB allocated. The marginal paper cost
between 10k and 85,729 points is **6.2 KB/point** — 4 KB of float32 vector, the rest HNSW,
sparse index and payload. Paper builds overlap uploads with encoding: the encoder waited
11.8 s on uploads in 13 minutes. The chunk pilot-1k ran before that overlap existed and
spent 71.5 s of 17m50s in upserts, and ~4 minutes sorting 3.4 M text rows twice; the
statistics pass no longer sorts.

The first 10k-scale chunk attempt exposed a worse cost in reading chunks back. A
server-side cursor is planned for its first tenth by default, and PostgreSQL chose to walk
the chunk primary key and fetch each row's text at random: ~450 rows/s, so **~2 hours per
pass** over 3.4 M chunks. The stream now plans for reading every row
(`cursor_tuple_fraction = 1.0`, `work_mem = 256MB`): 85 s to the first row, then ~20,000
rows/s, **about 4 minutes per pass** (commit `3acfbb8`). That attempt was stopped before it
encoded anything and restarted.

Full paper build: 4.45 GB peak RSS (model included); Qdrant RSS 864 MB afterwards; host
memory available 13.3 GB. BM25 statistics: 121,795 terms, average length 196.6 — the same
vocabulary size P2.1's in-memory oracle reported for this corpus.

The chunk pilot-10k confirms the extrapolation from 1k before the full build was started.
Its first 160 batches were the 1k pilot's, returned from the cache; the encoder waited
50.6 s on uploads in 1h49m of encoding. The marginal cost between the two chunk pilots is
**7.6 KB/point**. Qdrant's RSS afterwards was 1.22 GB; BM25 statistics hold 930,124 terms
over 409,600 chunks (average length 265.9).

**Projection for the full chunk collection (3,426,221 points):** ~26-30 GB of disk against
315 GB free; ~4-5 GB resident in Qdrant — the int8 copy of the dense vectors (~3.4 GB) and
the HNSW graph, with float32 originals, sparse index and payload on disk — against 13.4 GB
available; and ~14 hours to encode the 3,016,621 chunks the pilots did not already cache.
The build was started on these numbers, detached from any session, logging to
`${DATA_DIR}/runs/m2-20260924T095724Z-chunks.{log,json}`.

## BM25 served from Qdrant equals the oracle

```text
search compare-oracle --release m2-20260924T095724Z --sample 500 --limit 100
  -> 500 / 500 identical top-100 rankings | overlap 1.0 | max relative score error 2.2e-7
  -> latency p50 6.4 ms, p95 9.1 ms
```

Queries are 500 sampled paper titles (seed 42), run through the served sparse vectors and
through P2.1's exact in-memory BM25 over the same 85,729 texts. The residual score error is
float32 storage. The stored weights are final BM25 document weights and the collection has
no IDF modifier, so this is BM25 itself, not a provider variant.

## Dense smoke test (not an evaluation)

200 sampled titles as queries against the paper collection: the paper itself ranks first
for 97.5% and in the top 10 for 99.5%. Latency p50 24.9 ms, p95 42.5 ms, query encoding on
the GPU included. Retrieval quality is E3/E4's to measure; this only shows the index serves
what it was built from.

## Validation, switching and rollback evidence

Activation re-checks, and refuses with every reason found: the model identity; both
collections built and present; dense dimension and cosine distance; no sparse modifier;
exact point counts against the snapshot; build digests against the snapshot digests; the
pinned BM25 statistics file by sha256; the snapshot manifest unchanged since staging; no
measurement limit; and canary rankings recomputed with exact search. Canaries store their
query vectors, so validation needs no model and checks the index alone; re-checks fetch 20
results past the recorded limit so a tie that moved across it is found rather than excused.

`backend/tests/integration/test_index_release.py`, against the isolated test services with
the production chunk layout (on disk, int8, deferred HNSW), 15 passed:

| Acceptance case (plan P2.2) | Test |
|---|---|
| NaN / dimension mismatch rejected before writes | `test_nonfinite_vectors_are_refused_before_any_write`, unit `test_incompatible_vectors_fail_before_upsert` |
| same input upserts idempotently | `test_rebuilding_the_same_input_upserts_idempotently` |
| filtered query excludes wrong venue/year | `test_filters_exclude_the_wrong_venue_year_and_availability` (dense and sparse) |
| query/document model revision mismatch rejected | `test_a_query_from_another_model_is_refused` |
| crash between collection builds leaves previous release active | `test_a_crash_between_collection_builds_leaves_the_previous_release_serving` |
| requests in flight keep their original pair | `test_requests_in_flight_keep_the_pair_they_captured` |
| rollback restores both collections | `test_rollback_restores_both_collections_together` |
| restore yields matching canary rankings | `test_restored_collections_reproduce_the_canary_rankings` (Qdrant snapshot → delete → recover → validate → activate) |

Beyond the plan: drifted chunk text refuses activation by digest; a measurement build can
never activate; a collection that gained a point fails validation; a failed upload on the
uploader thread fails the build; the readiness endpoint returns 200 with the release id
once a validated pair is active; and a pilot's batches are cache hits for the full build.

A code review of this task found one real defect, since fixed: the canary comparison
exempted the last tie group from its identity check, so an all-tied canary whose documents
were wrong still passed. Every expected item is now checked.

## Deviations from the plan, and why

- `build_index(manifest, model, *, …)` takes the model object rather than a revision
  string, so the identity recorded on the release is the one that encoded the vectors.
- `search/sparse.py` is an extra file: a Qdrant-backed BM25 retriever in `lexical.py` would
  import `index.py`, which imports `lexical.py`.
- PyTorch and transformers are an opt-in `models` dependency group. CI installs only `dev`
  and was verified to pass mypy strict and the offline set with no torch installed.
- The fixture model is deterministic and two-dimensional, as the plan asks; tests never load
  weights.

## Still to do in P2.2

1. Finish the chunk collection of `m2-20260924T095724Z` (running), record its measurements
   here, then `search validate-index --release m2-20260924T095724Z` and
   `corpus activate --release m2-20260924T095724Z`.
2. A real-corpus restore drill: Qdrant snapshot of both collections, recover, re-validate.
3. ~~Drop the `pilot-*` releases once their numbers are recorded here.~~ Both dropped with
   `search drop-index`.

Known gap, not P2.2's to close: Qdrant writes collection snapshots to `/qdrant/snapshots`
inside the container, and `compose.yaml` does not bind-mount it under `DATA_DIR`. A real
snapshot would sit outside the deletable data root and vanish with the container. The
restore acceptance case runs inside one container's lifetime, so it is unaffected; P5.3's
local backup should mount `${DATA_DIR}/backups/qdrant` there.
