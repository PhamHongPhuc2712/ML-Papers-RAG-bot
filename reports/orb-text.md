# Open RAG Bench, text slice — evidence

Plan: `docs/superpowers/plans/2026-10-06-orb-text-benchmark.md`. Scope approved on
2026-10-07: **O1–O3, retrieval only.** O4 and O5 (the grounded generator and the judged
answering run, which spend money) wait for a separate go-ahead. Host: WSL2 Linux, 12 cores /
23 GB RAM, RTX 3080 Laptop 16 GB.

ORB is CC-BY-NC-4.0: nothing below quotes a question, an answer, a section or a PDF. The
repository holds ids, labels and the sample (`data/fixtures/orb/`); the text lives under
`${DATA_DIR}/benchmarks/orb/`.

## O1 — the benchmark, pinned and frozen (2026-10-07)

`eval orb-fetch` downloaded `vectara/open_ragbench` at commit `63f6b052`: the four label
files and 1,000 corpus files, 716 MB, checksums in `configs/evaluation.yaml`
(`benchmarks.orb`) and verified against the files on disk. A floating revision is refused.

| | |
|---|---|
| Papers | 1,000 arXiv PDFs, ids `2401`–`2503`, every one versioned (`2407.01528v3`) |
| Queries | 3,045, one gold document and one gold section each |
| Text slice | **1,914** (`source: text`): 1,021 extractive, 893 abstractive |
| Gold documents in the slice | 387, median 5 queries each |
| Answering sample | 400 frozen (200 per type) + 100 development (50 per type), at most 2 queries per gold document, seed 42, sorted before shuffled; byte-identical on a second build and under reversed input |

`eval orb-dataset` wrote the slice in the retrieval harness's shape: every query in one
`retrieval` split (a benchmark read as a whole never chooses; the dataset validator and
`eval retrieval` accept the split, and no decision is computed from it), a family per gold
document, the gold section in `sections.jsonl`, and the answering assignment in `qa.jsonl`.
`eval validate-dataset` passes: 1,914 queries, 1,914 qrels, 387 families.

## O2 — the corpus through our pipeline, as `orb-v1` (2026-10-07)

`eval orb-load` ran the 1,000 papers through identity resolution, the hardened downloader
(arxiv.org, one second between requests), `pypdf-text-v2` and `paragraph-pack-v1` into a
separate database, `copilot_orb`, one transaction per paper and resumable. The first run
died at paper 436 on a read timeout from arxiv.org that the loader had not caught; the
loader now retries a transport error twice with a 15 s pause and then records it as a typed
failure, and the resumed run skipped the 433 parsed papers and finished the rest without a
failure of that kind.

| | |
|---|---|
| Records | 1,000 → **997 papers**: three ORB documents are the same paper under adjacent arXiv versions (`2404.19707` v3 and v4, `2405.07102` v3 and v4, `2406.13839` v2 and v3) and the identity layer united each pair, both versions parsed; 0 conflicts, 0 quarantined |
| Parsed | 997 versions over 994 papers; **3 failed**, every one `download_policy:size` — the PDF exceeds the parser's size cap, recorded in `${DATA_DIR}/benchmarks/orb/load-failures.json` |
| Gold documents without text | **1** (`2411.08777v2`, oversized): **6 of 1,914 queries** have a gold paper the chunk collection cannot hold; the paper collection still carries its title and abstract |
| Chunks | 41,670 stored; max token count **1,200**, median 671 (the live corpus: 1,200 and comparable); kinds: body 21,791, references 5,713 (excluded from evidence by default), appendix 4,991, front 4,087, figure 2,052, table 1,309, abstract 1,231, acknowledgments 496 |
| Snapshot | `${DATA_DIR}/exports/orb-v1`: 997 papers, 41,463 chunk identities (one chosen version per paper, so the 207 chunks of the three duplicate versions are not in it), every chunk withheld (`redistribution: unknown`), `corpus validate` valid |
| Gold labels | `eval orb-dataset --database-url …copilot_orb` resolved all **1,914** qrels to paper ids by arXiv stem; a merged pair's two ids resolve to the same paper |

`search build-index` built `orb-v1` from the snapshot on the idle GPU (0% before, the
idle Windows-side process holding 1.3 GB throughout): the paper collection, 997 points, in
12 s; the chunk collection, 41,463 points, in 850 s (encode 831 s, 48 points/s, float16
BGE-M3), 466 MB on disk, 0 truncated inputs. Both digests equal the snapshot's
(`papers e56146d2…`, `chunks 2d4bcef4…`), and `search validate-index --release orb-v1`
reports valid with no problems.

The release `orb-v1` is registered in `copilot_orb` only. The API reads the active release of
`copilot_v2`, where `orb-v1` does not exist (`load_release` raises `release_unknown`), so it
cannot be served: not a namespace refusal, an absence. It is never activated anywhere.

## O3 — retrieval on all 1,914 text queries (2026-10-07)

Run `orb-text-retrieval-20261007T110552Z` (`reports/retrieval/orb-text/`), config
`configs/experiments/orb-text-retrieval.yaml`: the four modes, each with the paper
collection and with the chunk collection as the candidate source, under `configs/search.yaml`
as shipped, no decision block. Clean: GPU idle beforehand (p50 0%, 1,352 MiB held by the idle
Windows-side process), no other compute process during the run, **0 failed and 0 degraded
queries** in 38 minutes. Warm-up took the split's own last 10 queries, as the manifest
records; the latencies are a read, not a measurement of record. `judged@10` is 0.07–0.10
throughout: one gold document per query, so precision at 10 cannot exceed a tenth.

### Two attempts set aside, and what they found

The first two runs lost 661, then 751 queries of one variant, `hybrid_papers`, to
`candidates_unavailable`, while the same branches ran clean alone and under the reranker.
The harness recorded only the failure code, so it now carries each branch's error with the
failed row; the third attempt's precursor read `lexical:ResponseHandlingException,
dense:ResponseHandlingException`, and a direct replay gave the text: `Connection reset by
peer`, both branches, from about three seconds into the first variant that queried Qdrant
from two threads at once, for ten to twenty seconds. Qdrant never restarted and logged
nothing. The cause is in the client library: `qdrant-client` turns keep-alive off for any
`localhost` URL, so every request opened a new TCP connection; 300 queries left 175 more
sockets in `TIME-WAIT`, and a burst of a few thousand through Docker's port proxy on this
WSL host ends in resets. Every client is now opened through `search/client.py`, which passes
a bounded keep-alive pool; the same three-mode replay then ran 5,742 queries with 0 failures
and three established connections at the end. The API serves from the same URL and would
have met the same burst under load. Recorded as a decision in the tracker.

### Results

| Variant | Candidates | Recall@10 | Recall@50 | nDCG@10 | MRR@5 | Gold in pool | p50 / p95 |
|---|---|---|---|---|---|---|---|
| `bm25_papers` | titles and abstracts | 0.777 [0.753, 0.802] | 0.853 | 0.693 [0.669, 0.718] | 0.660 | 86.4% | 3 / 3 ms |
| `dense_papers` | titles and abstracts | 0.762 [0.738, 0.788] | 0.854 | 0.677 [0.653, 0.705] | 0.644 | 87.2% | 19 / 30 ms |
| `hybrid_papers` | titles and abstracts | 0.809 [0.787, 0.831] | 0.882 | 0.728 [0.706, 0.753] | 0.698 | 92.3% | 21 / 31 ms |
| `hybrid_rerank_papers` (P2.5's configuration) | titles and abstracts | 0.838 [0.816, 0.858] | 0.882 | 0.762 [0.740, 0.785] | 0.734 | 92.3% | 531 / 612 ms |
| `bm25` | chunks | 0.976 [0.967, 0.986] | 0.987 | 0.942 [0.929, 0.954] | 0.929 | 98.9% | 5 / 7 ms |
| `dense` | chunks | 0.963 [0.952, 0.974] | 0.981 | 0.917 [0.903, 0.930] | 0.900 | 98.7% | 24 / 32 ms |
| **`hybrid`** | chunks | **0.976** [0.966, 0.985] | **0.991** | **0.948** [0.936, 0.959] | **0.937** | 99.4% | 27 / 36 ms |
| `hybrid_rerank` (shipped) | chunks | 0.874 [0.855, 0.891] | 0.991 | 0.785 [0.763, 0.805] | 0.750 | 99.4% | 535 / 594 ms |

Paired differences, 95% intervals over query families (a family is a gold paper):

| Comparison | nDCG@10 | Recall@10 | Recall@50 | MRR@5 |
|---|---|---|---|---|
| `bm25` vs `bm25_papers` | +0.249 [+0.224, +0.274], 286 / 15 | +0.200 | +0.134 | +0.269 |
| `dense` vs `dense_papers` | +0.240 [+0.215, +0.263], 284 / 17 | +0.201 | +0.128 | +0.255 |
| `hybrid` vs `hybrid_papers` | +0.219 [+0.196, +0.242], 275 / 13 | +0.167 | +0.109 | +0.239 |
| `hybrid_rerank` vs `hybrid_rerank_papers` | +0.022 [+0.016, +0.028], 140 / 16 | +0.036 | +0.109 | +0.016 |
| `hybrid_rerank_papers` vs `hybrid_papers` | +0.034 [+0.021, +0.045], 163 / 84 | +0.029 | same head | +0.036 |
| **`hybrid_rerank` vs `hybrid`** | **−0.163 [−0.184, −0.143], 19 / 245** | −0.102 | same head | −0.187 |

1. **The P2.6 decision reproduces on a second corpus, three times as large.** Candidates found
   through chunks beat candidates found through titles and abstracts in every mode, by
   +0.22 to +0.25 nDCG@10 with intervals far above zero. Gold papers never pooled fall
   from 7.7% to 0.6% for hybrid. The effect is larger here than on LitSearch's author-written
   queries (+0.135 Recall@50 on our validation split) because every ORB question is written
   from one section of a paper's body, which is precisely what the abstract does not say.
   The 0.6% never pooled under chunks is 11 gold papers, 6 of them the oversized PDF with no
   text.
2. **The cross-encoder rerank over titles and abstracts loses the top 10 here.** With chunk
   candidates, `hybrid_rerank` scores 0.785 nDCG@10 against `hybrid`'s 0.948: −0.163
   [−0.184, −0.143], 19 wins to 245 losses, the same reranked head (Recall@50 identical).
   On paper-level candidates the same reranker helps (+0.034). The reason is what it reads:
   the first stage found the paper by a body chunk, and the reranker then scores the
   query against the title and abstract, which do not contain the section the question
   came from, so it demotes the right paper below papers whose abstracts happen to share
   the query's words. Extractive questions, the most section-bound, suffer most: 0.932 →
   0.662 against 0.960 → 0.858 for abstractive ones. On LitSearch, where a query describes a
   whole paper, the abstract is the right text and reranking gained +0.054 to +0.072; here
   it is the wrong text. **This is not a decision**: the shipped configuration was chosen
   on LitSearch validation and confirmed on the locked test, and this run reads it. It is a
   recorded finding with two consequences: the evidence retriever (O4 / P4.1) reranks chunk
   text, not abstracts, as planned; and a paper-search ablation that reranks each candidate's
   best chunk — or chunk plus title — is the next pre-registered experiment on our corpus,
   where both query shapes occur.
3. **Beside haiku.rag's ORB rows** (MAP over all 3,045 queries, which for one gold document
   is 1/rank at depth 5, so our MRR@5 over the 1,914 text queries is the comparable
   figure): their best retrieval stacks reach 0.977–0.990 with a 4B or multimodal embedder,
   a 4B or multimodal reranker and Docling's text. Our plain `hybrid` reads 0.937 with
   BGE-M3 and pypdf; our shipped `hybrid_rerank` reads 0.750 for the reason above. The
   samples differ (text slice against all queries; our parse against theirs), so this is a
   placement, not a comparison.
4. **Where the hit lands in the paper** (`sections.md`): 1,022 of 1,527 distinct gold
   (document, section) pairs mapped onto our chunks, covering 1,277 of 1,914 queries
   (66.7%); the rest are sections pypdf and the chunker cut differently from Mistral-OCR,
   or sections too short for the rule, and they are unknown, not misses. Over the mapped
   queries, `hybrid` puts the gold paper first 91.0% of the time and its best chunk in the
   gold section 66.8% of the time; given the right paper, the best chunk is in the gold
   section 72.8% of the time. Under the reranker, paper@1 drops to 66.2%.
5. **The dense branch alone is the weakest first stage here** (0.917 against BM25's 0.942),
   as on our corpus, and the two fuse to the best (0.948); BM25 is also the fastest branch
   by a factor of five. Nothing in this run is a latency measurement of record.

### Commands

```text
uv run --env-file .env --project backend python -m copilot.cli eval orb-fetch
uv run --env-file .env --project backend python -m copilot.cli eval orb-dataset --out data/fixtures/orb
uv run --env-file .env --project backend python -m copilot.cli eval orb-load --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli corpus export --run orb-v1 --out orb-v1 --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli corpus validate --manifest ${DATA_DIR}/exports/orb-v1/manifest.json
uv run --env-file .env --project backend python -m copilot.cli search build-index --manifest …/orb-v1/manifest.json --release orb-v1 --collections papers --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli search build-index --manifest …/orb-v1/manifest.json --release orb-v1 --collections chunks --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli search validate-index --release orb-v1 --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli eval orb-dataset --out data/fixtures/orb --database-url …copilot_orb
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/orb-text-retrieval.yaml --split retrieval --database-url …copilot_orb --out reports/retrieval/orb-text
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/orb-text
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/orb-text --split retrieval --focus hybrid_rerank
uv run --env-file .env --project backend python -m copilot.cli eval orb-sections --run reports/retrieval/orb-text --database-url …copilot_orb
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 596 passed
uv run --project backend pytest backend/tests -m "not integration" -q           -> 409 passed
uv run --project backend python -m copilot.cli eval smoke                       -> passed, reproducible
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 62 files
```

Recorded reports re-render unchanged. The locked test split is untouched. O4 and O5 are
not started.

## Reranking the chunk that found the paper (2026-10-08)

Finding 2 above said why the shipped reranker loses here: the first stage finds a paper by a
body chunk, then the cross-encoder scores the query against the title and abstract, which do
not contain the section the question came from. haiku.rag's reranker, which gains about one
point of MAP on the same benchmark, scores the chunks themselves (its search hands the
reranker chunk objects and expands context only afterwards). The search service now has the
same mechanism as a configurable: `rerank.text` in `configs/search.yaml` (`abstract`, the
shipped setting; `chunk`, the evidence chunk the paper was found by, the better-ranked
branch's when BM25 and dense disagree; `title_chunk`, that chunk with the title in front).
The setting is part of the cache identity and a run's search digest, needs
`candidates.source: chunks`, and a candidate no branch found through a chunk is read as its
abstract and counted in the trace. The model is unchanged: `bge-reranker-v2-m3` at the
1,024-token pair budget.

Run `orb-rerank-text-retrieval-20261008T061622Z` (`reports/retrieval/orb-rerank-text/`),
config `configs/experiments/orb-rerank-text.yaml`, pre-registered before any number existed,
no decision block. A 60-query probe before it (scratch, not recorded) had put chunk-pair
reranking at p95 1.39 s and max 1.47 s against the shipped 1.5 s rerank deadline, so each
chunk-text variant runs twice: under the shipped deadline (the serving path's reading) and
under a 2.5 s deadline (the quality reading: a query the reranker cannot finish falls back to
the RRF order, which here ranks *higher* than the shipped reranker and would inflate the
row). Clean: GPU 0% busy before the run (p50 over 5 s, 2,046 MiB held by the Windows-side
process), no failed query in any reranked variant, 0 degraded in the two relaxed-deadline
rows, 3 h 6 min. Two Windows-side processes (`[Not Found]` to WSL, pids 27 and 32) were
listed during the run; the two reference rows reproduce O3 (`hybrid_rerank` to four
decimals; `hybrid` to within one query, see below), which is the evidence the run was not
contended in a way that mattered.

| Variant | Rerank reads | Deadline | Recall@10 | nDCG@10 | MRR@5 | Degraded | p50 / p95 / max | rerank p95 |
|---|---|---|---|---|---|---|---|---|
| `hybrid` (O3: 0.976 / 0.948 / 0.937) | — | — | 0.976 [0.966, 0.985] | 0.947 [0.935, 0.959] | 0.937 | 1 (1 failed) | 40 / 71 / 1,117 ms | — |
| `hybrid_rerank` (shipped; O3 reproduced) | abstract | 1.5 s | 0.874 [0.855, 0.891] | 0.785 [0.763, 0.805] | 0.750 | 0 | 566 / 657 / 1,487 ms | 603 ms |
| `hybrid_rerank_chunk` | chunk | 1.5 s | 0.985 [0.978, 0.992] | 0.965 [0.954, 0.974] | 0.953 | 10 | 1,260 / 1,419 / 2,518 ms | 1,335 ms |
| `hybrid_rerank_title_chunk` | title + chunk | 1.5 s | 0.984 [0.977, 0.991] | 0.966 [0.958, 0.975] | 0.956 | 379 | 1,257 / 2,140 / 2,682 ms | 1,301 ms |
| **`hybrid_rerank_chunk_2500`** | chunk | 2.5 s | 0.985 [0.978, 0.992] | **0.965** [0.954, 0.974] | 0.953 | 0 | 1,235 / 1,409 / 2,323 ms | 1,349 ms |
| **`hybrid_rerank_title_chunk_2500`** | title + chunk | 2.5 s | 0.986 [0.979, 0.992] | **0.968** [0.959, 0.976] | 0.957 | 0 | 1,244 / 1,558 / 2,261 ms | 1,514 ms |

Paired differences, 95% intervals over query families (wins / losses / ties of 387):

| Comparison | nDCG@10 | Recall@10 | MRR@5 |
|---|---|---|---|
| `hybrid_rerank_chunk_2500` vs `hybrid_rerank` | **+0.180 [+0.160, +0.201]**, 251 / 15 / 121 | +0.111 [+0.095, +0.130] | +0.207 [+0.183, +0.230] |
| `hybrid_rerank_title_chunk_2500` vs `hybrid_rerank` | **+0.183 [+0.163, +0.204]**, 252 / 14 / 121 | +0.112 [+0.096, +0.130] | +0.212 [+0.189, +0.235] |
| `hybrid_rerank_chunk_2500` vs `hybrid` | +0.017 [+0.009, +0.027], 84 / 43 / 260 | +0.010 [+0.002, +0.018] | +0.020 [+0.010, +0.031] |
| `hybrid_rerank_title_chunk_2500` vs `hybrid` | +0.021 [+0.012, +0.030], 85 / 37 / 265 | +0.010 [+0.003, +0.018] | +0.025 [+0.014, +0.035] |
| `hybrid_rerank_title_chunk_2500` vs `hybrid_rerank_chunk_2500` | +0.003 [+0.001, +0.006], 33 / 15 / 339 | +0.001 [−0.001, +0.002] | +0.005 [+0.002, +0.009] |
| `hybrid_rerank_chunk` (1.5 s) vs its 2.5 s twin | 0.000, 0 / 0 / 387 | 0.000 | 0.000 |
| `hybrid_rerank_title_chunk` (1.5 s) vs its 2.5 s twin | −0.002 [−0.005, +0.002], 9 / 14 / 364 | −0.001 | −0.002 |

1. **Reading the chunk recovers the loss and then some.** Over the same reranked head
   (Recall@50 identical, 0.991), the reranker that reads the chunk gains +0.180 nDCG@10 over
   the one that reads the abstract, 251 families to 15, and now sits *above* plain fusion by
   +0.017 [+0.009, +0.027] (chunk) and +0.021 [+0.012, +0.030] (title and chunk). The gold
   paper is at rank 1 for 1,802–1,807 queries against `hybrid`'s 1,740 and the shipped
   reranker's 1,268. Extractive questions, which lost most, recover most: 0.662 → 0.953.
   The title in front of the chunk adds +0.003 [+0.001, +0.006], 33 to 15: real, small.
2. **Where the hit lands.** Over the 1,277 queries whose gold section mapped,
   `section_hit@1` is 69.5–69.7% (`hybrid` 66.8%, the shipped reranker 48.2%) and `paper@1`
   94.1–94.4%; given the right paper, the best chunk is in the gold section 73.1%, the same
   as before, because the reranker reorders papers and does not choose the chunk.
3. **Beside haiku.rag.** Its ORB rows with a reranker read MAP 0.983–0.990 over all 3,045
   queries; our chunk-reranked MRR@5 over the 1,914 text queries reads 0.953–0.957 against
   0.750 for the shipped stack and 0.937 for plain `hybrid`. Still a placement, not a
   comparison (text slice against all queries, pypdf against Docling, BGE-M3 and a 568M
   reranker against 4B models), but the mechanism is now the same, and the reranker now
   helps on this benchmark as theirs does.
4. **The cost is latency, and the shipped deadline does not fit it.** A chunk pair runs to
   the 1,024-token budget where an abstract pair is a few hundred tokens: rerank p95 goes
   from 0.60 s to 1.3–1.5 s, request p50 from 0.57 s to 1.24 s. Under the shipped 1.5 s
   rerank deadline the chunk row lost 10 queries (7 rerank timeouts, 3 dense timeouts while
   abandoned rerank work held the GPU) and the title-and-chunk row lost 379: its p50 sits
   at 1.26 s against the deadline, and once one rerank overruns, the abandoned call keeps
   the single rerank lane and the GPU busy so the next query's rerank, and sometimes its
   dense branch (p95 628 ms in that row against 45 ms elsewhere), time out behind it. The
   quality held either way — every one of the 10 degraded chunk-row queries had its gold
   paper at rank 1 under both orders, and the 379-query row reads within 0.002 of its twin —
   but a serving path that degrades a fifth of its requests is not the serving path. Under
   2.5 s both rows are clean with request p95 1.41 s and 1.56 s, inside the 3 s total. Two
   levers for serving, neither measured here: a 512-token pair budget (spec §7's own
   comparison; a chunk's median is 671 tokens) and a smaller rerank depth.
5. **One query failed in `hybrid`** (`0d9b8d7b…`): both branches timed out at the 1.0 s
   branch deadline (1.117 s total), the only failure in 11,484 queries; O3's `hybrid` had
   none. It is scored zero and stays in the denominator (0.947 against O3's 0.948); the five
   reranked rows, which ran the same branches, saw no such failure.

**This is not a decision.** ORB reads the shipped configuration; it never chooses. The
shipped reranker reads abstracts because LitSearch validation chose it, where a query
describes a whole paper and the abstract is the right text. What this run adds is that on
section-level questions the same model over the chunk that found the paper is the right
text, and that the serving path cannot carry it under its current rerank deadline. The
promotion path is a pre-registered rule on our corpus's validation split, where both query
shapes occur, with the deadline or pair budget settled first; the locked test split is
spent, so a shipped change needs its own release and the next locked run.

```text
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/orb-rerank-text.yaml --split retrieval --database-url …copilot_orb --out reports/retrieval/orb-rerank-text
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/orb-rerank-text
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/orb-rerank-text --split retrieval --focus hybrid_rerank_title_chunk_2500
uv run --env-file .env --project backend python -m copilot.cli eval orb-sections --run reports/retrieval/orb-rerank-text --database-url …copilot_orb
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 610 passed (14 new)
uv run --project backend pytest backend/tests -m "not integration" -q           -> 416 passed
uv run --project backend python -m copilot.cli eval smoke                       -> passed, reproducible
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 62 files
```

Recorded reports (`orb-text`, `e3-litsearch`, `m2-release`) re-render unchanged.

## O4 and O5: answering the text slice (2026-10-08, in progress)

O4 (`reports/m4-evidence.md`) built the evidence retriever and the grounded generator; O5
scores them. Each case: the shipped search's top 10 papers, evidence inside them under
`configs/evidence.yaml`, the generator's claims with evidence ids (one repair, then
insufficiency), then the answer-equivalence judge (`docs/judge-rubric.md`). Generator
`gpt-6-luna` at effort `none`; judge the same model at effort `low` — one provider, a
correlated-bias limitation this section states and the calibration below measures. An
abstention is scored wrong without a judge call; a judge error is excluded and counted;
cited MAP is 1/rank of the gold paper among cited papers. Config
`configs/experiments/orb-text-qa.yaml`, outputs under `reports/qa/orb-text/` (ids, labels,
verdicts and costs; the text under `${DATA_DIR}/eval/qa/`). Every run was clean: GPU 0%
busy beforehand, no failed generation, no budget stop.

### Pilot, 20 development cases (`orb-text-qa-development-20261008T100652Z`, `flat`)

19 judged of 20, accuracy 0.79 [0.57, 0.91] Wilson, cited MAP 0.90, provenance validity
1.00, 0 repairs, 0 abstentions, $0.0138 — $0.0007 a case against the plan's $0.0012
estimate; 7.3 s a case at the median. The five non-correct cases, read: two retrieval
misses (gold paper outside the top 10), one judge error, one strict verdict on an
omitted detail, one verdict that faulted correct added detail with outside knowledge. Two
fixes before the development pass, both to the judge: (1) the judge error was LaTeX
(`\phi`) written unescaped inside the verdict's JSON, invalid twice; both parsers now
take every backslash literally when the strict parse fails; (2) the rubric gained one
rule, to judge added detail against the reference alone and never from the judge's own
knowledge. The generation prompt is unchanged from O4.

### Development pass, 100 cases, both variants (`orb-text-qa-development-20261008T101136Z`)

| Variant | Judged | Accuracy (Wilson) [family bootstrap] | Extractive / abstractive | Cited MAP | Provenance | Abstained | Repairs | Judge errors | Gold in top 10 | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| `flat` | 100 / 100 | 0.77 [0.68, 0.84] [0.68, 0.85] | 0.74 / 0.80 | 0.83 | 1.00 | 11 | 0 | 0 | 84% | $0.054 |
| `section` | 100 / 100 | 0.76 [0.67, 0.83] [0.67, 0.84] | 0.72 / 0.80 | 0.84 | 0.99 | 8 | 1 | 0 | 84% | $0.066 |

Median 5.5 s a case; $0.0005–0.0007 a case. The judge's parser fix held: 0 judge errors
in 200 verdicts. The `flat` variant's 23 non-correct cases, by category: 14 retrieval
misses (the gold paper not in the search's top 10, so the evidence could not contain the
answer; 13 of these in `section` too), 1 abstention, and 8 answered with the gold paper
in hand. Of those 8, read: 6 are verdicts on an omitted qualifier or clause (the
reference, written by gpt-4o-mini, says "in France", "for high-skill workers", names a
second mechanism or a limitation; the answer has the main fact and not the qualifier),
1 answers a neighbouring case (the sinusoidal flux where the reference wants the general
expression) and 1 contradicts the reference outright. The one repair was a LaTeX answer
the strict parse refused before the wider fallback landed. **Two prompt passes were
made** (the pilot and this one); the generation prompt was not changed and the rubric
once. Whether the six omission verdicts are the judge's strictness or real misses is
the calibration's question, which is why the rubric is not touched again before it.

Two things this pass says about the pipeline, for the record and not as decisions:
the first stage bounds accuracy at 84% here (the shipped abstract-text reranker's
Recall@10 on ORB was 0.874; the chunk-text reranker read in the section above reaches
0.985), and section expansion, with the same budget, carries fewer pieces (median 6
against 8) for the same accuracy within noise on 100 cases.

### Calibration by a second judge (2026-10-09, the developer's decision)

The developer had no time to label, and decided to measure the judge the way haiku.rag
does: against a second judge model rather than against a person. Their page reports 95%
agreement, κ 0.90, between their new and previous judges on 120 ORB cases, and human
adjudication only on cases where two judges disagreed. Ours is `openai-gpt-6-sol`, a
different model family on the same provider ($2 / $0.20 / $10 per 1M tokens on the
provider's pricing page, read 2026-10-09; effort `none`, 1,024 output tokens), reading
the same question, reference and answer under the same rubric and prompt
(`answer-equivalence-v1`, same digest), through `eval qa-rejudge` under the ledger. Its
verdicts stand in for the labels in `eval calibration`'s arithmetic, with the κ floor of
0.6 unchanged. **This is model-model agreement:** two models can share a blind spot, and
this one shares the generator's provider too, so it is weaker evidence than 50 hand
labels, and every accuracy number below carries that caveat. The plan's step 7 records
the substitution.

Pilot on 2 cases ($0.002; the model accepted `reasoning_effort: none`, `temperature: 0`
and `seed`), then the `flat` run's 89 judged answers (the 11 abstentions were scored wrong
without a judge call and are not re-judged); 0 judge errors, $0.098 from the ledger
(`llm spend --day 2026-10-09`: 89 calls, 30,231 input and 3,983 output tokens, $0.100
with the pilot). Verdict sheet under `${DATA_DIR}/eval/qa/labels/`, summary beside the run
(`second-judge.openai-gpt-6-sol.json`, no text).

| | Second judge: equivalent | Second judge: not |
|---|---|---|
| **Judge of record: equivalent** | 75 | 2 |
| **Judge of record: not** | 2 | 10 |

**Agreement 95.5% (85 of 89), Cohen's κ 0.807**, above the 0.6 floor, beside haiku.rag's
95% and 0.90. The four disagreements, read: every one is the omitted-qualifier pattern the
development pass had already flagged (an answer with the reference's main fact and not a
qualifying clause or a second conclusion), two where the judge of record was stricter and
two where the second judge was. Neither judge is systematically the stricter one, and
nothing in the four suggests a rubric fix, so the rubric stays as it is and the development
accuracies stand as calibrated against a second model: 0.77 [0.68, 0.84] `flat` and 0.76
[0.67, 0.83] `section`. What remains uncertain is exactly the band the four cases sit in,
answers that omit a qualifier, which a human read would settle and this check cannot.

### The frozen 400: pending

The blind labelling sheet written for 50 development cases
(`${DATA_DIR}/eval/qa/labels/…flat.sheet.jsonl`) stays unlabelled; the second-judge check
above replaces it by decision. The frozen 400 have not been scored (`frozen-scorings.json`
absent). Before they are, the chunk-size read above argues for registering a third variant,
the 256-token corpus with the chunk-reading reranker, so the one scoring per configuration
is not spent on the weaker first stage; that is the developer's call.

## haiku.rag's chunk size, read on Open RAG Bench (2026-10-09)

**Why.** haiku.rag 0.95.0 (read from source) is a flat chunk index with an agent on top:
Docling parses to a layout tree, its chunker cuts 256-token chunks with no overlap that keep
their heading path, the embedded and full-text strings are the headings prepended to the
chunk, one search over all chunks fetches ten times the limit and the cross-encoder scores
each chunk's text, then every survivor is widened to its section within 5,000 characters. A
Pydantic AI agent may search up to five times, must call `cite` with retrieved chunk ids
before answering, and an empty list is a valid "ungrounded". Beside it, what our pipeline
still did differently after `candidates.source: chunks` and `rerank.text: chunk` was the
chunk itself: `paragraph-pack-v1` packs paragraphs to 800 tokens. Pre-registered question
(`configs/experiments/orb-c256-retrieval.yaml`): over the same corpus, parser, tokenizer and
models, does a 256-token chunk change first-stage recall, the fused ranking, the chunk-reading
reranker and its latency, and where the best chunk lands? A read of ORB, no decision block.

**How.** `eval orb-load --reparse` (new) re-chunks a loaded corpus from the PDFs kept under
`${DATA_DIR}/benchmarks/orb/pdfs`, offline, under another parsing file; it replaces each
version's chunks (`store_parsed_document` keeps one chunking per version) and skips a version
already under the target policy. It ran against `copilot_orb_c256`, a PostgreSQL template
copy of `copilot_orb`, so paper ids and the committed labels carry over and `orb-v1` is
untouched. Policy `configs/parsing-orb-c256.yaml`: `paragraph-pack-256-v1`, pack to 256,
split a paragraph past 256 on sentences, no overlap (haiku.rag's `HybridChunker` rule in our
policy; its tokenizer is Qwen's, ours BGE-M3's). 997 parsed, 3 recorded `reparse:not_parsed`
(the oversized PDFs of O2), 28 min; **119,983 chunks, median 213 tokens, max 256** (O2:
41,670, median 671, max 1,200); body 65,249, references 18,905, appendix 15,751, front
12,179, abstract 3,304, figure 2,052, table 1,315, acknowledgments 1,228. Snapshot
`${DATA_DIR}/exports/orb-c256` (119,369 chunk identities, papers digest identical to
`orb-v1`'s, `corpus validate` valid); release `orb-c256` built on the idle GPU in 806 s
(encode 733 s, 163 points/s), 852 MB, both digests equal to the snapshot's, `validate-index`
valid; never activated.

Run `orb-c256-retrieval-20261009T042830Z` (`reports/retrieval/orb-c256/`): GPU 0% busy
before it (p50 over 5 s, 1,771 MiB held by the Windows-side process, pid 27, the only other
compute process seen), **0 failed and 0 degraded queries in every row**, 75 min. Two rows read
`orb-v1` through the clone (its paper table is identical and neither reads chunk text from
the database), so the chunk-size comparison is paired within one run; both reproduce O3
(`hybrid` 0.948, `hybrid_rerank` 0.785). The `orb-v1` chunk-text rows cannot be re-read in
the clone, so they are placed against the 2026-10-08 run by query family with the harness's
own arithmetic (`pairing-vs-orb-v1.json`; `eval compare` refuses two releases by design).
A first run of the same config (`…T030428Z`, identical rankings) was set aside: the run
manifest had recorded the chunker version from the shipped parsing file. The harness now
records the experiment's own `parsing:` file (default `configs/parsing.yaml`), with a test.

| Variant | Chunks | Rerank reads | Deadline | Recall@10 | nDCG@10 | MRR@5 | Gold at rank 1 | p50 / p95 / max | rerank p95 |
|---|---|---|---|---|---|---|---|---|---|
| `hybrid_v1` (O3 reproduced) | 800 | — | — | 0.976 [0.966, 0.985] | 0.948 [0.936, 0.959] | 0.937 | 1,741 | 28 / 39 / 650 ms | — |
| `hybrid_rerank_v1` (shipped; O3 reproduced) | 800 | abstract | 1.5 s | 0.874 [0.855, 0.891] | 0.785 [0.763, 0.805] | 0.750 | 1,268 | 541 / 607 / 833 ms | 564 ms |
| **`hybrid`** | 256 | — | — | 0.980 [0.971, 0.989] | **0.959** [0.948, 0.969] | 0.951 | 1,773 | 32 / 41 / 333 ms | — |
| `hybrid_rerank` | 256 | abstract | 1.5 s | 0.872 [0.852, 0.890] | 0.784 [0.762, 0.805] | 0.750 | 1,270 | 542 / 605 / 822 ms | 564 ms |
| `hybrid_rerank_chunk` | 256 | chunk | 1.5 s | 0.983 [0.976, 0.991] | 0.958 [0.948, 0.967] | 0.948 | 1,765 | 379 / 403 / 617 ms | 360 ms |
| `hybrid_rerank_chunk_2500` | 256 | chunk | 2.5 s | 0.983 [0.976, 0.991] | 0.958 [0.948, 0.967] | 0.948 | 1,765 | 380 / 407 / 1,228 ms | 361 ms |
| **`hybrid_rerank_title_chunk_2500`** | 256 | title + chunk | 2.5 s | 0.986 [0.978, 0.993] | **0.965** [0.956, 0.974] | 0.957 | 1,783 | 407 / 432 / 1,015 ms | 388 ms |

Paired differences, 95% intervals over query families (wins / losses / ties of 387):

| Comparison | nDCG@10 | MRR@5 | Recall@10 |
|---|---|---|---|
| `hybrid` (256) vs `hybrid_v1` (800), within run | **+0.011 [+0.005, +0.019]**, 70 / 36 / 281 | +0.014 [+0.006, +0.023] | +0.004 [−0.000, +0.009] |
| `hybrid_rerank` (256) vs `hybrid_rerank_v1` (800), within run | −0.001 [−0.004, +0.002], 67 / 57 / 263 | +0.000 [−0.003, +0.004] | −0.002 [−0.009, +0.003] |
| `hybrid_rerank_chunk_2500`: 256 vs 800 (2026-10-08 run) | **−0.007 [−0.013, −0.001]**, 42 / 64 | −0.009 [−0.017, −0.001] | −0.002 [−0.007, +0.003] |
| `hybrid_rerank_title_chunk_2500`: 256 vs 800 (2026-10-08 run) | −0.003 [−0.009, +0.003], 36 / 53 | −0.004 [−0.011, +0.003] | +0.000 [−0.005, +0.005] |
| `hybrid_rerank_chunk_2500` vs `hybrid`, both 256 | −0.002 [−0.010, +0.006], 55 / 66 / 266 | −0.004 [−0.013, +0.006] | +0.003 [−0.003, +0.011] |
| `hybrid_rerank_title_chunk_2500` vs `hybrid`, both 256 | +0.006 [−0.003, +0.015], 63 / 53 / 271 | +0.006 [−0.004, +0.016] | +0.006 [−0.001, +0.013] |

1. **Smaller chunks help plain fusion.** Over the same 997 papers, a 256-token chunk finds the
   gold paper at rank 1 for 1,773 queries against 1,741: +0.011 [+0.005, +0.019] nDCG@10, 70
   families to 36; extractive questions gain most (0.932 → 0.946; abstractive 0.960 → 0.967).
   Recall@50 is unchanged within noise (0.991 → 0.993), so this is ordering, not reach.
2. **The abstract-reading reranker is indifferent to the chunk**, as it must be: it reads the
   same text either way (−0.001 [−0.004, +0.002]). O3's loss is the text, not the chunking.
3. **The chunk-reading reranker loses its margin over fusion.** Reading a 256-token chunk, it
   ranks slightly worse than reading an 800-token one (−0.007 [−0.013, −0.001], 42 to 64) and
   no longer beats fusion: −0.002 [−0.010, +0.006] for the chunk, +0.006 [−0.003, +0.015] with
   the title in front, against +0.017 and +0.021 at 800 tokens. Fusion got better and the
   cross-encoder sees less context per pair; the two meet. The best row overall is still the
   title-and-chunk reranker (0.965 [0.956, 0.974], MRR@5 0.957), but by a margin this read
   cannot distinguish from zero.
4. **Latency is the clear change.** A 256-token pair reranks at p95 360–388 ms against
   1,335–1,514 ms at 800 tokens and 564 ms for the shipped abstract pairs: request p95 403–432
   ms, 0 of 1,914 degraded under the shipped 1.5 s deadline, where the 800-token rows lost 10
   (chunk) and 379 (title and chunk). The 1.5 s and 2.5 s chunk rows are identical to the
   query. What the 2026-10-08 section left open, the serving cost of chunk-text reranking,
   has a chunk-size answer as well as a deadline or pair-budget one.
5. **Where the hit lands is not comparable across chunk sizes** (`sections.md`). The mapping
   rule needs a run of a gold section's sentences inside one chunk, so fewer sections map
   (987 of 1,527 pairs, 1,229 queries at 64.2%, against 1,022 and 66.7%), and a smaller chunk
   is a smaller target: `hybrid` puts its best chunk in the gold section for 57.1% of mapped
   queries (66.8% at 800 tokens) and 61.2% given the right paper (72.8%). `paper@1` moves the
   other way, 92.6% against 91.0%. The `orb-v1` rows read 0.0 here because their chunk ids
   are not in the clone; their figures are the 2026-10-08 run's. For evidence, the unit the
   generator receives is what matters, and that is where section expansion (`context.expansion:
   section`, the `section` QA variant) is meant to put back what a small chunk leaves out.
6. **Beside haiku.rag.** MRR@5 over the 1,914 text queries reads 0.951 (fusion) and 0.957
   (title and chunk) against their 0.983–0.990 MAP over all 3,045 with 4B models and Docling
   text. Candidate source, reranker text and chunk size are now the same mechanisms as theirs;
   what remains different is the parser (pypdf against Docling: a third of ORB's gold sections
   do not map onto our chunks at all) and the models (568M against 4B), and this read cannot
   separate the two. Still a placement, not a comparison.

Not a decision. A chunking change on our corpus is a release decision: a pre-registered rule
on the validation split, then a rebuild of 3,426,221 chunks (`corpus rechunk`, days of GPU
time for the index) and a locked run. What this read supports pre-registering there: the
fusion gain and the reranker's latency; what it warns about: the chunk-reading reranker's
margin, which here went to zero, and the section-hit metric, which must be re-based. The
answering run (O5) has not been repeated over `orb-c256`; it would be a new variant, about
$0.07 per 100 development cases, registered before the frozen split is scored.

Commands, in order (`.../copilot_orb_c256` is the clone's URL):

```text
psql -U copilot -d postgres -c "CREATE DATABASE copilot_orb_c256 TEMPLATE copilot_orb"
eval orb-load --database-url .../copilot_orb_c256 --parsing configs/parsing-orb-c256.yaml --reparse
corpus export --run orb-c256 --out orb-c256 --database-url .../copilot_orb_c256
corpus validate --manifest ${DATA_DIR}/exports/orb-c256/manifest.json
search build-index --manifest ${DATA_DIR}/exports/orb-c256/manifest.json --release orb-c256 --collections papers,chunks --database-url .../copilot_orb_c256
search validate-index --release orb-c256 --database-url .../copilot_orb_c256
eval retrieval --config configs/experiments/orb-c256-retrieval.yaml --split retrieval --database-url .../copilot_orb_c256 --out reports/retrieval/orb-c256
eval orb-sections --run reports/retrieval/orb-c256 --database-url .../copilot_orb_c256
```

## Why answering trailed haiku.rag, investigated (2026-10-09/10)

**Question.** Development accuracy read 0.77 / 0.76 against haiku.rag's published 0.92–0.97
on ORB. Where are the points lost, and what of haiku.rag's design closes the gap? Read from
haiku.rag 0.95.0's source: its judge rubric (`evaluations/evaluators/judge.py`), its search
(`client/search.py`, `store/repositories/chunk.py`), context expansion (`context.py`) and
the RAG agent's instructions (`capabilities/instructions/rag.md`). Everything below is on
the 100 development questions; the frozen 400 were not touched.

**1. Where the shipped run lost its 23 answers** (`flat`, judge of record):

| Stage | Questions | Correct |
|---|---|---|
| Gold paper outside the first stage's top 10 | 16 | 2 |
| Gold paper inside it | 84 | 75 (89%) |

Inside the top 10, the evidence was not the problem: re-retrieving each judged case's evidence
(the retriever is deterministic) and intersecting it with the gold section's chunks, the gold
section reached the generator in 51 of the 57 such cases whose section maps onto our chunks,
and the gold paper filled its three-piece cap in every case it was present. Of the 9 misses
with the paper in hand, 1 abstained and 8 were judged wrong, six of them on an omitted
qualifier (the pattern both judges split on in the calibration above).

**2. The judges' rubrics differ.** haiku.rag's rubric tells its judge not to penalise a
generated answer when "the expected answer is incomplete or narrow" and to "be tolerant of
different levels of detail if the core answer is preserved"; ours marks an answer that
"omits something that changes its meaning" not equivalent, and forbids the judge its own
knowledge of the subject. `prompts/judge/answer-equivalence-haiku-v1.yaml` is their rubric
verbatim (MIT, attributed) inside our reason-first output contract, and `eval qa-rejudge
--prompt` lets the judge of record re-read its own answers under it. Across the six runs
below, 38 verdicts moved to equivalent and 2 the other way; one of those two faulted a
correct answer from the judge's own knowledge, which their rubric does not forbid. κ between
the two rubrics on the same answers: 0.50–0.73.

**3. The first stage is the gap.** The answering run took its papers from the shipped search,
whose reranker reads titles and abstracts. The chunk-reading reranker of the 2026-10-08 read
puts the gold paper in the top 10 for 99 of the 100 questions (15 of the 16 misses), at either
chunk size. Two experiments, pre-registered as development reads with no decision block,
changed only the first stage (same generator, judge, prompts, evidence configuration and
variants), through a new `search_overrides` block in the QA experiment, parsed by the
retrieval harness's own rule and recorded in the manifest:

- `configs/experiments/orb-c256-qa.yaml`: release `orb-c256` (256-token chunks), reranker
  reads title and chunk, shipped 1.5 s deadline. Run `orb-c256-qa-development-20261009T091750Z`.
- `configs/experiments/orb-text-qa-chunk-rerank.yaml`: release `orb-v1` (800-token chunks),
  reranker reads title and chunk, 2.5 s rerank deadline (800-token pairs rerank at p95
  1.3–1.5 s). Run `orb-text-qa-chunk-rerank-development-20261010T075451Z`. A first attempt
  (`…20261009T094353Z`) was cut off by a laptop shutdown after its `flat` variant and wrote
  nothing; the rerun served `flat` from the response cache (same evidence, same request
  digests) and paid only for `section`.

Every run read exactly the first stage its retrieval run measured: each case's recorded top 10
equals that query's ranking in `reports/retrieval/orb-rerank-text/` or `…/orb-c256/` for 100 of
100 questions, so no search degraded. The two new runs waited behind an idle-GPU gate; the
second ran with the Windows-side desktop spiking the GPU for a second at a time (median 4%
over 30 s, no Linux compute process), which the runner's own rule does not count as busy.

| First stage | Chunks | Gold in top 10 | `flat`, our rubric | `flat`, haiku.rag's | `section`, our rubric | `section`, haiku.rag's | Cited MAP | Abstained |
|---|---|---|---|---|---|---|---|---|
| Shipped: reranker reads abstracts | 800 | 84 | 0.77 [0.68, 0.84] | 0.81 | 0.76 [0.67, 0.83] | 0.82 | 0.83 / 0.84 | 11 / 8 |
| Reranker reads title + chunk, 1.5 s | 256 | 99 | 0.79 [0.70, 0.86] | 0.87 | 0.84 [0.76, 0.90] | 0.90 | 0.955 / 0.97 | 2 / 2 |
| **Reranker reads title + chunk, 2.5 s** | 800 | 99 | **0.87** [0.79, 0.92] | **0.92** | **0.87** [0.79, 0.92] | **0.94** | 0.98 / 0.98 | 1 / 2 |

Wilson intervals; haiku.rag-rubric accuracy is the judge of record re-reading the same answers.

Paired on the same 100 questions, exact McNemar on the judge of record's verdicts, sign test
on cited MAP:

| Comparison | Accuracy | Right only in the first / only in the second | McNemar p | Cited MAP |
|---|---|---|---|---|
| Title + chunk at 800 vs shipped, `flat` | 0.77 → 0.87 | 13 / 3 | **0.021** | 0.83 → 0.98, 15–0, p 0.0001 |
| Title + chunk at 800 vs shipped, `section` | 0.76 → 0.87 | 12 / 1 | **0.003** | 0.84 → 0.98, 14–0, p 0.0001 |
| 800-token vs 256-token chunks, `flat` | 0.79 → 0.87 | 11 / 3 | 0.057 | 0.955 → 0.98, 3–0 |
| 800-token vs 256-token chunks, `section` | 0.84 → 0.87 | 7 / 4 | 0.55 | 0.97 → 0.98, 1–0 |
| `section` vs `flat`, title + chunk at 800 | 0.87 → 0.87 | 3 / 3 | 1.00 | 0.98 → 0.98 |

Findings, a read of the development split and no decision:

1. **The first stage was the gap.** Reading the chunk instead of the abstract lifts accuracy
   by 10–11 points and cited MAP from 0.83 to 0.98, the largest change any setting made, and
   McNemar rejects at p < 0.05 for both variants.
2. **Small chunks find the paper and starve the generator.** With 256-token chunks the gold
   paper is found as often (99 of 100), but on the 84 questions the shipped run already had the
   paper for, `flat` falls from 75 to 66 correct and its answers shorten (median 29.5 words
   against 39), dropping a clause or a second conclusion the reference holds. Section
   expansion recovers part of it (`section` 0.84), not all. haiku.rag pairs small chunks with
   expansion to 5,000 characters; our expansion shares one 6,000-token budget across 12 pieces.
3. **Given the paper, our pipeline is at haiku.rag's level.** With the chunk-reading reranker
   over 800-token chunks, 87 of the 99 questions with the gold paper in the top 10 are answered
   correctly under our rubric and 92–94 under haiku.rag's own, against their published
   0.92–0.97 over all 3,045 queries with 26–30B generators. Their agent loop (up to five
   searches), larger models and Docling are not where the points were: one question abstained
   with the paper in hand.
4. **The remaining difference in the headline number is the rubric**, about 5–7 points on the
   same answers. Which rubric the frozen run reports under is a methodology choice for the
   developer; both are recorded here.
5. **Serving cost.** The winning configuration needs a 2.5 s rerank deadline at 800 tokens; the
   shipped deadline is 1.5 s, which only 256-token pairs fit. Ranking on small chunks and
   handing the generator their 800-token parents would keep both properties; it is not built
   or measured.

Costs from the ledger (`llm spend --day`): 2026-10-09 $0.299 in all (the `orb-c256-qa` run
with its two re-judges $0.115, the interrupted first attempt $0.070, the `gpt-6-sol`
calibration $0.100, the rubric re-judges of the shipped run $0.013); 2026-10-10 $0.045 (the
rerun's `section` variant and its two re-judges; `flat` came from the cache). One call of the
interrupted attempt stayed `reserved` in the ledger, counted against 2026-10-09's cap only.
Per-case text and verdict sheets are under `${DATA_DIR}/eval/qa/`; the repository holds the
text-free `cases.parquet`, metrics and `second-judge.*.json` summaries.

What it means for the frozen 400: the pre-registered `orb-text-qa` rule compares `section`
with `flat` under the shipped first stage, which this read shows to be the weak part. Scoring
the frozen sample under the chunk-reading first stage instead needs that configuration
registered before any frozen scoring, and the rubric chosen; both are the developer's call.

```text
eval qa-rejudge --run <run>.flat --judge openai-gpt-6-luna-low --prompt prompts/judge/answer-equivalence-haiku-v1.yaml --out ${DATA_DIR}/eval/qa/labels/<run>.flat.rejudge-luna-haiku-rubric.jsonl --max-spend-usd 0.50
eval qa --config configs/experiments/orb-c256-qa.yaml --split development --database-url .../copilot_orb_c256 --out reports/qa/orb-c256 --max-spend-usd 1.50
eval qa --config configs/experiments/orb-text-qa-chunk-rerank.yaml --split development --database-url .../copilot_orb --out reports/qa/orb-text-chunk-rerank --max-spend-usd 1.50
eval pair --treated <run A>/cases.parquet --baseline <run B>/cases.parquet
```

### The frozen-run configuration, registered (2026-10-10)

The developer chose haiku.rag's rubric as the judge of record. The frozen 400 are scored
under `configs/experiments/orb-text-qa-v2.yaml`: the first stage's reranker reads title and
chunk at a 2.5 s rerank deadline, the judge reads `answer-equivalence-haiku-v1`, everything
else is v1's, the decision rule is unchanged, and the frozen answers are re-read under the v1
rubric as a pre-registered secondary. Registered and committed before any frozen scoring.

Development check of the exact configuration, run `orb-text-qa-v2-development-20261010T084915Z`:
`flat` 0.92, `section` 0.94, cited MAP 0.98 / 0.98, abstentions 1 / 2, every generation and
judge call served from the response cache ($0.00), the first stage identical to the retrieval
run's for 100 of 100 questions (GPU median 18–19% before it, from the Windows-side desktop, no
Linux compute process). The same figures as the rubric re-judge above, as they must be.

Calibration under the new rubric, the second judge (`openai-gpt-6-sol`) reading the same 99
judged `flat` answers with the same rubric: **agreement 98.0% (97 of 99), Cohen's κ 0.846**,
above the 0.6 floor; confusion 91 / 1 / 1 / 6, one disagreement each way; 0 judge errors;
$0.136. Model-model agreement, weaker evidence than hand labels, as before.

### The frozen 400, `flat` (2026-10-10)

Run `orb-text-qa-v2-frozen-20261010T090216Z` (`reports/qa/orb-text-v2/frozen/`), the first and
only scoring of the frozen sample under `orb-text-qa-v2`, variant `flat`, registered in
`reports/qa/orb-text-v2/frozen-scorings.json`. GPU 3% busy before it (p50 over 5 s, no Linux
compute process), 41 minutes, 0 failures, 0 judge errors, 1 repair.

| | `flat`, frozen 400 |
|---|---|
| **Accuracy, haiku.rag's rubric (judge of record)** | **0.88** [0.84, 0.91] Wilson; [0.84, 0.91] over 269 families; 352 of 400 |
| Accuracy, v1 rubric (pre-registered secondary, same judge re-reading) | 0.83 [0.79, 0.86]; 332 of 400 |
| By type, judge of record | extractive 0.87, abstractive 0.89 (200 each) |
| Gold paper in the first stage's top 10 | 392 of 400 |
| Correct given the gold paper | 351 of 392 (89.5%) |
| Abstained | 10 (6 with the gold paper in hand) |
| Cited MAP | 0.959 |
| Provenance validity | 0.9975 |
| Cost (ledger) | $0.269 answering and judging, $0.029 the secondary re-read |

The two rubrics disagreed on 28 of the 390 judged answers, 24 of them equivalent under
haiku.rag's and not under v1's. The first stage matched the retrieval run's ranking for 399 of
400 questions at rank 1 to 10 and for all 400 at rank 1; the one exception differs only at
ranks 8–10, with the gold paper present in both lists and the answer judged correct, which
looks like near-tied reranker scores reordering at the tail rather than a fallback (a degraded
rerank returns the fusion order for the whole list).

Development read 0.92 for this variant on 100 questions; the frozen 400 read 0.88, inside the
development interval ([0.85, 0.96]) and the frozen one overlapping it. Beside haiku.rag's
published 0.92–0.97 (all 3,045 queries including the multimodal ones, their 26–30B generators
and their own judge model) this is close but not level: the remaining errors are mostly
answers with the gold paper in hand (41 of 48), which the second half of this section reads
once `section` is scored. `section` is scored next, under the same registration; then
`eval pair` under the pre-registered rule.
