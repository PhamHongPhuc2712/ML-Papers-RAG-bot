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
