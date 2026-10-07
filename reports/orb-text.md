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
