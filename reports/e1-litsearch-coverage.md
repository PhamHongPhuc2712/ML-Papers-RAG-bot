# E1 — LitSearch pinned, and what it can actually measure

LitSearch (Ajith et al., EMNLP 2024) pinned at revision
`9573fb284a1026c998df47024b888a163f0f0e25`, fetched 2026-09-23 into
`${DATA_DIR}/benchmarks/litsearch` (2.85 GB, 208 s), every file checksummed into
`manifest.json` and `configs/evaluation.yaml`.

Measured against the finished corpus: 85,729 papers in `copilot_v2`.

## The headline

**41.9% of the 597 queries have their entire gold set in our corpus** — 250 covered,
0 partial, 347 missing.

That aggregate is the least useful number in this report. The coverage is not spread
across the benchmark; it is almost binary by query set.

| Query set | Covered | Of | Share | What the queries are |
|---|---:|---:|---:|---|
| `manual_acl` | 155 | 155 | **100.0%** | Author-written, about their own recent paper |
| `manual_iclr` | 88 | 91 | **96.7%** | Author-written, ICLR |
| `inline_nonacl` | 7 | 253 | 2.8% | GPT-4-generated from a citation sentence |
| `inline_acl` | 0 | 98 | 0.0% | GPT-4-generated from a citation sentence |

The split is not a defect in either dataset. An inline query is built from a sentence
that *cites* something, so its gold paper is the cited work — typically older than the
paper citing it, and therefore outside a 2023–2026 window by construction. A manual query
is an author describing the paper they just published, which is exactly what our corpus
holds.

**This supersedes the 68% figure recorded in the benchmarks plan.** That was a sample of
the 150 queries whose gold happened to fall in one of six shards, matched on titles alone
against a corpus that was 47% built. It was biased toward the manual sets, and it was
labelled as a sample. 41.9% is the measurement.

## Consequence for the plan

The plan set a threshold: below roughly 40% coverage, the in-domain slice could not carry
G2. 41.9% clears it, but the structure gives a better answer than the threshold assumed.

**Use the two author-written sets — 243 queries — as the in-domain evaluation slice**,
rather than a 42% scatter across all four. That slice is 99% covered, which means recall
measured on it is bounded by retrieval quality rather than by coverage. The other 354
queries run against LitSearch's own corpus in E3, where they are fully answerable and the
numbers compare with published work.

Two limits to state wherever the in-domain slice is reported:

- **It is a narrower behaviour.** Author-written queries describe a specific known paper.
  Citation-derived queries describe a capability and ask what has it. A system tuned on
  the first is not thereby good at the second.
- **243 queries is a smaller instrument.** Bootstrap intervals will be wider than on the
  full 597, and E3's numbers stay the primary G2 evidence.

## How documents were matched

The plan assumed a strong-ID join. It does not exist: LitSearch carries `externalids`
(`acl` on 59,385 documents, `doi` on 34,891, `arxiv` on 15,697) and **our corpus carries
none of them** — only papercli registry ids (85,732) and OpenReview forum ids (33,121).

The one bridge is the ACL Anthology. 9,127 of our papers record an `aclanthology.org`
PDF URL with the anthology id inside it, which is the same id LitSearch stores.

| | |
|---|---:|
| Matched | 1,962 of 64,183 documents (3.1%) |
| by ACL id | 790 |
| by normalized title | 1,172 |
| Ambiguous titles (resolved to neither) | **0** |

3.1% corpus-wide overlap is expected and unimportant — LitSearch's corpus is mostly the
older work recent papers cite. What matters is that the *gold* papers are covered, which
the table above measures.

### Is the title matcher trustworthy?

Two checks, because title matching is a guess and the evaluation depends on it.

**Agreement where both methods fire: 765 of 765, 100.00%.** Every document with both an
ACL id in our corpus and an unambiguous title match resolved to the same paper by both
routes.

**Hand-checked sample of 50 title-only matches, seed 42: 0 false positives.** 45 pairs are
byte-identical ignoring case. The 5 that differ are PDF-extraction artifacts in the
LitSearch title — `LAYER GRAFTED PRE-TRAINING: BRIDGING CON- TRASTIVE LEARNING`,
`intra-and inter-area` — which normalization removes and which are plainly the same paper.

A normalized title shared by two of our papers resolves to **neither**, and there were
none. Guessing between duplicates would inject a false gold into the evaluation, which is
worse than a smaller slice.

## License: unresolved, so local-use-only

The dataset repo declares no license — `cardData.license` is null and no license tag is
present on the hub. The paper and code repository were not reachable for a stronger answer
at time of writing.

Until that is resolved: the data stays under `DATA_DIR`, never in the repository, and
`redistribution: unknown` keeps it out of every corpus export by the same rule that
governs paper full text.

## Annotation semantics: undocumented here

`specificity` (0 for 155 queries, 1 for 442) and `quality` (1 for 294, 2 for 303) are
described in the dataset card only as "specificity and quality annotations". Their exact
scales are not stated in the artifacts fetched, so **E2 stratifies splits on `query_set`
alone** and carries the two fields through as metadata for slicing rather than using them
to balance anything.

## Reproducing

```bash
uv run --env-file .env --project backend python -c "
from pathlib import Path; import os
from copilot.evaluation.litsearch import fetch_litsearch
fetch_litsearch(Path(os.environ['DATA_DIR']) / 'benchmarks' / 'litsearch')"
```

The fetch refuses any revision that is not a 40-character commit: a benchmark pulled from
`main` cannot be compared with a run from last month, and the failure would be silent.
