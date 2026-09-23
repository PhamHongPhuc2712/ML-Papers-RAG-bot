# Corpus rebuild under `paragraph-pack-v1`

The whole corpus re-parsed from its source PDFs and re-chunked under the paragraph-aware
policy adopted on 2026-09-17, into a fresh `copilot_v2` database. The previous corpus,
chunked with `fixed-window-v2`, stays intact in `copilot`.

Run: `corpus run --database-url …copilot_v2 --state ${DATA_DIR}/runs/venues-v2-state.json
--workers 8`, started 2026-09-18T02:49Z, completed 2026-09-22T16:41Z.

## Result

| | |
|---|---|
| Venue-years | **42 / 42** |
| Papers | **85,729** |
| Chunks | **3,426,221** |
| Documents with chunks | 81,929 |
| Chunker version | `paragraph-pack-v1`, single value across every row |
| Failed download jobs | 7 (0.008%) |

Chunk counts fell from ~4.45 M under `fixed-window-v2` to 3.43 M — a 23% reduction, in
line with the 16% predicted on the 40-paper comparison set and larger because that set was
ACL-only.

## Verification

| Check | Result |
|---|---|
| Max token count | **1,200** — the configured ceiling, exactly |
| Chunks above the ceiling | **0** of 3,426,221 |
| Ordinals start at 0 | 81,929 of 81,929 documents |
| Non-contiguous ordinal ranges | **0** |
| Duplicate ordinals within a document | **0** |
| Largest single document | 704 chunks |
| Mean / median tokens | 467 / 447 |
| Chunks per paper | 41.8 mean |

The 704-chunk maximum is the check that mattered most. The overlap defect fixed in
`e74c64c` produced a 7,975-chunk document from a 254 KB paper by carrying each chunk
forward whole; nothing resembling it survives in 3.4 million chunks.

Mid-sentence endings, measured as chunk text not ending in sentence-final punctuation:

| Kind | Chunks | Ends mid-sentence |
|---|---|---|
| body | 1,567,696 | **15.2%** |
| references | 716,634 | 19.7% |
| appendix | 304,171 | 13.4% |
| figure | 261,944 | 12.5% |
| table | 260,256 | 13.1% |

Body prose at 15.2% against the 14.4% measured on 40 papers before adoption, and against
`fixed-window-v2`'s 47.2%. The prediction held at roughly 2,000× the sample size.

## The gap: two venue-years are incomplete

3,802 papers carry no chunks. They are not spread across the corpus:

| Venue-year | Papers without text | Of | Share |
|---|---|---|---|
| NeurIPS 2025 | 2,027 | 5,286 | **38.3%** |
| ICLR 2026 | 1,736 | 5,351 | **32.4%** |
| All other 40 venue-years | 39 | 75,092 | 0.05% |

**3,759 of the 3,802 are zero-byte PDFs, and they are zero bytes at the source.** Checked
against `GenAI4ELab/papercli-papers-neurips` at revision `21153c96710fbbc0c5`:
`get_hf_file_metadata` reports `size: 0` for the affected members, and the index records
their checksum as `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` —
the SHA-256 of the empty string. The mirror published placeholders for the two newest
venue-years, presumably captured before those proceedings were final.

The pipeline downloaded them, checksummed them, failed to parse them and kept them for
retry, which is correct behaviour — but it typed them as parse failures rather than as a
distinct "source served an empty file" class, which is why the pattern was only visible
once the run finished.

Recovery through the recorded `pdf_url` does not work: all three sampled OpenReview URLs
return **HTTP 403 with an HTML block page**, the same `ChallengeRequiredError` behaviour
that made the mirror necessary. Recovering these papers needs either a repopulated
upstream shard or a different source for those two venue-years.

## Interruptions

The run stopped twice and resumed from `venues-v2-state.json` both times.

| When | Cause | Cost |
|---|---|---|
| 2026-09-18 18:03 → 2026-09-19 19:46 | Windows host suspended the WSL VM; no traceback, 24 hours with no job processed, then Postgres exited with the VM | ~2 days |
| 2026-09-21 16:09 | `HfHubHTTPError: 503` from `HfApi().dataset_info()` for `papercli-papers-cvpr`, propagated through `run_campaign` | ~18 hours |

The second was a defect and is fixed in `e10ec83`: the revision lookup now retries five
times with exponential backoff. The first is an operating-environment property, not a code
path — a run of this length needs the host kept awake.

Throughput when running: 51–104 papers/minute depending on venue weight, 0.4–1.7 seconds
per paper. Roughly 20 hours of actual compute spread across four days.

## Commands

```bash
uv run --env-file .env --project backend python -m copilot.cli corpus run \
  --database-url postgresql+psycopg://copilot:***@localhost:5432/copilot_v2 \
  --state ${DATA_DIR}/runs/venues-v2-state.json --workers 8
```

Verification queries ran against `copilot_v2` directly; each table above is a single
aggregate over `chunks` joined to `paper_versions`, with no sampling.
