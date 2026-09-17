# Chunking policies compared — fixed window vs paragraph packing

Date: 2026-09-17
Sample: 40 real ACL papers, seed 2026, parsed with `pypdf-text-v2`, counted in
the pinned BGE-M3 tokenizer. Reproduce with:

```powershell
uv run --project backend python -m copilot.cli corpus compare-chunkers `
  --source ${DATA_DIR}/sources/papercli/pdfs/pdfs/acl --sample 40
```

## The two policies

**`fixed-window`** (in production): split each section into windows of
`target_tokens` with `overlap_tokens` repeated from the previous window, capped
at `hard_cap_tokens`. Currently 450 / 60 / 600.

**`paragraph-pack`** (new): split each section into paragraphs, then add whole
paragraphs to a chunk until it reaches `target_tokens` or the next paragraph
would carry it past `max_tokens`. A paragraph larger than
`paragraph_max_tokens` is split on sentence boundaries; text with no sentence
boundary at all is cut on the token window as a last resort. Each chunk repeats
the previous chunk's last `overlap_sentences` sentences, bounded to a quarter of
the budget. Defaults 800 / 900 / 1200 / 2.

Both keep windows inside a section, return the same chunk shape and use the same
chunk-identity scheme, so the persistence layer is indifferent to the choice.

## Measured

| | fixed-window | paragraph-pack |
|---|---:|---:|
| Chunks per paper | 47.5 | **40.0** |
| Token median (all chunks) | 372 | 308 |
| Token mean | 293 | 354 |
| Token p90 | 450 | 745 |
| Token max | 450 | 1,200 |
| Chunks over the policy ceiling | 0 | **0** |
| Prose chunks | 1,103 | 924 |
| Prose token median | 363 | 345 |
| **Prose chunks ending mid-sentence** | **47.2%** | **14.4%** |
| **Prose chunks starting mid-sentence** | **35.9%** | **10.4%** |

Paragraph detection found a median of 39 paragraphs per paper.

## Reading it

The headline is where the policies cut. The fixed window ends **roughly half**
its prose chunks in the middle of a sentence and begins a third of them there;
paragraph packing cuts that to about a seventh and a tenth. For evidence-grounded
answering that is the difference between quoting a claim and quoting half of one.

It also produces **16% fewer chunks**, which is less to embed and less index to
hold — on the full corpus that would be roughly 600k fewer chunk vectors.

The token distributions say something subtler. Median chunk size *falls* (372 →
308) while the mean *rises* (293 → 354) and p90 nearly doubles (450 → 745).
Packing does not make every chunk bigger; it makes prose chunks bigger and
leaves short sections — captions, table blocks, acknowledgements — exactly as
short as they were. Neither policy merges across a section, so a 30-token
caption stays a 30-token chunk under both.

## Two limitations worth stating

**Paragraphs are inferred, not read.** pypdf's extracted text contains no blank
lines whatsoever, so there is no paragraph delimiter to parse. The detector uses
the typographic signal that survives: a paragraph's last line both ends a
sentence and falls short of the column width. That is why 14.4% of chunks still
end mid-sentence rather than ~0% — a wrapped line that happens to end with a
period and fall short is a false boundary. A parser that preserved layout would
lift this further, and it would also fix the page-span precision limitation
recorded against the current corpus.

**The ceiling is enforced, but it took two attempts.** Measuring on real papers
caught what the unit tests did not: reference lists contain "sentences" of 700+
tokens, and carrying two of them as overlap started the next chunk already over
budget — one chunk reached 3,052 tokens against a 1,200 ceiling. The overlap
carry is now bounded to a quarter of the budget, and every emitted slice is
re-measured after cutting, because a slice re-tokenizes a token or two higher
than the window it came from. The comparison above reports 0 chunks over the
ceiling for both policies.

## Status

`paragraph-pack` is implemented, tested (18 unit tests) and selectable through
`configs/parsing.yaml`, but **the production policy is unchanged**: the corpus
is still built with `fixed-window`. Switching would change `chunker_version` and
therefore every chunk ID, which means re-chunking — and re-chunking needs the
PDFs, which the venue sweep deletes. So adopting it is a decision to make
deliberately, ideally at the same time as any other re-parse.
