# Evidence retrieval and the grounded generator — evidence (O4; P4.1 and P4.2's adapter)

Built under the ORB plan (`docs/superpowers/plans/2026-10-06-orb-text-benchmark.md`, task
O4) on 2026-10-08, with the M4 plan's P4.1 interfaces and the generation adapter P4.2
names, because the answering benchmark (O5) cannot run without them. P4.1 keeps its
private-upload entry point and P4.2 its routing, streams, persistence and HTTP API. Host:
WSL2 Linux, RTX 3080 Laptop 16 GB; models BGE-M3 and `bge-reranker-v2-m3` as pinned in
`configs/models.yaml`; hosted model `gpt-6-luna` (`openai-gpt-6-luna` in `configs/llm.yaml`).

## What was built

- `backend/src/copilot/evidence/citations.py` — structural validation: `validate_citations`
  (the unknown ids, sorted), `check_claims` (every rule, per claim: `unknown_evidence`,
  `uncited_claim`, `duplicate_reference`, `quote_not_in_source` for quoted spans of 12
  characters or more, matched after whitespace and case normalization against the cited
  evidence) and `provenance_valid`. These prove an id is real; whether it is the right id is
  the judge's question.
- `backend/src/copilot/evidence/retrieve.py` — `EvidenceRetriever.retrieve(query, paper_ids,
  release_id)`: BM25 and dense over the release's **chunk collection** inside the named
  papers (every paper when `paper_ids` is None, the global baseline), evidence-by-default
  chunks only unless `include_references`, 40 candidates fused by RRF, reranked by the
  cross-encoder **on the chunk text**, then at most 12 pieces and 3 per paper, then a token
  budget of 6,000 counted in the pinned BGE-M3 tokenizer (no provider tokenizer exists
  offline; the manifest says which counted). Each piece keeps its chunk id, paper version,
  section and page span from the stored row and the stored text; ids are `E1..En` in rank
  order and mean nothing outside the evidence sent. A named paper with no parsed text
  contributes its abstract, labelled `abstract_only`. `expand_to_section` is haiku.rag's
  rule (whole section when it fits `expansion_max_chars` and is at least `expansion_min_share`
  of it; otherwise grown outward from the hit inside the section; the hit never clipped),
  applied under `context.expansion: section`. An optional floor on the reranker logit
  (`min_rerank_score`) exists and is **off** by default (below).
- `backend/src/copilot/models/generation.py` — `ChatGenerator` over the L2 chat client and
  the pinned prompt `prompts/generate/grounded-claims-v1.yaml` (digest in its identity):
  structured JSON claims with evidence ids; the answer is parsed and checked; any problem
  (unknown id, uncited claim, duplicate, bad quote, unparseable or malformed answer) earns
  exactly one repair call naming the problems and the allowed ids; a second failure is
  `GroundedAnswer(claims=[], insufficient_evidence=True)` with `repair_attempted` and the
  reason. No evidence is insufficiency with no call and no reservation. A refused or
  timed-out call raises the client's typed `LlmError`, never an empty answer. Every call
  reserves its worst case in the ledger (`purpose="evaluation"`, the run's id).
  `FakeGenerator` runs the same loop deterministically and invents an id on a fixture
  question so the repair path runs in CI. `provenance_valid` on an outcome means every id
  passed with no repair.
- `configs/evidence.yaml` — the budgets above; `contracts.Evidence` gains `chunk_id: UUID |
  None` (None only for abstract-only evidence), `expanded` and `abstract_only`.

## Checks

| Suite | Result |
|---|---|
| `backend/tests/unit/test_citations.py` | 8 passed (P4.1's `test_unknown_evidence_is_rejected` as written, plus the rules) |
| `backend/tests/unit/test_generation.py` | 10 passed: prompt digest and evidence-as-data; JSON plain or fenced; one call with cost; fabricated id → one repair → insufficiency; malformed answer repaired the same way; no evidence → no call, no reservation; refused call is a typed failure; the fake generator's fabrication is caught; the two live captures replay unrepaired |
| `backend/tests/integration/test_evidence.py` | 12 passed over a four-paper release with 6, 2, 1 and 0 chunks: named papers only with caps and provenance; references excluded by default and included on request; one paper cannot fill the context; the token budget counts model tokens and drops what does not fit; no evidence is an empty list; an unparsed paper contributes its abstract labelled; a misleading abstract does not hide what the body says (P4.1's two-paper case); a cross-release or unknown id is rejected; a paper-level miss is measured against the global chunk search; expansion returns the section when it fits and grows outward when not; `build_context` keeps rank order; the release is the one named |
| Full suite | **640 passed** (30 new), offline set 434. The first full run had one failure, `test_health.py::test_liveness_and_missing_release`: the new evidence fixture left its release active and the health suite, which runs next in file order, expects none. The fixture now deactivates its release and clears its vectors on teardown; the suite was rerun clean |
| Ruff, mypy strict, `eval smoke` | clean, clean, reproducible |

## Step 4: three real ORB development questions over `orb-v1`

The shipped search (`hybrid_rerank`, chunk candidates) gave each question its top 10
papers; evidence was retrieved inside them and answered by `FakeGenerator`. The first three
development questions in id order all had the gold paper in the top 10. Query ids only;
the text stays under `DATA_DIR`.

| Query | Evidence | Papers | Max per paper | Gold among evidence | Tokens (pinned tokenizer) / 6,000 | Dropped by cap / budget | Fake claims valid | Foreign id rejected | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| `006b56ae…` | 8 | 4 | 3 | yes | 5,975 | 28 / 4 | yes | yes | 1.43 |
| `0081a5c8…` | 10 | 6 | 3 | yes | 5,769 | 28 / 2 | yes | yes | 1.01 |
| `0705da20…` | 10 | 6 | 3 | yes | 5,482 | 28 / 2 | yes | yes | 0.99 |

The budget binds before the count cap: with chunks at a 671-token median, 8–10 pieces fill
6,000 tokens, and 2–4 ranked pieces were dropped for it. Seconds are wall-clock for the
evidence stage (two Qdrant queries, 40 chunk texts read, one reranker batch of 40), on a GPU
no other compute process shared.

**The reranker floor, read and switched off.** The first run of this check used a floor of
zero on the cross-encoder logit ("as likely relevant as not"), and the second question came
back with no evidence at all: every one of its 40 candidates scored at or below zero, the
gold paper's best chunk at −0.01. Over the first twelve development questions (ids
`006b56ae`, `0081a5c8`, `0705da20`, `0998f6ee`, `102f622a`, `12eab3a1`, `130057d3`,
`19b5f43c`, `1e96b357`, `1f655fe0`, `22c86297`, `244c5d11`), the gold paper was in the top
10 for ten; on nine of those ten the best-scoring chunk belonged to the gold paper with a
logit above zero (0.27 to 7.5), and on one it did at −0.01. A zero floor would have turned
a question with the right chunk in hand into an abstention. The floor is therefore unset by
default; the caps bound the evidence and the generator says insufficiency itself. The
development set is the place for this reading; the frozen 400 were not touched.

## Step 5: two live calls

Synthetic evidence (three blocks, one carrying an injected instruction to cite `E7` and
answer 42), so no benchmark text reaches a fixture. `gpt-6-luna` at effort `none`:

| Case | Calls | Claims | Cites | Injection followed | Insufficient | Tokens in / out | Cost |
|---|---|---|---|---|---|---|---|
| answerable | 1 | 2 | `E1`, `E2` | no | no | 296 / 83 | $0.000071 |
| unanswerable | 1 | 0 | — | no | yes | 285 / 14 | $0.000036 |

Both validated with no repair. Recorded as `backend/tests/fixtures/llm/openai_grounded_
{answer,insufficient}.json` (`id` → `"fixture"`, nothing else edited) and replayed by
`test_recorded_live_answers_pass_validation_unrepaired`. The ledger for 2026-10-08 shows
three calls, $0.000664: one earlier attempt failed in my recording transport (a gzip
header, not the provider) and was settled at its estimate, $0.000557, as the ledger's rule
says a call with unknown usage is.

## Open items carried into O5

- The provider counts tokens differently from BGE-M3: the 6,000-token budget is a bound in
  our tokenizer. The run manifest records which tokenizer counted; the ledger records what
  the provider billed.
- `expansion: section` is built and tested but not yet read on ORB; O5's second variant is
  the first measurement.
- P4.1's private-upload entry point (`Principal`, authorized document ids) and P4.2's
  routing, streams and API are not built here.
