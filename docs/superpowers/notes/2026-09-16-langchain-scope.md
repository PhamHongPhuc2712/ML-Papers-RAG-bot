# Note: what LangChain could replace in this project

Date: 2026-09-16. Status: assessment, not a decision to adopt.

## Question

The chat and evidence path (spec §9, plan M4) and the bounded research agent
(plan P6.2) are hand-written. LangChain builds RAG applications, so which parts
of this system could it stand in for, and what would remain project code either
way?

## Pieces LangChain has a generic version of

| Project component | LangChain counterpart | What the counterpart does not do |
| --- | --- | --- |
| RRF fusion in application code (§7: `sum(1/(60+rank))`, dedupe within a list, ties by canonical paper ID, cap 50 to the reranker) | `EnsembleRetriever` (RRF, constant 60, per-retriever weights) | Dedupe within a list, canonical tie-break, rerank cap, single-branch degradation with a warning |
| Exact BM25 as Qdrant sparse vectors, per release | `BM25Retriever` (in-memory `rank_bm25` over a document list) | Persist beyond process memory, scale to millions of chunks, pin a release |
| Dense retrieval against `paper_*_<release>` collections with a request-scoped release pointer | Qdrant vector-store wrapper (dense query, filters, hybrid mode) | Immutable per-release collections, active-release pointer, named-vector layout the project owns |
| Cross-encoder rerank with timeout fallback to RRF order | Contextual-compression rerankers | Timeout-then-fallback behaviour, explicit warning in the response |
| Section-aware chunker, 450 tokens, cap 600, overlap 60, versioned chunker ID | Recursive / token text splitters | Section awareness, chunker version for reproducible reindexing |
| `Generator` Protocol over the hosted model | Chat-model wrappers (uniform provider interface, retries, usage) | Per-request cost logging, daily spend cap (P5.2), deterministic fake for CI |
| Bounded recent conversation loaded from `messages` | Memory classes | Ownership checks, idempotent `request_id`, replay of completed answers |
| Prompt assembly over typed `Evidence` | Prompt templates | Nothing missing; the value is small |
| Parse of model output into typed claims | Output parsers | Structural validation against the authorized evidence set, quote checks, one bounded repair |

Net: LangChain could replace the fusion call, the vector-store query and the
provider adapter with generic components that behave slightly differently from
the spec. Each would need overriding to meet §7 and §9 as written.

## Pieces with no LangChain equivalent

- Paper identity resolution, alias table, merge redirects, DOI/arXiv normalisation (P1.2).
- Hierarchical retrieval: 10 papers first, then chunks inside them, at most 3 per paper (§9).
- Citation validation: IDs must exist in selected evidence and the authorized release or documents; quoted spans checked against stored text; one repair then an insufficient-evidence response (§9, P4.1).
- Streaming contract: status/evidence first, buffered answer until validated, idempotent `request_id`, disconnect handling (§9, P4.2).
- Public/private separation: research mode never silently searches uploads; `user_documents_*` filtered by server-derived `user_id` (§2, §4).
- Recommendations: decayed feedback profile, bounded heuristic score, MMR, reasons from real score components (§8).
- Corpus pipeline: source adapters, leased job queue, hardened download, Parquet export with redistribution eligibility (§4, P1.4–P1.5).
- Cost logging and the operator spend cap (P5.2).

## Assessment

For this system the distinctive work is the retrieval, provenance and
evaluation logic, which LangChain does not provide. Adopting it would swap a
few hundred lines of glue the project controls for a large dependency tree
whose defaults must be overridden, and it would put framework types at the
`contracts.py` boundary that §6 keeps provider-free. Conversation memory does
not need a framework: it is a bounded query over the `messages` table.

Recommendation: no LangChain in M4 or elsewhere. LangGraph remains optional for
the P6.2 agent as a graph runner over project-owned state, budgets and tool
dispatch; a plain loop is sufficient for its six-call, two-search, 30-second
scope, and P6.2 already requires measuring graph versus direct routing before
keeping it.

## If the decision is revisited

Triggers that would change the answer: the agent grows to need interrupts,
human-in-the-loop pauses or resumable runs (LangGraph checkpointing to the
self-hosted Postgres would keep the `DATA_DIR` guarantee); or the project
chooses to demonstrate framework fluency as a portfolio goal, in which case a
separate small demo is the cleaner place for it.
