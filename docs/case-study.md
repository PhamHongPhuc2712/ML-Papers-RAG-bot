# Case study: search and cited answers over 85,729 ML papers

Every number here is copied from the report it links to. The decision rules were committed
before the runs they decided, and the frozen and locked test splits were each scored once.

## The problem

A researcher wants two things from the ML literature:
- the papers about a topic;
- an answer to a question, with each claim traced to the passage that supports it.

The constraints shaped every choice:
- **Infrastructure:** zero budget. PostgreSQL, Qdrant and the models run on one laptop, an RTX
  3080 with 16 GB shared with other workloads.
- **The one paid resource:** a hosted LLM, under a daily cap of $3.
- **The corpus:** 85,729 accepted main-conference papers from 42 venue-years, parsed from PDFs
  into 3,426,221 chunks (`docs/data-card.md`).

The product is modelled on Consensus and haiku.rag (spec §1). It has no recommendations,
uploads or agent: those were removed on 2026-10-11 to finish search and grounded chat well.

## Decisions, and the evidence for each

### Find candidates through chunks, not abstracts (P2.6)

At first both search branches searched title-and-abstract vectors. Many misses described
something only a paper's body says. Candidates found through the chunk collection, with each
paper scored by its best evidence chunk, were registered and promoted on validation on
2026-10-06. That evidence was LitSearch-based and was removed with LitSearch on 2026-10-14
(git history keeps it). The decision stands on Open RAG Bench, which reproduced it: chunk-level
candidates beat title-and-abstract ones by +0.22 to +0.25 nDCG@10 (`reports/orb-text.md`).

### What the cross-encoder reads: title + chunk, the one first stage since 2026-10-15

On Open RAG Bench's section-level questions, the reranker reading titles and abstracts *lost*
0.163 nDCG@10 against plain fusion (`hybrid_rerank` 0.785, `hybrid` 0.948). Reading the title
plus the chunk that found the paper gained **+0.183** [+0.163, +0.204] over the abstract reader
(`reports/orb-text.md`).

- **The chat** answers passage-level questions, so it reads title + chunk at a 2.5 s rerank
  deadline. That is the first stage of the frozen answering result.
- **The search page** was decided separately, on our corpus (S2). The candidate gained
  **+0.348** [+0.292, +0.408] nDCG@10 on S1's 190 model-written passage questions, but failed
  the registered rule's paper-level guard, which was measured on LitSearch's 52 in-domain
  queries and removed with them. So the search page kept reading **abstracts**
  (`reports/m5-rerank-text.md`) until the search page itself was removed.
- **One stage (U1, 2026-10-15).** With RAG only, the chat's setting became the only one. U4
  measured its cost on topic requests: on S3's 349 model-written "find papers" queries the
  abstract reader ranks the source paper higher (nDCG@10 0.663 against 0.563), while on S1's
  passage questions title + chunk wins by 0.348. Each set was written from the text one reader
  reads (S3 from abstracts), so both gaps are uncertain in size; a judged measurement is the
  open follow-up (`reports/u4-single-stage.md`).

### The answer judge's rubric

On the same development answers, our first rubric and haiku.rag's differ by about 5–7 points.
haiku.rag's rubric tolerates a narrower or broader answer that keeps the core point, while ours
marked an omitted qualifier wrong. κ between the two rubrics on the same answers is 0.50–0.73.
The developer chose haiku.rag's rubric, verbatim and attributed, as the judge of record for
the frozen run, so the headline is comparable with theirs.

Calibration used a second model judge rather than hand labels, again the developer's
decision: agreement 95.5% (85 of 89), Cohen's κ 0.807, above the 0.6 floor
(`reports/orb-text.md`). Frozen 400, `flat`: **0.880** [0.844, 0.908].

### Structured generation (C4)

The chat generates through LangChain's `ChatOpenAI` with a strict JSON schema, behind the spend
ledger (`StructuredGenerator`). It was registered against v2's own HTTP client on the same
frozen 400 cases: **0.884** [0.849, 0.912] against 0.880, exact McNemar p = 0.824. Cited MAP
was 0.965 against 0.959. None of the three rejection conditions held, so it was adopted
(`reports/orb-text.md`).

### Condensing follow-ups (C7)

Should a follow-up like "and why is it significant?" be rewritten into a standalone query
before searching? The registered rule kept condensing only if it beat searching the follow-up
as typed at p < 0.05. On 43 frozen conversations:
- condensed **0.791** [0.648, 0.886]; raw 0.674 [0.525, 0.795]; standalone ceiling 0.930;
- 8 cases only condensed got right, 3 only raw, p = 0.227.

So the rule turned condensing off: a rule outcome on an underpowered set, not evidence that
condensing does not help. Every secondary measure favoured it (`reports/chat-followups.md`). On
2026-10-14 the developer turned it back on as a recorded decision. With it off, a follow-up like
"How large was its batch size?" was searched without its subject and came back "not enough
evidence"; with it on, the same follow-up was searched as "How large was SimCSE's batch size?"
and answered with a citation.

### The turn pipeline

The developer asked for LangChain where it fits. On 2026-10-22, the linear LangGraph turn
and LangChain retriever adapter were replaced with three explicit Python steps: condense →
retrieve → generate. Structured `ChatOpenAI` generation remains. Each step's timing is stored with the
message; earlier turns showed retrieval dominating at 2.6–8.8 s (`reports/m3-chat.md`). Turning
condensing off, and on again, was a configuration change, not a code change. Generation stays a single
structured call. There is no agent loop: one search per turn was the developer's answer, and
`reports/orb-text.md` found that given the right paper, our pipeline answers at haiku.rag's
level without one.

## What failed and was set aside

- **Paper-level reranking for passage questions.** The abstract-reading reranker was the
  answering gap. Under it, the gold paper was outside the top 10 for 16 of 100 ORB
  development questions, and only 2 of those were answered. Given the paper, 75 of 84 were
  right. Reading the chunk put the gold paper in the top 10 for 99 of 100 and lifted
  development accuracy from 0.77 to 0.87, McNemar p = 0.021 (`reports/orb-text.md`).
- **256-token chunks for answering.** haiku.rag's chunk size found the paper just as often,
  99 of 100, but starved the generator. On the 84 questions the shipped run already had the
  paper for, `flat` fell from 75 to 66 correct, and answers shortened from a median of 39 to
  29.5 words. The live corpus stays at about 800 tokens (`paragraph-pack-v1`).
- **A retry alone did not fix the cold start.** After a Qdrant restart, the first chat turns
  failed even with one retry of the candidate stage. The cause was that the API took requests
  before Qdrant had loaded its collection. A startup warm-up fixed it: after a Qdrant restart,
  startup took 138 s, and both chat turns then completed where before both failed
  (`reports/m5-drills.md`).

## Costs

The hosted-LLM ledger, all time, read from `llm_calls` on 2026-10-14:

| Purpose | Calls | Cost | Dates |
|---|---|---|---|
| `evaluation` (answering runs, judges, question writing) | 8,049 | $2.0384 | 2026-10-02 to 2026-10-12 |
| `chat` (development and drills) | 8 | $0.0041 | 2026-10-11 to 2026-10-14 |

That is $2.04 in total. The single largest item was the frozen 400: $0.2690 for v2 and
$0.2721 for v3 (`reports/orb-text.md`). Everything else is self-hosted and costs nothing to
run: storage, search, embeddings and reranking.

## RAG only (2026-10-15/16)

On 2026-10-15 the developer decided to drop the separate search mode and make every request a
cited-answer turn, as Consensus does: it shows the searches it ran and the papers it read, then
a cited report, with the paper list behind References. The trigger was a measured inconsistency.
The search page and the chat ran one search service with different reranker settings, so a
"find papers" request gave two different lists: only 4/9, 3/10 and 6/8 of the chat's papers were
in the search page's top 10. The chat's first stage became the only configuration (U1). A
structured overview for paper-finding requests kept accuracy on ORB but was not adopted: its
registered hand read found one claim its citation does not support (U2, `reports/u2-overview.md`).
The web app became one page with References (U3), and U4 measured the stage on S1 and on a topic
query set built from abstracts, then walked the real stack (`reports/u4-single-stage.md`). Plan:
`docs/superpowers/plans/2026-10-15-rag-only.md`.

## What a deployment would add

These are deferred, not deleted (spec §2):
- **P5.3's staged deployment** and its backup recovery drill (RPO/RTO);
- **Public rate limiting and per-user quotas** on top of the global daily cap;
- **Row-level security** in PostgreSQL. Ownership is currently enforced and tested in one
  store module;
- **Re-warming search when Qdrant restarts under a running API**, which today meets a cold
  index until it warms.

R1–R3 already built what a deployment would rely on:
- request logs that carry no user text, cost by purpose and `chat stats`;
- account deletion, and backup and restore of user data;
- eight failure drills on the real stack.
