# Evidence Assistant and Multimodal Documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add grounded research chat, private PDF QA and image interpretation with traceable evidence.

**Architecture:** Paper/chunk retrieval supplies typed evidence to generation. Citation validation, bounded routing, persistent stream states and private upload jobs enforce clear source boundaries.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL, Qdrant, Next.js/TypeScript, pytest, Playwright; subsystem additions are specified below.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md). Read the relevant sections and global contracts before implementing.

**Status:** Draft for review. No implementation tasks completed.

## Global Constraints

- Python 3.12; Node.js 22; TypeScript strict mode; UTF-8 files; UTC timestamps.
- Lock Python and JavaScript dependencies; pin container images and model revisions before benchmark or deployment runs.
- Hugging Face is an offline corpus/artifact destination and is never queried during user requests.
- Public corpus data and private user documents use separate Qdrant collections and storage namespaces.
- Derive user identity from a verified token; never trust a client-supplied user_id.
- All benchmark runs record corpus, split, model, configuration, code, hardware, and seed versions.
- No paid cloud resources, public publishing, or application implementation occur while preparing these documents.
- Postgres and Qdrant are self-hosted via Docker on the developer's own machine by default; no managed database, vector, or storage subscription is used unless a future, explicitly-recorded decision changes this.
- All persistent local data lives under one configurable root directory, default `C:\ml-copilot-data\` on the developer's machine, so the entire dataset can be deleted by removing that directory.
- The hosted LLM/multimodal generation API is the one paid resource in the project, funded personally by the developer; its cost tracking and daily spend cap from P5.2 remain required because real personal money is involved.

## Entry checkpoint and working conventions

Entry: the search/library/recommendation alpha works; retrieval reports exist. Read spec §§9–12. Keep generation provider calls behind a typed adapter; ordinary tests are offline. Readiness for public beta requires P5 isolation/load/restore gates too.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P4.1: Retrieve bounded evidence and validate citation provenance

**Depends on:** P2.3, P3.1

**Files:**

- Create `backend/src/copilot/evidence/{retrieve,citations}.py`, `configs/evidence.yaml`.
- Create `backend/tests/unit/test_citations.py`, `backend/tests/integration/test_evidence.py`.

**Interfaces:** `retrieve_evidence(query: str, paper_ids: list[UUID], release_id: str) -> list[Evidence]`; `validate_citations(claims: list[dict], allowed_ids: set[str]) -> list[str]` returns sorted unknown IDs; `build_context(evidence: list[Evidence], token_budget: int, tokenizer) -> list[Evidence]`. Production uses typed Claim models and serialization at the validator boundary.

- [ ] Write the regression test below in `backend/tests/unit/test_citations.py` and add the additional acceptance cases listed for this task.

```python
from copilot.evidence.citations import validate_citations

def test_unknown_evidence_is_rejected():
    claims = [{"text": "A supported claim", "evidence_ids": ["E1"]},
              {"text": "Invented source", "evidence_ids": ["E999"]}]
    assert validate_citations(claims, {"E1", "E2"}) == ["E999"]
    assert validate_citations(claims[:1], {"E1"}) == []
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_citations.py backend/tests/integration/test_evidence.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def validate_citations(claims: list[dict], allowed_ids: set[str]) -> list[str]:
    referenced = {source for claim in claims for source in claim["evidence_ids"]}
    return sorted(referenced - allowed_ids)
```

Implement paper-first retrieval: top 10 papers, up to 40 chunk candidates, rerank, select at most 12/max three per paper and 6,000 generation tokens. Preserve source versions, section/page spans and original text; do not use embedding prefix text as evidence quotes. Match exact quotes to stored source spans and record abstract-only provenance. Treat empty evidence as insufficient. Add direct global chunk retrieval and union fallback as experiment configurations, not unconditional extra requests. Private retrieval has a separate entry point with Principal and authorized document IDs; public function cannot reference user collection names. Validator also requires at least one evidence ID per factual research claim, unique evidence references and valid ownership/source mapping.

- [ ] Verify the additional acceptance cases: unknown IDs; cross-release IDs; unauthorized uploads; absent pages stay null; quotes not in source rejected; references excluded by default; one paper cannot fill all context; token budget counts model tokenizer; no evidence returns insufficiency; paper-level miss compared against global chunk baseline.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_citations.py backend/tests/integration/test_evidence.py -q` again, then run evidence integration against a synthetic two-paper corpus with a known answer and a deliberately misleading abstract. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m4-evidence.md` showing selected chunks, excluded candidates and provenance checks; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: retrieve bounded evidence and validate citation provenance`.

## P4.2: Add controlled routing, generation and interruption-safe streams

**Depends on:** P4.1, P3.1

**Files:**

- Create `backend/src/copilot/chat/{router,service,api}.py`, `backend/src/copilot/models/generation.py`.
- Create `backend/migrations/versions/0004_chat_uploads.py`, `backend/tests/unit/test_router.py`, `backend/tests/integration/test_chat_stream.py`.
- Modify app/contracts/db models and OpenAPI schema export.

**Interfaces:** `choose_mode(requested: str, document_ids: list[str], query: str) -> str`; `Generator.answer` and ChatRequest per spec; `ChatService.stream(principal: Principal, request: ChatRequest) -> AsyncIterator[str]` yields SSE frames. The persisted message state is pending/generating/complete/interrupted/failed.

- [ ] Write the regression test below in `backend/tests/unit/test_router.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.chat.router import choose_mode

def test_explicit_mode_and_document_scope():
    assert choose_mode("general", [], "What is gradient descent?") == "general"
    assert choose_mode("auto", ["doc"], "Explain the method") == "document"
    assert choose_mode("research", [], "recent retrieval papers") == "research"
    with pytest.raises(ValueError, match="document_required"):
        choose_mode("document", [], "Explain this")
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_router.py backend/tests/integration/test_chat_stream.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def choose_mode(requested: str, document_ids: list[str], query: str) -> str:
    if requested == "document" and not document_ids:
        raise ValueError("document_required")
    if requested != "auto":
        return requested
    if document_ids:
        return "document"
    research_terms = ("paper", "research", "recent", "related work", "study")
    return "research" if any(term in query.casefold() for term in research_terms) else "general"
```

The heuristic is a transparent first baseline; test ambiguous phrases and provide an explicit mode control rather than pretending perfect intent classification. Research/document generation receives only retrieved Evidence plus bounded recent conversation. Hosted provider and exact generation model remain configuration selected against the quality/cost gate; deterministic fake emits a typed GroundedAnswer in CI. Structure prompts to treat retrieved documents as data. Allow one repair for malformed IDs/schema then return insufficiency. Send evidence/status before generation, buffer grounded answer until validation, stream validated text then persist final atomically. General mode streams raw model deltas. Store request_id uniqueness and replay completed final answers; duplicate in-flight request returns 409. Validate chat ownership before reads/model calls. Cancel on disconnect; persist interrupted state and usage actually reported. Traces omit raw private text by default.

- [ ] Verify the additional acceptance cases: explicit modes honored; attachment ownership checked before provider; unknown mode rejected by schema; empty research evidence abstains; injected paper instructions cannot invoke tools or expose secrets; one repair maximum; disconnect before final not complete; retry request doesn't double-charge through duplicate generation; SSE chunks survive arbitrary transport boundaries; provider timeout returns typed stream error.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_router.py backend/tests/integration/test_chat_stream.py -q` again, then run router/stream integration with a deterministic provider that simulates delay, malformed JSON, fabricated citations and disconnects. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m4-chat.md` with event ordering, cancellation and idempotency traces; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add grounded chat routing and safe stream lifecycle`.

## P4.3: Process private PDF uploads with ownership and deletion guarantees

**Depends on:** P4.2, P1.3–P1.4

**Files:**

- Create `backend/src/copilot/uploads/{storage,service,api}.py`, `backend/tests/unit/test_upload_validation.py`.
- Create `backend/tests/integration/test_upload_lifecycle.py`; extend worker handlers and migration 0004 before release.

**Interfaces:** `validate_pdf_header(data: bytes, max_bytes: int=26214400) -> None`; `UploadStorage.put(user_id: UUID, data: bytes) -> str`, `.read(user_id: UUID, key: str) -> bytes`, `.delete(user_id: UUID, key: str) -> None`; `submit_upload(principal, file) -> tuple[UUID,UUID]`; `delete_upload(principal, document_id) -> UUID` returns deletion job ID.

- [ ] Write the regression test below in `backend/tests/unit/test_upload_validation.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.uploads.service import validate_pdf_header

def test_content_validation_does_not_trust_extension():
    validate_pdf_header(b"%PDF-1.7\ncontent")
    with pytest.raises(ValueError, match="unsupported_pdf"):
        validate_pdf_header(b"<html>fake.pdf</html>")
    with pytest.raises(ValueError, match="upload_too_large"):
        validate_pdf_header(b"%PDF-" + b"x" * 10, max_bytes=10)
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_upload_validation.py backend/tests/integration/test_upload_lifecycle.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def validate_pdf_header(data: bytes, max_bytes: int = 26214400) -> None:
    if len(data) > max_bytes:
        raise ValueError("upload_too_large")
    if not data.startswith(b"%PDF-"):
        raise ValueError("unsupported_pdf")
```

Use streaming bounded receive before fully loading bytes; header check is only the first validation, not proof of a valid PDF. Parse in isolated worker with page/time/memory limits, check encrypted/corrupt content, then write dense-only vectors with server user_id/document_id. Local storage resolves random keys within an owned root under `${DATA_DIR}/uploads/`; this local directory adapter is the default regardless of deployment target, and managed object storage is deferred. Duplicate content per owner returns the existing active document/job. GET status returns progress and typed error; retrieval blocks until ready and fails closed on DB ownership uncertainty. Delete marks hidden transactionally, then removes private vectors and blob with retries; retrieval rechecks not-deleted state. Default expiry is 30 days, physical deletion SLA 24 hours. Implement cleanup command and test worker crashes between DB/vector/blob actions.

- [ ] Verify the additional acceptance cases: cross-user get/query/delete denied; guessed document ID doesn't reveal existence; fake PDF rejected; actual malformed/encrypted/>100-page PDF classified; 25 MiB boundary; upload cancellation; worker time limit; duplicate upload per owner; deletion immediately blocks search even if vector cleanup fails; retry removes all storage copies; no private HF export.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_upload_validation.py backend/tests/integration/test_upload_lifecycle.py -q` again, then run complete upload lifecycle against isolated local storage, DB and Qdrant, including worker failure/recovery. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m4-uploads.md` with ownership matrix, deletion retry evidence and parser limits; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add private pdf processing and deletion lifecycle`.

## P4.4: Expose chat, evidence and multimodal document interactions

**Depends on:** P4.3, P3.4

**Files:**

- Create `frontend/src/features/{chat,uploads}/`, `frontend/src/app/chat/[id]/page.tsx`, `frontend/tests/chat-documents.spec.ts`.
- Create `backend/src/copilot/uploads/images.py`, `backend/tests/unit/test_images.py`, `backend/tests/integration/test_multimodal.py`.

**Interfaces:** `validate_image(width: int, height: int, encoded_bytes: int, mime: str) -> None`; image decoder passes verified dimensions/mime, not client values. Extend Generator adapter with `answer_image(question: str, images: list[bytes], evidence: list[Evidence]) -> GroundedAnswer`; all rendered PDF images retain owned document/page references.

- [ ] Write the regression test below in `backend/tests/unit/test_images.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.uploads.images import validate_image

def test_decoded_image_limit():
    validate_image(1000, 1000, 100000, "image/png")
    with pytest.raises(ValueError, match="image_dimensions"):
        validate_image(10000, 10000, 100000, "image/png")
    with pytest.raises(ValueError, match="image_type"):
        validate_image(100, 100, 1000, "image/svg+xml")
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_images.py backend/tests/integration/test_multimodal.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def validate_image(width: int, height: int, encoded_bytes: int, mime: str) -> None:
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("image_type")
    if width <= 0 or height <= 0 or width * height > 20_000_000:
        raise ValueError("image_dimensions")
    if encoded_bytes > 10 * 1024 * 1024:
        raise ValueError("image_too_large")
```

Decode with bounded image library limits, strip unnecessary metadata, and never accept SVG as an image upload. Add chat composer, mode selector, upload progress, selected-doc scope, cited paper cards and evidence drawer. Browser SSE parser accumulates text across partial chunks, handles final/error explicitly and renders markdown/math with unsafe HTML disabled. Use fetch streaming; EventSource alone cannot POST this authenticated request body. PDF figure questions retrieve page text and rasterize at most three authorized relevant pages under size/time limits. Label model interpretation and absent numeric evidence. Image bytes go only to configured multimodal provider after authorization, not into public corpus. UI lets users delete uploads and inspect retention settings.

- [ ] Verify the additional acceptance cases: rendered malicious markdown cannot execute; citation drawer points to correct paper/page; incomplete stream isn't shown as finished; image-only question visibly labeled; figure question includes right page; image decode bomb blocked; other user's image inaccessible; keyboard upload/evidence interaction; upload retry and timeout messaging.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_images.py backend/tests/integration/test_multimodal.py -q` again, then run `npm --prefix frontend run test:e2e -- chat-documents.spec.ts`, frontend typecheck/build and multimodal fixture tests. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m4-multimodal.md` with PDF/image journeys, page evidence and unsupported-case examples; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add evidence chat interface and multimodal document qa`.

## P4.5: Calibrate grounded-answer evaluation and regression gates

**Depends on:** P4.4, P2.5

**Files:**

- Create `backend/src/copilot/evaluation/{rag,judge}.py`, `configs/experiments/rag.yaml`.
- Create `backend/tests/unit/test_judge_schema.py`, `backend/tests/integration/test_rag_eval.py`, `docs/judge-rubric.md`.
- Create `data/fixtures/rag/cases.jsonl`, `reports/rag/README.md`; extend evaluation CLI.

**Interfaces:** `JudgeResult` Pydantic model with supported_claims:int, factual_claims:int, citation_correctness:float, citation_completeness:float, answer_relevance:int, insufficient_evidence_handled:bool and rationale:str; `run_rag(config: Path, split: str, out: Path) -> dict`. Judge input contains question, evidence, answer, citations and rubric.

- [ ] Write the regression test below in `backend/tests/unit/test_judge_schema.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from pydantic import ValidationError
from copilot.evaluation.judge import JudgeResult

def test_judge_cannot_report_impossible_claim_counts():
    with pytest.raises(ValidationError):
        JudgeResult(supported_claims=5, factual_claims=3,
                    citation_correctness=1.0, citation_completeness=1.0,
                    answer_relevance=5, insufficient_evidence_handled=True,
                    rationale="invalid counts")
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_judge_schema.py backend/tests/integration/test_rag_eval.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from pydantic import BaseModel, Field, model_validator

class JudgeResult(BaseModel):
    supported_claims: int = Field(ge=0)
    factual_claims: int = Field(ge=0)
    citation_correctness: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    answer_relevance: int = Field(ge=1, le=5)
    insufficient_evidence_handled: bool
    rationale: str

    @model_validator(mode="after")
    def counts(self):
        if self.supported_claims > self.factual_claims:
            raise ValueError("invalid_claim_counts")
        return self
```

Create the 60-case family-split dataset with answerable, multi-paper, absent-evidence and injection cases. Compare flat chunks, reranked chunks and hierarchical retrieval under equal evidence budgets, plus global fallback. Judge uses a separate configured model and a versioned rubric; deterministic fake only validates harness behavior. Human-label 30 development/validation answers first, increase to 50–100 for stronger claims, compute weighted agreement and examine systematic disagreement. Provenance validity and semantic support are separate metrics. Zero factual claims makes support rate undefined; abstention accuracy is reported separately to prevent winning by always refusing. CI uses fixed fixture answers and structural citations; manual paid judge run requires runtime budget configuration.

- [ ] Verify the additional acceptance cases: invalid judge output bounded retry; fake references caught deterministically; unsupported claims lower support; unanswerable cases not scored against fabricated reference answers; evidence injection isolated; abstention denominator honest; generation/judge model versions recorded; cost ceiling stops runs resumably; no test answers used to tune rubric.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_judge_schema.py backend/tests/integration/test_rag_eval.py -q` again, then run `uv run --project backend python -m copilot.cli eval rag --config configs/experiments/rag.yaml --split validation --out reports/rag/pilot` and the deterministic smoke suite. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/rag/pilot/report.md`, calibration disagreement table and G4 quality decision; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: calibrate rag evaluation and citation regression gates`.

## Exit checkpoint

G4 requires corpus citation provenance, human-supported-claim audit, streaming lifecycle tests, private upload deletion and multimodal journeys. A model judge score alone cannot pass this gate. Next: Plan 5.

