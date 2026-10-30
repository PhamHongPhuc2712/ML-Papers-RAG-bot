"""The faithfulness score: is each claim supported by the passages it cites? (2026-10-29)

Every evaluated answer, of every variant, is scored claim by claim by a judge model with its own
prompt (``prompts/judge/claim-support-v1.yaml``), against the same evidence the answer was
written from. Per answer: claims, supported claims, their rate. An abstention has no claims and
makes no call. The judge is the same model family as the agent's self-check, a stated limitation;
``rejudge_agreement`` measures it against a second scorer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from ..agent.check import CheckError, CheckPrompt, LedgerChecker
from ..contracts import Claim, Evidence, GroundedAnswer
from ..models.llm import LlmError
from .pairing import cohen_kappa


@dataclass(frozen=True)
class FaithfulnessScore:
    claims: int
    supported: int
    cost_usd: float = 0.0
    error: str | None = None
    verdicts: tuple[bool, ...] = ()

    @property
    def rate(self) -> float | None:
        return self.supported / self.claims if self.claims and self.error is None else None

    def fields(self) -> dict[str, Any]:
        """The case row's columns."""

        return {
            "claims": self.claims,
            "supported_claims": self.supported if self.error is None else None,
            "supported_rate": self.rate,
            "faithfulness_error": self.error,
            "faithfulness_cost_usd": self.cost_usd,
            "supported_verdicts": list(self.verdicts),
        }


class FaithfulnessScorer:
    def __init__(self, chat: Any, prompt: CheckPrompt, *, purpose: str = "evaluation") -> None:
        self._checker = LedgerChecker(chat, prompt, purpose)

    @property
    def identity(self) -> str:
        return self._checker.identity

    def score(
        self,
        question: str,
        answer: GroundedAnswer,
        evidence: Sequence[Evidence],
        *,
        request_id: UUID | None,
        run_id: str | None,
    ) -> FaithfulnessScore:
        if not answer.claims:
            return FaithfulnessScore(0, 0)
        try:
            verdicts, cost = self._checker.check(
                question, answer, evidence, request_id=request_id, run_id=run_id
            )
        except LlmError as error:
            return FaithfulnessScore(len(answer.claims), 0, error=f"faithfulness:{error.code}")
        except CheckError:
            return FaithfulnessScore(len(answer.claims), 0, error="faithfulness:unparseable")
        flags = tuple(v.supported for v in verdicts)
        return FaithfulnessScore(len(flags), sum(flags), cost, None, flags)


def faithfulness_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean supported rate over answers with claims and a score; the share fully supported."""

    rates = [float(r["supported_rate"]) for r in rows if r.get("supported_rate") is not None]
    return {
        "supported_rate": float(np.mean(rates)) if rates else None,
        "fully_supported": float(np.mean([rate == 1.0 for rate in rates])) if rates else None,
        "scored": len(rates),
        "claims": int(sum(int(r.get("claims") or 0) for r in rows)),
        "errors": sum(1 for r in rows if r.get("faithfulness_error")),
    }


def paired_rate_difference(
    treated: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    key: str,
    by: str | None = None,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """The supported-rate difference over items both scored, with a bootstrap interval over
    items, or over ``by`` groups (MTRAG: conversations), resampling whole groups."""

    base = {str(row[key]): row for row in baseline}
    pairs = [
        (row, base[str(row[key])])
        for row in treated
        if str(row[key]) in base
        and row.get("supported_rate") is not None
        and base[str(row[key])].get("supported_rate") is not None
    ]
    if not pairs:
        return {"items": 0, "groups": 0, "difference": None, "interval": [None, None]}
    groups: dict[str, list[float]] = {}
    for index, (t, b) in enumerate(pairs):
        group = str(t[by]) if by else str(index)
        groups.setdefault(group, []).append(float(t["supported_rate"]) - float(b["supported_rate"]))
    blocks = [np.array(values) for _, values in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        means[i] = float(np.concatenate([blocks[j] for j in chosen]).mean())
    tail = (1.0 - confidence) / 2.0
    return {
        "items": len(pairs),
        "groups": len(blocks),
        "treated": float(np.mean([float(t["supported_rate"]) for t, _ in pairs])),
        "baseline": float(np.mean([float(b["supported_rate"]) for _, b in pairs])),
        "difference": float(np.concatenate(blocks).mean()),
        "interval": [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))],
    }


def rejudge_agreement(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    """Per-claim agreement and Cohen's kappa between two scorers' verdicts."""

    if len(first) != len(second):
        raise ValueError("verdict lists differ in length")
    agree = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    return {
        "claims": len(first),
        "agreement": agree / len(first) if first else None,
        "kappa": cohen_kappa(first, second) if first else None,
        "first_supported": sum(first),
        "second_supported": sum(second),
    }


def faithfulness_input(answer: GroundedAnswer, by_id: Mapping[str, Evidence]) -> dict[str, Any]:
    """What was scored: the claims and the passages they cite. Text, for DATA_DIR only."""

    cited = dict.fromkeys(label for claim in answer.claims for label in claim.evidence_ids)
    return {
        "claims": [claim.model_dump() for claim in answer.claims],
        "evidence": [by_id[label].model_dump(mode="json") for label in cited if label in by_id],
    }


def run_faithfulness_rejudge(
    run_dir: str | Path,
    *,
    scorer_key: str,
    out: Path,
    max_spend_usd: float,
    limit: int | None = None,
    scorer: Any | None = None,
    prompt: str | Path = "prompts/judge/claim-support-v1.yaml",
) -> dict[str, Any]:
    """Re-score one variant's scored answers with a second model, claim by claim, and report
    agreement and kappa against the run's own verdicts. The sheet (text) goes to ``out``,
    under DATA_DIR; the summary carries no text."""

    from ..agent.check import load_check_prompt
    from ..models.spend import estimate_usd, load_llm_config
    from .qa import GENERATION_OVERHEAD_CHARS, PROMPT_CHARS_PER_TOKEN, QaError

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("faithfulness") or {}).get("key") == scorer_key:
        raise QaError("same_scorer", scorer_key)
    text_path = (
        Path(manifest["text_dir"]) / manifest["run_id"] / manifest["variant"] / "cases.jsonl"
    )
    rows = [
        json.loads(line)
        for line in text_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = sorted(
        (
            row
            for row in rows
            if row.get("faithfulness_input")
            and row.get("supported_verdicts")
            and not row.get("faithfulness_error")
        ),
        key=lambda row: str(row["query_id"]),
    )[: limit if limit is not None else None]
    if not rows:
        raise QaError("no_cases", str(text_path))
    model = load_llm_config("configs/llm.yaml").models[scorer_key]
    evidence_chars = 6_000 * PROMPT_CHARS_PER_TOKEN + GENERATION_OVERHEAD_CHARS
    worst = len(rows) * estimate_usd(model.price, evidence_chars, model.max_output_tokens)
    if worst > max_spend_usd:
        raise QaError(
            "run_budget_exceeded",
            f"worst case ${worst:.4f} for {len(rows)} cases > budget ${max_spend_usd:.4f}",
        )
    if scorer is None:
        from ..config import Settings
        from ..db.session import make_engine
        from ..models.lc_chat import build_ledger_chat_model
        from ..models.spend import SpendLedger

        settings = Settings()  # type: ignore[call-arg]
        ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
        chat = build_ledger_chat_model(scorer_key, settings, ledger, timeout=60.0, cache=None)
        scorer = FaithfulnessScorer(chat, load_check_prompt(prompt))
    run_id = f"{manifest['run_id']}.{manifest['variant']}.faithfulness-rejudge"
    first: list[bool] = []
    second: list[bool] = []
    sheet = []
    errors = 0
    cost = 0.0
    for row in rows:
        data = row["faithfulness_input"]
        answer = GroundedAnswer(
            claims=[Claim(**claim) for claim in data["claims"]], insufficient_evidence=False
        )
        evidence = [Evidence(**item) for item in data["evidence"]]
        scored = scorer.score(row["question"], answer, evidence, request_id=None, run_id=run_id)
        cost += scored.cost_usd
        if scored.error is not None:
            errors += 1
            continue
        first.extend(bool(v) for v in row["supported_verdicts"])
        second.extend(scored.verdicts)
        sheet.append(
            {
                "query_id": row["query_id"],
                "first": row["supported_verdicts"],
                "second": list(scored.verdicts),
                "claims": [claim["text"] for claim in data["claims"]],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(item) + "\n" for item in sheet), encoding="utf-8")
    return {
        "answers": len(sheet),
        "errors": errors,
        "cost_usd": cost,
        "first": manifest.get("faithfulness"),
        "second": scorer.identity,
        **rejudge_agreement(first, second),
    }
