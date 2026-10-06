"""The Markdown report of a retrieval experiment, rendered from its recorded runs.

Every number comes from ``<out>/<split>/metrics.json``; nothing is typed in by
hand. Interpretation belongs in ``<out>/analysis.md``, which is included
verbatim, so re-rendering after a new run never overwrites it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .datasets import SPLITS

_QUALITY = ("recall@10", "recall@50", "ndcg@10", "mrr@10")


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _interval(entry: Mapping[str, Any]) -> str:
    return f"{_number(entry['mean'])} [{_number(entry['low'])}, {_number(entry['high'])}]"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _runs(out: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for split in SPLITS:
        path = out / split / "metrics.json"
        if path.is_file():
            runs[split] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _run_table(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    splits = list(runs)
    lines = ["| | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]

    def row(label: str, render: Any) -> None:
        cells = " | ".join(render(runs[split]["manifest"]) for split in splits)
        lines.append(f"| {label} | {cells} |")

    row("Run", lambda m: f"`{m['run_id']}`")
    row(
        "Code",
        lambda m: f"`{str(m['code']['git_sha'])[:10]}`"
        + (f" + uncommitted `{str(m['code']['diff_sha256'])[:10]}`" if m["code"]["dirty"] else ""),
    )
    row("Corpus release", lambda m: f"`{m['corpus']['release_id']}`")
    row("Dataset", lambda m: f"`{m['dataset']['digest'][:12]}`, {m['dataset']['slice']}")
    row(
        "Queries",
        lambda m: str(m["dataset"]["queries"])
        + (f" (limited to {m['dataset']['limited_to']})" if m["dataset"].get("limited_to") else ""),
    )
    row("Hardware", lambda m: f"{m['hardware']['gpu']}; torch {m['hardware']['torch']}")

    def gpu(m: Mapping[str, Any]) -> str:
        sampled = m["timing"]["gpu"]
        if not sampled.get("available"):
            return "not sampled"
        text = (
            f"util p50 {sampled['utilization_percent']['p50']:.0f}% / max "
            f"{sampled['utilization_percent']['max']}%; memory "
            f"{sampled['memory_mib']['min']}–{sampled['memory_mib']['max']} MiB; "
            f"{len(sampled['compute_processes_seen'])} GPU processes seen"
        )
        before = sampled.get("before_run") or {}
        if before.get("available"):
            text += f"; before the run {before['utilization_percent_p50']:.0f}% busy" + (
                " — **contended**" if before.get("busy") else ""
            )
        return text

    row("GPU during the run", gpu)
    row("Locked test", lambda m: "yes" if m.get("locked_test") else "no")
    return lines


def _split_section(split: str, run: Mapping[str, Any]) -> list[str]:
    manifest = run["manifest"]
    variants = run["variants"]
    lines = [f"## {split.capitalize()} — {manifest['dataset']['queries']} queries", ""]
    lines += [
        "Mean over queries with a 95% bootstrap interval over query families. Failed queries "
        "score zero and stay in every denominator.",
        "",
        "| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall "
        "| Failures | Degraded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in variants.items():
        summary = entry["summary"]
        cells = [_interval(summary[metric]) for metric in _QUALITY if metric in summary]
        judged = summary.get("judged@10", {}).get("mean")
        lines.append(
            f"| `{name}` | "
            + " | ".join(cells)
            + f" | {_number(judged)} | {_number(summary.get('candidate_recall'))} | "
            + f"{summary['failures']['count']} | {summary['degraded']['count']} |"
        )
    lines += [
        "",
        "Latency of `rank()` in milliseconds; stage columns are p95.",
        "",
    ]
    # LLM columns appear only when a variant called a hosted LLM, so every report
    # recorded before them re-renders unchanged.
    llm = {
        name: manifest["variants"][name]["llm"]
        for name in variants
        if isinstance(manifest["variants"][name].get("llm"), dict)
    }
    stage_names = ("lexical", "dense", "rerank", "llm") if llm else ("lexical", "dense", "rerank")
    header = "| Variant | p50 | p95 | max | " + " | ".join(stage_names) + " | warm-up s |"
    if llm:
        header += " LLM cost USD | LLM calls (cached) |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for name, entry in variants.items():
        summary = entry["summary"]
        latency, stages = summary["latency"], summary["stages"]
        warm = manifest["variants"][name].get("warmup_seconds")
        row = (
            f"| `{name}` | {latency['p50_ms']} | {latency['p95_ms']} | {latency['max_ms']} | "
            + " | ".join(str(stages.get(stage, {}).get("p95_ms", "—")) for stage in stage_names)
            + f" | {warm} |"
        )
        if llm:
            facts = llm.get(name)
            row += (
                f" {facts['cost_usd']:.4f} | {facts['calls']} ({facts['cache_hits']}) |"
                if facts
                else " — | — |"
            )
        lines.append(row)
    for facet, label in (("by_query_set", "query set"), ("by_specificity", "specificity")):
        lines += _facet_table(variants, facet, label)
    comparisons = run.get("comparisons") or {}
    if comparisons:
        lines += [
            "",
            "Paired differences, candidate minus baseline, 95% interval over the same family "
            "resamples:",
            "",
            "| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] "
            "| Wins / losses / ties |",
            "|---|---|---|---|---|---|",
        ]
        for comparison in comparisons.values():
            for metric in ("ndcg@10", "recall@50", "mrr@10"):
                item = comparison["metrics"].get(metric)
                if item is None:
                    continue
                lines.append(
                    f"| `{comparison['candidate']}` vs `{comparison['baseline']}` | {metric} | "
                    f"{_number(item['baseline'])} | {_number(item['candidate'])} | "
                    f"{_signed(item['difference'])} "
                    f"[{_signed(item['low'])}, {_signed(item['high'])}] | "
                    f"{item['wins']} / {item['losses']} / {item['ties']} |"
                )
    decision = run.get("decision")
    if decision:
        lines += _decision(decision)
    return lines


def _facet_table(variants: Mapping[str, Any], facet: str, label: str) -> list[str]:
    """nDCG@10 per value of one query attribute, with each value's query count."""

    values = sorted({v for entry in variants.values() for v in entry["summary"].get(facet, {})})
    if len(values) < 2:
        return []
    lines = ["", f"nDCG@10 by {label}:", ""]
    lines += ["| Variant | " + " | ".join(f"{value} (n)" for value in values) + " |"]
    lines += ["|---|" + "---|" * len(values)]
    for name, entry in variants.items():
        groups = entry["summary"].get(facet, {})
        cells = [
            f"{_number(groups[value]['ndcg@10'])} ({groups[value]['queries']})"
            if value in groups
            else "—"
            for value in values
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def _decision(decision: Mapping[str, Any]) -> list[str]:
    rules = decision["rules"]
    strict = bool(rules.get("require_undegraded"))
    guards = list(rules.get("guards") or [])
    lines = [
        "",
        "### Decision",
        "",
        f"Pre-registered rules (`decision` in the config): primary metric `{decision['primary']}`; "
        f"a costlier mode is promoted only if its paired interval lies above zero and its p95 is "
        f"within {rules.get('latency_p95_seconds', 3.0)} s; a cheaper ablation is adopted only if "
        f"its interval rules out losing more than {rules.get('max_drop', 0.03)}."
        + (
            " A candidate with any degraded query is never chosen (`require_undegraded`)."
            if strict
            else ""
        )
        + (
            " Guards: a step is promoted only if "
            + " and ".join(
                f"its `{guard['metric']}` interval rules out losing more than "
                f"{guard.get('max_drop', rules.get('max_drop', 0.03))}"
                for guard in guards
            )
            + "."
            if guards
            else ""
        ),
        "",
        "| Step | Difference [interval] | p95 s | Gain supported | Fits budget | "
        + ("Degraded | " if strict else "")
        + ("Guards | " if guards else "")
        + "Promoted |",
        "|---|---|---|---|---|" + ("---|" if strict else "") + ("---|" if guards else "") + "---|",
    ]
    for step in decision["steps"]:
        item = step["comparison"]
        guard_cell = "; ".join(
            f"`{check['metric']}` {_signed(check['comparison']['low'])} "
            f"{'ok' if check['holds'] else '**fails**'}"
            for check in step.get("guards", [])
        )
        lines.append(
            f"| `{step['from']}` → `{step['to']}` | {_signed(item['difference'])} "
            f"[{_signed(item['low'])}, {_signed(item['high'])}] | {step['p95_seconds']:.2f} | "
            f"{'yes' if step['gain_supported'] else 'no'} | "
            f"{'yes' if step['fits_budget'] else 'no'} | "
            + (f"{step.get('degraded', 0)} | " if strict else "")
            + (f"{guard_cell} | " if guards else "")
            + f"{'**yes**' if step['promoted'] else 'no'} |"
        )
    if decision["ablations"]:
        lines += [
            "",
            "| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice "
            "| Adopted |",
            "|---|---|---|---|---|---|",
        ]
        for item in decision["ablations"]:
            comparison = item["comparison"]
            lines.append(
                f"| `{item['variant']}` vs `{item['baseline']}` | {item['changes']} | "
                f"{_signed(comparison['difference'])} [{_signed(comparison['low'])}, "
                f"{_signed(comparison['high'])}] | {'yes' if item['noninferior'] else 'no'} | "
                f"{'yes' if item['applies'] else 'no'} | "
                f"{'**yes**' if item['adopt'] and item['applies'] else 'no'} |"
            )
    chosen = decision["chosen"]
    adopted = ", ".join(f"`{name}`" for name in chosen["adopted_ablations"]) or "none"
    lines += [
        "",
        f"**Chosen on validation: `{chosen['variant']}` (mode `{chosen['mode']}`); ablations "
        f"adopted: {adopted}.**",
    ]
    return lines


def _cost_line(runs: Mapping[str, Any]) -> str:
    """One line of cost: the first split's when nothing was metered, every split's when it was."""

    costs = {split: run["manifest"]["cost"] for split, run in runs.items()}
    first = next(iter(costs.values()))
    if not any(cost.get("by_variant") for cost in costs.values()):
        return f"Cost: {first['metered_usd']:.2f} USD metered — {first['note']}."
    total = sum(float(cost["metered_usd"]) for cost in costs.values())
    parts = ", ".join(f"{split} {float(cost['metered_usd']):.4f}" for split, cost in costs.items())
    note = next(cost["note"] for cost in costs.values() if cost.get("by_variant"))
    return f"Cost: {total:.4f} USD metered ({parts}) — {note}."


def render_report(out: Path) -> Path:
    """Write ``<out>/report.md`` from every recorded split, plus ``analysis.md`` if present."""

    runs = _runs(out)
    if not runs:
        raise FileNotFoundError(f"no recorded runs under {out}")
    first = next(iter(runs.values()))["manifest"]
    lines = [
        f"# Retrieval experiment — {first['experiment']['name']}",
        "",
        f"Rendered by `eval report` from `{out}/<split>/metrics.json`; edit `analysis.md`, not "
        "this file. Per-query rows are in `<split>/per_query.parquet`, manifests in "
        "`<split>/manifest.json`.",
        "",
        *_run_table(runs),
        "",
        f"Timing: {first['timing']['methodology']}",
        "",
        _cost_line(runs),
    ]
    for split, run in runs.items():
        lines += ["", *_split_section(split, run)]
    analysis = out / "analysis.md"
    if analysis.is_file():
        lines += ["", analysis.read_text(encoding="utf-8").rstrip()]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
