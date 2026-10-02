"""Regression thresholds and the rules that make two runs comparable (spec §11).

A retrieval number is evidence only with its manifest: which corpus release,
which frozen split, which models, which code. ``validate_manifest`` rejects a
run missing any of those, and ``comparable`` refuses to set two runs side by
side unless they scored the same corpus on the same split — a difference
between them would otherwise measure the data, not the system.

The regression threshold is absolute: a drop of more than 0.03 in Recall@10 or
nDCG@10 fails, whatever the baseline was (spec §11).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_DROP = 0.03
GATED_METRICS = ("recall@10", "ndcg@10")

# Every field a run must carry to count as evidence. Dotted paths into the manifest.
REQUIRED_FIELDS = (
    "run_id",
    "created_at",
    "code.git_sha",
    "code.dirty",
    "corpus.release_id",
    "corpus.manifest_sha256",
    "corpus.parser_version",
    "corpus.chunker_version",
    "dataset.digest",
    "dataset.split",
    "dataset.slice",
    "dataset.queries",
    "experiment.config_sha256",
    "search.config_sha256",
    "seed",
    "hardware.cpu",
    "hardware.gpu",
    "hardware.torch",
    "timing.methodology",
)
VARIANT_FIELDS = ("mode", "release_id", "embedding", "reranker")


class ManifestError(ValueError):
    """A run that cannot be used as evidence, or two that cannot be compared."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def regressed(baseline: float, candidate: float, max_drop: float = MAX_DROP) -> bool:
    """True when the candidate fell more than ``max_drop`` below the baseline, absolutely."""

    if not 0 <= baseline <= 1 or not 0 <= candidate <= 1:
        raise ValueError("invalid_metric")
    return baseline - candidate > max_drop + 1e-12


def _field(manifest: Mapping[str, Any], path: str) -> Any:
    value: Any = manifest
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse a run that does not say exactly what produced it."""

    for path in REQUIRED_FIELDS:
        if _missing(_field(manifest, path)):
            raise ManifestError("manifest_incomplete", path)
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        raise ManifestError("manifest_incomplete", "variants")
    for name, variant in variants.items():
        for key in VARIANT_FIELDS:
            if not isinstance(variant, Mapping) or _missing(variant.get(key)):
                raise ManifestError("manifest_incomplete", f"variants.{name}.{key}")


_IDENTITY = {
    "corpus": ("corpus.release_id", "corpus.manifest_sha256"),
    "dataset": ("dataset.digest",),
    "split": ("dataset.split", "dataset.slice"),
}


def comparable(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Refuse to compare runs over different corpora, datasets or splits."""

    validate_manifest(baseline)
    validate_manifest(candidate)
    for what, paths in _IDENTITY.items():
        for path in paths:
            if _field(baseline, path) != _field(candidate, path):
                raise ManifestError("incomparable_runs", f"{what}: {path}")


def variant_search(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    """What one variant searched under. Runs before P2.6 record only the shared file's values."""

    variant = manifest["variants"][name]
    recorded = variant.get("search") if isinstance(variant, Mapping) else None
    if recorded is None:
        recorded = (manifest.get("search") or {}).get("values") or {}
    return dict(recorded)


def variant_llm(manifest: Mapping[str, Any], name: str) -> tuple[Any, Any] | None:
    """The served models and prompt digest of an LLM variant; None for any other."""

    variant = manifest["variants"][name]
    llm = variant.get("llm") if isinstance(variant, Mapping) else None
    if not isinstance(llm, Mapping):
        return None
    return sorted(llm.get("served_models") or []), llm.get("prompt_sha256")


def compare_runs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    metrics: Sequence[str] = GATED_METRICS,
    max_drop: float = MAX_DROP,
) -> dict[str, Any]:
    """Per variant, per gated metric: both means, the change, and whether it regressed.

    Each argument is a run's ``metrics.json``: its manifest and its variants'
    summaries. Only variants present in both runs are compared.
    """

    comparable(baseline["manifest"], candidate["manifest"])
    shared = sorted(set(baseline["variants"]) & set(candidate["variants"]))
    variants: dict[str, Any] = {}
    for name in shared:
        rows = {}
        for metric in metrics:
            before = float(baseline["variants"][name]["summary"][metric]["mean"])
            after = float(candidate["variants"][name]["summary"][metric]["mean"])
            rows[metric] = {
                "baseline": before,
                "candidate": after,
                "change": after - before,
                "regressed": regressed(before, after, max_drop),
            }
        variants[name] = rows
    return {
        "baseline_run": baseline["manifest"]["run_id"],
        "candidate_run": candidate["manifest"]["run_id"],
        "max_drop": max_drop,
        "variants": variants,
        # Same name, different depth, candidate count or deadline: a change of system,
        # reported as such rather than read as a like-for-like regression check.
        "search_changed": [
            name
            for name in shared
            if variant_search(baseline["manifest"], name)
            != variant_search(candidate["manifest"], name)
        ],
        # A hosted model can move under an alias, and a prompt can be edited: either one
        # makes the "same" LLM variant a different ranker.
        "llm_changed": [
            name
            for name in shared
            if variant_llm(baseline["manifest"], name) != variant_llm(candidate["manifest"], name)
        ],
        "regressed": any(row["regressed"] for rows in variants.values() for row in rows.values()),
    }
