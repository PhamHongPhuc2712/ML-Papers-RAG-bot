"""Cross-encoder reranking of fused candidates, and the checks its scores must pass.

A cross-encoder reads the query and one candidate together and returns a
relevance logit; it produces no embedding, so it can only reorder candidates a
retriever already found. Its scores are ranking signals, not calibrated
probabilities, and are never compared with RRF scores (spec §7).

The pinned model is ``BAAI/bge-reranker-v2-m3``, fetched offline and verified
by sha256 like the embedder. Each pair is truncated on the document side only,
so the query survives whole (spec §7: "preserve query tokens").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.embeddings import (
    EmbeddingError,
    PinnedFile,
    model_section,
    pinned_files,
    pinned_revision,
    verify_model_files,
)
from .lexical import tokenize


class RerankError(RuntimeError):
    """A reranker returned something that cannot order the candidates."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def validate_scores(scores: Sequence[float], expected: int) -> list[float]:
    """One finite score per candidate, or nothing is reordered."""

    if len(scores) != expected:
        raise RerankError("score_count_mismatch", f"{len(scores)} for {expected} candidates")
    values = [float(score) for score in scores]
    if not all(math.isfinite(value) for value in values):
        raise RerankError("nonfinite_score")
    return values


def rerank_order(candidates: Sequence[str], scores: Sequence[float]) -> list[tuple[str, float]]:
    """Candidates by score, highest first; equal scores keep their incoming order.

    Scores are zipped to the IDs they were computed for, so the only thing the
    reranker decides is the order — it cannot add, drop or swap a candidate.
    """

    values = validate_scores(scores, len(candidates))
    paired = list(zip(candidates, values, strict=True))
    # sorted() is stable, so a tie keeps the fused order it arrived in.
    return sorted(paired, key=lambda pair: -pair[1])


@dataclass(frozen=True)
class RerankerSpec:
    """The pinned cross-encoder, as ``configs/models.yaml`` declares it."""

    repo: str
    revision: str
    files: tuple[PinnedFile, ...]
    pair_max_tokens: int
    batch_size: int
    precision: Mapping[str, str]

    def model_dir(self, data_dir: str | Path) -> Path:
        return Path(data_dir) / "models" / "rerankers" / self.repo / self.revision


def load_reranker_spec(path: str | Path) -> RerankerSpec:
    section = model_section(path, "reranker")
    precision = section.get("precision") or {}
    try:
        return RerankerSpec(
            repo=str(section["repo"]),
            revision=pinned_revision(section),
            files=pinned_files(section),
            pair_max_tokens=int(section["pair_max_tokens"]),
            batch_size=int(section.get("batch_size", 16)),
            precision={str(k): str(v) for k, v in dict(precision).items()},
        )
    except KeyError as error:
        raise EmbeddingError("models_config_invalid", f"reranker.{error.args[0]}") from error


class FixtureReranker:
    """Deterministic scorer for tests: the share of query terms a text contains."""

    identity = "fixture/term-overlap@v1"

    def score(self, query: str, texts: list[str]) -> list[float]:
        wanted = set(tokenize(query))
        if not wanted:
            return [0.0 for _ in texts]
        return [len(wanted & set(tokenize(text))) / len(wanted) for text in texts]


class CrossEncoderReranker:
    """``bge-reranker-v2-m3``: one logit per (query, text) pair, in input order."""

    def __init__(
        self,
        spec: RerankerSpec,
        data_dir: str | Path,
        *,
        device: str = "auto",
        precision: str | None = None,
        pair_max_tokens: int | None = None,
    ) -> None:
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModelForSequenceClassification

        directory = verify_model_files(spec, data_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        default = spec.precision.get(device.split(":")[0]) or "float32"
        self.precision = precision or default
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if self.precision not in dtype:
            raise EmbeddingError("precision_unsupported", self.precision)
        self.pair_max_tokens = pair_max_tokens or spec.pair_max_tokens
        # The pair budget is part of the identity: 512 and 1,024 tokens are
        # different rankers for the purpose of comparing runs.
        self.identity = f"{spec.repo}@{spec.revision}#pair{self.pair_max_tokens}"
        self._torch = torch
        self._batch_size = spec.batch_size
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=self.pair_max_tokens, strategy="only_second")
        pad = self._tokenizer.token_to_id("<pad>")
        if pad is None:
            raise EmbeddingError("tokenizer_invalid", "no <pad> token")
        self._pad_id = pad
        model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), dtype=dtype[self.precision]
        )
        model.eval()
        self._model = model.to(device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch([(query, text) for text in texts])
        order = sorted(range(len(encodings)), key=lambda index: len(encodings[index].ids))
        scores = np.zeros(len(encodings), dtype=np.float32)
        for start in range(0, len(order), self._batch_size):
            chosen = order[start : start + self._batch_size]
            scores[chosen] = self._forward([encodings[index].ids for index in chosen])
        return validate_scores(scores.tolist(), len(texts))

    def _forward(self, sequences: list[list[int]]) -> np.ndarray:
        torch = self._torch
        width = max(len(ids) for ids in sequences)
        ids = torch.full((len(sequences), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, : len(sequence)] = 1
        with torch.inference_mode():
            logits = self._model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            ).logits
        result: np.ndarray = logits[:, 0].float().cpu().numpy()
        return result
