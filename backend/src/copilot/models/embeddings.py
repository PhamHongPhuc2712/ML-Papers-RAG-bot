"""Pinned embedding adapters and the checks every vector passes before a write.

Two models implement one shape. The production adapter runs BGE-M3's dense
head — the last layer's ``[CLS]`` state, L2-normalized — from weights pinned by
revision and sha256 under ``DATA_DIR``; nothing is fetched at encode time. The
fixture model is deterministic and two-dimensional, so the offline suite and
the integration tests exercise the whole index lifecycle with no weights and no
network (spec §11: CI never downloads models).

Every vector carries its model's ``identity`` into the release it is written
to. A query encoded under any other identity is a different vector space, and
the retrievers refuse it rather than return confident nonsense.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray

from ..search.lexical import tokenize

Matrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """A model cannot be loaded or cannot encode this input."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def validate_vectors(vectors: Sequence[Sequence[float]] | Matrix, dimensions: int) -> None:
    """Reject a wrong dimension or a nonfinite value before anything is written."""

    if isinstance(vectors, np.ndarray):
        if vectors.ndim != 2 or (vectors.shape[0] and vectors.shape[1] != dimensions):
            raise ValueError("dimension_mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite_vector")
        return
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("dimension_mismatch")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("nonfinite_vector")


@dataclass(frozen=True)
class EncodedBatch:
    """Vectors in input order, and how many inputs were cut to fit the model."""

    vectors: Matrix
    truncated: int


class VectorModel(Protocol):
    """What the index build and the retrievers need from an embedding model."""

    identity: str
    dimensions: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_array(
        self, texts: Sequence[str], *, max_tokens: int | None = None
    ) -> EncodedBatch: ...


@dataclass(frozen=True)
class PinnedFile:
    file: str
    sha256: str


@dataclass(frozen=True)
class EmbeddingSpec:
    """One pinned embedding model, as ``configs/models.yaml`` declares it."""

    repo: str
    revision: str
    dimensions: int
    preprocessing: str
    files: tuple[PinnedFile, ...]
    max_tokens: Mapping[str, int]
    batch_size: int
    precision: Mapping[str, str]

    @property
    def identity(self) -> str:
        """Model revision plus preprocessing: what query and document must share."""

        return f"{self.repo}@{self.revision}#{self.preprocessing}"

    def model_dir(self, data_dir: str | Path) -> Path:
        """Pinned weights live beside the tokenizers under the one data root."""

        return Path(data_dir) / "models" / "embeddings" / self.repo / self.revision


def _require(mapping: Mapping[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise EmbeddingError("models_config_invalid", key)
    return value


def load_embedding_spec(path: str | Path) -> EmbeddingSpec:
    """Read and validate the pinned embedding model; an unpinned one is refused."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("embedding"), Mapping):
        raise EmbeddingError("models_config_invalid", "embedding")
    embedding = raw["embedding"]
    files = embedding.get("files")
    if not isinstance(files, list) or not files:
        raise EmbeddingError("models_config_invalid", "files")
    pinned: list[PinnedFile] = []
    for entry in files:
        if not isinstance(entry, Mapping) or not entry.get("file") or not entry.get("sha256"):
            raise EmbeddingError("models_config_invalid", "files")
        pinned.append(PinnedFile(str(entry["file"]), str(entry["sha256"])))
    revision = _require(embedding, "revision", str)
    if len(revision) != 40:
        # A branch name is not a pin: "main" moves, a commit does not.
        raise EmbeddingError("models_config_invalid", "revision")
    max_tokens = _require(embedding, "max_tokens", Mapping)
    precision = _require(embedding, "precision", Mapping)
    return EmbeddingSpec(
        repo=_require(embedding, "repo", str),
        revision=revision,
        dimensions=_require(embedding, "dimensions", int),
        preprocessing=_require(embedding, "preprocessing", str),
        files=tuple(pinned),
        max_tokens={str(k): int(v) for k, v in max_tokens.items()},
        batch_size=_require(embedding, "batch_size", int),
        precision={str(k): str(v) for k, v in precision.items()},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_files(spec: EmbeddingSpec, data_dir: str | Path) -> Path:
    """Check every pinned file is present and unaltered; return the model directory."""

    directory = spec.model_dir(data_dir)
    for pinned in spec.files:
        path = directory / pinned.file
        if not path.is_file():
            raise EmbeddingError("model_missing", str(path))
        actual = _sha256(path)
        if actual != pinned.sha256:
            raise EmbeddingError("model_checksum_mismatch", f"{path}: {actual}")
    return directory


def fetch_model(spec: EmbeddingSpec, data_dir: str | Path) -> Path:
    """Offline capture of the pinned files at their revision, then verification.

    Like ``corpus mirror``, this runs from the CLI and never in the serving
    path: Hugging Face is an artifact store, not a request-time dependency.
    """

    from huggingface_hub import hf_hub_download

    directory = spec.model_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for pinned in spec.files:
        if (directory / pinned.file).is_file():
            continue
        hf_hub_download(
            repo_id=spec.repo,
            filename=pinned.file,
            revision=spec.revision,
            local_dir=str(directory),
        )
    return verify_model_files(spec, data_dir)


def _normalize(rows: NDArray[np.float64]) -> Matrix:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    return (rows / safe).astype(np.float32)


class FixtureEmbedding:
    """Deterministic two-dimensional model for tests: no weights, no network.

    Each term points at an angle fixed by its sha256; a text is the normalized
    sum of its terms. Identical texts get identical vectors and a query equal to
    a document's text ranks that document first, which is all a lifecycle test
    needs from a model.
    """

    dimensions = 2

    def __init__(self, identity: str = "fixture/hashed-2d@v1#l2") -> None:
        self.identity = identity

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.encode_array(texts).vectors]

    def encode_array(self, texts: Sequence[str], *, max_tokens: int | None = None) -> EncodedBatch:
        rows: list[list[float]] = []
        truncated = 0
        for text in texts:
            terms = tokenize(text)
            if max_tokens is not None and len(terms) > max_tokens:
                terms = terms[:max_tokens]
                truncated += 1
            x = y = 0.0
            for term in terms:
                turn = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big") / 2**32
                x += math.cos(2 * math.pi * turn)
                y += math.sin(2 * math.pi * turn)
            rows.append([x, y] if math.hypot(x, y) > 1e-12 else [1.0, 0.0])
        matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), self.dimensions)
        return EncodedBatch(_normalize(matrix), truncated)


class TransformerEmbedding:
    """BGE-M3's dense head: ``[CLS]`` of the last layer, L2-normalized in float32.

    Inputs are sorted by length before batching so padding stays small, then
    returned in their original order. A CUDA out-of-memory error halves the
    batch and retries it rather than failing a multi-hour build.
    """

    def __init__(
        self,
        spec: EmbeddingSpec,
        data_dir: str | Path,
        *,
        device: str = "auto",
        precision: str | None = None,
    ) -> None:
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModel

        directory = verify_model_files(spec, data_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        default = spec.precision.get(device.split(":")[0]) or "float32"
        self.precision = precision or default
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if self.precision not in dtype:
            raise EmbeddingError("precision_unsupported", self.precision)
        self.spec = spec
        self.identity = spec.identity
        self.dimensions = spec.dimensions
        self._torch = torch
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._pad_id = self._tokenizer.token_to_id("<pad>")
        if self._pad_id is None:
            raise EmbeddingError("tokenizer_invalid", "no <pad> token")
        model = AutoModel.from_pretrained(
            str(directory), dtype=dtype[self.precision], add_pooling_layer=False
        )
        model.eval()
        self._model = model.to(device)
        self._batch_size = spec.batch_size

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.encode_array(texts).vectors]

    def encode_array(self, texts: Sequence[str], *, max_tokens: int | None = None) -> EncodedBatch:
        limit = max_tokens or max(self.spec.max_tokens.values())
        self._tokenizer.enable_truncation(max_length=limit)
        encodings = self._tokenizer.encode_batch(list(texts))
        # A truncated encoding reports what it dropped; one tokenization pass
        # yields both the input ids and the truncation rate spec §7 asks for.
        truncated = sum(1 for encoding in encodings if encoding.overflowing)
        order = sorted(range(len(encodings)), key=lambda index: len(encodings[index].ids))
        output = np.zeros((len(encodings), self.dimensions), dtype=np.float32)
        start = 0
        batch = self._batch_size
        while start < len(order):
            chosen = order[start : start + batch]
            try:
                output[chosen] = self._forward([encodings[index].ids for index in chosen])
            except self._torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                self._torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                continue
            start += len(chosen)
        return EncodedBatch(output, truncated)

    def _forward(self, sequences: list[list[int]]) -> Matrix:
        torch = self._torch
        width = max(len(ids) for ids in sequences)
        ids = torch.full((len(sequences), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, : len(sequence)] = 1
        with torch.inference_mode():
            hidden = self._model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            ).last_hidden_state
            cls = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
        result: Matrix = cls.cpu().numpy()
        return result


@dataclass
class EmbeddingCache:
    """Encoded batches on disk, so an interrupted build resumes without re-encoding.

    The directory is keyed by model identity and each entry by the sha256 of
    every text in the batch plus the token limit, so a different model or any
    changed text is a miss rather than a stale hit.
    """

    root: Path
    identity: str
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    @property
    def directory(self) -> Path:
        return self.root / hashlib.sha256(self.identity.encode()).hexdigest()[:16]

    @staticmethod
    def key(text_hashes: Sequence[str], max_tokens: int | None) -> str:
        digest = hashlib.sha256(f"max_tokens={max_tokens}\n".encode())
        for text_hash in text_hashes:
            digest.update(f"{text_hash}\n".encode())
        return digest.hexdigest()

    def load(self, key: str) -> EncodedBatch | None:
        path = self.directory / key[:2] / f"{key}.npz"
        if not path.is_file():
            self.misses += 1
            return None
        with np.load(path) as stored:
            if str(stored["identity"]) != self.identity:
                self.misses += 1
                return None
            batch = EncodedBatch(stored["vectors"].astype(np.float32), int(stored["truncated"]))
        self.hits += 1
        return batch

    def store(self, key: str, batch: EncodedBatch) -> None:
        path = self.directory / key[:2] / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        (self.directory / "identity.json").write_text(
            json.dumps({"identity": self.identity}), encoding="utf-8"
        )
        staged = path.with_suffix(".partial.npz")
        np.savez(
            staged,
            vectors=batch.vectors,
            truncated=np.int64(batch.truncated),
            identity=np.str_(self.identity),
        )
        staged.replace(path)
