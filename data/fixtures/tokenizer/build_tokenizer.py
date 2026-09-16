"""Generate a tiny deterministic BPE tokenizer for the chunking suite.

Production measures chunk windows in the pinned embedding model's tokens, so
the offline suite has to exercise that same code path. Downloading BGE-M3's
17 MB tokenizer in CI would break the deterministic-offline rule, so this
trains a small byte-level BPE over the repository's own CC0 fixture text.

The property the tests need is that one word is several tokens, which is what
makes a word-counted window wrong. Training is deterministic: same corpus,
same parameters, same file.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "papers" / "fixture.txt"
TARGET = HERE / "tokenizer.json"
VOCAB_SIZE = 180


def build() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(
        [str(SOURCE)],
        trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=["<unk>"],
            min_frequency=1,
            show_progress=False,
        ),
    )
    return tokenizer


if __name__ == "__main__":
    build().save(str(TARGET))
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes)")
