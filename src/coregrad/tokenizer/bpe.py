"""Public byte-level BPE tokenizer wrapper and trainer for coregrad."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from coregrad.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)

# GPT-4/Llama-3 style split regex with single-digit splitting (\p{N}) so numbers
# and arithmetic expressions tokenize cleanly and consistently.
PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def train_bpe_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int = 16384,
    min_frequency: int = 2,
    output_path: Path | None = None,
) -> CoreGradTokenizer:
    """Train a byte-level BPE tokenizer from a stream of public text documents."""
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(PRETOKENIZER_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(ALL_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iterator, trainer=trainer)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(output_path))
    return CoreGradTokenizer(tok)


class CoreGradTokenizer:
    """Thin, fast wrapper around Hugging Face `tokenizers.Tokenizer`."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def from_file(cls, path: Path | str) -> CoreGradTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(f"Unknown token: {token!r}")
        return tid

    @property
    def pad_id(self) -> int:
        return self.token_to_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_to_id(EOS_TOKEN)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=True).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        encoded = self._tok.encode_batch(texts, add_special_tokens=True)
        if add_eos:
            eos = self.eos_id
            return [[*enc.ids, eos] for enc in encoded]
        return [enc.ids for enc in encoded]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


def measure_fertility(
    tokenizer: CoreGradTokenizer, samples_by_domain: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Compute bytes/token and tokens/char per domain for tokenizer comparison."""
    report: dict[str, dict[str, float]] = {}
    for domain, texts in samples_by_domain.items():
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        total_chars = sum(len(t) for t in texts)
        encoded = tokenizer.encode_batch(texts, add_eos=False)
        total_tokens = sum(len(ids) for ids in encoded)
        report[domain] = {
            "bytes": float(total_bytes),
            "tokens": float(total_tokens),
            "bytes_per_token": total_bytes / max(total_tokens, 1),
            "tokens_per_char": total_tokens / max(total_chars, 1),
        }
    return report
