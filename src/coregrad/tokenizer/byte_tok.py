"""Raw UTF-8 byte tokenizer with native single-ID special protocol tokens (vocab_size=288).

ID layout:
  0..255   : Raw UTF-8 bytes (universal across every Unicode script with zero OOV)
  256..287 : Exact 32 coregrad special protocol slots (PAD, BOS, EOS, <|system|>, <|user|>,
             <|assistant|>, <|think|>, <|call|>, <|result|>, <|cite|>, ...)
"""

from __future__ import annotations

import re

from coregrad.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS_TOKEN,
    EOS_TOKEN,
    NAMED_SPECIAL_TOKENS,
    PAD_TOKEN,
)

BYTE_OFFSET = 256
BYTE_VOCAB_SIZE = BYTE_OFFSET + len(ALL_SPECIAL_TOKENS)  # 288 (multiple of 32)

_SPECIAL_TO_ID: dict[str, int] = {tok: BYTE_OFFSET + i for i, tok in enumerate(ALL_SPECIAL_TOKENS)}
_ID_TO_SPECIAL: dict[int, str] = {v: k for k, v in _SPECIAL_TO_ID.items()}
# Longest-first alternation so special tags split cleanly without ever touching raw UTF-8 bytes
_SPECIAL_RE = re.compile(
    "(" + "|".join(re.escape(tok) for tok in sorted(NAMED_SPECIAL_TOKENS, key=len, reverse=True)) + ")"
)


class ByteTokenizer:
    """Drop-in UTF-8 byte + special-token tokenizer matching CoreGradTokenizer's public API."""

    def __init__(self) -> None:
        self.vocab_size: int = BYTE_VOCAB_SIZE
        self.pad_id: int = _SPECIAL_TO_ID[PAD_TOKEN]
        self.bos_id: int = _SPECIAL_TO_ID[BOS_TOKEN]
        self.eos_id: int = _SPECIAL_TO_ID[EOS_TOKEN]

    def token_to_id(self, token: str) -> int:
        if token in _SPECIAL_TO_ID:
            return _SPECIAL_TO_ID[token]
        raw = token.encode("utf-8")
        if len(raw) == 1:
            return raw[0]
        raise KeyError(f"Unknown single token in ByteTokenizer: {token!r}")

    def id_to_token(self, token_id: int) -> str:
        if token_id in _ID_TO_SPECIAL:
            return _ID_TO_SPECIAL[token_id]
        if 0 <= token_id < BYTE_OFFSET:
            return bytes([token_id]).decode("utf-8", errors="replace")
        raise KeyError(f"Invalid byte token ID: {token_id}")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            sid = _SPECIAL_TO_ID.get(part)
            if sid is not None:
                ids.append(sid)
            else:
                ids.extend(part.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for tid in ids:
            if 0 <= tid < BYTE_OFFSET:
                buf.append(tid)
            else:
                flush()
                if not skip_special_tokens and tid in _ID_TO_SPECIAL:
                    out.append(_ID_TO_SPECIAL[tid])
        flush()
        return "".join(out)
