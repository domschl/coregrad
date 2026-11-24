"""Immutable special-token registry for coregrad.

Special tokens occupy IDs 0..31 so chat, chain-of-thought, tool-calling,
and citation tokens are native single tokens from Phase 1 onward.
"""

from __future__ import annotations

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
SYSTEM_START = "<|system|>"
SYSTEM_END = "<|/system|>"
USER_START = "<|user|>"
USER_END = "<|/user|>"
ASSISTANT_START = "<|assistant|>"
ASSISTANT_END = "<|/assistant|>"
THINK_START = "<|think|>"
THINK_END = "<|/think|>"
CALL_START = "<|call|>"
CALL_END = "<|/call|>"
RESULT_START = "<|result|>"
RESULT_END = "<|/result|>"
CITE_START = "<|cite|>"
CITE_END = "<|/cite|>"

NAMED_SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SYSTEM_START,
    SYSTEM_END,
    USER_START,
    USER_END,
    ASSISTANT_START,
    ASSISTANT_END,
    THINK_START,
    THINK_END,
    CALL_START,
    CALL_END,
    RESULT_START,
    RESULT_END,
    CITE_START,
    CITE_END,
)

NUM_SPECIAL_SLOTS = 32
RESERVED_SPECIAL_TOKENS: tuple[str, ...] = tuple(
    f"<|reserved_{i}|>" for i in range(len(NAMED_SPECIAL_TOKENS), NUM_SPECIAL_SLOTS)
)

ALL_SPECIAL_TOKENS: tuple[str, ...] = NAMED_SPECIAL_TOKENS + RESERVED_SPECIAL_TOKENS
SPECIAL_TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(ALL_SPECIAL_TOKENS)}
