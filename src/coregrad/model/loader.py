"""Universal checkpoint loader supporting both BPE CoreGradLM and UTF-8 CoreGradByteLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from coregrad.config import load_local_paths
from coregrad.model.hnet import CoreGradByteLM, HNetConfig
from coregrad.model.transformer import CoreGradLM, ModelConfig
from coregrad.tokenizer import ByteTokenizer, CoreGradTokenizer


def load_checkpoint_model_and_tokenizer(
    checkpoint_path: Path | str,
    device: torch.device,
    *,
    tokenizer_path: Path | None = None,
    min_seq_len: int | None = None,
    activation_checkpointing: bool | None = None,
    rope_theta_override: float | None = None,
) -> tuple[CoreGradLM | CoreGradByteLM, CoreGradTokenizer | ByteTokenizer, dict[str, Any]]:
    """Load any coregrad checkpoint and its matching tokenizer (BPE or UTF-8 Byte)."""
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mcfg_dict = dict(ckpt["model_config"])
    if min_seq_len is not None:
        mcfg_dict["max_seq_len"] = max(int(mcfg_dict.get("max_seq_len", min_seq_len)), min_seq_len)
    if rope_theta_override is not None:
        mcfg_dict["rope_theta"] = rope_theta_override

    if ckpt.get("model_type") == "hnet_byte" or "n_enc" in mcfg_dict:
        mcfg = HNetConfig(**mcfg_dict)
        model: CoreGradLM | CoreGradByteLM = CoreGradByteLM(mcfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        tokenizer: CoreGradTokenizer | ByteTokenizer = ByteTokenizer()
    else:
        if activation_checkpointing is not None:
            mcfg_dict["activation_checkpointing"] = activation_checkpointing
        mcfg_bpe = ModelConfig(**mcfg_dict)
        model = CoreGradLM(mcfg_bpe).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        tok_p = tokenizer_path or (load_local_paths().data_dir / "tokenizer" / "tokenizer.json")
        tokenizer = CoreGradTokenizer.from_file(tok_p)

    return model, tokenizer, ckpt
