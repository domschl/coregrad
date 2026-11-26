"""Clean, modern decoder-only Transformer for coregrad.

Features:
- Pre-RMSNorm + optional QK-Norm for training stability
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA) via PyTorch Flash SDPA
- SwiGLU MLP (bias-free linear layers)
- Tied input/output embeddings
- Optional activation checkpointing for tight VRAM budgets
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from coregrad.config import load_yaml


@dataclass
class ModelConfig:
    name: str = "coregrad-125m"
    vocab_size: int = 16384
    max_seq_len: int = 2048
    n_layer: int = 16
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    activation_checkpointing: bool = False
    n_loops: int = 1
    loop_start: int = 0
    loop_end: int | None = None
    loop_step_embed: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {self.n_loops}")
        end = self.n_layer if self.loop_end is None else self.loop_end
        if not (0 <= self.loop_start < end <= self.n_layer):
            raise ValueError(
                f"Invalid loop slice [{self.loop_start}:{end}] for n_layer={self.n_layer}"
            )

    @property
    def resolved_loop_end(self) -> int:
        return self.n_layer if self.loop_end is None else self.loop_end

    @property
    def effective_layers(self) -> int:
        end = self.resolved_loop_end
        core = end - self.loop_start
        return self.loop_start + self.n_loops * core + (self.n_layer - end)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> ModelConfig:
        data = load_yaml(Path(path))
        data.update(overrides)
        return cls(**data)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim // 2)
    cos = freqs.cos()[None, None, :, :]  # (1, 1, T, head_dim // 2)
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to x of shape (B, H, T, D)."""
    x_dtype = x.dtype
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2).to(x_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.use_gqa = self.n_kv_heads != self.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.use_gqa)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class CoreGradLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.loop_emb: nn.Parameter | None = (
            nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model))
            if (cfg.n_loops > 1 and cfg.loop_step_embed)
            else None
        )

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual output projections by effective execution depth
        residual_std = 0.02 / math.sqrt(2.0 * cfg.effective_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        # Deduplicate by Parameter object identity (works on meta device too).
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def iter_execution_steps(self) -> list[tuple[int | None, TransformerBlock]]:
        """Return ordered (loop_emb_idx_or_none, block) steps matching forward() execution.

        loop_emb_idx is non-None only on the first block of each recurrent iteration when
        loop_step_embed is active.
        """
        if self.cfg.n_loops == 1:
            return [(None, block) for block in self.blocks]
        start = self.cfg.loop_start
        end = self.cfg.resolved_loop_end
        steps: list[tuple[int | None, TransformerBlock]] = [(None, self.blocks[i]) for i in range(start)]
        for loop_idx in range(self.cfg.n_loops):
            for offset, i in enumerate(range(start, end)):
                emb_idx = loop_idx if (offset == 0 and self.loop_emb is not None) else None
                steps.append((emb_idx, self.blocks[i]))
        for i in range(end, self.cfg.n_layer):
            steps.append((None, self.blocks[i]))
        return steps

    def _run_block(
        self, block: TransformerBlock, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(block, x, cos, sin, use_reentrant=False)
        return block(x, cos, sin)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        cos = self.rope_cos[:, :, :seq_len, :]
        sin = self.rope_sin[:, :, :seq_len, :]

        if self.cfg.n_loops == 1:
            for block in self.blocks:
                x = self._run_block(block, x, cos, sin)
        else:
            start = self.cfg.loop_start
            end = self.cfg.resolved_loop_end
            for i in range(start):
                x = self._run_block(self.blocks[i], x, cos, sin)
            for loop_idx in range(self.cfg.n_loops):
                if self.loop_emb is not None:
                    x = x + self.loop_emb[loop_idx]
                for i in range(start, end):
                    x = self._run_block(self.blocks[i], x, cos, sin)
            for i in range(end, self.cfg.n_layer):
                x = self._run_block(self.blocks[i], x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return idx
