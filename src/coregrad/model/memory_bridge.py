"""Phase 4B: Retrofittable Latent Memory Bridge for coregrad.

Implements two complementary middle-layer latent memory mechanisms bolted onto
a pretrained CoreGradLM without breaking base behavior at initialization:
1. LatentQueryHead: projects middle-layer hidden states h_L -> unit 768d query
   vectors aligned with EmbeddingGemma 2's semantic space.
2. GatedCrossAttention + CoreGradMemoryBridge: zero-initialized gated cross-attention
   (x = x + tanh(alpha) * CrossAttn(x, mem)) injected at middle/upper layers,
   supporting both:
   - Variant A (xRAG-style): 1 EmbeddingGemma 2 vector (768d) projected to memory slots
   - Variant B (RETRO-style): N pooled latent tokens per passage from frozen bottom layers
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from coregrad.model.transformer import CoreGradLM, RMSNorm, apply_rotary_emb


@dataclass
class MemoryBridgeConfig:
    bridge_layers: tuple[int, ...] = (9, 12, 15)
    query_layer: int = 9
    mem_input_dim: int = 768
    mem_expansion_slots: int = 4  # for Variant A (1 vector -> k learned memory slots)
    retro_encoder_layers: int = 6
    retro_slots_per_passage: int = 8
    query_dim: int = 768
    freeze_base: bool = True


class LatentQueryHead(nn.Module):
    """Projects a middle-layer hidden representation into EmbeddingGemma 2's 768d query space."""

    def __init__(self, d_model: int, query_dim: int = 768, hidden_mult: int = 2) -> None:
        super().__init__()
        h_dim = d_model * hidden_mult
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, h_dim, bias=False)
        self.fc2 = nn.Linear(h_dim, query_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        out = self.fc2(F.silu(self.fc1(x)))
        return F.normalize(out.float(), p=2, dim=-1)


class VectorMemoryProjector(nn.Module):
    """Projects single external passage embeddings (B, K, mem_dim) into (B, K*S, d_model) memory slots."""

    def __init__(self, mem_dim: int, d_model: int, expansion_slots: int = 4) -> None:
        super().__init__()
        self.expansion_slots = expansion_slots
        self.d_model = d_model
        self.norm = RMSNorm(mem_dim)
        self.fc1 = nn.Linear(mem_dim, d_model * 2, bias=False)
        self.fc2 = nn.Linear(d_model * 2, d_model * expansion_slots, bias=False)

    def forward(self, vecs: torch.Tensor) -> torch.Tensor:
        bsz, n_passages, _ = vecs.shape
        x = F.silu(self.fc1(self.norm(vecs)))
        out = self.fc2(x)
        return out.view(bsz, n_passages * self.expansion_slots, self.d_model)


class GatedCrossAttention(nn.Module):
    """Zero-initialized gated cross-attention block (Flamingo / RETRO style).

    Starts as an exact identity (tanh(0) == 0) so retrofitting onto a pretrained
    checkpoint never perturbs initial loss or generations.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_gqa = n_kv_heads != n_heads

        self.x_norm = RMSNorm(d_model, eps=norm_eps)
        self.m_norm = RMSNorm(d_model, eps=norm_eps)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        # Zero-initialized gate guarantees exact identity at step 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        mem_len = memory.shape[1]

        q = self.q_proj(self.x_norm(x)).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        m = self.m_norm(memory)
        k = self.k_proj(m).view(bsz, mem_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(m).view(bsz, mem_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = None
        if memory_mask is not None:
            # memory_mask: (B, M) bool where True = valid memory slot
            attn_mask = memory_mask[:, None, None, :].expand(bsz, 1, seq_len, mem_len)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=self.use_gqa
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return x + torch.tanh(self.gate) * self.o_proj(y)


class CoreGradMemoryBridge(nn.Module):
    """Wraps a pretrained CoreGradLM with middle-layer gated cross-attention memory."""

    def __init__(
        self,
        base_model: CoreGradLM,
        bridge_cfg: MemoryBridgeConfig,
        memory_mode: str = "embeddinggemma",
    ) -> None:
        super().__init__()
        if memory_mode not in ("embeddinggemma", "retro", "retro_tokens"):
            raise ValueError(f"Unsupported memory_mode: {memory_mode!r}")
        self.base = base_model
        self.bridge_cfg = bridge_cfg
        self.memory_mode = memory_mode
        d_model = base_model.cfg.d_model

        if bridge_cfg.freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        if memory_mode == "embeddinggemma":
            self.mem_proj: nn.Module = VectorMemoryProjector(
                mem_dim=bridge_cfg.mem_input_dim,
                d_model=d_model,
                expansion_slots=bridge_cfg.mem_expansion_slots,
            )
        else:
            # Lightweight adapter projecting frozen bottom-layer RETRO tokens (pooled or unpooled)
            self.mem_proj = nn.Sequential(
                RMSNorm(d_model, eps=base_model.cfg.norm_eps),
                nn.Linear(d_model, d_model, bias=False),
            )

        self.cross_attns = nn.ModuleDict(
            {
                str(layer_idx): GatedCrossAttention(
                    d_model=d_model,
                    n_heads=base_model.cfg.n_heads,
                    n_kv_heads=base_model.cfg.n_kv_heads,
                    norm_eps=base_model.cfg.norm_eps,
                )
                for layer_idx in bridge_cfg.bridge_layers
                if 0 <= layer_idx < base_model.cfg.n_layer
            }
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gate_values(self) -> dict[str, float]:
        return {k: float(torch.tanh(m.gate).item()) for k, m in self.cross_attns.items()}

    @torch.no_grad()
    def extract_layer_hidden(
        self, idx: torch.Tensor, layer_indices: tuple[int, ...]
    ) -> dict[int, torch.Tensor]:
        """Extract intermediate hidden states after specified 1-based layer depths."""
        _, seq_len = idx.shape
        x = self.base.tok_emb(idx)
        cos = self.base.rope_cos[:, :, :seq_len, :]
        sin = self.base.rope_sin[:, :, :seq_len, :]
        want = set(layer_indices)
        max_layer = max(want)
        out: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.base.blocks, start=1):
            x = block(x, cos, sin)
            if i in want:
                out[i] = x.detach()
            if i >= max_layer:
                break
        return out

    @torch.no_grad()
    def encode_retro_passages(
        self,
        passage_ids: torch.Tensor,
        slots_per_passage: int | None = None,
        lengths: list[int] | None = None,
    ) -> torch.Tensor:
        """Encode tokenized passages (N, P) via frozen bottom layers and pool to (N, S, d_model)."""
        slots = slots_per_passage or self.bridge_cfg.retro_slots_per_passage
        bsz, seq_len = passage_ids.shape
        x = self.base.tok_emb(passage_ids)
        cos = self.base.rope_cos[:, :, :seq_len, :]
        sin = self.base.rope_sin[:, :, :seq_len, :]
        n_enc = min(self.bridge_cfg.retro_encoder_layers, self.base.cfg.n_layer)
        for i in range(n_enc):
            x = self.base.blocks[i](x, cos, sin)
        if lengths is None:
            pooled = F.adaptive_avg_pool1d(x.transpose(1, 2).float(), slots).transpose(1, 2)
            return pooled.to(x.dtype)
        # Pool strictly over valid (non-pad) tokens per passage
        out = torch.empty((bsz, slots, x.shape[-1]), dtype=x.dtype, device=x.device)
        for i, n_tok in enumerate(lengths):
            valid = x[i : i + 1, : max(1, n_tok), :].transpose(1, 2).float()
            out[i] = F.adaptive_avg_pool1d(valid, slots).transpose(1, 2).squeeze(0).to(x.dtype)
        return out

    @torch.no_grad()
    def encode_retro_token_states(self, passage_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized passages (N, P) via frozen bottom layers without pooling -> (N, P, d_model)."""
        _, seq_len = passage_ids.shape
        x = self.base.tok_emb(passage_ids)
        cos = self.base.rope_cos[:, :, :seq_len, :]
        sin = self.base.rope_sin[:, :, :seq_len, :]
        n_enc = min(self.bridge_cfg.retro_encoder_layers, self.base.cfg.n_layer)
        for i in range(n_enc):
            x = self.base.blocks[i](x, cos, sin)
        return x

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.base.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.base.cfg.max_seq_len}")

        mem_tokens: torch.Tensor | None = None
        eff_mask = memory_mask
        if memory is not None:
            mem_tokens = self.mem_proj(memory)
            if self.memory_mode == "embeddinggemma" and memory_mask is not None:
                # Expand passage-level mask (B, K) -> (B, K * expansion_slots)
                eff_mask = memory_mask.repeat_interleave(
                    self.bridge_cfg.mem_expansion_slots, dim=1
                )

        x = self.base.tok_emb(idx)
        cos = self.base.rope_cos[:, :, :seq_len, :]
        sin = self.base.rope_sin[:, :, :seq_len, :]

        for layer_idx, block in enumerate(self.base.blocks):
            x = block(x, cos, sin)
            key = str(layer_idx)
            if mem_tokens is not None and key in self.cross_attns:
                x = self.cross_attns[key](x, mem_tokens, memory_mask=eff_mask)

        x = self.base.final_norm(x)
        logits = self.base.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss


@torch.no_grad()
def encode_kv_passage(
    model: CoreGradLM,
    passage_ids: torch.Tensor,
    *,
    drop_bos: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Offline per-passage encoder producing pre-RoPE GQA (K_raw, V) tensors for every layer.

    Each passage is encoded independently (TurboRAG / Block-Attention style) so passages
    can be cached once on disk/RAM and composed in arbitrary combinations at query time
    with zero query-time passage prefill compute.

    Returns a list of n_layer tuples (k_raw, v), each of shape (B, n_kv_heads, P', head_dim).
    """
    bsz, seq_len = passage_ids.shape
    x = model.tok_emb(passage_ids)
    cos = model.rope_cos[:, :, :seq_len, :]
    sin = model.rope_sin[:, :, :seq_len, :]
    kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    start = 1 if (drop_bos and seq_len > 1) else 0

    for block in model.blocks:
        attn = block.attn
        xn = block.attn_norm(x)
        q = attn.q_proj(xn).view(bsz, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
        k_raw = attn.k_proj(xn).view(bsz, seq_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(xn).view(bsz, seq_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        if attn.qk_norm:
            q = attn.q_norm(q)
            k_raw = attn.k_norm(k_raw)
        kv_list.append((k_raw[:, :, start:, :].detach(), v[:, :, start:, :].detach()))
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k_raw, cos, sin)
        y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True, enable_gqa=attn.use_gqa)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, attn.n_heads * attn.head_dim)
        x = x + attn.o_proj(y)
        x = x + block.mlp(block.mlp_norm(x))
    return kv_list


def forward_with_kv_bank(
    model: CoreGradLM,
    query_ids: torch.Tensor,
    kv_bank: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run CoreGradLM on query_ids attending over precomputed pre-RoPE latent GQA KV memory.

    Memory tokens occupy contiguous RoPE positions [0, M) and query tokens occupy [M, M+Q).
    Requires zero new parameters and leverages the pretrained model's existing induction heads.
    """
    if kv_bank is None:
        return model(query_ids, targets=targets)

    bsz, q_len = query_ids.shape
    mem_len = kv_bank[0][0].shape[2]
    total_len = mem_len + q_len
    if total_len > model.cfg.max_seq_len:
        raise ValueError(f"KV memory ({mem_len}) + query ({q_len}) exceeds max_seq_len ({model.cfg.max_seq_len})")

    x = model.tok_emb(query_ids)
    cos_m = model.rope_cos[:, :, :mem_len, :]
    sin_m = model.rope_sin[:, :, :mem_len, :]
    cos_q = model.rope_cos[:, :, mem_len:total_len, :]
    sin_q = model.rope_sin[:, :, mem_len:total_len, :]

    # Query tokens attend to all M memory tokens + causally to query tokens
    q_causal = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=query_ids.device))
    mem_allow = torch.ones((q_len, mem_len), dtype=torch.bool, device=query_ids.device)
    attn_mask = torch.cat([mem_allow, q_causal], dim=1)[None, None, :, :]

    for block, (k_mem_raw, v_mem) in zip(model.blocks, kv_bank, strict=True):
        attn = block.attn
        xn = block.attn_norm(x)
        q = attn.q_proj(xn).view(bsz, q_len, attn.n_heads, attn.head_dim).transpose(1, 2)
        k_q = attn.k_proj(xn).view(bsz, q_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        v_q = attn.v_proj(xn).view(bsz, q_len, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        if attn.qk_norm:
            q = attn.q_norm(q)
            k_q = attn.k_norm(k_q)

        q_rot = apply_rotary_emb(q, cos_q, sin_q)
        k_mem_rot = apply_rotary_emb(k_mem_raw.to(dtype=q.dtype), cos_m, sin_m)
        k_q_rot = apply_rotary_emb(k_q, cos_q, sin_q)

        k_full = torch.cat([k_mem_rot, k_q_rot], dim=2)
        v_full = torch.cat([v_mem.to(dtype=v_q.dtype), v_q], dim=2)

        y = F.scaled_dot_product_attention(
            q_rot, k_full, v_full, attn_mask=attn_mask, is_causal=False, enable_gqa=attn.use_gqa
        )
        y = y.transpose(1, 2).contiguous().view(bsz, q_len, attn.n_heads * attn.head_dim)
        x = x + attn.o_proj(y)
        x = x + block.mlp(block.mlp_norm(x))

    logits = model.lm_head(model.final_norm(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    return logits, loss
