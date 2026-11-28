"""coregrad-Byte: H-Net End-to-End Dynamic UTF-8 Patching + Outer-GRU Looped Cognitive Core.

Architecture (strictly causal at both byte and chunk resolution):
  1. Raw UTF-8 + special-token embedding (vocab=288 -> ~0.15M params, zero BPE lock-in)
  2. Local Byte Encoder: n_enc causal sliding-window Transformer blocks (window_size=256)
  3. H-Net Cosine-Similarity Boundary Router:
       p_t = 0.5 * (1 - cos(q_t, k_{t-1})), p_0 = 1.0, b_t = (p_t >= 0.5)
       Ratio load-balancing loss guides mean compression toward target_ratio (~4.5 bytes/chunk)
  4. Downsampling -> Global n_core x n_loops Outer-GRU Depth-RNN Core at chunk resolution
     (valid chunks sit contiguously on the left, so standard causal Flash SDPA is exact!)
  5. Chunk-level parallel O(log M) causal EMA smoothing + Straight-Through Estimator (STE)
     confidence-gated upsampling + residual skip from byte encoder
  6. Local Byte Decoder: n_dec causal sliding-window Transformer blocks -> 288-way head
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_COMPILED_FLEX_ATTN: Any = None


def _get_compiled_flex_attn() -> Any:
    global _COMPILED_FLEX_ATTN
    if _COMPILED_FLEX_ATTN is None:
        _COMPILED_FLEX_ATTN = torch.compile(flex_attention)
    return _COMPILED_FLEX_ATTN

from coregrad.config import load_yaml
from coregrad.model.transformer import (
    ModelConfig,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rotary_emb,
    precompute_rope_freqs,
)
from coregrad.tokenizer.byte_tok import BYTE_VOCAB_SIZE


@dataclass
class HNetConfig:
    name: str = "coregrad-32m-hnet-gru"
    vocab_size: int = BYTE_VOCAB_SIZE
    max_seq_len: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1408
    n_enc: int = 2
    n_core: int = 4
    n_loops: int = 3
    n_dec: int = 2
    window_size: int = 256
    target_ratio: float = 4.5
    ratio_loss_weight: float = 0.03
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    loop_step_embed: bool = True
    loop_mode: str = "gru"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be > 1.0, got {self.target_ratio}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def effective_layers(self) -> int:
        return self.n_enc + self.n_core * self.n_loops + self.n_dec

    def to_block_cfg(self) -> ModelConfig:
        return ModelConfig(
            name=self.name,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            n_layer=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            norm_eps=self.norm_eps,
            qk_norm=self.qk_norm,
            tie_embeddings=self.tie_embeddings,
        )

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> HNetConfig:
        data = load_yaml(Path(path))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class SlidingWindowAttention(nn.Module):
    """Strictly causal local sliding-window GQA using flex_attention on CUDA and masked SDPA on CPU."""

    def __init__(self, cfg: HNetConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = cfg.window_size
        self.use_gqa = self.n_kv_heads != self.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: Any = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if seq_len <= self.window_size:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.use_gqa)
        elif block_mask is not None and x.is_cuda:
            y = _get_compiled_flex_attn()(q, k, v, block_mask=block_mask, enable_gqa=self.use_gqa)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=self.use_gqa)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y)


class LocalByteBlock(nn.Module):
    """Pre-RMSNorm sliding-window causal Transformer block for local byte encoding/decoding."""

    def __init__(self, cfg: HNetConfig) -> None:
        super().__init__()
        block_cfg = cfg.to_block_cfg()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = SlidingWindowAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(block_cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: Any = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, block_mask=block_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class HNetRouter(nn.Module):
    """End-to-end differentiable cosine-similarity boundary router (Hwang, Wang & Gu, 2025)."""

    def __init__(self, d_model: int, target_ratio: float = 4.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.target_ratio = target_ratio
        self.eps = eps
        self.norm = RMSNorm(d_model, eps=eps)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Learnable threshold shift lets ratio loss center R ~ target_ratio within ~20 steps
        self.bias = nn.Parameter(torch.zeros(()))
        # Initialize close to identity so initial cosine similarity is meaningful and well-conditioned
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (boundary_probs (B, L), boundary_mask (B, L) bool, ratio_loss scalar)."""
        bsz, seq_len, _ = h.shape
        hn = self.norm(h).float()
        q = F.normalize(self.q_proj(hn), dim=-1, eps=self.eps)
        k = F.normalize(self.k_proj(hn), dim=-1, eps=self.eps)

        # Strictly causal: p_t = 0.5 * (1 - cos(q_t, k_{t-1}) + bias), forced p_0 = 1.0
        cos_sim = (q[:, 1:, :] * k[:, :-1, :]).sum(dim=-1)
        p_rest = (0.5 * (1.0 - cos_sim + self.bias)).clamp(1e-5, 1.0 - 1e-5)
        p0 = torch.ones((bsz, 1), device=h.device, dtype=torch.float32)
        p = torch.cat([p0, p_rest], dim=1)  # (B, L)

        b = p >= 0.5
        # H-Net ratio load-balancing loss (exact gradient: dL/dG propto N*F - 1)
        n = self.target_ratio
        if seq_len > 1:
            f_actual = b[:, 1:].float().mean().detach()
            g_prob = p[:, 1:].mean()
            ratio_loss = (n / (n - 1.0)) * ((n - 1.0) * f_actual * g_prob + (1.0 - f_actual) * (1.0 - g_prob))
        else:
            ratio_loss = torch.zeros((), device=h.device, dtype=torch.float32)
        return p, b, ratio_loss


def parallel_ema_scan(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Exact causal EMA smoothing z_tilde[m] = p[m]*z[m] + (1-p[m])*z_tilde[m-1] in O(log M) GPU ops.

    Args:
        z: Chunk latent sequence of shape (B, M, D).
        p: Boundary probabilities of shape (B, M) with p[:, 0] == 1.0.
    """
    m_len = z.size(1)
    if m_len <= 1:
        return z
    a = (1.0 - p.float()).unsqueeze(-1)  # (B, M, 1), a[:, 0] = 0
    u = p.float().unsqueeze(-1) * z.float()  # (B, M, D)
    stride = 1
    while stride < m_len:
        u_prev = u[:, :-stride]
        a_prev = a[:, :-stride]
        u_cur = u[:, stride:]
        a_cur = a[:, stride:]
        u = torch.cat([u[:, :stride], a_cur * u_prev + u_cur], dim=1)
        a = torch.cat([a[:, :stride], a_cur * a_prev], dim=1)
        stride *= 2
    return u.to(z.dtype)


class CoreGradByteLM(nn.Module):
    """H-Net End-to-End Dynamic UTF-8 Byte Language Model with Outer-GRU Looped Core."""

    def __init__(self, cfg: HNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        block_cfg = cfg.to_block_cfg()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.enc_blocks = nn.ModuleList([LocalByteBlock(cfg) for _ in range(cfg.n_enc)])
        self.router = HNetRouter(cfg.d_model, target_ratio=cfg.target_ratio, eps=cfg.norm_eps)

        # Global chunk-level Outer-GRU looped cognitive core
        self.core_blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(cfg.n_core)])
        self.loop_emb: nn.Parameter | None = (
            nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model))
            if (cfg.n_loops > 1 and cfg.loop_step_embed)
            else None
        )
        self.use_gru_loop = cfg.n_loops > 1 and cfg.loop_mode == "gru"
        if self.use_gru_loop:
            self.loop_e0_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.loop_s_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.loop_in_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
            self.loop_gate_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=True)

        # Residual bridge from byte encoder around global core into byte decoder
        self.res_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dec_blocks = nn.ModuleList([LocalByteBlock(cfg) for _ in range(cfg.n_dec)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._block_mask_cache: dict[tuple[int, str], Any] = {}
        self._cpu_mask_cache: dict[int, torch.Tensor] = {}

        self._init_all_weights()

    def _init_all_weights(self) -> None:
        for name, module in self.named_modules():
            if name.startswith("router"):
                continue
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        if self.use_gru_loop:
            nn.init.constant_(self.loop_gate_proj.bias, 0.5)
            nn.init.normal_(self.loop_in_proj.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.cfg.n_loops))
        # Initialize residual skip slightly damped so global chunk latent carries primary signal
        nn.init.normal_(self.res_proj.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2.0 * self.cfg.effective_layers)
        for block in [*self.enc_blocks, *self.core_blocks, *self.dec_blocks]:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=residual_std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def _get_local_masks(self, seq_len: int, device: torch.device) -> tuple[Any, torch.Tensor | None]:
        if seq_len <= self.cfg.window_size:
            return None, None
        if device.type == "cuda":
            key = (seq_len, str(device))
            bm = self._block_mask_cache.get(key)
            if bm is None:
                win = self.cfg.window_size

                def sw_mask(_b: Any, _h: Any, q_idx: Any, kv_idx: Any) -> Any:
                    return (q_idx >= kv_idx) & (q_idx - kv_idx < win)

                bm = create_block_mask(sw_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
                self._block_mask_cache[key] = bm
            return bm, None
        cm = self._cpu_mask_cache.get(seq_len)
        if cm is None:
            idx = torch.arange(seq_len)
            diff = idx[:, None] - idx[None, :]
            cm = (diff >= 0) & (diff < self.cfg.window_size)
            self._cpu_mask_cache[seq_len] = cm
        return None, cm.to(device)

    def _run_global_core(self, z: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Run n_core x n_loops Outer-GRU depth core over padded chunk tensor (B, M, D)."""
        m_len = z.size(1)
        cos = self.rope_cos[:, :, :m_len, :]
        sin = self.rope_sin[:, :, :m_len, :]

        e0_norm = self.loop_e0_norm(z) if self.use_gru_loop else None
        s = z
        loop_cos_val = 1.0
        gate_mean_val = 0.0
        prev_state: torch.Tensor | None = None

        for loop_idx in range(self.cfg.n_loops):
            if self.use_gru_loop:
                assert e0_norm is not None
                s_norm = self.loop_s_norm(s)
                cat = torch.cat([s_norm, e0_norm], dim=-1)
                h = s_norm + self.loop_in_proj(F.silu(cat))
                if self.loop_emb is not None:
                    h = h + self.loop_emb[loop_idx].to(h.dtype)
                for block in self.core_blocks:
                    h = block(h, cos, sin)
                gate = torch.sigmoid(self.loop_gate_proj(torch.cat([s_norm, self.loop_s_norm(h)], dim=-1)))
                s = (1.0 - gate) * s + gate * h
                if loop_idx == self.cfg.n_loops - 1:
                    gate_mean_val = float(gate.detach().float().mean().item())
            else:
                if self.loop_emb is not None:
                    s = s + self.loop_emb[loop_idx].to(s.dtype)
                for block in self.core_blocks:
                    s = block(s, cos, sin)
            if loop_idx == 1 and prev_state is not None:
                loop_cos_val = float(
                    F.cosine_similarity(
                        prev_state.detach().float().reshape(-1, s.size(-1)),
                        s.detach().float().reshape(-1, s.size(-1)),
                        dim=-1,
                    )
                    .mean()
                    .item()
                )
            if loop_idx == 0:
                prev_state = s.detach()
        return s, loop_cos_val, gate_mean_val

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos[:, :, :seq_len, :]
        sin = self.rope_sin[:, :, :seq_len, :]
        block_mask, attn_mask = self._get_local_masks(seq_len, input_ids.device)

        # 1. Local byte encoder
        h_enc = self.tok_emb(input_ids)
        for block in self.enc_blocks:
            h_enc = block(h_enc, cos, sin, block_mask=block_mask, attn_mask=attn_mask)

        # 2. End-to-end differentiable cosine boundary router
        p, b_mask, ratio_loss = self.router(h_enc)  # p: (B, L) fp32, b_mask: (B, L) bool

        # 3. Downsample selected boundary vectors contiguously onto left side of (B, M, D)
        slot = b_mask.long().cumsum(dim=1) - 1  # (B, L) in [0, count_i - 1]
        counts = b_mask.long().sum(dim=1)  # (B,)
        max_chunks = max(1, int(counts.max().item()))

        z_in = h_enc.new_zeros((bsz, max_chunks, self.cfg.d_model))
        p_sel = p.new_ones((bsz, max_chunks))
        batch_idx, pos_idx = b_mask.nonzero(as_tuple=True)
        slot_idx = slot[batch_idx, pos_idx]
        z_in[batch_idx, slot_idx] = h_enc[batch_idx, pos_idx]
        p_sel[batch_idx, slot_idx] = p[batch_idx, pos_idx]

        # 4. Global Outer-GRU core at chunk resolution (strictly causal: right padding never attended)
        z_out, loop_cos_val, gate_mean_val = self._run_global_core(z_in)

        # 5. Parallel O(log M) chunk EMA smoothing + STE confidence-gated upsampling + residual skip
        z_smooth = parallel_ema_scan(z_out, p_sel)  # (B, M, D)
        gather_idx = slot.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
        z_up = torch.gather(z_smooth, dim=1, index=gather_idx)  # (B, L, D)

        # Confidence for chosen action: p_t if boundary else (1 - p_t); STE gate has forward value 1.0
        conf = torch.where(b_mask, p, 1.0 - p)
        ste_gate = (1.0 + (conf - conf.detach())).to(z_up.dtype).unsqueeze(-1)
        h_dec = ste_gate * z_up + self.res_proj(h_enc)

        # 6. Local byte decoder -> 288-way logits
        for block in self.dec_blocks:
            h_dec = block(h_dec, cos, sin, block_mask=block_mask, attn_mask=attn_mask)

        logits = self.lm_head(self.final_norm(h_dec))
        ce_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            total_loss = (
                ce_loss + self.cfg.ratio_loss_weight * ratio_loss.to(ce_loss.dtype)
                if self.training
                else ce_loss
            )

        if return_aux:
            actual_ratio = float(seq_len) / max(float(counts.float().mean().item()), 1.0)
            aux = {
                "ce_loss": float(ce_loss.detach().item()) if ce_loss is not None else 0.0,
                "ratio_loss": float(ratio_loss.detach().item()),
                "compression_ratio": round(actual_ratio, 3),
                "mean_chunks": round(float(counts.float().mean().item()), 1),
                "loop_cos": round(loop_cos_val, 4),
                "loop_gate_mean": round(gate_mean_val, 4),
                "boundary_mask": b_mask.detach(),
                "boundary_probs": p.detach(),
            }
            return logits, total_loss, aux
        return logits, total_loss

    @torch.no_grad()
    def compute_health_vitals(self, input_ids: torch.Tensor) -> dict[str, float]:
        was_training = self.training
        self.eval()
        max_w_rms = 0.0
        for name, p in self.named_parameters():
            if p.ndim >= 2 and "tok_emb" not in name and "lm_head" not in name and "router" not in name:
                wrms = float(p.detach().float().pow(2).mean().sqrt().item())
                max_w_rms = max(max_w_rms, wrms)
        amp = input_ids.is_cuda
        with torch.amp.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _, aux = self(input_ids, return_aux=True)  # type: ignore[misc]
        max_logit = float(logits.detach().float().abs().max().item())
        # Measure final hidden RMS before final_norm via quick check on logits proxy or enc/dec
        _bsz, seq_len = input_ids.shape
        cos = self.rope_cos[:, :, :seq_len, :]
        sin = self.rope_sin[:, :, :seq_len, :]
        bm, am = self._get_local_masks(seq_len, input_ids.device)
        with torch.amp.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=amp):
            h = self.tok_emb(input_ids)
            for blk in self.enc_blocks:
                h = blk(h, cos, sin, block_mask=bm, attn_mask=am)
        act_rms = float(h.detach().float().pow(2).mean().sqrt().item())
        if was_training:
            self.train()
        return {
            "max_w_rms": round(max_w_rms, 5),
            "act_rms": round(act_rms, 3),
            "max_logit": round(max_logit, 2),
            "loop_cos": float(aux["loop_cos"]),
            "loop_gate_mean": float(aux["loop_gate_mean"]),
            "compression_ratio": float(aux["compression_ratio"]),
        }

    @torch.no_grad()
    def inspect_chunks(self, text: str, tokenizer: Any, device: torch.device) -> list[str]:
        """Return the learned UTF-8 byte chunks produced by the H-Net router on a sample string."""
        self.eval()
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if not ids:
            return []
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, _, aux = self(x, return_aux=True)  # type: ignore[misc]
        b_mask = aux["boundary_mask"][0].tolist()
        chunks: list[list[int]] = []
        for tid, is_start in zip(ids, b_mask, strict=True):
            if is_start or not chunks:
                chunks.append([tid])
            else:
                chunks[-1].append(tid)
        return [tokenizer.decode(c, skip_special_tokens=False) for c in chunks]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 40,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        amp = out.is_cuda
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.max_seq_len :]
            with torch.amp.autocast(device_type=ctx.device.type, dtype=torch.bfloat16, enabled=amp):
                logits, _ = self(ctx)
            next_logits = logits[:, -1, :].float()
            if temperature <= 0.0:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=1)
            if eos_id is not None and bool((next_tok == eos_id).all()):
                break
        return out
