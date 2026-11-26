"""Phase 4A unit tests: looped/recursive Transformer configs, Muon exclusion, and KV-bank compatibility."""

from __future__ import annotations

from pathlib import Path

import torch

from coregrad.model import CoreGradLM, ModelConfig
from coregrad.model.memory_bridge import encode_kv_passage, forward_with_kv_bank
from coregrad.train.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]


def test_25m_configs_are_iso_parameter_and_valid() -> None:
    cfg_dense = ModelConfig.from_yaml(ROOT / "configs" / "model_25m_dense.yaml")
    cfg_l12 = ModelConfig.from_yaml(ROOT / "configs" / "model_25m_looped12.yaml")
    cfg_l14 = ModelConfig.from_yaml(ROOT / "configs" / "model_25m_looped14.yaml")

    assert cfg_dense.effective_layers == 12
    assert cfg_l12.effective_layers == 12
    assert cfg_l14.effective_layers == 14

    with torch.device("meta"):
        p_dense = CoreGradLM(cfg_dense).num_parameters()
        p_l12 = CoreGradLM(cfg_l12).num_parameters()
        p_l14 = CoreGradLM(cfg_l14).num_parameters()

    assert 24_000_000 <= p_dense <= 27_000_000
    assert 24_000_000 <= p_l12 <= 27_000_000
    assert 24_000_000 <= p_l14 <= 27_000_000
    # Looped12 and Looped14 share exact same 6 unique blocks (differ only by 512 floats in loop_emb)
    assert abs(p_l14 - p_l12) == 512


def test_looped_forward_backward_muon_and_kv_bank_identity() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(
        name="tiny-looped",
        vocab_size=128,
        max_seq_len=64,
        n_layer=6,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_loops=3,
        loop_start=1,
        loop_end=5,
        loop_step_embed=True,
    )
    assert cfg.effective_layers == 14
    model = CoreGradLM(cfg)
    assert model.loop_emb is not None and model.loop_emb.shape == (3, 64)

    # loop_emb must be routed to AdamW (not Muon)
    opt = build_optimizer(model, optimizer_type="muon", lr=1e-3, muon_lr=0.02, use_fused=False)
    muon_ids = {id(p) for pg in opt.optimizers[0].param_groups for p in pg["params"]}
    assert id(model.loop_emb) not in muon_ids

    x = torch.randint(0, 128, (2, 12))
    y = torch.randint(0, 128, (2, 12))
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    assert model.loop_emb.grad is not None and model.loop_emb.grad.abs().sum().item() > 0.0

    # Verify KV-bank injection matches full joint prefill on looped model too!
    model.eval()
    p_ids = torch.randint(0, 128, (1, 10))
    q_ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        full_logits, _ = model(torch.cat([p_ids, q_ids], dim=1))
        kv_bank = encode_kv_passage(model, p_ids)
        assert len(kv_bank) == 14
        kv_logits, _ = forward_with_kv_bank(model, q_ids, kv_bank=kv_bank)
    assert torch.allclose(full_logits[:, 10:, :], kv_logits, atol=1e-5)
