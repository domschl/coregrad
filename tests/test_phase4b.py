"""Phase 4B unit tests: zero-init gated cross-attention identity, gradient flow, query head, RETRO pooling."""

from __future__ import annotations

import torch

from coregrad.model import CoreGradLM, ModelConfig
from coregrad.model.memory_bridge import (
    CoreGradMemoryBridge,
    LatentQueryHead,
    MemoryBridgeConfig,
    encode_kv_passage,
    forward_with_kv_bank,
)


def _tiny_base() -> CoreGradLM:
    cfg = ModelConfig(
        name="tiny-bridge",
        vocab_size=128,
        max_seq_len=64,
        n_layer=6,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )
    return CoreGradLM(cfg)


def test_zero_init_bridge_preserves_exact_base_logits() -> None:
    torch.manual_seed(0)
    base = _tiny_base().eval()
    bridge = CoreGradMemoryBridge(
        base,
        MemoryBridgeConfig(bridge_layers=(2, 4), mem_input_dim=32, mem_expansion_slots=2, freeze_base=True),
        memory_mode="embeddinggemma",
    ).eval()

    x = torch.randint(0, 128, (2, 12))
    mem = torch.randn(2, 3, 32)

    with torch.no_grad():
        base_logits, _ = base(x)
        off_logits, _ = bridge(x, memory=None)
        mem_logits, _ = bridge(x, memory=mem)

    assert torch.allclose(base_logits, off_logits, atol=1e-6)
    # Zero-initialized tanh(0) gate must produce exact base logits at step 0
    assert torch.allclose(base_logits, mem_logits, atol=1e-6)


def test_bridge_gradients_flow_to_gates_and_base_stays_frozen() -> None:
    torch.manual_seed(1)
    base = _tiny_base()
    bridge = CoreGradMemoryBridge(
        base,
        MemoryBridgeConfig(bridge_layers=(2, 4), mem_input_dim=32, mem_expansion_slots=2, freeze_base=True),
        memory_mode="embeddinggemma",
    )
    x = torch.randint(0, 128, (2, 10))
    y = torch.randint(0, 128, (2, 10))
    mem = torch.randn(2, 2, 32)

    _, loss = bridge(x, y, memory=mem)
    assert loss is not None
    loss.backward()

    # Frozen base receives no gradients; gate parameters receive non-zero gradients
    assert all(p.grad is None for p in bridge.base.parameters())
    gate_grads = [m.gate.grad for m in bridge.cross_attns.values()]
    assert all(g is not None and g.abs().item() > 0.0 for g in gate_grads)


def test_retro_passage_encoder_and_latent_query_head() -> None:
    torch.manual_seed(2)
    base = _tiny_base().eval()
    bridge = CoreGradMemoryBridge(
        base,
        MemoryBridgeConfig(bridge_layers=(2, 4), retro_encoder_layers=3, retro_slots_per_passage=6),
        memory_mode="retro",
    )
    passages = torch.randint(0, 128, (4, 24))
    pooled = bridge.encode_retro_passages(passages, slots_per_passage=6)
    assert pooled.shape == (4, 6, 64)

    x = torch.randint(0, 128, (2, 10))
    # 2 passages per batch item -> 12 memory slots
    mem = pooled.view(2, 12, 64)
    logits, _ = bridge(x, memory=mem)
    assert logits.shape == (2, 10, 128)

    hidden = bridge.extract_layer_hidden(x, (3, 6))
    assert hidden[3].shape == (2, 10, 64)
    head = LatentQueryHead(d_model=64, query_dim=32)
    q = head(hidden[3][:, -1, :])
    assert q.shape == (2, 32)
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_retro_tokens_with_memory_mask() -> None:
    torch.manual_seed(3)
    base = _tiny_base().eval()
    bridge = CoreGradMemoryBridge(
        base,
        MemoryBridgeConfig(bridge_layers=(2, 4), retro_encoder_layers=3),
        memory_mode="retro_tokens",
    )
    passages = torch.randint(0, 128, (2, 16))
    states = bridge.encode_retro_token_states(passages)
    assert states.shape == (2, 16, 64)
    mask = torch.zeros((2, 16), dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :12] = True
    x = torch.randint(0, 128, (2, 8))
    logits, _ = bridge(x, memory=states, memory_mask=mask)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()


def test_kv_bank_matches_exact_full_text_prefill() -> None:
    torch.manual_seed(4)
    base = _tiny_base().eval()
    p_ids = torch.randint(0, 128, (1, 14))
    q_ids = torch.randint(0, 128, (1, 8))
    full_ids = torch.cat([p_ids, q_ids], dim=1)

    with torch.no_grad():
        full_logits, _ = base(full_ids)
        kv_bank = encode_kv_passage(base, p_ids)
        kv_logits, _ = forward_with_kv_bank(base, q_ids, kv_bank=kv_bank)

    # Single-passage KV injection must exactly match full joint prefill on query positions
    assert torch.allclose(full_logits[:, 14:, :], kv_logits, atol=1e-5)
