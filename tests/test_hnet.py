"""Unit tests for Gate 4 coregrad-Byte (ByteTokenizer + H-Net dynamic patching + Outer GRU)."""

from __future__ import annotations

import torch

from coregrad.model.hnet import CoreGradByteLM, HNetConfig, parallel_ema_scan
from coregrad.tokenizer.byte_tok import BYTE_VOCAB_SIZE, ByteTokenizer
from coregrad.tokenizer.special_tokens import CALL_END, CALL_START, USER_END, USER_START


def test_byte_tokenizer_multiscript_and_special_tokens() -> None:
    tok = ByteTokenizer()
    assert tok.vocab_size == BYTE_VOCAB_SIZE == 288
    sample = (
        f"{USER_START}EN hello · DE Über · RU Привет · AR مرحبا · HI नमस्ते · ZH 你好 · JA こんにちは{USER_END}"
        f"{CALL_START}calc(expr=\"1..5 |> sum\"){CALL_END}"
    )
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.token_to_id(USER_START) in ids
    assert tok.token_to_id(CALL_START) in ids
    # Exact round-trip with special tokens preserved
    assert tok.decode(ids[1:-1], skip_special_tokens=False) == sample
    # Clean strip of special tokens
    plain = tok.decode(ids, skip_special_tokens=True)
    assert USER_START not in plain and "Привет" in plain and "你好" in plain and "مرحبا" in plain


def test_parallel_ema_scan_matches_sequential_exact() -> None:
    torch.manual_seed(7)
    z = torch.randn(3, 65, 16)
    p = torch.rand(3, 65).clamp(0.05, 0.95)
    p[:, 0] = 1.0

    fast = parallel_ema_scan(z, p)
    # Reference sequential EMA
    ref = torch.zeros_like(z)
    ref[:, 0] = z[:, 0]
    for m in range(1, z.size(1)):
        pm = p[:, m].unsqueeze(-1)
        ref[:, m] = pm * z[:, m] + (1.0 - pm) * ref[:, m - 1]
    assert torch.allclose(fast, ref, atol=1e-5)


def test_hnet_strict_causality_and_router_gradient_flow() -> None:
    torch.manual_seed(42)
    cfg = HNetConfig(
        name="tiny-hnet",
        vocab_size=BYTE_VOCAB_SIZE,
        max_seq_len=128,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_enc=1,
        n_core=2,
        n_loops=2,
        n_dec=1,
        window_size=32,
        target_ratio=4.0,
    )
    model = CoreGradByteLM(cfg)
    x = torch.randint(32, 122, (2, 48))
    y = torch.randint(32, 122, (2, 48))

    # 1. Strict causality check: modifying byte t=24 must NOT change logits for any position < 24
    model.eval()
    x_mod = x.clone()
    x_mod[:, 24:] = (x_mod[:, 24:] + 13) % 256
    with torch.no_grad():
        logits_a, _ = model(x)
        logits_b, _ = model(x_mod)
    max_prefix_diff = float((logits_a[:, :24] - logits_b[:, :24]).abs().max().item())
    assert max_prefix_diff < 1e-5, f"Causality leak detected: max diff={max_prefix_diff}"

    # 2. End-to-end gradient flow into H-Net router (q_proj, k_proj), global core, and byte enc/dec
    model.train()
    _logits, loss, aux = model(x, y, return_aux=True)  # type: ignore[misc]
    assert loss is not None and torch.isfinite(loss)
    assert aux["compression_ratio"] >= 1.0
    loss.backward()
    for pname in (
        "router.q_proj.weight",
        "router.k_proj.weight",
        "enc_blocks.0.attn.q_proj.weight",
        "core_blocks.0.attn.q_proj.weight",
        "loop_gate_proj.weight",
        "dec_blocks.0.attn.q_proj.weight",
    ):
        p = dict(model.named_parameters())[pname]
        assert p.grad is not None and float(p.grad.abs().sum().item()) > 0.0, f"Zero grad on {pname}"

    # 3. Generation & chunk inspection smoke test
    tok = ByteTokenizer()
    chunks = model.inspect_chunks("Hello coregrad!", tok, device=torch.device("cpu"))
    assert "".join(chunks) == "Hello coregrad!"
    gen = model.generate(x[:1, :8], max_new_tokens=6, temperature=0.0)
    assert gen.shape == (1, 14)
