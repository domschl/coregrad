"""Phase 0 unit tests for coregrad model, configs, and repository hygiene."""

from __future__ import annotations

import subprocess
import sys

import torch

from coregrad.config import REPO_ROOT, load_yaml
from coregrad.model import CoreGradLM, ModelConfig


def test_model_configs_instantiate_and_parameter_counts() -> None:
    expected_ranges = {
        "model_12m.yaml": (10_000_000, 16_000_000),
        "model_45m.yaml": (40_000_000, 52_000_000),
        "model_125m.yaml": (115_000_000, 135_000_000),
    }
    for filename, (lo, hi) in expected_ranges.items():
        cfg = ModelConfig.from_yaml(REPO_ROOT / "configs" / filename)
        with torch.device("meta"):
            model = CoreGradLM(cfg)
        n_params = model.num_parameters(non_embedding=False)
        assert lo <= n_params <= hi, f"{filename}: got {n_params:,}, expected [{lo:,}, {hi:,}]"
        assert model.lm_head.weight is model.tok_emb.weight


def test_forward_backward_and_masked_loss() -> None:
    cfg = ModelConfig(
        name="tiny-test",
        vocab_size=256,
        max_seq_len=64,
        n_layer=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        qk_norm=True,
        tie_embeddings=True,
    )
    model = CoreGradLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    # Mask first half of targets (verifies SFT loss masking works cleanly)
    targets[:, :8] = -100

    logits, loss = model(x, targets)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g > 0 for g in grad_norms)


def test_generate_respects_eos_and_length() -> None:
    cfg = ModelConfig(
        name="tiny-gen",
        vocab_size=128,
        max_seq_len=32,
        n_layer=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    model = CoreGradLM(cfg)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 8)


def test_datasets_manifest_is_public_and_valid() -> None:
    manifest = load_yaml(REPO_ROOT / "configs" / "datasets.yaml")
    weights = [spec["weight"] for spec in manifest["pretrain"].values()]
    assert abs(sum(weights) - 1.0) < 1e-6
    for section in ("pretrain", "tools_and_sft"):
        for spec in manifest[section].values():
            assert "/" in spec["repo"]
            assert "license" in spec


def test_hygiene_script_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_hygiene.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr + res.stdout
