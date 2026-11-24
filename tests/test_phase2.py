"""Phase 2 unit tests: WSD schedule, AdamW/Muon optimizers, and resumable training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from coregrad.data import pack_domain_shards
from coregrad.model import CoreGradLM, ModelConfig
from coregrad.tokenizer import train_bpe_tokenizer
from coregrad.train import TrainConfig, build_optimizer, train_pretrain, wsd_lr
from coregrad.train.monitor import load_run_payload


def test_wsd_schedule_phases() -> None:
    # Warmup
    assert wsd_lr(0, 100, peak_lr=1e-3, min_lr=1e-4, warmup_steps=10, decay_steps=20) == pytest.approx(1e-4)
    assert wsd_lr(9, 100, peak_lr=1e-3, min_lr=1e-4, warmup_steps=10, decay_steps=20) == pytest.approx(1e-3)
    # Stable
    assert wsd_lr(50, 100, peak_lr=1e-3, min_lr=1e-4, warmup_steps=10, decay_steps=20) == pytest.approx(1e-3)
    # Decay end
    assert wsd_lr(100, 100, peak_lr=1e-3, min_lr=1e-4, warmup_steps=10, decay_steps=20) == pytest.approx(1e-4)


@pytest.mark.parametrize("opt_type", ["adamw", "muon"])
def test_optimizer_reduces_loss(opt_type: str) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        name="tiny-opt",
        vocab_size=128,
        max_seq_len=32,
        n_layer=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )
    model = CoreGradLM(cfg)
    opt = build_optimizer(model, optimizer_type=opt_type, lr=3e-3, muon_lr=0.02, use_fused=False)

    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))
    losses: list[float] = []
    for _ in range(8):
        opt.zero_grad()
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.7


def test_train_pretrain_end_to_end_and_resume(tmp_path: Path) -> None:
    docs = [
        f"The quick brown fox jumps over the lazy dog in experiment {i}. Water boils at 100 C."
        for i in range(150)
    ]
    tok_path = tmp_path / "tokenizer.json"
    tok = train_bpe_tokenizer(iter(docs), vocab_size=256, output_path=tok_path)

    shards_root = tmp_path / "shards"
    dm = pack_domain_shards(
        domain="edu-en",
        texts=iter(docs),
        tokenizer=tok,
        out_dir=shards_root,
        max_train_tokens=50_000,
        max_val_tokens=10_000,
        val_fraction=0.25,
        shard_tokens=20_000,
    )
    manifest = {
        "vocab_size": tok.vocab_size,
        "domains": {
            "edu-en": {
                "train": {
                    "shards": dm.train.shards,
                    "tokens": dm.train.tokens,
                    "utf8_bytes": dm.train.utf8_bytes,
                    "docs": dm.train.docs,
                },
                "val": {
                    "shards": dm.val.shards,
                    "tokens": dm.val.tokens,
                    "utf8_bytes": dm.val.utf8_bytes,
                    "docs": dm.val.docs,
                },
            }
        },
    }
    (shards_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model_yaml = tmp_path / "tiny_model.yaml"
    model_yaml.write_text(
        "name: tiny\nvocab_size: 256\nmax_seq_len: 64\nn_layer: 2\n"
        "d_model: 32\nn_heads: 4\nn_kv_heads: 2\nd_ff: 64\n",
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    cfg = TrainConfig(
        run_name="tiny_smoke",
        model_config=str(model_yaml),
        optimizer="muon",
        total_tokens=256,
        seq_len=16,
        micro_batch=2,
        grad_accum=2,
        log_interval=2,
        eval_interval=4,
        eval_val_batches=2,
        sample_count=2,
        sample_max_tokens=8,
        save_interval=0,
        compile_model=False,
        mixture={"edu-en": 1.0},
    )
    run_dir = train_pretrain(
        cfg,
        shards_dir=shards_root,
        tokenizer_path=tok_path,
        runs_dir=runs_dir,
        device=torch.device("cpu"),
    )
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()
    payload = load_run_payload(runs_dir, "tiny_smoke")
    assert len(payload["metrics"]) >= 2
    assert len(payload["evals"]) >= 2
    assert len(payload["samples"]) >= 2

    # Resume for 4 more steps (512 total tokens)
    cfg.total_tokens = 512
    train_pretrain(
        cfg,
        resume=True,
        shards_dir=shards_root,
        tokenizer_path=tok_path,
        runs_dir=runs_dir,
        device=torch.device("cpu"),
    )
    payload2 = load_run_payload(runs_dir, "tiny_smoke")
    assert payload2["metrics"][-1]["step"] == 8
