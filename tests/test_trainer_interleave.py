"""Unit tests for interleaved assistant-masked WSD mid-training and optimizer-state branching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from coregrad.model import CoreGradLM, ModelConfig
from coregrad.synthetic.dsl_gen import generate_dsl_tasks
from coregrad.tokenizer import train_bpe_tokenizer
from coregrad.train.trainer import TrainConfig, _PackedSFTCycle, train_pretrain


def test_packed_sft_cycle_and_interleaved_pretrain_step(tmp_path: Path) -> None:
    # Build tiny synthetic corpus, tokenizer, binary shards, and SFT jsonl
    tasks = generate_dsl_tasks(12, seed=42)
    corpus = [
        " ".join(t.content for t in task.turns)
        for task in tasks
    ]
    tok_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=512, output_path=tok_path)

    sft_path = tmp_path / "sft.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            # Keep system prompt short for tiny 512-vocab unit test
            turns = [
                {"role": t.role, "content": "You are coregrad." if t.role == "system" else t.content}
                for t in task.turns
            ]
            f.write(json.dumps({"turns": turns}) + "\n")

    cycle = _PackedSFTCycle(sft_path, tokenizer=tokenizer, seq_len=192, batch_size=2, seed=7)
    x, y = cycle.next_batch(device=torch.device("cpu"))
    assert x.shape == (2, 192)
    assert y.shape == (2, 192)
    # Prompt/result tokens must be masked to -100 while assistant/call tokens remain active
    assert int((y == -100).sum().item()) > 0
    assert int((y != -100).sum().item()) > 0

    # Write minimal single-domain pretraining shard + manifest for end-to-end interleaved trainer test
    shards_dir = tmp_path / "shards"
    (shards_dir / "edu-en").mkdir(parents=True)
    toks = [np.array(tokenizer.encode(c)[:80], dtype=np.uint16) for c in corpus]
    flat = np.concatenate(toks * 12)
    (shards_dir / "edu-en" / "train_0000.bin").write_bytes(flat.tobytes())
    (shards_dir / "edu-en" / "val_0000.bin").write_bytes(flat.tobytes())
    manifest = {
        "vocab_size": tokenizer.vocab_size,
        "domains": {
            "edu-en": {
                "train": {"tokens": int(flat.size), "utf8_bytes": int(flat.size * 3), "shards": ["train_0000.bin"]},
                "val": {"tokens": int(flat.size), "utf8_bytes": int(flat.size * 3), "shards": ["val_0000.bin"]},
            }
        },
    }
    (shards_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model_yaml = tmp_path / "model_tiny.yaml"
    model_yaml.write_text(
        "name: tiny-gru\nn_layer: 4\nd_model: 64\nn_heads: 4\nn_kv_heads: 2\nd_ff: 128\n"
        "vocab_size: 512\nmax_seq_len: 256\nloop_start: 1\nloop_end: 3\nn_loops: 2\nloop_mode: gru\n",
        encoding="utf-8",
    )

    base_cfg = TrainConfig(
        run_name="tiny_base",
        model_config=str(model_yaml),
        optimizer="muon",
        total_tokens=1536,
        seq_len=192,
        micro_batch=2,
        grad_accum=2,
        warmup_fraction=0.0,
        decay_fraction=0.0,
        compile_model=False,
        eval_interval=2,
        eval_val_batches=1,
        sample_count=1,
        sample_max_tokens=8,
        log_interval=1,
        save_interval=0,
        mixture={"edu-en": 1.0},
    )
    runs_dir = tmp_path / "runs"
    base_dir = train_pretrain(
        base_cfg, shards_dir=shards_dir, tokenizer_path=tok_path, runs_dir=runs_dir, device=torch.device("cpu")
    )
    assert (base_dir / "latest.pt").exists()

    # Branch an interleaved decay inheriting optimizer state
    decay_cfg = TrainConfig(
        run_name="tiny_interleaved",
        model_config=str(model_yaml),
        optimizer="muon",
        total_tokens=1536,
        seq_len=192,
        micro_batch=2,
        grad_accum=2,
        warmup_fraction=0.0,
        decay_fraction=1.0,
        init_optimizer=True,
        interleave_sft_jsonl=str(sft_path),
        interleave_sft_ratio=0.5,
        compile_model=False,
        eval_interval=2,
        eval_val_batches=1,
        sample_count=1,
        sample_max_tokens=8,
        log_interval=1,
        save_interval=0,
        mixture={"edu-en": 1.0},
    )
    dec_dir = train_pretrain(
        decay_cfg,
        init_from=base_dir / "latest.pt",
        shards_dir=shards_dir,
        tokenizer_path=tok_path,
        runs_dir=runs_dir,
        device=torch.device("cpu"),
    )
    ckpt = torch.load(dec_dir / "latest.pt", map_location="cpu", weights_only=False)
    model = CoreGradLM(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["step"] == 2
