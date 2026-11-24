"""Tests for RunLogger, dashboard payload loading, and static HTML export."""

from __future__ import annotations

from pathlib import Path

from coregrad.train.monitor import RunLogger, list_runs, load_run_payload, render_static_report


def test_run_logger_and_static_report(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    logger = RunLogger(runs_dir / "smoke_test", meta={"model": "coregrad-12m", "total_tokens": 100_000})
    logger.log_step(
        step=10,
        tokens_seen=16384,
        train_loss=5.12,
        lr=3e-4,
        grad_norm=1.2,
        tok_per_sec=120_000.0,
        vram_gb=2.1,
        elapsed_s=1.5,
    )
    logger.log_eval(
        step=10,
        tokens_seen=16384,
        mean_bpb=1.85,
        domain_bpb={"edu-en": 1.80, "cosmo": 1.90},
        grammar_acc=0.625,
        completion_acc=0.375,
    )
    logger.log_samples(
        step=10,
        tokens_seen=16384,
        samples=[{"prompt": "The sky is blue because", "completion": " sunlight scatters."}],
    )

    assert list_runs(runs_dir) == ["smoke_test"]
    payload = load_run_payload(runs_dir, "smoke_test")
    assert len(payload["metrics"]) == 1
    assert len(payload["evals"]) == 1
    assert len(payload["samples"]) == 1

    out_html = render_static_report(runs_dir, "smoke_test", tmp_path / "report.html")
    content = out_html.read_text(encoding="utf-8")
    assert "__EMBEDDED_PAYLOAD__" in content
    assert "The sky is blue because" in content
