#!/usr/bin/env python3
"""Launch the live coregrad training monitor or export a static HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path

from coregrad.config import load_local_paths
from coregrad.train.monitor import list_runs, load_run_payload, render_static_report, serve_monitor


def print_cli_summary(runs_dir: Path, run_name: str | None) -> int:
    runs = list_runs(runs_dir)
    if not runs:
        print(f"No runs found in {runs_dir}")
        return 0
    target = run_name or runs[0]
    data = load_run_payload(runs_dir, target)
    metrics = data.get("metrics", [])
    evals = data.get("evals", [])
    samples = data.get("samples", [])
    print(f"Run: {target} ({len(metrics)} logged steps, {len(evals)} evals, {len(samples)} sample snapshots)")
    if metrics:
        m = metrics[-1]
        print(
            f"  Latest step={m['step']:,} | tokens={m['tokens']/1e6:.2f}M | "
            f"loss={m['loss']:.4f} | lr={m['lr']:.2e} | {m['tok_s']:,.0f} tok/s | "
            f"vram={m['vram_gb']:.2f} GB"
        )
    if evals:
        e = evals[-1]
        print(
            f"  Latest eval step={e['step']:,} | mean_bpb={e['mean_bpb']:.4f} | "
            f"grammar={e.get('grammar_acc', 0)*100:.1f}% | 4-way={e.get('completion_acc', 0)*100:.1f}%"
        )
    if samples:
        snap = samples[-1]
        print(f"\nLatest sample generations (step {snap['step']}):")
        for item in snap["samples"][:3]:
            print(f"  Prompt: {item['prompt']}")
            print(f"  Output: {item['completion']}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live training monitor & report exporter for coregrad.")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cli", action="store_true", help="Print latest run summary to terminal")
    parser.add_argument("--run", type=str, default=None, help="Run name for --cli or --export")
    parser.add_argument("--export", type=Path, default=None, help="Export static HTML report path")
    args = parser.parse_args()

    runs_dir = args.runs_dir or load_local_paths().runs_dir

    if args.cli:
        return print_cli_summary(runs_dir, args.run)

    if args.export is not None:
        runs = list_runs(runs_dir)
        if not runs:
            raise SystemExit(f"No runs found in {runs_dir}")
        target = args.run or runs[0]
        out = render_static_report(runs_dir, target, args.export)
        print(f"Exported static report for '{target}' -> {out}")
        return 0

    serve_monitor(runs_dir, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
