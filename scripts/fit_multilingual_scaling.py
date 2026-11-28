#!/usr/bin/env python3
"""Fit empirical multilingual capacity scaling law from Phase 5.3B Stage 0 grid runs.

Reads eval.jsonl from runs/stage0_{15m,26m,55m}_{k1_en,k2_ende,k4_4script}/ and fits:
    BPB_EN(N, K) = L_inf + A * (N / K^gamma)^(-alpha)
separately for same-script (K=2: EN+DE -> gamma_same) and cross-script (K=4: EN+RU+AR+ZH -> gamma_cross),
then prints a markdown summary table and predicted iso-quality flagship parameter budgets N*(K).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from coregrad.config import load_local_paths

GRID_SIZES = [
    ("15m", 14.14),
    ("26m", 25.60),
    ("55m", 54.72),
]

GRID_MIXS = [
    ("k1_en", 1, ["en"]),
    ("k2_ende", 2, ["en", "de"]),
    ("k4_4script", 4, ["en", "ru", "ar", "zh"]),
]


def _load_final_eval(run_dir: Path) -> dict[str, object] | None:
    eval_path = run_dir / "eval.jsonl"
    if not eval_path.exists():
        return None
    lines = [ln.strip() for ln in eval_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    return dict(json.loads(lines[-1]))


def _fit_power_law_3pt(ns: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Fit y = L_inf + A * N^(-alpha) over 3+ points by grid search over L_inf + closed-form log-linear fit."""
    best = (float("inf"), 0.5, 1.0, 0.2)
    y_min = float(ys.min())
    for l_inf in np.linspace(max(0.1, y_min - 0.8), y_min - 0.02, 400):
        shifted = ys - l_inf
        if np.any(shifted <= 1e-6):
            continue
        lx = np.log(ns)
        ly = np.log(shifted)
        slope, intercept = np.polyfit(lx, ly, 1)
        alpha = -float(slope)
        if not (0.02 <= alpha <= 1.5):
            continue
        a_coef = math.exp(float(intercept))
        pred = l_inf + a_coef * (ns ** (-alpha))
        err = float(np.mean((pred - ys) ** 2))
        if err < best[0]:
            best = (err, float(l_inf), a_coef, alpha)
    return best[1], best[2], best[3]


def _infer_gamma(n_val: float, y_k: float, k: int, l_inf: float, a_coef: float, alpha: float) -> float:
    """Solve y_k = L_inf + A * (N / K^gamma)^(-alpha) for gamma."""
    if y_k <= l_inf + 1e-6:
        return 0.0
    n_eff = (a_coef / (y_k - l_inf)) ** (1.0 / alpha)
    ratio = n_val / max(n_eff, 1e-6)
    return math.log(max(ratio, 1e-6)) / math.log(k)


def main() -> int:
    runs_dir = load_local_paths().runs_dir
    table: dict[tuple[str, str], dict[str, object]] = {}

    print("| Size (`N`) | Mix (`K`) | Trained Mean BPB | `en` BPB | `de` BPB | `ru` BPB | `ar` BPB | `zh` BPB | `R` (val) | `wRMS` |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stag, n_m in GRID_SIZES:
        for mtag, k, trained in GRID_MIXS:
            rdir = runs_dir / f"stage0_{stag}_{mtag}"
            ev = _load_final_eval(rdir)
            if ev is None:
                print(f"| `{n_m:.2f}M` | `{mtag}` (`K={k}`) | *missing* | — | — | — | — | — | — | — |")
                continue
            table[(stag, mtag)] = ev
            doms = ev.get("domain_bpb", {})
            assert isinstance(doms, dict)
            vitals = ev.get("vitals", {})
            assert isinstance(vitals, dict)

            def _fmt(lang: str, d_map: dict[str, object] = doms, tr: list[str] = trained) -> str:
                v = d_map.get(lang)
                if v is None:
                    return "—"
                s = f"{float(v):.4f}"
                return f"**{s}**" if lang in tr else f"*{s}*"

            print(
                f"| `{n_m:.2f}M` | `{mtag}` (`K={k}`) | **{float(ev['mean_bpb']):.4f}** | "
                f"{_fmt('en')} | {_fmt('de')} | {_fmt('ru')} | {_fmt('ar')} | {_fmt('zh')} | "
                f"{float(vitals.get('compression_ratio', 0.0)):.2f} | {float(vitals.get('max_w_rms', 0.0)):.3f} |"
            )

    # Fit EN K=1 power law
    k1_pts = [
        (n_m, float(table[(stag, "k1_en")]["domain_bpb"]["en"]))  # type: ignore[index]
        for stag, n_m in GRID_SIZES
        if (stag, "k1_en") in table
    ]
    if len(k1_pts) < 3:
        print("\nNeed all 3 K=1 points to fit scaling law.")
        return 0

    ns = np.array([p[0] for p in k1_pts], dtype=np.float64)
    ys = np.array([p[1] for p in k1_pts], dtype=np.float64)
    l_inf, a_coef, alpha = _fit_power_law_3pt(ns, ys)
    print(f"\n### Fitted English K=1 Power Law: BPB_EN(N, 1) = {l_inf:.4f} + {a_coef:.4f} * N^(-{alpha:.4f})")

    for label, mtag, k in [("Same-Script (EN+DE)", "k2_ende", 2), ("Cross-Script (EN+RU+AR+ZH)", "k4_4script", 4)]:
        gammas = []
        deltas = []
        for stag, n_m in GRID_SIZES:
            if (stag, mtag) in table and (stag, "k1_en") in table:
                y1 = float(table[(stag, "k1_en")]["domain_bpb"]["en"])  # type: ignore[index]
                yk = float(table[(stag, mtag)]["domain_bpb"]["en"])  # type: ignore[index]
                g = _infer_gamma(n_m, yk, k, l_inf, a_coef, alpha)
                gammas.append((n_m, g, yk - y1))
                deltas.append(yk - y1)
        if gammas:
            mean_g = sum(x[1] for x in gammas) / len(gammas)
            print(f"\n**{label} (`K={k}`):** mean `gamma = {mean_g:.3f}` | mean `ΔBPB_EN = {sum(deltas)/len(deltas):+.4f}`")
            for n_m, g, d in gammas:
                print(f"  - N={n_m:5.2f}M: ΔBPB_EN = {d:+.4f} -> inferred gamma = {g:.3f}")
            print(f"  - Predicted iso-quality capacity vs 25.6M EN-only (`N*(K) = 25.6 * K^{mean_g:.3f}`):")
            for k_pred in [4, 6, 8, 10, 12]:
                print(f"      K={k_pred:2d} languages -> ~{25.6 * (k_pred ** max(mean_g, 0.0)):5.1f}M params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
