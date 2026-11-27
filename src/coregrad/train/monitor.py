"""Lightweight, zero-dependency training logger and live web monitor for coregrad.

Writes append-only JSONL logs per run:
- metrics.jsonl : step, tokens_seen, train_loss, lr, grad_norm, tok_per_sec, vram_gb
- evals.jsonl   : step, tokens_seen, mean_bpb, per-domain bpb, grammar_acc, completion_acc
- samples.jsonl : step, tokens_seen, list of {prompt, completion} generations

Serves an auto-refreshing browser dashboard (loss/BPB/LR/probe plots + sample text timeline)
using only Python's standard library.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


class RunLogger:
    """Append-only JSONL logger for training metrics, evals, and sample generations."""

    def __init__(self, run_dir: Path, meta: dict[str, Any] | None = None) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.evals_path = self.run_dir / "evals.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.meta_path = self.run_dir / "meta.json"

        if meta is not None:
            existing: dict[str, Any] = {}
            if self.meta_path.exists():
                existing = json.loads(self.meta_path.read_text(encoding="utf-8"))
            existing.update(meta)
            existing.setdefault("created_at", int(time.time()))
            existing["updated_at"] = int(time.time())
            self.meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_step(
        self,
        *,
        step: int,
        tokens_seen: int,
        train_loss: float,
        lr: float,
        grad_norm: float,
        tok_per_sec: float,
        vram_gb: float,
        elapsed_s: float,
        max_layer_gn: float | None = None,
        max_w_rms: float | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "t": round(time.time(), 2),
            "step": step,
            "tokens": tokens_seen,
            "loss": round(train_loss, 5),
            "lr": lr,
            "grad_norm": round(grad_norm, 4),
            "tok_s": round(tok_per_sec, 1),
            "vram_gb": round(vram_gb, 3),
            "elapsed_s": round(elapsed_s, 1),
        }
        if max_layer_gn is not None:
            rec["max_layer_gn"] = round(max_layer_gn, 4)
        if max_w_rms is not None:
            rec["max_w_rms"] = round(max_w_rms, 5)
        self._append(self.metrics_path, rec)

    def log_eval(
        self,
        *,
        step: int,
        tokens_seen: int,
        mean_bpb: float,
        domain_bpb: dict[str, float],
        grammar_acc: float | None = None,
        completion_acc: float | None = None,
        vitals: dict[str, float] | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "t": round(time.time(), 2),
            "step": step,
            "tokens": tokens_seen,
            "mean_bpb": round(mean_bpb, 5),
            "domain_bpb": {k: round(v, 5) for k, v in domain_bpb.items()},
        }
        if grammar_acc is not None:
            rec["grammar_acc"] = round(grammar_acc, 4)
        if completion_acc is not None:
            rec["completion_acc"] = round(completion_acc, 4)
        if vitals:
            for k, v in vitals.items():
                rec[k] = v
        self._append(self.evals_path, rec)

    def log_samples(
        self, *, step: int, tokens_seen: int, samples: list[dict[str, str]]
    ) -> None:
        self._append(
            self.samples_path,
            {
                "t": round(time.time(), 2),
                "step": step,
                "tokens": tokens_seen,
                "samples": samples,
            },
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def list_runs(runs_dir: Path) -> list[str]:
    if not runs_dir.exists():
        return []
    found: list[tuple[float, str]] = []
    for p in runs_dir.iterdir():
        if p.is_dir() and ((p / "metrics.jsonl").exists() or (p / "meta.json").exists()):
            mtime = (
                (p / "metrics.jsonl").stat().st_mtime
                if (p / "metrics.jsonl").exists()
                else p.stat().st_mtime
            )
            found.append((mtime, p.name))
    found.sort(reverse=True)
    return [name for _, name in found]


def load_run_payload(runs_dir: Path, run_name: str) -> dict[str, Any]:
    run_dir = runs_dir / run_name
    if not run_dir.exists() or run_dir.parent.resolve() != runs_dir.resolve():
        return {"error": f"Unknown run: {run_name}"}
    meta = (
        json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        if (run_dir / "meta.json").exists()
        else {"name": run_name}
    )
    return {
        "name": run_name,
        "meta": meta,
        "metrics": _read_jsonl(run_dir / "metrics.jsonl"),
        "evals": _read_jsonl(run_dir / "evals.jsonl"),
        "samples": _read_jsonl(run_dir / "samples.jsonl"),
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>coregrad — Training Monitor</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --amber: #d29922; --purple: #bc8cff; --red: #f85149;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--text); padding: 18px 22px; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
           border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px; }
  h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.02em; }
  select, button { background: var(--card); color: var(--text); border: 1px solid var(--border);
                   border-radius: 6px; padding: 6px 10px; font-size: 0.9rem; cursor: pointer; }
  .pill { font-size: 0.8rem; color: var(--muted); margin-left: auto; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px; margin-bottom: 16px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .kpi .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
  .kpi .val { font-size: 1.25rem; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 14px; margin-bottom: 18px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .card h2 { margin: 0 0 8px 0; font-size: 0.95rem; color: var(--muted); font-weight: 600; }
  svg { width: 100%; height: 240px; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.78rem; color: var(--muted); margin-top: 6px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
  .sample-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 12px; }
  .sample-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
  .sample-box { background: #0b0f14; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }
  .sample-prompt { color: var(--accent); font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;
                   white-space: pre-wrap; font-family: ui-monospace, monospace; }
  .sample-out { font-size: 0.85rem; line-height: 1.45; white-space: pre-wrap;
                font-family: ui-monospace, monospace; color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>coregrad Monitor</h1>
  <select id="runSelect"></select>
  <button id="refreshBtn">Refresh</button>
  <label style="font-size:0.85rem;color:var(--muted);">
    <input type="checkbox" id="logScale" checked /> Log Y (Loss/BPB)
  </label>
  <label style="font-size:0.85rem;color:var(--muted);">
    <input type="checkbox" id="autoRefresh" checked /> Auto (5s)
  </label>
  <span class="pill" id="statusPill">Waiting for runs…</span>
</header>

<section class="kpis" id="kpiRow"></section>

<section class="charts">
  <div class="card">
    <h2>Training Loss (Raw + EMA) vs. Tokens (M)</h2>
    <svg id="lossChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="lossLegend"></div>
  </div>
  <div class="card">
    <h2>Validation BPB (Mean &amp; Per-Domain) vs. Tokens (M)</h2>
    <svg id="bpbChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="bpbLegend"></div>
  </div>
  <div class="card">
    <h2>Gate 1 Probe Accuracy (%) vs. Tokens (M)</h2>
    <svg id="probeChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="probeLegend"></div>
  </div>
  <div class="card">
    <h2>Learning Rate &amp; Gradient Norms (Global + Max Layer)</h2>
    <svg id="lrChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="lrLegend"></div>
  </div>
  <div class="card">
    <h2>Health Vitals (Early Warning: Weight RMS, Act RMS, Max Logit)</h2>
    <svg id="vitalsChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="vitalsLegend"></div>
  </div>
  <div class="card">
    <h2>Repetition &amp; Loop Health (%)</h2>
    <svg id="repChart" viewBox="0 0 600 240"></svg>
    <div class="legend" id="repLegend"></div>
  </div>
</section>

<section class="card">
  <h2>Sample Text Generations</h2>
  <div class="sample-controls">
    <label for="sampleStepSelect" style="font-size:0.85rem;color:var(--muted);">Checkpoint:</label>
    <select id="sampleStepSelect"></select>
  </div>
  <div class="sample-grid" id="sampleGrid"></div>
</section>

<script>
const COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149", "#39c5cf"];
let currentRun = null;
let lastPayload = null;

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function ema(vals, alpha=0.1) {
  const out = [];
  let prev = null;
  for (const v of vals) {
    prev = prev === null ? v : alpha * v + (1 - alpha) * prev;
    out.push(prev);
  }
  return out;
}

function plotSVG(svgId, legendId, series, opts={}) {
  const svg = document.getElementById(svgId);
  const leg = document.getElementById(legendId);
  const nonEmpty = series.filter(s => s.pts.length > 0);
  if (!nonEmpty.length) {
    svg.innerHTML = '<text x="300" y="125" text-anchor="middle" fill="#8b949e" font-size="13">No data yet</text>';
    leg.innerHTML = "";
    return;
  }
  const W = 600, H = 240, padL = 52, padR = 16, padT = 14, padB = 28;
  const useLog = Boolean(opts.logY);
  const ty = y => useLog ? Math.log10(Math.max(y, 1e-6)) : y;
  const xs = nonEmpty.flatMap(s => s.pts.map(p => p[0]));
  const ys = nonEmpty.flatMap(s => s.pts.map(p => ty(p[1])));
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = opts.minY !== undefined ? ty(opts.minY) : Math.min(...ys);
  let maxY = opts.maxY !== undefined ? ty(opts.maxY) : Math.max(...ys);
  if (maxX === minX) maxX = minX + 1;
  if (maxY === minY) { minY -= 0.1; maxY += 0.1; }
  else {
    const dy = (maxY - minY) * 0.08;
    if (opts.minY === undefined) minY -= dy;
    if (opts.maxY === undefined) maxY += dy;
  }
  const sx = x => padL + ((x - minX) / (maxX - minX)) * (W - padL - padR);
  const sy = y => H - padB - ((ty(y) - minY) / (maxY - minY)) * (H - padT - padB);

  let html = "";
  for (let i = 0; i <= 4; i++) {
    const yT = minY + (i / 4) * (maxY - minY);
    const yv = useLog ? Math.pow(10, yT) : yT;
    const yp = H - padB - (i / 4) * (H - padT - padB);
    html += `<line x1="${padL}" y1="${yp}" x2="${W-padR}" y2="${yp}" stroke="#21262d" stroke-width="1"/>`;
    html += `<text x="${padL-6}" y="${yp+4}" text-anchor="end" fill="#8b949e" font-size="10">${yv.toPrecision(3)}</text>`;
  }
  for (let i = 0; i <= 4; i++) {
    const xv = minX + (i / 4) * (maxX - minX);
    const xp = sx(xv);
    html += `<text x="${xp}" y="${H-8}" text-anchor="middle" fill="#8b949e" font-size="10">${xv.toFixed(1)}M</text>`;
  }
  for (const s of nonEmpty) {
    const d = s.pts.map((p, idx) => `${idx===0?"M":"L"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
    const op = s.opacity ?? 1.0;
    const sw = s.width ?? 2;
    html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${sw}" stroke-opacity="${op}" />`;
    if (s.pts.length <= 40) {
      for (const p of s.pts) {
        html += `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="2.6" fill="${s.color}" />`;
      }
    }
  }
  svg.innerHTML = html;
  leg.innerHTML = nonEmpty.map(s => {
    const last = s.pts[s.pts.length - 1][1];
    return `<span><i class="dot" style="background:${s.color}"></i>${esc(s.label)}: <b>${last.toPrecision(4)}</b></span>`;
  }).join("");
}

function renderSamples(payload) {
  const sel = document.getElementById("sampleStepSelect");
  const grid = document.getElementById("sampleGrid");
  const snaps = payload.samples || [];
  if (!snaps.length) {
    sel.innerHTML = "";
    grid.innerHTML = '<div style="color:var(--muted);font-size:0.9rem;">No sample generations logged yet.</div>';
    return;
  }
  const prevIdx = sel.value;
  sel.innerHTML = snaps.map((s, i) =>
    `<option value="${i}">Step ${s.step} (${(s.tokens/1e6).toFixed(2)}M tok)</option>`
  ).join("");
  const chosen = (prevIdx !== "" && Number(prevIdx) < snaps.length) ? Number(prevIdx) : snaps.length - 1;
  sel.value = String(chosen);

  const snap = snaps[chosen];
  grid.innerHTML = snap.samples.map(item => `
    <div class="sample-box">
      <div class="sample-prompt">${esc(item.prompt)}</div>
      <div class="sample-out">${esc(item.completion)}</div>
    </div>
  `).join("");
}

function renderRun(payload) {
  lastPayload = payload;
  const metrics = payload.metrics || [];
  const evals = payload.evals || [];
  const meta = payload.meta || {};
  const lastM = metrics[metrics.length - 1];
  const lastE = evals[evals.length - 1];
  const bestBpb = evals.length ? Math.min(...evals.map(e => e.mean_bpb)) : null;

  const targetTok = meta.total_tokens || 0;
  let etaStr = "—";
  if (lastM && targetTok > lastM.tokens && lastM.tok_s > 0) {
    const remH = ((targetTok - lastM.tokens) / lastM.tok_s) / 3600;
    etaStr = remH >= 1 ? `${remH.toFixed(1)} h` : `${(remH*60).toFixed(0)} m`;
  }

  const wRms = lastM && lastM.max_w_rms !== undefined ? lastM.max_w_rms : (lastE ? lastE.max_w_rms : undefined);
  const actRms = lastE ? lastE.act_rms : undefined;
  const rep0 = lastE ? lastE.rep4_t0 : undefined;
  const healthStr = (wRms !== undefined || actRms !== undefined)
    ? `wRMS ${wRms !== undefined ? wRms.toFixed(3) : "—"} · act ${actRms !== undefined ? actRms.toFixed(1) : "—"}`
    : "—";
  const healthWarn = (wRms !== undefined && wRms > 0.5) || (actRms !== undefined && actRms > 100);

  const kpis = [
    ["Step", lastM ? lastM.step.toLocaleString() : "—"],
    ["Tokens", lastM ? `${(lastM.tokens/1e6).toFixed(2)}M` + (targetTok ? ` / ${(targetTok/1e6).toFixed(0)}M` : "") : "—"],
    ["Train Loss", lastM ? lastM.loss.toFixed(4) : "—"],
    ["Best Val BPB", bestBpb !== null ? bestBpb.toFixed(4) : "—"],
    ["Throughput", lastM ? `${Math.round(lastM.tok_s).toLocaleString()} tok/s` : "—"],
    ["Peak VRAM", lastM ? `${lastM.vram_gb.toFixed(2)} GB` : "—"],
    ["Grammar / 4-Way", lastE && lastE.grammar_acc !== undefined
      ? `${(lastE.grammar_acc*100).toFixed(0)}% / ${((lastE.completion_acc||0)*100).toFixed(0)}%` : "—"],
    ["Health (wRMS · act)", healthStr, healthWarn ? "#f85149" : "#3fb950"],
    ["Greedy Rep4 (T=0)", rep0 !== undefined ? `${(rep0*100).toFixed(1)}%` : "—", rep0 !== undefined && rep0 > 0.35 ? "#d29922" : undefined],
    ["Est. Remaining", etaStr],
  ];
  document.getElementById("kpiRow").innerHTML = kpis.map(([l, v, col]) =>
    `<div class="kpi"><div class="label">${l}</div><div class="val" style="${col ? `color:${col}` : ""}">${v}</div></div>`
  ).join("");

  const logY = document.getElementById("logScale").checked;
  const xM = metrics.map(m => m.tokens / 1e6);
  const rawLoss = metrics.map(m => m.loss);
  const smLoss = ema(rawLoss, 0.12);
  plotSVG("lossChart", "lossLegend", [
    { label: "Raw Loss", color: "#58a6ff", opacity: 0.35, width: 1.4, pts: xM.map((x, i) => [x, rawLoss[i]]) },
    { label: "EMA Loss", color: "#58a6ff", width: 2.3, pts: xM.map((x, i) => [x, smLoss[i]]) },
  ], { logY });

  const domains = new Set();
  for (const e of evals) Object.keys(e.domain_bpb || {}).forEach(d => domains.add(d));
  const bpbSeries = [
    { label: "mean_bpb", color: "#ffffff", width: 2.5, pts: evals.map(e => [e.tokens/1e6, e.mean_bpb]) },
  ];
  [...domains].sort().forEach((d, idx) => {
    bpbSeries.push({
      label: d, color: COLORS[idx % COLORS.length], width: 1.8,
      pts: evals.filter(e => e.domain_bpb && e.domain_bpb[d] !== undefined).map(e => [e.tokens/1e6, e.domain_bpb[d]]),
    });
  });
  plotSVG("bpbChart", "bpbLegend", bpbSeries, { logY });

  plotSVG("probeChart", "probeLegend", [
    { label: "Grammar %", color: "#3fb950", pts: evals.filter(e => e.grammar_acc !== undefined).map(e => [e.tokens/1e6, e.grammar_acc*100]) },
    { label: "4-Way %", color: "#bc8cff", pts: evals.filter(e => e.completion_acc !== undefined).map(e => [e.tokens/1e6, e.completion_acc*100]) },
  ], { minY: 0, maxY: 100 });

  plotSVG("lrChart", "lrLegend", [
    { label: "LR (×1e4)", color: "#d29922", pts: metrics.map(m => [m.tokens/1e6, m.lr * 1e4]) },
    { label: "Global Grad Norm", color: "#f85149", opacity: 0.65, width: 1.5, pts: metrics.map(m => [m.tokens/1e6, m.grad_norm]) },
    { label: "Max Layer Grad Norm", color: "#bc8cff", opacity: 0.85, width: 1.7, pts: metrics.filter(m => m.max_layer_gn !== undefined).map(m => [m.tokens/1e6, m.max_layer_gn]) },
  ]);

  plotSVG("vitalsChart", "vitalsLegend", [
    { label: "Max Weight RMS (×100)", color: "#39c5cf", pts: metrics.filter(m => m.max_w_rms !== undefined).map(m => [m.tokens/1e6, m.max_w_rms * 100]) },
    { label: "Act RMS", color: "#f85149", pts: evals.filter(e => e.act_rms !== undefined).map(e => [e.tokens/1e6, e.act_rms]) },
    { label: "Max Logit", color: "#d29922", pts: evals.filter(e => e.max_logit !== undefined).map(e => [e.tokens/1e6, e.max_logit]) },
  ], { logY });

  plotSVG("repChart", "repLegend", [
    { label: "Greedy T=0 Rep4 %", color: "#f85149", pts: evals.filter(e => e.rep4_t0 !== undefined).map(e => [e.tokens/1e6, e.rep4_t0 * 100]) },
    { label: "Sampled T=0.7 Rep4 %", color: "#d29922", pts: evals.filter(e => e.rep4_t07 !== undefined).map(e => [e.tokens/1e6, e.rep4_t07 * 100]) },
    { label: "Loop Cos Sim %", color: "#bc8cff", pts: evals.filter(e => e.loop_cos !== undefined).map(e => [e.tokens/1e6, e.loop_cos * 100]) },
    { label: "GRU Gate Mean %", color: "#3fb950", pts: evals.filter(e => e.loop_gate_mean !== undefined).map(e => [e.tokens/1e6, e.loop_gate_mean * 100]) },
  ], { minY: 0, maxY: 100 });

  renderSamples(payload);
  document.getElementById("statusPill").textContent =
    `Run: ${payload.name} | ${metrics.length} steps, ${evals.length} evals | Updated ${new Date().toLocaleTimeString()}`;
}

async function fetchRunsAndRender() {
  if (window.__EMBEDDED_PAYLOAD__) {
    document.getElementById("runSelect").innerHTML = `<option>${esc(window.__EMBEDDED_PAYLOAD__.name)}</option>`;
    renderRun(window.__EMBEDDED_PAYLOAD__);
    return;
  }
  const r = await fetch("/api/runs");
  const { runs } = await r.json();
  const sel = document.getElementById("runSelect");
  if (!runs.length) {
    document.getElementById("statusPill").textContent = "No training runs found yet.";
    return;
  }
  if (!currentRun || !runs.includes(currentRun)) currentRun = runs[0];
  sel.innerHTML = runs.map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  sel.value = currentRun;
  const data = await (await fetch(`/api/run?name=${encodeURIComponent(currentRun)}`)).json();
  renderRun(data);
}

document.getElementById("runSelect").addEventListener("change", e => { currentRun = e.target.value; fetchRunsAndRender(); });
document.getElementById("logScale").addEventListener("change", () => { if (lastPayload) renderRun(lastPayload); });
document.getElementById("sampleStepSelect").addEventListener("change", () => { if (lastPayload) renderSamples(lastPayload); });
document.getElementById("refreshBtn").addEventListener("click", fetchRunsAndRender);
setInterval(() => { if (document.getElementById("autoRefresh").checked && !window.__EMBEDDED_PAYLOAD__) fetchRunsAndRender(); }, 5000);
fetchRunsAndRender();
</script>
</body>
</html>
"""


def render_static_report(runs_dir: Path, run_name: str, out_html: Path) -> Path:
    """Export a standalone self-contained HTML report for a single run."""
    payload = load_run_payload(runs_dir, run_name)
    inject = f"<script>window.__EMBEDDED_PAYLOAD__ = {json.dumps(payload)};</script>\n</head>"
    html = DASHBOARD_HTML.replace("</head>", inject, 1)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def serve_monitor(runs_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the live training monitor over HTTP."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/runs":
                payload = json.dumps({"runs": list_runs(runs_dir)}).encode("utf-8")
                self._send(200, payload, "application/json")
            elif parsed.path == "/api/run":
                qs = parse_qs(parsed.query)
                name = (qs.get("name") or [""])[0]
                payload = json.dumps(load_run_payload(runs_dir, name)).encode("utf-8")
                self._send(200, payload, "application/json")
            else:
                self._send(404, b"Not found", "text/plain")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"coregrad monitor listening on http://{host}:{port} (runs_dir={runs_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down monitor.")
    finally:
        server.server_close()
