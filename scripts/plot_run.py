#!/usr/bin/env python3
"""Render the run-of-record figure as SVG (and PNG when resvg-py is available)
from results/<date>/: the descent of the first answer token for every fact,
and the engine's first-token probability with the fact's own overlay and with
the merged one.

Usage: uv run --with resvg-py python scripts/plot_run.py 2026-09-05
Writes docs/img/run-<date>.svg and docs/img/run-<date>.png.
"""
import json
import math
import sys
from pathlib import Path

date = sys.argv[1] if len(sys.argv) > 1 else "2026-09-05"
root = Path("results") / date
s = json.load(open(root / "summary.json"))
ec = json.load(open(root / "engine_check.json"))
fj = json.load(open("facts/facts.json"))
facts = {f["id"]: f for f in (fj["facts"] if isinstance(fj, dict) else fj)}
order = s["order"]

W, H = 1800, 900
BG, INK, MUTED, GRID = "#0b1220", "#e5e7eb", "#8b93a7", "#1f2937"
PAL = ["#22d3ee", "#38bdf8", "#34d399", "#a3e635", "#a78bfa", "#f472b6", "#fb7185", "#fbbf24"]
FONT = "Inter, 'Noto Sans', 'Segoe UI', Helvetica, Arial, sans-serif"

out = []
add = out.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
add('<defs>'
    '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="4" result="b"/>'
    '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    '<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="10"/></filter>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# ---- headline -------------------------------------------------------------
n_ok = sum(ec["q8"]["facts"][f]["answer_reproduced"] for f in order)
add(f'<text x="70" y="86" font-size="44" font-weight="800" fill="{INK}">{n_ok} of {len(order)} facts took.</text>')
add(f'<text x="70" y="122" font-size="18" fill="{MUTED}">ENGRAFT on Qwen3.8-Flash-Next (125B MoE) · gradient on 8 rows of the n-gram table per fact · '
    f'CPU-only · run of record {date}</text>')

# ---- left panel: descent curves ------------------------------------------
L, T, R, B = 90, 190, 980, 720  # plot box
xmax = 300


def X(step):
    return L + (R - L) * step / xmax


def Y(p):
    return B - (B - T) * p


for gy in (0, 0.25, 0.5, 0.75, 1.0):
    add(f'<line x1="{L}" y1="{Y(gy):.1f}" x2="{R}" y2="{Y(gy):.1f}" stroke="{GRID}" stroke-width="1"/>')
    add(f'<text x="{L-12}" y="{Y(gy)+5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end">{gy:g}</text>')
for gx in (0, 100, 200, 300):
    add(f'<text x="{X(gx):.1f}" y="{B+26}" font-size="13" fill="{MUTED}" text-anchor="middle">{gx}</text>')
add(f'<line x1="{L}" y1="{Y(0.95):.1f}" x2="{R}" y2="{Y(0.95):.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 5"/>')
add(f'<text x="{R}" y="{Y(0.95)-8:.1f}" font-size="13" fill="{MUTED}" text-anchor="end">stop: p = 0.95 under free routing</text>')
add(f'<text x="{L}" y="{T-28}" font-size="20" font-weight="700" fill="{INK}">Descent of the first answer token</text>')
add(f'<text x="{L}" y="{T-8}" font-size="13" fill="{MUTED}">p(answer | trigger) on the CPU replica, expert routing refreshed at every step</text>')
add(f'<text x="{(L+R)/2:.0f}" y="{B+52}" font-size="14" fill="{MUTED}" text-anchor="middle">descent step</text>')

ends = []
for i, fid in enumerate(order):
    rows = [json.loads(l) for l in open(root / "facts" / fid / f"descend_{fid}_0.jsonl")]
    pts = [(X(r["step"]), Y(math.exp(r["logp_y"]))) for r in rows]
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    col = PAL[i % len(PAL)]
    cf = facts[fid].get("kind") == "counterfactual"
    dash = ' stroke-dasharray="7 5"' if cf else ""
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow)"{dash}/>')
    ends.append((pts[-1], col, fid, rows[-1]))

# labels: a column right of the plot, one per curve, aligned to the end point and pushed apart
ends.sort(key=lambda e: e[0][1])
placed = []
for (x, y), col, fid, last in ends:
    ly = y
    for py in placed:
        if abs(ly - py) < 24:
            ly = py + 24
    placed.append(ly)
    f = facts[fid]
    tag = f"{f['answer']} · {f['trigger']}"
    if f.get("kind") == "counterfactual":
        tag += " (counterfactual)"
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" filter="url(#glow)"/>')
    add(f'<circle cx="{R+22}" cy="{ly-4:.1f}" r="4" fill="{col}"/>')
    add(f'<text x="{R+32}" y="{ly:.1f}" font-size="12.5" fill="{col}">{tag}</text>')

# annotation for a fact stopped at the cap
for fid in order:
    g0 = s["facts"][fid]["grafts"][0]
    if g0["stop_reason"] == "max_steps":
        add(f'<text x="{X(296):.0f}" y="{Y(0.06):.0f}" font-size="13" fill="#fb7185" text-anchor="end">never takes: strong prior of the base model at this trigger</text>')

# ---- right panel: engine bars --------------------------------------------
bx, by = 1330, 190
add(f'<text x="{bx}" y="{by-28}" font-size="20" font-weight="700" fill="{INK}">On the real llama.cpp engine</text>')
add(f'<text x="{bx}" y="{by-8}" font-size="13" fill="{MUTED}">p(first answer token), own overlay · merged overlay: same digits</text>')
bw = 300
rowh = 58
for i, fid in enumerate(order):
    p_own = ec["q8"]["facts"][fid]["p_first"]
    p_m = ec["q8"]["merged"]["facts"][fid]["p_first"]
    col = PAL[i % len(PAL)]
    y = by + 20 + i * rowh
    f = facts[fid]
    add(f'<text x="{bx}" y="{y+14}" font-size="13" fill="{INK}">{f["answer"]}</text>')
    add(f'<text x="{bx+bw}" y="{y+14}" font-size="11" fill="{MUTED}" text-anchor="end">{f["trigger"]}</text>')
    add(f'<rect x="{bx}" y="{y+22}" width="{bw}" height="14" rx="7" fill="{GRID}"/>')
    add(f'<rect x="{bx}" y="{y+22}" width="{max(6, bw*p_own):.1f}" height="14" rx="7" fill="{col}" filter="url(#glow)"/>')
    same = "= merged" if p_own == p_m else f"merged {p_m:.3f}"
    add(f'<text x="{bx+bw+16}" y="{y+34}" font-size="20" font-weight="700" fill="{col}">{p_own:.2f}</text>')
    add(f'<text x="{bx+bw+80}" y="{y+34}" font-size="11" fill="{MUTED}">{same}</text>')

# ---- footer chips ---------------------------------------------------------
chips = [
    ("replica = F32 engine on 17/17 grafts", "#34d399"),
    ("sister triggers: Δ = 0.0", "#22d3ee"),
    ("no weight changed · GGUF untouched", "#a78bfa"),
    ("67 GB RAM peak · 7–17 min per fact", "#fbbf24"),
]
cx = 70
for text, col in chips:
    wch = 12 + 8.2 * len(text)
    add(f'<rect x="{cx}" y="{H-90}" width="{wch:.0f}" height="36" rx="18" fill="none" stroke="{col}" stroke-opacity="0.7"/>')
    add(f'<text x="{cx+wch/2:.0f}" y="{H-67}" font-size="13" fill="{col}" text-anchor="middle">{text}</text>')
    cx += wch + 14
add(f'<text x="{W-70}" y="{H-30}" font-size="12" fill="{MUTED}" text-anchor="end">plotted from results/{date} · github.com/fulvian/engraft-ngram</text>')
add("</svg>")

svg = "\n".join(out)
Path("docs/img").mkdir(parents=True, exist_ok=True)
svg_path = Path("docs/img") / f"run-{date}.svg"
svg_path.write_text(svg)
print(svg_path)
try:
    import resvg_py

    png = resvg_py.svg_to_bytes(svg_string=svg, width=W, font_dirs=["/usr/share/fonts"], font_family="Noto Sans",
                                sans_serif_family="Noto Sans", skip_system_fonts=False)
    (Path("docs/img") / f"run-{date}.png").write_bytes(png)
    print(Path("docs/img") / f"run-{date}.png")
except ImportError:
    print("resvg-py not installed: SVG only")
