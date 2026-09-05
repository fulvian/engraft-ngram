#!/usr/bin/env python3
"""Plot the run-of-record figure (docs/img/run-<date>.png) from results/<date>/:
descent curves of the first answer token for every fact, and the engine's
first-token probability with the fact's own overlay and with the merged one.

Usage: uv run --with matplotlib --with numpy python scripts/plot_run.py 2026-09-05
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

date = sys.argv[1] if len(sys.argv) > 1 else "2026-09-05"
root = f"results/{date}"
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans", "sans-serif"], "font.size": 11,
                     "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155", "xtick.color": "#64748b",
                     "ytick.color": "#64748b", "axes.titlesize": 14, "axes.titlecolor": "#0f172a"})
s = json.load(open(f"{root}/summary.json"))
ec = json.load(open(f"{root}/engine_check.json"))
facts = {f["id"]: f for f in json.load(open("facts/facts.json"))}
order = s["order"]


def label(fid):
    f = facts[fid]
    kind = " (counterfactual)" if f.get("kind") == "counterfactual" else f" ({f['lang']})"
    return f"{f['trigger']}: {f['answer']}{kind}"


pal = ["#2563eb", "#0ea5e9", "#10b981", "#84cc16", "#8b5cf6", "#c026d3", "#f43f5e", "#f97316"]
fig = plt.figure(figsize=(15, 7.2), facecolor="#f8fafc")
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.2, left=0.055, right=0.985, top=0.78, bottom=0.27)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (ax1, ax2):
    ax.set_facecolor("#ffffff")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", lw=1)
    ax.set_axisbelow(True)
for i, fid in enumerate(order):
    rows = [json.loads(l) for l in open(f"{root}/facts/{fid}/descend_{fid}_0.jsonl")]
    x = [r["step"] for r in rows]
    y = [np.exp(r["logp_y"]) for r in rows]
    cf = facts[fid].get("kind") == "counterfactual"
    ax1.plot(x, y, color=pal[i % len(pal)], lw=2.6, ls=(0, (4, 2)) if cf else "-", label=label(fid))
    ax1.scatter([x[-1]], [y[-1]], color=pal[i % len(pal)], s=36, zorder=5, edgecolor="white", lw=1.2)
ax1.axhline(0.95, color="#94a3b8", lw=1.2, ls=":")
ax1.text(3, 0.965, "stop at p = 0.95", fontsize=10, color="#64748b")
ax1.set_xlim(0, 300)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel("descent step  (expert routing refreshed at every step)")
ax1.set_ylabel("p(first answer token | trigger), CPU replica")
ax1.set_title("Gradient descent on the 8 trigram rows, first answer token", loc="left")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9, ncol=2, handlelength=2.4)
for fid in order:
    g0 = s["facts"][fid]["grafts"][0]
    if g0["stop_reason"] == "max_steps":
        ax1.annotate("never takes: strong base prior at the trigger", xy=(g0["n_steps"] - 4, g0["final_p_free"]),
                     xytext=(232, 0.17), fontsize=9.5, color="#e11d48", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1))
pf = [ec["q8"]["facts"][f]["p_first"] for f in order]
pm = [ec["q8"]["merged"]["facts"][f]["p_first"] for f in order]
xs = np.arange(len(order))
w = 0.36
ax2.bar(xs - w / 2, pf, w, color=pal[: len(order)], edgecolor="none")
ax2.bar(xs + w / 2, pm, w, color=pal[: len(order)], alpha=0.35, edgecolor="none", hatch="////")
ax2.set_xticks(xs)
ax2.set_xticklabels([f.replace("_", "\n") for f in order], fontsize=9)
ax2.set_ylim(0, 1.06)
ax2.set_ylabel("p(first answer token), real llama.cpp engine")
n_ok = sum(ec["q8"]["facts"][f]["answer_reproduced"] for f in order)
ax2.set_title(f"Real engine: {n_ok} of {len(order)} take, merged = single", loc="left")
for xi, v in zip(xs, pf):
    ax2.text(xi - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#334155")
ax2.legend(handles=[Patch(color="#334155", label="own overlay"),
                    Patch(facecolor="#334155", alpha=0.35, hatch="////", label="merged overlay of all facts")],
           loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9.5, ncol=2)
fig.text(0.055, 0.93, "ENGRAFT on Qwen3.8-Flash-Next (125B MoE)", fontsize=20, color="#0f172a")
fig.text(0.055, 0.875, f"{len(order)} facts written into the n-gram table by gradient on 8 rows each · "
         f"CPU-only descent · run of record {date}", fontsize=11, color="#64748b")
fig.text(0.055, 0.03, f"Plotted from summary.json, engine_check.json and the per-step descend_*.jsonl files in "
         f"results/{date} · github.com/fulvian/engraft-ngram", fontsize=9, color="#94a3b8")
fig.savefig(f"docs/img/run-{date}.png", dpi=130, facecolor=fig.get_facecolor())
print(f"docs/img/run-{date}.png")
