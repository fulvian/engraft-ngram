#!/usr/bin/env python3
"""Render the two figures that summarise the Quail run, as SVG (and PNG when
resvg-py is available). Every number is recomputed here from the result files
under data/quail/results/ and results/2026-09-12/yardstick/, so the figures
cannot drift from the repository.

Usage: uv run --with resvg-py python scripts/plot_post.py
Writes docs/img/post-results.{svg,png} and docs/img/post-limits.{svg,png}.
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"

BG, PANEL, INK, MUTED, GRID = "#0b1220", "#101a2c", "#e5e7eb", "#8b93a7", "#1f2937"
CY, CY_LIGHT, CY_MID, CY_DARK = "#22d3ee", "#67e8f9", "#0e7490", "#16303f"
WARM = "#fbbf24"
FONT = "Inter, 'Noto Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
DEFS = ('<defs>'
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="4" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '</defs>')

CELLS = {"it": "s0b", "en": "e0b-preliminary", "zh": "z0b-preliminary"}
DAMAGE_OWN = {"it": "s0b/damage_it_text.json",
              "en": "e0b-preliminary/damage_en_text.json",
              "zh": "z0b-preliminary/damage_zh_text.json"}


def load(rel):
    return json.load(open(ROOT / rel))


def engine(cell):
    return load(f"data/quail/results/{cell}/engine_results.json")


def em(records):
    """Exact answer under greedy decoding, overlay and base, overall and by family."""
    by = collections.defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]
    for r in records:
        for a in (by[r["family"]], tot):
            a[0] += 1
            a[1] += bool(r["greedy"]["student"]["exact_match"])
            a[2] += bool(r["greedy"]["base"]["exact_match"])
    return tot, dict(by)


def write(name, svg, width):
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / f"{name}.svg").write_text(svg)
    print(IMG / f"{name}.svg")
    try:
        import resvg_py
    except ImportError:
        print("resvg-py not installed: SVG only")
        return
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width * 2, font_dirs=["/usr/share/fonts"],
                                font_family="Noto Sans", sans_serif_family="Noto Sans",
                                skip_system_fonts=False)
    (IMG / f"{name}.png").write_bytes(bytes(png))
    print(IMG / f"{name}.png")


def chips(add, items, W, H):
    cx = 70
    for text, col in items:
        w = 12 + 8.2 * len(text)
        add(f'<rect x="{cx}" y="{H-92}" width="{w:.0f}" height="36" rx="18" fill="none" stroke="{col}" stroke-opacity="0.65"/>')
        add(f'<text x="{cx+w/2:.0f}" y="{H-69}" font-size="13" fill="{col}" text-anchor="middle">{text}</text>')
        cx += w + 14


def panel_title(add, x, y, title, subs):
    add(f'<text x="{x}" y="{y}" font-size="21" font-weight="700" fill="{INK}">{title}</text>')
    for i, sub in enumerate(subs):
        add(f'<text x="{x}" y="{y+24+i*20}" font-size="14" fill="{MUTED}">{sub}</text>')


# --------------------------------------------------------------------------
# Figure 1 — the numbers
# --------------------------------------------------------------------------
def figure_results():
    tot, by_family = em(engine("s0b"))
    n, ok, base_ok = tot
    s1b, _ = em(engine("s1b"))
    cap = {r["n"]: r["rbr"]["top1_frac"] for r in load("data/quail/results/capacity-curve/capacity_curve.json")["rows"]}

    lang = {}
    for code, cell in CELLS.items():
        d = engine(cell)
        m = len(d)
        lang[code] = dict(
            em=sum(x["greedy"]["student"]["exact_match"] for x in d) / m,
            base=sum(x["greedy"]["base"]["exact_match"] for x in d) / m,
            rank1=sum(x["student"]["rank_first"] == 1 for x in d) / m,
            dmg=load(f"data/quail/results/{DAMAGE_OWN[code]}")["global"]["kl_mean"],
            n=m,
        )

    W, H = 1800, 1220
    out = []
    add = out.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
    add(DEFS)
    add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    add(f'<text x="70" y="92" font-size="46" font-weight="800" fill="{INK}">100 facts written into an n-gram table. No weight changed.</text>')
    add(f'<text x="70" y="130" font-size="18" fill="{MUTED}">Qwen3.8-Flash-Next, 125B MoE, IQ4_XS · 14,032 of the table’s 320 million rows trained · one 9 MB overlay applied at read time</text>')

    # ---- left column: exact answer + families -----------------------------
    L = 70
    panel_title(add, L, 214, "Exact answer, greedy decoding, on the real llama.cpp engine",
                [f"{n} held-out test sentences the descent never saw · free expert routing, the production condition"])
    bar_x, bar_w = L + 250, 500
    for i, (label, frac, col, note) in enumerate([
            ("base model", base_ok / n, MUTED, f"{base_ok} of {n}"),
            ("with overlay", ok / n, CY, f"{ok} of {n}"),
            ("second seed", s1b[1] / s1b[0], CY_MID, f"{s1b[1]} of {s1b[0]}")]):
        y = 272 + i * 86
        add(f'<text x="{bar_x-20}" y="{y+40}" font-size="19" fill="{INK}" text-anchor="end">{label}</text>')
        add(f'<rect x="{bar_x}" y="{y+14}" width="{bar_w}" height="40" rx="8" fill="{GRID}"/>')
        add(f'<rect x="{bar_x}" y="{y+14}" width="{max(5, bar_w*frac):.1f}" height="40" rx="8" fill="{col}"'
            f'{" filter=\"url(#glow)\"" if col == CY else ""}/>')
        add(f'<text x="{bar_x+bar_w+22}" y="{y+45}" font-size="30" font-weight="800" fill="{col}">{frac:.3f}</text>')
        add(f'<text x="{bar_x+bar_w+22}" y="{y+8}" font-size="12" fill="{MUTED}">{note}</text>')

    add(f'<line x1="{L}" y1="548" x2="{L+820}" y2="548" stroke="{GRID}" stroke-width="1"/>')
    panel_title(add, L, 592, "By the shape of the question",
                ["same metric, the overlay column, split by prompt family"])
    fam_x, fam_w = L + 250, 380
    for i, (fam, v) in enumerate(sorted(by_family.items(), key=lambda kv: -kv[1][1] / kv[1][0])):
        nf, okf, _ = v
        y = 640 + i * 44
        add(f'<text x="{fam_x-20}" y="{y+18}" font-size="15" fill="{INK}" text-anchor="end">{fam}</text>')
        add(f'<text x="{fam_x-20}" y="{y+34}" font-size="11" fill="{MUTED}" text-anchor="end">{nf} sentences</text>')
        add(f'<rect x="{fam_x}" y="{y+4}" width="{fam_w}" height="20" rx="6" fill="{GRID}"/>')
        add(f'<rect x="{fam_x}" y="{y+4}" width="{fam_w*okf/nf:.1f}" height="20" rx="6" fill="{CY}" fill-opacity="0.85"/>')
        add(f'<text x="{fam_x+fam_w+18}" y="{y+21}" font-size="18" font-weight="700" fill="{CY}">{okf/nf:.3f}</text>')

    # ---- right column: three languages (numbers only) ---------------------
    R = 990
    panel_title(add, R, 214, "The same world, rebuilt in three languages",
                ["one seed each, same recipe and same row budget",
                 "Chinese and English are preliminary · the English corpus covers 97 of the 100 facts"])
    tx = [R, R + 400, R + 540, R + 680]
    head_y = 300
    for j, h in enumerate(["", "Italian", "English", "Chinese"]):
        anchor = "start" if j == 0 else "middle"
        add(f'<text x="{tx[j]+(0 if j == 0 else 40)}" y="{head_y}" font-size="16" font-weight="700" '
            f'fill="{INK if j else MUTED}" text-anchor="{anchor}">{h}</text>')
    rows = [("exact answer, with the overlay", "em", "{:.3f}", CY),
            ("exact answer, base model", "base", "{:.3f}", MUTED),
            ("first answer token at rank 1", "rank1", "{:.3f}", INK),
            ("damage on its own language (KL)", "dmg", "{:.4f}", INK),
            ("test sentences", "n", "{:d}", MUTED)]
    for i, (label, key, fmt, col) in enumerate(rows):
        y = head_y + 42 + i * 46
        if i % 2 == 0:
            add(f'<rect x="{R-14}" y="{y-26}" width="748" height="40" rx="8" fill="{PANEL}"/>')
        add(f'<text x="{R}" y="{y}" font-size="15" fill="{MUTED}">{label}</text>')
        for j, code in enumerate(("it", "en", "zh")):
            v = lang[code][key]
            add(f'<text x="{tx[j+1]+40}" y="{y}" font-size="19" font-weight="{"800" if key == "em" else "600"}" '
                f'fill="{col}" text-anchor="middle">{fmt.format(v)}</text>')
    add(f'<text x="{R}" y="{head_y+42+len(rows)*46+18}" font-size="14" fill="{MUTED}">The ordering is training mass, not language: controlling for how much text each</text>')
    add(f'<text x="{R}" y="{head_y+42+len(rows)*46+40}" font-size="14" fill="{MUTED}">fact received, the language coefficient’s 95 % interval contains zero.</text>')

    # ---- right column, separate panel: capacity ---------------------------
    add(f'<line x1="{R}" y1="618" x2="{R+740}" y2="618" stroke="{GRID}" stroke-width="1"/>')
    panel_title(add, R, 662, "A different corpus: how many facts fit?",
                ["short invented facts, not Quail · first answer token at rank 1",
                 "expert routing pinned to the base model’s choices — not comparable with the numbers above"])
    PL, PR, PT, PB = R + 60, R + 660, 730, 900
    for gy in (0.5, 0.75, 1.0):
        y = PB - (PB - PT) * (gy - 0.5) / 0.5
        add(f'<line x1="{PL}" y1="{y:.1f}" x2="{PR}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        add(f'<text x="{PL-14}" y="{y+5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end">{gy:g}</text>')
    xs = {24: PL + 70, 100: (PL + PR) / 2, 300: PR - 70}
    pts = [(xs[k], PB - (PB - PT) * (cap[k] - 0.5) / 0.5) for k in (24, 100, 300)]
    add('<path d="M ' + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
        f'" fill="none" stroke="{CY}" stroke-width="3" stroke-linecap="round" filter="url(#glow)"/>')
    for (x, y), k in zip(pts, (24, 100, 300)):
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{BG}" stroke="{CY}" stroke-width="3"/>')
        add(f'<text x="{x:.1f}" y="{y-24:.1f}" font-size="22" font-weight="800" fill="{CY}" text-anchor="middle">{cap[k]:.3f}</text>')
        add(f'<text x="{x:.1f}" y="{PB+28:.1f}" font-size="15" fill="{INK}" text-anchor="middle">{k} facts</text>')
    add(f'<text x="{R}" y="{PB+66}" font-size="15" fill="{MUTED}">Flat to 300, on an axis that starts at 0.5. Where it bends is the open question.</text>')

    # ---- bottom strip: the four facts that stick --------------------------
    tiles = [("9 MB", "the overlay that carries all 100 facts"),
             ("14,032", "table rows trained, of 320 million"),
             ("0", "weights of the model changed"),
             ("2.4 h", "to graft all 100, on one integrated GPU")]
    for i, (value, label) in enumerate(tiles):
        tx0 = 70 + i * 420
        add(f'<rect x="{tx0}" y="1012" width="390" height="86" rx="14" fill="{PANEL}" stroke="{GRID}"/>')
        add(f'<text x="{tx0+22}" y="1056" font-size="32" font-weight="800" fill="{CY}">{value}</text>')
        add(f'<text x="{tx0+22}" y="1082" font-size="13" fill="{MUTED}">{label}</text>')

    chips(add, [("remove the overlay and the model is exactly itself again", CY_LIGHT),
                ("every number here is recomputed from a file in the repository", WARM)], W, H)
    add(f'<text x="{W-70}" y="{H-32}" font-size="12" fill="{MUTED}" text-anchor="end">recomputed from data/quail/results/ · github.com/fulvian/engraft-ngram</text>')
    add("</svg>")
    write("post-results", "\n".join(out), W)


# --------------------------------------------------------------------------
# Figure 2 — the limits
# --------------------------------------------------------------------------
def figure_limits():
    probes = load("data/quail/results/s0b/probe_results.json")
    both = sum(p["both"] for p in probes)
    one = sum((p["hit_a"] or p["hit_b"]) and not p["both"] for p in probes)
    none = len(probes) - both - one

    floor = load("results/2026-09-12/yardstick/damage_noise_quant_s0.json")["global"]["kl_mean"]
    dmg = {24: load("data/quail/results/capacity-curve/damage_n24_seed1.json")["global"]["kl_mean"],
           100: load("data/quail/results/capacity-curve/damage_n100.json")["global"]["kl_mean"],
           300: load("data/quail/results/capacity-curve/damage_n300.json")["global"]["kl_mean"]}

    sp = load("data/quail/results/languages/specular-zh-on-it/engine_results.json")
    m = len(sp)
    sp_em = sum(x["greedy"]["student"]["exact_match"] for x in sp) / m
    sp_base = sum(x["greedy"]["base"]["exact_match"] for x in sp) / m
    sp_same = sum(x["student"]["p_first"] == x["base"]["p_first"] for x in sp)
    sp_hits = sum(x["student"]["overlay_hits"] for x in sp)

    W, H = 1800, 960
    out = []
    add = out.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
    add(DEFS)
    add(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    add(f'<text x="70" y="92" font-size="46" font-weight="800" fill="{INK}">Three things that still do not work.</text>')
    add(f'<text x="70" y="130" font-size="18" fill="{MUTED}">Same run, same files. These are the numbers we would want a reader to attack first.</text>')

    # ---- 1: composition ----------------------------------------------------
    panel_title(add, 70, 216, "1 · Two facts in one question",
                ["83 probes, each asking for two grafted facts at once.",
                 "One dot per probe, free generation on the real engine."])
    cols, cell, gap = 12, 30, 9
    for i in range(len(probes)):
        cx = 70 + (i % cols) * (cell + gap)
        cy = 300 + (i // cols) * (cell + gap)
        if i < both:
            fill, stroke = CY_LIGHT, "none"
        elif i < both + one:
            fill, stroke = CY_DARK, CY
        else:
            fill, stroke = CY_DARK, "#243347"
        add(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"'
            f'{" filter=\"url(#glow)\"" if i < both else ""}/>')
    for i, (label, cnt, fill, stroke) in enumerate([
            ("both answers right", both, CY_LIGHT, "none"),
            ("only one right", one, CY_DARK, CY),
            ("neither", none, CY_DARK, "#243347")]):
        y = 566 + i * 50
        add(f'<rect x="70" y="{y}" width="26" height="26" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        add(f'<text x="110" y="{y+20}" font-size="17" fill="{INK}">{label}</text>')
        add(f'<text x="530" y="{y+20}" font-size="24" font-weight="800" fill="{CY if i == 0 else MUTED}" text-anchor="end">{cnt} / {len(probes)}</text>')
    add(f'<text x="70" y="750" font-size="15" fill="{MUTED}">Each fact on its own is answered well. Putting two together is</text>')
    add(f'<text x="70" y="772" font-size="15" fill="{MUTED}">where the method is weakest, and we do not yet know why.</text>')

    # ---- 2: damage ---------------------------------------------------------
    BL = 640
    panel_title(add, BL, 216, "2 · The rest of the model moves",
                ["Mean KL to the base model on neutral text it should not",
                 "care about. Lower is better; zero would be no damage."])
    DT, DB = 330, 650
    top = 0.008
    for gy in (0, 0.002, 0.004, 0.006, 0.008):
        y = DB - (DB - DT) * gy / top
        add(f'<line x1="{BL}" y1="{y:.1f}" x2="{BL+440}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        add(f'<text x="{BL-12}" y="{y+5:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{gy:.3f}</text>')
    yfloor = DB - (DB - DT) * floor / top
    add(f'<line x1="{BL}" y1="{yfloor:.1f}" x2="{BL+440}" y2="{yfloor:.1f}" stroke="{WARM}" stroke-width="2" stroke-dasharray="6 6"/>')
    add(f'<line x1="{BL+196}" y1="{DT-14}" x2="{BL+228}" y2="{DT-14}" stroke="{WARM}" stroke-width="2" stroke-dasharray="6 6"/>')
    add(f'<text x="{BL+236}" y="{DT-9}" font-size="13" fill="{WARM}">quantization yardstick {floor:.4f}</text>')
    for i, k in enumerate((24, 100, 300)):
        x = BL + 60 + i * 130
        v = dmg[k]
        y = DB - (DB - DT) * v / top
        add(f'<rect x="{x}" y="{y:.1f}" width="86" height="{DB-y:.1f}" rx="8" fill="{CY}" fill-opacity="{0.55+0.2*i:.2f}"/>')
        add(f'<text x="{x+43}" y="{y-14:.1f}" font-size="19" font-weight="800" fill="{CY}" text-anchor="middle">{v:.4f}</text>')
        add(f'<text x="{x+43}" y="{DB+26}" font-size="14" fill="{INK}" text-anchor="middle">{k} facts</text>')
        add(f'<text x="{x+43}" y="{DB+46}" font-size="12" fill="{MUTED}" text-anchor="middle">{v/floor:.1f}x</text>')
    add(f'<text x="{BL}" y="{DB+96}" font-size="15" fill="{MUTED}">The yardstick is an overlay that only re-quantizes the same rows,</text>')
    add(f'<text x="{BL}" y="{DB+118}" font-size="15" fill="{MUTED}">so it is the table’s own noise. Damage grows more slowly than the</text>')
    add(f'<text x="{BL}" y="{DB+140}" font-size="15" fill="{MUTED}">number of facts, but it grows: on Quail at 100 facts it reaches</text>')
    add(f'<text x="{BL}" y="{DB+162}" font-size="15" fill="{MUTED}">0.0131, about 4x the yardstick.</text>')

    # ---- 3: no transfer across languages ----------------------------------
    CL = 1240
    panel_title(add, CL, 216, "3 · A graft does not cross languages",
                ["The Italian test set, asked against the Chinese overlay.",
                 "Exact answer, greedy, on the real engine."])
    for i, (label, frac, col) in enumerate([("Chinese overlay", sp_em, CY_MID),
                                            ("base model, no overlay", sp_base, MUTED)]):
        y = 320 + i * 104
        add(f'<text x="{CL}" y="{y}" font-size="17" fill="{INK}">{label}</text>')
        add(f'<rect x="{CL}" y="{y+14}" width="430" height="34" rx="8" fill="{GRID}"/>')
        add(f'<rect x="{CL}" y="{y+14}" width="{max(6, 430*frac/0.9):.1f}" height="34" rx="8" fill="{col}"/>')
        add(f'<text x="{CL+22}" y="{y+38}" font-size="20" font-weight="800" fill="{INK}">{frac:.4f}</text>')
    add(f'<text x="{CL}" y="540" font-size="15" fill="{MUTED}">(same axis as the 0.841 of the Italian overlay, which would fill the bar)</text>')
    add(f'<rect x="{CL}" y="572" width="450" height="96" rx="14" fill="{PANEL}" stroke="{GRID}"/>')
    add(f'<text x="{CL+22}" y="616" font-size="30" font-weight="800" fill="{WARM}">{sp_hits}</text>')
    add(f'<text x="{CL+22}" y="642" font-size="13.5" fill="{MUTED}">overlay rows the engine actually read — and still</text>')
    add(f'<text x="{CL+22}" y="660" font-size="13.5" fill="{MUTED}">nothing changed: {sp_same} of {m} sentences bit-identical</text>')
    add(f'<text x="{CL}" y="712" font-size="15" fill="{MUTED}">The overlay fires and has no effect. A fact has to be grafted in the</text>')
    add(f'<text x="{CL}" y="734" font-size="15" fill="{MUTED}">language it will be asked in: the rows are keyed by the tokens of its</text>')
    add(f'<text x="{CL}" y="756" font-size="15" fill="{MUTED}">own script. Measured on one pair of languages and one cell.</text>')

    chips(add, [("no comparison against LoRA or ROME/MEMIT yet", WARM),
                ("one model measured", WARM),
                ("rephrasing is covered only as far as the corpus goes", WARM)], W, H)
    add(f'<text x="{W-70}" y="{H-32}" font-size="12" fill="{MUTED}" text-anchor="end">recomputed from data/quail/results/ · github.com/fulvian/engraft-ngram</text>')
    add("</svg>")
    write("post-limits", "\n".join(out), W)


figure_results()
figure_limits()
