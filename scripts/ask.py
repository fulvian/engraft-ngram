#!/usr/bin/env python3
"""Ask the real engine one prompt, with and without a `.pleo` overlay.

The quickest way to see an overlay do something: the same prompt is sent to the
`fork-ple` lens twice (no overlay, then the overlay), and for each you get the top
next tokens with their probabilities and a greedy continuation. Nothing on disk is
modified; the overlay rows are substituted at gather time for that request only.

    uv run python scripts/ask.py --overlay results/2026-09-12/s0/merged.pleo \
        --prompt "La sferoglifica di Gabrinsu fu scoperta a"

Engine and model paths come from `engraft.toml` (`engine.lens_bin`, `model.shards`,
`model.tokenizer`), exactly as `scripts/window.sh` uses them. `--lens-cmd` replaces the
whole engine command (used by the test with a fake engine; then `--tokens` replaces
`--prompt` and no tokenizer is needed). `--json` prints one JSON object instead of text.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engraft.engine import ENGINE_CFG, LensClient, logsoftmax64, run_job  # noqa: E402


def build_lens_cmd(cfg) -> list[str]:
    """The engine command `scripts/window.sh` builds, plus the quantized-engine flags."""
    lens = str(cfg.get_path("engine.lens_bin"))
    shard = str(cfg.get_list("model.shards")[0])
    return [lens, "-m", shard, "--ngram-on-disk", "-ngl", "99", "-c", "4096", "-b", "4096", "-ub", "4096"] + list(
        ENGINE_CFG["q8"]["args"]
    )


def top_tokens(row: np.ndarray, k: int) -> list[dict]:
    lp = logsoftmax64(np.asarray(row, dtype=np.float64))
    idx = np.argsort(-lp)[:k]
    return [{"token": int(i), "p": float(np.exp(lp[i]))} for i in idx]


def greedy_ids(client: LensClient, raw_dir: Path, tokens: list[int], overlay: str | None, n: int, log, tag: str) -> list[int]:
    seq = list(tokens)
    gen: list[int] = []
    for i in range(n):
        job = {"id": f"{tag}_greedy{i}", "text": "", "tokens": seq, "overlay": overlay, "capture": [], "logits": "last"}
        _result, row, _meta = run_job(client, raw_dir, job, log)
        nxt = int(np.argmax(row))
        gen.append(nxt)
        seq.append(nxt)
    return gen


def ask(client: LensClient, raw_dir: Path, tokens: list[int], overlay: str | None, n: int, k: int, log) -> dict:
    """Runs the prompt without and with the overlay; returns both sides."""
    out: dict = {"tokens": list(tokens), "overlay": overlay}
    for side, ov in (("base", None), ("overlay", overlay)):
        if side == "overlay" and overlay is None:
            continue
        job = {"id": f"ask_{side}", "text": "", "tokens": list(tokens), "overlay": ov, "capture": [], "logits": "last"}
        result, row, _meta = run_job(client, raw_dir, job, log)
        out[side] = {
            "overlay_hits": result.get("overlay_hits"),
            "top": top_tokens(row, k),
            "greedy": greedy_ids(client, raw_dir, tokens, ov, n, log, f"ask_{side}") if n > 0 else [],
        }
    return out


def render_text(res: dict, decode) -> str:
    lines = []
    for side in ("base", "overlay"):
        if side not in res:
            continue
        r = res[side]
        lines.append(f"[{side}] overlay rows hit: {r['overlay_hits']}")
        for t in r["top"]:
            label = decode([t["token"]]) if decode else ""
            lines.append(f"    p={t['p']:.4f}  token {t['token']:>7}  {label!r}")
        if r["greedy"]:
            text = decode(r["greedy"]) if decode else ""
            lines.append(f"    greedy: {r['greedy']}  {text!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to engraft.toml (default: ./engraft.toml)")
    p.add_argument("--lens-cmd", default=None, help="full engine command (replaces the one built from engraft.toml)")
    p.add_argument("--overlay", default=None, help=".pleo overlay; without it only the base side runs")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", default=None, help="text, tokenized with model.tokenizer from engraft.toml")
    g.add_argument("--tokens", type=int, nargs="+", default=None, help="token ids (no tokenizer needed)")
    p.add_argument("--n", type=int, default=8, help="greedy continuation length (0 = none)")
    p.add_argument("--top", type=int, default=5, help="how many next tokens to show")
    p.add_argument("--json", action="store_true", help="print one JSON object")
    args = p.parse_args(argv)

    log = logging.getLogger("ask")
    log.addHandler(logging.NullHandler())

    decode = None
    tokens = args.tokens
    cfg = None
    if args.prompt is not None or args.lens_cmd is None:
        from engraft.config import load as load_config

        cfg = load_config(args.config)
    if args.prompt is not None:
        from engraft.table import PleTokenizer

        tok = PleTokenizer(cfg.get_path("model.tokenizer"))
        tokens = tok.encode(args.prompt)
        decode = lambda ids: tok._tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    if not tokens:
        p.error("empty prompt")

    overlay = str(Path(args.overlay).resolve()) if args.overlay else None
    if overlay and not Path(overlay).is_file():
        p.error(f"overlay not found: {overlay}")

    with tempfile.TemporaryDirectory(prefix="engraft-ask-") as tmp:
        raw_dir = Path(tmp) / "raw"
        cmd = shlex.split(args.lens_cmd) if args.lens_cmd else build_lens_cmd(cfg)
        cmd = cmd + ["--jobs", "-", "--out", str(raw_dir)]
        client = LensClient(cmd, raw_dir, Path(tmp) / "engine.log")
        try:
            res = ask(client, raw_dir, list(tokens), overlay, args.n, args.top, log)
        finally:
            client.close()

    if decode:
        for side in ("base", "overlay"):
            if side in res:
                res[side]["top_text"] = [decode([t["token"]]) for t in res[side]["top"]]
                res[side]["greedy_text"] = decode(res[side]["greedy"]) if res[side]["greedy"] else ""
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print(render_text(res, decode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
