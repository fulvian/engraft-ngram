"""Replica-vs-engine validation: Q1a, Q1b, Q2.

Q1a: routing imposed from a frozen reference (`base_a.bin`, already frozen
bit-for-bit), replica logits compared against `f32/base.json` (top-10 in
order, |delta-logp_y| <= 0.02).
Q1b: free (live) replica routing compared against the same reference, per
layer and position (layers 0-46 all positions, layer 47 only the last);
counts disagreements at the last position (threshold >= 46/48) and a relative
margin for each.
Q2: analytic gradient (frozen routing) against precomputed finite differences
(`pairs_f32_frozen_0.02.json`), Pearson correlation / slope / sign agreement.

If Q1a fails: no descent is run (a declared stop, not worked around). Q1b is
still computed and reported to localize the first diverging layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from engraft.replica.model import Replica, PrefixState


def _top10_order_match(logits_top10: list[int], reference_top10: list[int]) -> bool:
    return list(logits_top10) == list(reference_top10)


def run_q1a(
    replica: Replica,
    tokens: list[int],
    rows_true: torch.Tensor,
    routing_source: dict[int, np.ndarray],
    base_json: dict,
) -> dict:
    state = replica.prefix(tokens, routing_source=routing_source)
    with torch.no_grad():
        logits = replica.last_step(tokens, state, rows_true, routing_source=routing_source, persist_experts=True)

    logp = torch.log_softmax(logits.to(torch.float64), dim=-1)
    y = int(base_json["y"])
    logp_y_replica = float(logp[y].item())
    logp_y_ref = float(base_json["logp_y"])
    diff = abs(logp_y_replica - logp_y_ref)

    top10_replica = torch.topk(logits, 10).indices.tolist()
    top10_ref = list(base_json["top10"])
    order_match = _top10_order_match(top10_replica, top10_ref)

    return {
        "y": y,
        "logp_y_replica": logp_y_replica,
        "logp_y_ref": logp_y_ref,
        "diff_logp_y": diff,
        "threshold": 0.02,
        "top10_replica": top10_replica,
        "top10_ref": top10_ref,
        "top10_order_match": order_match,
        "pass": order_match and diff <= 0.02,
    }


def run_q1b(
    replica: Replica,
    tokens: list[int],
    rows_true: torch.Tensor,
    routing_engine: dict[int, np.ndarray],
) -> dict:
    routing_replica, diag_prefix, diag_last = replica.routing_free_full(tokens, rows_true)

    n_layer = replica.hp.n_layer
    t_last = len(tokens) - 1
    per_layer = {}
    last_pos_discordant = 0
    last_pos_total = 0

    for il in range(n_layer):
        eng = routing_engine.get(il)
        rep = routing_replica.get(il)
        if eng is None or rep is None:
            continue
        entries = []
        n_pos = eng.shape[0]
        for t in range(n_pos):
            set_eng = set(int(x) for x in eng[t])
            set_rep = set(int(x) for x in rep[t])
            discordant = set_eng != set_rep
            entry = {"pos": t if n_pos > 1 else t_last, "discordant": discordant}
            if discordant:
                diag = diag_last if n_pos == 1 else diag_prefix
                diag_row = 0 if n_pos == 1 else t
                if il in diag:
                    idx11, val11 = diag[il]
                    top10_val = float(val11[diag_row, 9])
                    excluded_val = float(val11[diag_row, 10])
                    margin_rel = (top10_val - excluded_val) / top10_val if top10_val != 0 else float("nan")
                    entry["margin_relative_10th_vs_11th"] = margin_rel
            entries.append(entry)
            is_last_position = (n_pos == 1) or (t == n_pos - 1 and n_pos == len(tokens))
            if is_last_position:
                last_pos_total += 1
                if discordant:
                    last_pos_discordant += 1
        per_layer[il] = entries

    return {
        "per_layer": per_layer,
        "last_position_discordant": last_pos_discordant,
        "last_position_total": last_pos_total,
        "last_position_match": last_pos_total - last_pos_discordant,
        "threshold_min_match": 46,
        "pass": (last_pos_total - last_pos_discordant) >= 46,
    }


def run_q2(
    replica: Replica,
    tokens: list[int],
    rows_true: torch.Tensor,
    routing_source: dict[int, np.ndarray],
    pairs: dict,
) -> dict:
    tokens_11 = pairs["tokens_11"]
    d_plus = pairs["d_plus"]
    d_minus = pairs["d_minus"]
    coords = sorted(int(c) for c in d_plus.keys())

    state = replica.prefix(tokens, routing_source=routing_source)
    rows = rows_true.clone().requires_grad_(True)
    logits = replica.last_step(tokens, state, rows, routing_source=routing_source, persist_experts=True)
    logp = torch.log_softmax(logits.to(torch.float64), dim=-1)

    grads = {}
    for j, tok in enumerate(tokens_11):
        retain = j < len(tokens_11) - 1
        (g,) = torch.autograd.grad(logp[tok], rows, retain_graph=retain)
        grads[tok] = g.reshape(-1).numpy()

    analytic = np.zeros((len(coords), len(tokens_11)))
    finite = np.zeros((len(coords), len(tokens_11)))
    for ci, c in enumerate(coords):
        for j, tok in enumerate(tokens_11):
            analytic[ci, j] = grads[tok][c]
            finite[ci, j] = (d_plus[str(c)][j] - d_minus[str(c)][j]) / 0.04

    def stats(a: np.ndarray, f: np.ndarray) -> dict:
        a_flat, f_flat = a.reshape(-1), f.reshape(-1)
        if np.std(a_flat) == 0 or np.std(f_flat) == 0:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(a_flat, f_flat)[0, 1])
        # regression slope (f as predictor, a as response -- scale ratio)
        if np.sum(f_flat ** 2) > 0:
            slope = float(np.sum(a_flat * f_flat) / np.sum(f_flat ** 2))
        else:
            slope = float("nan")
        sign_match = float(np.mean(np.sign(a_flat) == np.sign(f_flat)))
        return {"pearson": pearson, "slope": slope, "sign_match": sign_match, "n": int(a_flat.size)}

    y_idx = tokens_11.index(pairs.get("y", tokens_11[0]))
    overall = stats(analytic, finite)
    per_y = stats(analytic[:, y_idx : y_idx + 1], finite[:, y_idx : y_idx + 1])
    per_token = {
        str(tok): stats(analytic[:, j : j + 1], finite[:, j : j + 1]) for j, tok in enumerate(tokens_11)
    }

    return {
        "coords": coords,
        "tokens_11": tokens_11,
        "overall": overall,
        "target_token": per_y,
        "per_token": per_token,
        "pass": overall["pearson"] >= 0.95 and 0.8 <= overall["slope"] <= 1.25 and overall["sign_match"] >= 0.9,
    }
