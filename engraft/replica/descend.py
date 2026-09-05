"""Constrained gradient descent over the 16 trigger rows, with refreshed routing.

Two runs: lambda=10 (sister KL penalty active) and lambda=0 (control, no
sister penalty). Adam, at most 300 steps, stops at p_replica >= 0.7. Every
step records logp_y, p, per-sister KL, per-sister delta-logp(argmax_base) and
argmax (the checkpoint criterion, not just diagnostics), and the max norm
ratio.

Checkpoints: p thresholds {0.05, 0.1, 0.2, 0.35, 0.5} plus the final step; a
checkpoint is the *first* step where p_replica clears the threshold *and*
every sister has an unchanged argmax and delta-logp_s(argmax_base) >= -0.05
nat (a margin on the engine's -0.1); if the threshold clears without the
sisters passing, it is recorded as "threshold reached, sisters out" and the
descent continues.

`refresh_every`: the last position's routing is recomputed live
(`routing_source=None`) at step 0 and at every multiple of `refresh_every`,
and frozen (`routing_current`) between refreshes. With `refresh_every=None`
routing_source is always `routing_trigger`, no `.plert1` file is written, and
checkpoints/stop use default thresholds/p_stop. With an integer
`refresh_every`, checkpoints and stopping are based on `p_free` (p measured
only at refresh steps, where the loss forward is already at free routing),
not on `p` at every step; the final step is always a refresh step (an extra
gradient-free forward if needed).

`on_step`: optional hook called with each step's `record`, right after it is
written to the jsonl. Serves a runner-side step-time guardian: if `on_step`
raises, `descend` closes the log file and propagates, writing no checkpoint
for the interrupted step. `on_step=None` (default) leaves behavior bit-for-bit
unchanged -- no existing test is affected.

`plateau_metric` (default `"logp"`) picks the quantity that resets the
no-improvement refresh counter when it clears the best value seen so far by
more than its margin (0.05 nat for `"logp"`, 0.01 absolute for `"p_free"`,
selectable for compatibility): with p measured in absolute terms, a 0.01
improvement is impossible by construction once p is already below 0.01 (p =
exp(logp) saturates near 0), so for a fact whose trigger starts at very low
probability the counter never resets and the plateau criterion degenerates
into a hard step cap -- a false plateau. `"logp"` tracks real progress on the
quantity the descent actually minimizes.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from engraft.lens import RowSet, write_pleo, write_plert1
from engraft.table import ROW_LEN

from engraft.replica.model import Replica

THRESHOLDS = [0.05, 0.1, 0.2, 0.35, 0.5]
MU = 0.1
MAX_STEPS = 300
P_STOP = 0.7
SISTER_MARGIN_NAT = -0.05


def _kl(p_base: torch.Tensor, p_overlay: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sum(p_base * (torch.log(p_base + eps) - torch.log(p_overlay + eps)))


def _last_row(arr: np.ndarray) -> np.ndarray:
    """Last-position row of a per-layer PLERT1 array: [1,k] if the file records
    only the last position (layer n_layer-1), otherwise the last row of [T,k]
    (the prefix positions, indexed 0..T-1)."""
    a = np.asarray(arr)
    return a[0] if a.shape[0] == 1 else a[-1]


def _expert_set(row: np.ndarray) -> frozenset:
    return frozenset(int(x) for x in np.asarray(row).reshape(-1))


def _count_diverging_layers(cap_a: dict[int, np.ndarray], cap_b: dict[int, np.ndarray], n_layer: int) -> int:
    """Number of layers where the last-position expert set differs
    between two captured routings (`routing_changed_vs_prev`/`routing_changed_vs_base`)."""
    n = 0
    for il in range(n_layer):
        if il not in cap_a or il not in cap_b:
            continue
        if _expert_set(_last_row(cap_a[il])) != _expert_set(_last_row(cap_b[il])):
            n += 1
    return n


def _build_plert1_layers(
    routing_trigger: dict[int, np.ndarray], cap: dict[int, np.ndarray], n_layer: int
) -> dict[int, np.ndarray]:
    """Prefix rows from `routing_trigger[il][:-1]` (unchanged: the prefix does
    not depend on the rows) concatenated with the just-captured last-position
    row; for layer n_layer-1 (which consumes no routing in the prefix) just
    the captured row."""
    layers: dict[int, np.ndarray] = {}
    for il in range(n_layer):
        cap_row = _last_row(cap[il]).reshape(1, -1).astype(np.int32)
        if il == n_layer - 1:
            layers[il] = cap_row
        else:
            prefix = np.asarray(routing_trigger[il][:-1], dtype=np.int32)
            layers[il] = np.concatenate([prefix, cap_row], axis=0)
    return layers


def descend(
    replica: Replica,
    trigger_tokens: list[int],
    trigger_state,
    routing_trigger: dict[int, np.ndarray],
    rows_true: torch.Tensor,  # [16,160] righe vere del trigger (punto base)
    y: int,
    sisters: dict[str, dict],  # {sid: {"tokens","state","routing","rows_true","base_logits"}}
    trigger_rows_global: np.ndarray,
    lam: float,
    out_dir: Path,
    log_path: Path,
    row_mask: np.ndarray | None = None,
    refresh_every: int | None = None,
    thresholds: list[float] | None = None,
    p_stop: float | None = None,
    tag: str | None = None,
    plateau_steps: int = 50,
    rows_start: torch.Tensor | None = None,
    on_step: "Callable[[dict], None] | None" = None,
    plateau_metric: str = "logp",
) -> dict:
    thresholds = THRESHOLDS if thresholds is None else thresholds
    p_stop_val = P_STOP if p_stop is None else p_stop
    tag_val = str(lam) if tag is None else tag
    n_layer = len(routing_trigger)
    if plateau_metric not in ("p_free", "logp"):
        raise ValueError(f"unrecognized plateau_metric: {plateau_metric!r}")
    plateau_margin = 0.01 if plateau_metric == "p_free" else 0.05

    rows0 = rows_true.clone()  # regolarizzatore e diagnostiche: sempre le righe vere
    rows_init = rows_true if rows_start is None else rows_start
    rows = rows_init.clone().requires_grad_(True)

    mean_row_norm = float(rows0.norm(dim=1).mean().item())
    lr = 0.01 * mean_row_norm / (ROW_LEN ** 0.5)
    opt = torch.optim.Adam([rows], lr=lr)

    sister_base = {}
    for sid, s in sisters.items():
        p_base = torch.softmax(s["base_logits"], dim=-1)
        argmax_base = int(torch.argmax(s["base_logits"]).item())
        logp_base = torch.log_softmax(s["base_logits"].to(torch.float64), dim=-1)
        sister_base[sid] = {"p_base": p_base, "argmax_base": argmax_base, "logp_base": logp_base}

    # `plateau_refreshes = max(3, ceil(plateau_steps / k))`, None if
    # refresh_every is None (no plateau-based stop in that case: unchanged
    # behavior, the only stop conditions are p_stop / MAX_STEPS).
    plateau_refreshes = None
    if refresh_every is not None:
        plateau_refreshes = max(3, math.ceil(plateau_steps / refresh_every))

    steps = []
    checkpoints: dict[str, dict] = {}
    thresholds_with_sisters_out: list[float] = []
    regress_count = 0
    prev_logp_y = None

    # refresh state
    routing_current: dict[int, np.ndarray] = routing_trigger
    prev_cap: dict[int, np.ndarray] = routing_trigger
    n_refreshes = 0
    routing_changed_vs_prev_hist: list[int] = []
    best_metric = float("-inf")  # p_free_val or logp_y_val, per plateau_metric
    refreshes_since_improvement = 0
    stop_reason = None

    t_start = time.time()
    log_f = open(log_path, "a")

    reached_p_stop = False
    rows_pre = None
    for step in range(MAX_STEPS):
        opt.zero_grad()

        refreshed = (refresh_every is not None) and (step % refresh_every == 0)
        cap: dict[int, np.ndarray] | None = {} if refreshed else None
        routing_source_step = None if refreshed else routing_current

        # Ogni termine costruisce il proprio grafo a 48 strati e lo scarica subito con
        # il proprio backward() (retain_graph=False): il picco di RAM osservato il
        # 2026-09-05 (104 GB, corsa uccisa dal guardiano di memoria) veniva da un unico
        # backward() finale che teneva vivi contemporaneamente i quattro grafi (trigger
        # + 3 sorelle, ~48 strati ciascuno con gli esperti dell'ultima posizione).
        # `rows[0:8]` va ricalcolato fresco per ogni sorella (non un'unica variabile
        # condivisa): altrimenti il nodo di taglio sarebbe un antenato comune a tutti i
        # grafi delle sorelle e il primo backward(retain_graph=False) lo libererebbe,
        # rompendo i successivi. Il gradiente si accumula comunque su `rows.grad`
        # (unico leaf), la somma dei backward separati è identica a quella di un unico
        # backward sulla somma dei termini.
        trigger_logits = replica.last_step(
            trigger_tokens, trigger_state, rows, routing_source=routing_source_step,
            persist_experts=True, capture_routing=cap,
        )
        logp_trigger = torch.log_softmax(trigger_logits.to(torch.float64), dim=-1)
        logp_y = logp_trigger[y]
        p = torch.exp(logp_y)
        logp_y_val = float(logp_y.item())
        p_val = float(p.item())
        (-logp_y).backward(retain_graph=False)

        p_free_val = None
        changed_vs_base = None
        changed_vs_prev = None
        if refreshed:
            routing_current = cap
            p_free_val = p_val
            n_refreshes += 1
            changed_vs_base = _count_diverging_layers(cap, routing_trigger, n_layer)
            changed_vs_prev = _count_diverging_layers(cap, prev_cap, n_layer)
            prev_cap = cap
            routing_changed_vs_prev_hist.append(changed_vs_prev)

        kl_total_val = 0.0
        sister_metrics = {}
        for sid, s in sisters.items():
            rows_sister = torch.cat([rows[0:8], s["rows_true"][8:16]], dim=0)
            logits_s = replica.last_step(
                s["tokens"], s["state"], rows_sister, routing_source=s["routing"], persist_experts=True
            )
            p_overlay = torch.softmax(logits_s, dim=-1)
            logp_overlay = torch.log_softmax(logits_s.to(torch.float64), dim=-1)
            kl = _kl(sister_base[sid]["p_base"].to(torch.float64), p_overlay.to(torch.float64))

            argmax_base = sister_base[sid]["argmax_base"]
            argmax_overlay = int(torch.argmax(logits_s).item())
            delta_logp_argmax_base = float(
                (logp_overlay[argmax_base] - sister_base[sid]["logp_base"][argmax_base]).item()
            )
            sister_metrics[sid] = {
                "kl": float(kl.item()),
                "argmax_base": argmax_base,
                "argmax_overlay": argmax_overlay,
                "argmax_unchanged": argmax_overlay == argmax_base,
                "delta_logp_argmax_base": delta_logp_argmax_base,
            }
            kl_total_val += float(kl.item())

            if lam != 0.0:
                (lam * kl).backward(retain_graph=False)

        reg = MU * ((rows - rows0).pow(2).sum() / rows0.pow(2).sum())
        reg.backward()
        reg_val = float(reg.item())

        loss_val = -logp_y_val + lam * kl_total_val + reg_val

        # Le righe salvate nei punti di controllo sono quelle su cui p e le sorelle sono
        # stati misurati in questo passo (pre-aggiornamento), non quelle dopo opt.step():
        # altrimenti l'overlay sarebbe un passo avanti rispetto ai numeri registrati.
        rows_pre = rows.detach().clone()
        if row_mask is not None:
            # `--rows T8`: heads outside the mask stay at the true rows.
            # Gradiente azzerato -> Adam non le muove (momenti nulli), e per sicurezza
            # si ripristinano dopo il passo.
            frozen = torch.from_numpy(~np.asarray(row_mask, dtype=bool))
            rows.grad[frozen] = 0.0
        opt.step()
        if row_mask is not None:
            with torch.no_grad():
                rows[frozen] = rows0[frozen]

        norm_ratio_max = float((rows.detach().norm(dim=1) / rows0.norm(dim=1).clamp_min(1e-12)).max().item())

        if prev_logp_y is not None and logp_y_val < prev_logp_y:
            regress_count += 1
        prev_logp_y = logp_y_val

        record = {
            "step": step,
            "logp_y": logp_y_val,
            "p": p_val,
            "loss": loss_val,
            "kl_total": kl_total_val,
            "sisters": sister_metrics,
            "norm_ratio_max": norm_ratio_max,
            "t_s": time.time() - t_start,
            "refreshed": refreshed,
            "p_free": p_free_val,
            "routing_changed_vs_prev": changed_vs_prev,
            "routing_changed_vs_base": changed_vs_base,
        }
        steps.append(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

        # Minimal, additive extension (`on_step=None` by default -> unchanged,
        # bit-for-bit behavior). The caller may raise to interrupt (e.g. a step
        # too slow for too many consecutive steps): the log file is closed
        # correctly before propagating, so a resume never finds it half
        # written without a final flush.
        if on_step is not None:
            try:
                on_step(record)
            except BaseException:
                log_f.close()
                raise

        sisters_ok = all(m["argmax_unchanged"] and m["delta_logp_argmax_base"] >= SISTER_MARGIN_NAT for m in sister_metrics.values())

        if refresh_every is None:
            # comportamento 2f invariato: soglie e arresto su `p_val`, a ogni passo.
            for thr in thresholds:
                key = f"p{thr}"
                if p_val >= thr:
                    if sisters_ok and key not in checkpoints:
                        checkpoints[key] = {"step": step, "threshold": thr, "p": p_val, "sisters_ok": True}
                        rows_np = rows_pre.numpy().copy()
                        pleo_path = out_dir / f"ckpt_{tag_val}_{key}.pleo"
                        write_pleo(pleo_path, trigger_rows_global, rows_np)
                        np.save(out_dir / f"ckpt_{tag_val}_{key}.npy", rows_np)
                    elif not sisters_ok and thr not in thresholds_with_sisters_out:
                        thresholds_with_sisters_out.append(thr)
                        # Salvato comunque, con suffisso: serve a Q4 (regione di fiducia),
                        # non a Q3. Il rapporto li tiene distinti dai punti con sorelle ok.
                        checkpoints[key + "_sistersout"] = {"step": step, "threshold": thr, "p": p_val, "sisters_ok": False}
                        rows_np = rows_pre.numpy().copy()
                        write_pleo(out_dir / f"ckpt_{tag_val}_{key}_sistersout.pleo", trigger_rows_global, rows_np)
                        np.save(out_dir / f"ckpt_{tag_val}_{key}_sistersout.npy", rows_np)

            if p_val >= p_stop_val:
                reached_p_stop = True
                stop_reason = "p_stop"
                break
        elif refreshed:
            # thresholds and stop based on `p_free`, measured only at
            # rinfresco.
            for thr in thresholds:
                key = f"p{thr}"
                if p_free_val >= thr:
                    if sisters_ok and key not in checkpoints:
                        checkpoints[key] = {"step": step, "threshold": thr, "p_free": p_free_val, "sisters_ok": True}
                        rows_np = rows_pre.numpy().copy()
                        pleo_path = out_dir / f"ckpt_{tag_val}_{key}.pleo"
                        write_pleo(pleo_path, trigger_rows_global, rows_np)
                        np.save(out_dir / f"ckpt_{tag_val}_{key}.npy", rows_np)
                        write_plert1(
                            out_dir / f"ckpt_{tag_val}_{key}.plert1",
                            _build_plert1_layers(routing_trigger, cap, n_layer),
                        )
                    elif not sisters_ok and thr not in thresholds_with_sisters_out:
                        thresholds_with_sisters_out.append(thr)
                        checkpoints[key + "_sistersout"] = {"step": step, "threshold": thr, "p_free": p_free_val, "sisters_ok": False}
                        rows_np = rows_pre.numpy().copy()
                        write_pleo(out_dir / f"ckpt_{tag_val}_{key}_sistersout.pleo", trigger_rows_global, rows_np)
                        np.save(out_dir / f"ckpt_{tag_val}_{key}_sistersout.npy", rows_np)

            if p_free_val >= p_stop_val:
                reached_p_stop = True
                stop_reason = "p_stop"
                break

            metric_val = p_free_val if plateau_metric == "p_free" else logp_y_val
            if metric_val > best_metric + plateau_margin:
                best_metric = metric_val
                refreshes_since_improvement = 0
            else:
                refreshes_since_improvement += 1
            if refreshes_since_improvement >= plateau_refreshes:
                stop_reason = "plateau"
                break

    log_f.close()

    if not steps:
        return {
            "lambda": lam, "tag": tag_val, "row_mask": None if row_mask is None else [bool(x) for x in row_mask],
            "n_steps": 0, "reached_p_stop": False, "final_logp_y": None, "final_p": 0.0,
            "n_regress_steps": 0, "checkpoints": {}, "n_effective_checkpoints": 0,
            "thresholds_with_sisters_out": [], "diverging_layers_at_final": [],
            "total_time_s": time.time() - t_start, "lr": lr, "refresh_every": refresh_every,
            "stop_reason": None, "plateau_refreshes": plateau_refreshes, "n_refreshes": 0,
            "plateau_metric": plateau_metric,
            "final_p_free": None, "final_refreshed_by_extra_forward": False,
            "routing_changed_vs_prev_hist": [], "cos_min_final": None, "norm_ratio_max_final": None,
        }

    if refresh_every is None and stop_reason is None:
        stop_reason = "p_stop" if reached_p_stop else "max_steps"

    if refresh_every is not None and stop_reason is None:
        stop_reason = "max_steps"

    # punto finale
    final_p = steps[-1]["p"] if steps else 0.0
    final_sisters_ok = all(
        m["argmax_unchanged"] and m["delta_logp_argmax_base"] >= SISTER_MARGIN_NAT
        for m in (steps[-1]["sisters"].values() if steps else [])
    )

    final_p_free = None
    final_refreshed_by_extra_forward = False
    final_routing_layers = None

    if refresh_every is None:
        rows_np = rows_pre.numpy().copy()
        write_pleo(out_dir / f"ckpt_{tag_val}_final.pleo", trigger_rows_global, rows_np)
        np.save(out_dir / f"ckpt_{tag_val}_final.npy", rows_np)
        checkpoints["final"] = {
            "step": steps[-1]["step"], "p": final_p, "sisters_ok": final_sisters_ok,
        }
    else:
        # spec B5: il finale e' sempre un passo di rinfresco. Se l'ultimo passo non lo
        # era, un forward extra senza gradiente su `rows_pre` di quel passo lo fornisce.
        if steps[-1]["refreshed"]:
            final_p_free = steps[-1]["p_free"]
            final_routing_layers = _build_plert1_layers(routing_trigger, prev_cap, n_layer)
        else:
            cap_extra: dict[int, np.ndarray] = {}
            with torch.no_grad():
                logits_extra = replica.last_step(
                    trigger_tokens, trigger_state, rows_pre, routing_source=None,
                    persist_experts=False, capture_routing=cap_extra,
                )
            logp_extra = torch.log_softmax(logits_extra.to(torch.float64), dim=-1)
            final_p_free = float(torch.exp(logp_extra[y]).item())
            final_refreshed_by_extra_forward = True
            final_routing_layers = _build_plert1_layers(routing_trigger, cap_extra, n_layer)

        rows_np = rows_pre.numpy().copy()
        write_pleo(out_dir / f"ckpt_{tag_val}_final.pleo", trigger_rows_global, rows_np)
        np.save(out_dir / f"ckpt_{tag_val}_final.npy", rows_np)
        write_plert1(out_dir / f"ckpt_{tag_val}_final.plert1", final_routing_layers)
        checkpoints["final"] = {
            "step": steps[-1]["step"], "p_free": final_p_free, "sisters_ok": final_sisters_ok,
            "refreshed_by_extra_forward": final_refreshed_by_extra_forward,
        }

    # minimum cosine / max norm ratio on the final step:
    # sulle sole righe in maschera se `row_mask` e' dato, altrimenti su tutte.
    rows_pre_np = rows_pre.numpy()
    rows0_np = rows0.numpy()
    mask_idx = np.asarray(row_mask, dtype=bool) if row_mask is not None else np.ones(rows0_np.shape[0], dtype=bool)
    a = rows_pre_np[mask_idx]
    b = rows0_np[mask_idx]
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    cos_min_final = float(cos.min()) if a.shape[0] else None
    norm_ratio_max_final = float((np.linalg.norm(a, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)).max()) if a.shape[0] else None

    # Free routing at the final point: layers where it diverges from the frozen
    # one, on the last position (diagnostic). Reuses `trigger_state` (the
    # prefix does not depend on `rows`, verified by check_precondition): do NOT
    # call routing_free_full, which would recompute a whole 48-layer prefix
    # from scratch just for this diagnostic.
    final_routing: dict[int, np.ndarray] = {}
    with torch.no_grad():
        replica.last_step(
            trigger_tokens, trigger_state, rows.detach(), routing_source=None,
            persist_experts=False, capture_routing=final_routing,
        )
    diverging_layers = []
    for il, arr_frozen in routing_trigger.items():
        arr_free = final_routing.get(il)
        if arr_free is None:
            continue
        last_frozen = _expert_set(_last_row(arr_frozen))
        last_free = _expert_set(_last_row(arr_free))
        if last_frozen != last_free:
            diverging_layers.append(il)

    return {
        "lambda": lam,
        "tag": tag_val,
        "row_mask": None if row_mask is None else [bool(x) for x in row_mask],
        "n_steps": len(steps),
        "reached_p_stop": reached_p_stop,
        "final_logp_y": steps[-1]["logp_y"] if steps else None,
        "final_p": final_p,
        "n_regress_steps": regress_count,
        "checkpoints": checkpoints,
        "n_effective_checkpoints": len([k for k in checkpoints if k != "final" and not k.endswith("_sistersout")]),
        "thresholds_with_sisters_out": thresholds_with_sisters_out,
        "diverging_layers_at_final": diverging_layers,
        "total_time_s": time.time() - t_start,
        "lr": lr,
        "refresh_every": refresh_every,
        "stop_reason": stop_reason,
        "plateau_refreshes": plateau_refreshes,
        "n_refreshes": n_refreshes,
        "final_p_free": final_p_free,
        "final_refreshed_by_extra_forward": final_refreshed_by_extra_forward,
        "routing_changed_vs_prev_hist": routing_changed_vs_prev_hist,
        "cos_min_final": cos_min_final,
        "norm_ratio_max_final": norm_ratio_max_final,
        "plateau_metric": plateau_metric,
    }
