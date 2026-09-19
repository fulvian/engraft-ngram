"""Mixed-routing descent: pure arbiter `RegimeArbiter` + `replay_regime` -- G-B synthetic tests (spec §5 G-B).

To run tests:
uv run --with pytest --with torch --with numpy python3 -m pytest \\
    tests/test_regime.py -p no:cacheprovider -q
"""
from __future__ import annotations

import pytest

import engraft.replica.regime as R

# --------------------------------------------------------------------------
# G-B: synthetic tests (spec §5 G-B).
# --------------------------------------------------------------------------


def test_locked_rows_growing_without_pause_never_switches():
    """g >= 0.07 at every reading (norm_ratio_max growing by 8% per reading,
    eval_every=20): condition A always false -- never switch, regardless of B."""
    norm = [1.0]
    for _ in range(9):
        norm.append(norm[-1] * 1.08)
    acc = [0.1 + 0.05 * i for i in range(10)]
    series = [{"step": 20 * i, "acc_rbr": acc[i], "norm_ratio_max": norm[i]} for i in range(10)]
    decisions = R.replay_regime(series)
    assert not any(d.event == "switch" for d in decisions)
    assert all(d.next_state == R.STATE_LOCKED for d in decisions)


def test_locked_flat_rows_with_steep_rising_accuracy_never_switches():
    """Flat rows (A always true, norm_ratio_max constant) but accuracy
    linearly rising: r_4 = 4/k (k = reading index, 1-based in the local_rate_ratio
    formula) stays above phi_s=0.6 for this entire short series (k<=5:
    r_4 in {1.333; 1.0; 0.8}) -- B always false, never switch: proof that
    the dual condition is required (A alone is not sufficient)."""
    norm = [1.0] * 5
    acc = [0.5 + 0.05 * i for i in range(5)]
    series = [{"step": 20 * i, "acc_rbr": acc[i], "norm_ratio_max": norm[i]} for i in range(5)]
    decisions = R.replay_regime(series)
    assert not any(d.event == "switch" for d in decisions)


def test_locked_both_true_from_first_computable_reading_switches_exactly_at_reading_4():
    """B is computable for the first time at k=3 (4 readings, `switch_window`); with
    A true from k=1 (norm_ratio_max constant) and B true already at k=3 (r_4=0.4) and at k=4
    (r_4=-6.8, positive gain 0.05>0), the persistence P_s=2 fires on the FIRST
    available consecutive pair -- switch exactly at reading 4, step 80 with eval_every=20
    (spec §1.1: "switch cannot fire before k=4")."""
    norm = [1.0] * 6
    acc = [0.5, 0.9, 0.5, 0.7, 0.55, 0.5]
    series = [{"step": 20 * i, "acc_rbr": acc[i], "norm_ratio_max": norm[i]} for i in range(6)]
    decisions = R.replay_regime(series)
    idx = next(i for i, d in enumerate(decisions) if d.event == "switch")
    assert idx == 4
    assert series[idx]["step"] == 80


def test_locked_both_true_once_then_a_false_never_switches():
    """Same accuracy series as the previous test (B true at k=3 and k=4), but
    `norm_ratio_max` jumps at k=4 (doubles, g=1.0 >= gamma): persistence resets at k=4
    (a single reading with A and B both true is not enough, P_s=2) -- no
    switch in the entire series."""
    norm = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    acc = [0.5, 0.9, 0.5, 0.7, 0.55, 0.5]
    series = [{"step": 20 * i, "acc_rbr": acc[i], "norm_ratio_max": norm[i]} for i in range(6)]
    decisions = R.replay_regime(series)
    assert not any(d.event == "switch" for d in decisions)


def test_locked_accuracy_at_or_below_a0_never_switches_floor_guard():
    """Constant accuracy = a_0 (gain always 0): `local_rate_ratio`
    returns `None` (gain guard, spec §1.1 B) -- B not computable
    therefore always false, no switch even with perfectly flat rows
    (A always true)."""
    norm = [1.0] * 7
    acc = [0.5] * 7
    series = [{"step": 20 * i, "acc_rbr": acc[i], "norm_ratio_max": norm[i]} for i in range(7)]
    decisions = R.replay_regime(series)
    assert not any(d.event == "switch" for d in decisions)


def test_locked_none_readings_ignored_and_counted():
    """A reading with `acc_rbr=None` (no heldout position in that bin) is not
    a live reading -- it does not enter the locked series (step 10 is absent
    from `_locked_steps`) and is counted in `n_skipped`."""
    arb = R.RegimeArbiter()
    readings = [
        (0, 0.5, 1.0), (10, None, 1.0), (20, 0.6, 1.0),
    ]
    for step, acc_rbr, norm_ratio_max in readings:
        arb.push(step, acc_rbr, None, norm_ratio_max, None)
    assert arb._locked_steps == [0, 20]
    assert arb.n_skipped == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"switch_norm_growth": 0.0},
        {"switch_norm_growth_per_steps": 0},
        {"switch_window": 1},
        {"switch_phi": 0.0},
        {"switch_persistence": 0},
        {"instab_drop": -0.01},
        {"instab_persistence": 0},
        {"instab_grad_factor": 1.0},
        {"reentry_readings": 0},
        {"eval_every_free": 0},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        R.RegimeArbiter(**kwargs)


# --------------------------------------------------------------------------
# G-B: free/reentry phases -- constructed by forcing an early switch with the
# same locked series as `test_locked_both_true_from_first_computable_reading_...`
# (switch at step 80), then manually pushing free readings.
# --------------------------------------------------------------------------


def _switched_arbiter(free0: float, grad_kd_ref: "float | None" = None, **kwargs) -> R.RegimeArbiter:
    """Arbiter just transitioned to FREE at step 80 (same locked series as tests above),
    with `acc_free` of the switch reading = `free0` and
    gradient reference (spec §1.1, "average over last 10 steps S-9..S") =
    `grad_kd_ref`."""
    norm = [1.0] * 5
    acc = [0.5, 0.9, 0.5, 0.7, 0.55]
    arb = R.RegimeArbiter(**kwargs)
    for i in range(5):
        is_switch_reading = i == 4
        arb.push(
            20 * i, acc[i], free0 if is_switch_reading else None, norm[i],
            grad_kd_ref if is_switch_reading else None,
        )
    assert arb.state == R.STATE_FREE
    assert arb.switch_step == 80
    return arb


def test_free_saturating_series_stops_no_earlier_than_free_reading_4():
    """Free series that rises then flattens (same "floor-rise-flat" pattern
    as existing exit-criterion tests, without floor here since the rise starts
    from the switch reading itself): exit detection (default LocalRateDetector,
    L=4/phi=0.05/P=2) does not fire before free reading 4 (step S+40) -- here
    it fires at reading 6 (step 140)."""
    arb = _switched_arbiter(free0=0.75)
    free_after = [0.85, 0.9, 0.9, 0.9, 0.9, 0.9]  # free readings 1..6
    step = arb.switch_step
    last = None
    for i, v in enumerate(free_after, start=1):
        step = arb.switch_step + 10 * i
        last = arb.push(step, None, v, None, 0.0)
        if last.next_state == R.STATE_STOP:
            break
    assert last.next_state == R.STATE_STOP
    assert last.reason == R.STOP_REASON_PLATEAU_FREE
    assert step >= arb.switch_step + 40
    assert step == 140


def test_free_single_drop_below_threshold_does_not_trigger_reentry():
    """A single drop (>delta) for ONE reading only, followed by recovery: persistence
    (P=2) not satisfied -- remains in FREE."""
    arb = _switched_arbiter(free0=0.5, exit_persistence=1_000_000)  # disable exit detector
    free_after = [0.8, 0.75, 0.85, 0.9]  # drop only at k=2 (0.8-0.75=0.05>0.02), recovery at k=3
    step = arb.switch_step
    d = None
    for i, v in enumerate(free_after, start=1):
        step = arb.switch_step + 10 * i
        d = arb.push(step, None, v, None, 0.0)
    assert arb.state == R.STATE_FREE
    assert d.next_state == R.STATE_FREE


def test_free_two_consecutive_drops_of_003_trigger_reentry_at_reading_2():
    """Two consecutive readings below the maximum by more than INSTAB_DROP (0.02;
    here 0.03, as in the spec example) trigger REENTRY -- the drop is
    computable for the first time at free reading 2 (spec §1.3), so
    REENTRY fires at free reading 3 (second consecutive drop)."""
    arb = _switched_arbiter(free0=0.5, exit_persistence=1_000_000)
    free_after = [0.8, 0.75, 0.74]  # max=0.8 after k1; drop 0.05 at k2, 0.06 at k3 (both > 0.03)
    step = arb.switch_step
    decisions = []
    for i, v in enumerate(free_after, start=1):
        step = arb.switch_step + 10 * i
        decisions.append((i, arb.push(step, None, v, None, 0.0)))
    assert decisions[1][1].next_state == R.STATE_FREE  # k=2: single drop, not yet persistent
    assert decisions[2][1].next_state == R.STATE_REENTRY  # k=3: second consecutive drop
    assert decisions[2][1].reason == "drop"
    assert arb.n_reentries == 1


def test_reentry_returns_to_free_after_r_readings_then_second_instability_stops():
    """After R=4 reentry readings (spec §1.3, `REENTRY_READINGS`) return to
    FREE (reference maximum and drop persistence reset); a
    SECOND pair of consecutive drops triggers `STOP unstable`
    (second instability trigger, never a third REENTRY). T4-bis (debt 3
    of final review, spec rev. 2 §1.3/§2): `n_reentries` counts REENTRY phases
    actually opened (one only: the second trigger goes directly to
    STOP), `n_instability_triggers` counts EVERY trigger (two) -- the two
    accountings diverge precisely here."""
    arb = _switched_arbiter(free0=0.5, exit_persistence=1_000_000)
    free_first_drop = [0.8, 0.75, 0.74]
    step = arb.switch_step
    for i, v in enumerate(free_first_drop, start=1):
        step = arb.switch_step + 10 * i
        d = arb.push(step, None, v, None, 0.0)
    assert arb.state == R.STATE_REENTRY
    assert arb.n_reentries == 1
    assert arb.n_instability_triggers == 1

    for _ in range(arb.reentry_readings):
        step += 10
        d = arb.push(step, None, 0.74, None, 0.0)
    assert arb.state == R.STATE_FREE
    assert d.event == "reentry_end"

    free_second_drop = [0.9, 0.85, 0.8]  # rise then two consecutive drops (0.05 and 0.1 > 0.02)
    for v in free_second_drop:
        step += 10
        d = arb.push(step, None, v, None, 0.0)
    assert d.next_state == R.STATE_STOP
    assert d.reason == R.STOP_REASON_UNSTABLE
    assert arb.n_reentries == 1
    assert arb.n_instability_triggers == 2
    assert sum(1 for f in arb.phases if f.state == R.STATE_REENTRY) == 1


def test_free_gradient_factor_25_triggers_reentry_15_does_not_reference_zero_never():
    """Average of `grad_kd_norm` over last 10 steps > kappa (2.0) times
    reference set at switch -- 2.5x triggers REENTRY, 1.5x does not; reference
    0 never triggers (explicit guard, spec §1.3)."""
    arb_no = _switched_arbiter(free0=0.5, grad_kd_ref=1.0, exit_persistence=1_000_000)
    d_no = arb_no.push(arb_no.switch_step + 10, None, 0.6, None, 1.5)
    assert d_no.next_state == R.STATE_FREE

    arb_yes = _switched_arbiter(free0=0.5, grad_kd_ref=1.0, exit_persistence=1_000_000)
    d_yes = arb_yes.push(arb_yes.switch_step + 10, None, 0.6, None, 2.5)
    assert d_yes.next_state == R.STATE_REENTRY
    assert d_yes.reason == "gradient"

    arb_ref0 = _switched_arbiter(free0=0.5, grad_kd_ref=0.0, exit_persistence=1_000_000)
    d_ref0 = arb_ref0.push(arb_ref0.switch_step + 10, None, 0.6, None, 999.0)
    assert d_ref0.next_state == R.STATE_FREE
