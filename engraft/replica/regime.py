"""Mixed-routing descent: state machine locked -> free -> stop.

Pure module (no torch), alongside `plateau.py`: the arbiter is consulted ONLY at
evaluation steps (as `LocalRateDetector`/`PlateauDetector` are in the descent
harness), reads the step's readings and decides the next step's state -- never a
second formula for condition B (computed with the SAME function
`local_rate_ratio`, imported from `plateau.py`).

Reading convention (applies here too): a "live reading" is a non-`None` value;
readings from the locked phase are indexed by `k` 0-based in order of arrival
(not by absolute step -- the step is recorded separately in `phases`/
`switch_step`).

States: "LOCKED" -> "FREE" -> "STOP", with "REENTRY" (locked by time, a safety
net) between "FREE" and "FREE"/"STOP"."""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

from engraft.replica.plateau import LocalRateDetector, local_rate_ratio

# --------------------------------------------------------------------------
# Constants -- ONLY numeric literals in the module.
# --------------------------------------------------------------------------

SWITCH_NORM_GROWTH = 0.04  # gamma: max growth of norm_ratio_max per SWITCH_NORM_GROWTH_PER_STEPS
SWITCH_NORM_GROWTH_PER_STEPS = 20  # normalization of gamma across steps between two consecutive live readings
SWITCH_WINDOW = 4  # window L of local_rate_ratio for condition B (r_4), in readings
SWITCH_PHI = 0.6  # phi_s: threshold of r_4 (looser than the exit criterion, phi=0.05)
SWITCH_PERSISTENCE = 2  # P_s: consecutive live readings with A and B both true
INSTAB_DROP = 0.02  # delta: drop below the maximum of the free series
INSTAB_PERSISTENCE = 2  # consecutive live readings of drop before triggering REENTRY
INSTAB_GRAD_FACTOR = 2.0  # kappa: threshold on grad_kd_norm / reference ratio
INSTAB_GRAD_STEPS = 10  # window in STEPS over which grad_kd_norm_recent is already averaged by the caller
REENTRY_READINGS = 4  # R: locked reentry readings before retrying FREE
EVAL_EVERY_FREE = 10  # evaluation cadence (in steps) of the free/reentry phase
INSTAB_MAX_TRIGGERS = 2  # at the second instability trigger -> STOP (never a second REENTRY)
INSTAB_DROP_MIN_READINGS = 3  # drop is computable from k=2 (0-based) -- 3 readings in the segment

STOP_REASON_PLATEAU_FREE = "plateau_acc_free_rate"  # a genuine stop
STOP_REASON_UNSTABLE = "free_unstable"  # second instability trigger

STATE_LOCKED = "LOCKED"
STATE_FREE = "FREE"
STATE_REENTRY = "REENTRY"
STATE_STOP = "STOP"


class Decision(NamedTuple):
    next_state: str
    event: "str | None"
    reason: "str | None"


def _validate_params(
    switch_norm_growth: float, switch_norm_growth_per_steps: int, switch_window: int,
    switch_phi: float, switch_persistence: int, instab_drop: float, instab_persistence: int,
    instab_grad_factor: float, reentry_readings: int, eval_every_free: int,
) -> None:
    if not (switch_norm_growth > 0):
        raise ValueError(f"regime: switch_norm_growth={switch_norm_growth!r} -- must be > 0")
    if switch_norm_growth_per_steps < 1:
        raise ValueError(
            f"regime: switch_norm_growth_per_steps={switch_norm_growth_per_steps!r} -- must be >= 1"
        )
    if switch_window < 2:
        raise ValueError(f"regime: switch_window={switch_window!r} -- must be >= 2 (window of local_rate_ratio)")
    if not (switch_phi > 0):
        raise ValueError(f"regime: switch_phi={switch_phi!r} -- must be > 0")
    if switch_persistence < 1:
        raise ValueError(f"regime: switch_persistence={switch_persistence!r} -- must be >= 1")
    if instab_drop < 0:
        raise ValueError(f"regime: instab_drop={instab_drop!r} -- must be >= 0")
    if instab_persistence < 1:
        raise ValueError(f"regime: instab_persistence={instab_persistence!r} -- must be >= 1")
    if not (instab_grad_factor > 1):
        raise ValueError(f"regime: instab_grad_factor={instab_grad_factor!r} -- must be > 1")
    if reentry_readings < 1:
        raise ValueError(f"regime: reentry_readings={reentry_readings!r} -- must be >= 1")
    if eval_every_free < 1:
        raise ValueError(f"regime: eval_every_free={eval_every_free!r} -- must be >= 1")


@dataclasses.dataclass
class _Phase:
    state: str
    from_step: int
    to_step: "int | None" = None
    event: "str | None" = None
    reason: "str | None" = None


class RegimeArbiter:
    """State machine LOCKED -> FREE -> STOP, with REENTRY. `push` is the
    only entry point: one call per evaluation step, with the readings OF
    THAT STEP (unfiltered -- `None` is handled here with the same "live
    reading" predicate as the rest of the pipeline)."""

    def __init__(
        self, *,
        switch_norm_growth: float = SWITCH_NORM_GROWTH,
        switch_norm_growth_per_steps: int = SWITCH_NORM_GROWTH_PER_STEPS,
        switch_window: int = SWITCH_WINDOW,
        switch_phi: float = SWITCH_PHI,
        switch_persistence: int = SWITCH_PERSISTENCE,
        instab_drop: float = INSTAB_DROP,
        instab_persistence: int = INSTAB_PERSISTENCE,
        instab_grad_factor: float = INSTAB_GRAD_FACTOR,
        reentry_readings: int = REENTRY_READINGS,
        eval_every_free: int = EVAL_EVERY_FREE,
        exit_window: "int | None" = None,  # None -> LOCAL_RATE_WINDOW of the exit criterion (plateau.py)
        exit_phi: "float | None" = None,  # idem LOCAL_RATE_PHI
        exit_persistence: "int | None" = None,  # idem LOCAL_RATE_PERSISTENCE
    ) -> None:
        _validate_params(
            switch_norm_growth, switch_norm_growth_per_steps, switch_window, switch_phi,
            switch_persistence, instab_drop, instab_persistence, instab_grad_factor,
            reentry_readings, eval_every_free,
        )
        # Import here (not at module level, "no second formula"): defaults
        # for the exit criterion are LOCAL_RATE_WINDOW/PHI/PERSISTENCE from
        # plateau.py -- one single place declaring them, never recopied as
        # literals here.
        from engraft.replica.plateau import LOCAL_RATE_PERSISTENCE, LOCAL_RATE_PHI, LOCAL_RATE_WINDOW
        self._exit_window = exit_window if exit_window is not None else LOCAL_RATE_WINDOW
        self._exit_phi = exit_phi if exit_phi is not None else LOCAL_RATE_PHI
        self._exit_persistence = exit_persistence if exit_persistence is not None else LOCAL_RATE_PERSISTENCE

        self.switch_norm_growth = switch_norm_growth
        self.switch_norm_growth_per_steps = switch_norm_growth_per_steps
        self.switch_window = switch_window
        self.switch_phi = switch_phi
        self.switch_persistence = switch_persistence
        self.instab_drop = instab_drop
        self.instab_persistence = instab_persistence
        self.instab_grad_factor = instab_grad_factor
        self.reentry_readings = reentry_readings
        self.eval_every_free = eval_every_free

        self.state = STATE_LOCKED
        self.phases: "list[_Phase]" = [_Phase(state=STATE_LOCKED, from_step=0)]

        # Locked series (live readings only).
        self._locked_steps: "list[int]" = []
        self._locked_norm: "list[float]" = []
        self._locked_acc: "list[float]" = []
        self._switch_persist_count = 0

        # Public read attributes.
        self.switch_step: "int | None" = None
        self.switch_norm_growth_at: "float | None" = None
        self.switch_rate_ratio_at: "float | None" = None
        self.switch_acc_rbr: "float | None" = None
        self.switch_acc_free: "float | None" = None
        self.switch_grad_kd_ref: "float | None" = None
        self.n_reentries = 0  # REENTRY phases actually opened
        self.n_instability_triggers = 0  # instability triggers (drop or gradient), open or not
        # "non-live" readings ignored (same predicate as PlateauDetector/
        # LocalRateDetector) -- in LOCKED counts missing acc_rbr/norm_ratio_max,
        # in FREE/REENTRY counts missing acc_free (beyond those already counted
        # by the internal exit detector, which has its own separate n_skipped).
        self.n_skipped = 0

        # Free series.
        self._exit_detector: "LocalRateDetector | None" = None
        self._free_segment_readings = 0
        self._free_max: "float | None" = None
        self._drop_persist_count = 0

    # ------------------------------------------------------------------
    def _close_phase(self, step: int, event: "str | None", reason: "str | None") -> None:
        self.phases[-1].to_step = step
        self.phases[-1].event = event
        self.phases[-1].reason = reason

    def _open_phase(self, state: str, step: int) -> None:
        self.phases.append(_Phase(state=state, from_step=step))

    # ------------------------------------------------------------------
    def push(
        self, step: int, acc_rbr: "float | None", acc_free: "float | None",
        norm_ratio_max: "float | None", grad_kd_norm_recent: "float | None",
    ) -> Decision:
        if self.state == STATE_STOP:
            return Decision(STATE_STOP, None, None)
        if self.state == STATE_LOCKED:
            return self._push_locked(step, acc_rbr, acc_free, norm_ratio_max, grad_kd_norm_recent)
        return self._push_free_or_reentry(step, acc_free, grad_kd_norm_recent)

    # ------------------------------------------------------------------
    def _push_locked(
        self, step: int, acc_rbr: "float | None", acc_free: "float | None", norm_ratio_max: "float | None",
        grad_kd_norm_recent: "float | None",
    ) -> Decision:
        if acc_rbr is None or norm_ratio_max is None:
            self.n_skipped += 1  # not a live reading
            return Decision(STATE_LOCKED, None, None)

        self._locked_steps.append(int(step))
        self._locked_norm.append(float(norm_ratio_max))
        self._locked_acc.append(float(acc_rbr))
        k = len(self._locked_steps) - 1

        cond_a = False
        g = None
        if k >= 1:
            delta_steps = self._locked_steps[k] - self._locked_steps[k - 1]
            g = (
                (self._locked_norm[k] / self._locked_norm[k - 1] - 1.0)
                * self.switch_norm_growth_per_steps / delta_steps
            )
            cond_a = g < self.switch_norm_growth

        cond_b = False
        r4 = None
        if k >= self.switch_window - 1:
            window = self._locked_acc[k - self.switch_window + 1: k + 1]
            gain = self._locked_acc[k] - self._locked_acc[0]
            r4 = local_rate_ratio(window, gain)
            cond_b = (r4 is not None) and (r4 < self.switch_phi)

        if cond_a and cond_b:
            self._switch_persist_count += 1
        else:
            self._switch_persist_count = 0

        if self._switch_persist_count >= self.switch_persistence:
            return self._switch(step, g, r4, acc_rbr, acc_free, grad_kd_norm_recent)
        return Decision(STATE_LOCKED, None, None)

    def _switch(
        self, step: int, g: "float | None", r4: "float | None", acc_rbr: float, acc_free: "float | None",
        grad_kd_norm_recent: "float | None",
    ) -> Decision:
        # acc_free of THIS step (the two forward passes are computed in the
        # SAME reading, in every phase) is the k=0 reading of the free series
        # -- a_0^free is the exit detector's first reading, in the same call.
        self.switch_step = step
        self.switch_norm_growth_at = g
        self.switch_rate_ratio_at = r4
        self.switch_acc_rbr = acc_rbr
        self.switch_acc_free = acc_free
        self.switch_grad_kd_ref = grad_kd_norm_recent
        self._close_phase(step, "switch", "norm_growth_and_rate_ratio_below_threshold_persistent")
        self._open_phase(STATE_FREE, step)
        self.state = STATE_FREE
        self._exit_detector = LocalRateDetector(self._exit_window, self._exit_phi, self._exit_persistence)
        self._exit_detector.push(step, acc_free)
        self._free_max = acc_free
        self._free_segment_readings = 1 if acc_free is not None else 0
        return Decision(STATE_FREE, "switch", "norm_growth_and_rate_ratio_below_threshold_persistent")

    # ------------------------------------------------------------------
    def _push_free_or_reentry(
        self, step: int, acc_free: "float | None", grad_kd_norm_recent: "float | None",
    ) -> Decision:
        assert self._exit_detector is not None
        if acc_free is None:
            self.n_skipped += 1
        if self._exit_detector.push(step, acc_free):
            self._close_phase(step, "stop", STOP_REASON_PLATEAU_FREE)
            self.state = STATE_STOP
            return Decision(STATE_STOP, "stop", STOP_REASON_PLATEAU_FREE)

        if self.state == STATE_FREE:
            return self._push_free_instability(step, acc_free, grad_kd_norm_recent)
        return self._push_reentry(step, acc_free)

    def _push_free_instability(
        self, step: int, acc_free: "float | None", grad_kd_norm_recent: "float | None",
    ) -> Decision:
        if acc_free is not None:
            self._free_segment_readings += 1
            if self._free_max is None:
                self._free_max = acc_free
            # "Computable from k=2" -- requires at least a maximum established
            # over 2 readings BEFORE the current one (segment_readings also counts
            # the current one, so >=3 == k>=2, 0-based, k=0 is the transition reading).
            drop = (
                self._free_segment_readings >= INSTAB_DROP_MIN_READINGS
                and (self._free_max - acc_free) > self.instab_drop
            )
            self._free_max = max(self._free_max, acc_free)
            if drop:
                self._drop_persist_count += 1
            else:
                self._drop_persist_count = 0

        cond_grad = (
            self.switch_grad_kd_ref is not None and self.switch_grad_kd_ref > 0
            and grad_kd_norm_recent is not None
            and grad_kd_norm_recent > self.instab_grad_factor * self.switch_grad_kd_ref
        )
        cond_drop = self._drop_persist_count >= self.instab_persistence

        if not (cond_drop or cond_grad):
            return Decision(STATE_FREE, None, None)

        reason = "drop" if cond_drop else "gradient"
        # `n_instability_triggers` counts EVERY trigger (even the second,
        # which goes directly to STOP without opening a second REENTRY);
        # `n_reentries` counts only REENTRY phases actually opened -- the two
        # accountings diverge precisely at the second trigger.
        self.n_instability_triggers += 1
        if self.n_instability_triggers >= INSTAB_MAX_TRIGGERS:
            self._close_phase(step, "stop", STOP_REASON_UNSTABLE)
            self.state = STATE_STOP
            return Decision(STATE_STOP, "stop", STOP_REASON_UNSTABLE)

        self.n_reentries += 1
        self._close_phase(step, "reentry", reason)
        self._open_phase(STATE_REENTRY, step)
        self.state = STATE_REENTRY
        self._reentry_remaining = self.reentry_readings
        self._drop_persist_count = 0
        self._free_segment_readings = 0
        self._free_max = None
        return Decision(STATE_REENTRY, "reentry", reason)

    def _push_reentry(self, step: int, acc_free: "float | None") -> Decision:
        self._reentry_remaining -= 1
        if self._reentry_remaining > 0:
            return Decision(STATE_REENTRY, None, None)
        self._close_phase(step, "reentry_end", "reentry_readings_exhausted")
        self._open_phase(STATE_FREE, step)
        self.state = STATE_FREE
        # The reference maximum restarts from reentry to FREE -- reseeded
        # with the current reading (the first one again in FREE).
        self._free_max = acc_free
        self._free_segment_readings = 1 if acc_free is not None else 0
        self._drop_persist_count = 0
        return Decision(STATE_FREE, "reentry_end", "reentry_readings_exhausted")

    # ------------------------------------------------------------------
    @property
    def a0_free(self) -> "float | None":
        return getattr(self._exit_detector, "a0", None) if self._exit_detector is not None else None

    @property
    def exit_rate_ratio(self) -> "float | None":
        return getattr(self._exit_detector, "rate_ratio", None) if self._exit_detector is not None else None

    @property
    def exit_stopped_step(self) -> "int | None":
        return getattr(self._exit_detector, "stopped_step", None) if self._exit_detector is not None else None

    def phases_as_dicts(self) -> "list[dict]":
        """Manifest field `phases`: one dict per phase, without exposing
        `_Phase` (private) to callers -- the only place that serializes
        phases."""
        return [dataclasses.asdict(f) for f in self.phases]


def replay_regime(series: "list[dict]", **arbiter_kwargs) -> "list[Decision]":
    """Offline use: `series` is the ORDERED list of readings, each a dict
    with keys `step`, `acc_rbr`, `acc_free`, `norm_ratio_max`,
    `grad_kd_norm_recent` (missing keys treated as `None` -- same predicate
    as `push`). Constructs ONE `RegimeArbiter(**arbiter_kwargs)`, same class
    as inline use, returns the list of `Decision`, one per reading -- stops
    at the first `STOP` (subsequent readings ignored, as the real caller
    would do)."""
    arbiter = RegimeArbiter(**arbiter_kwargs)
    decisions: "list[Decision]" = []
    for reading in series:
        decision = arbiter.push(
            reading["step"], reading.get("acc_rbr"), reading.get("acc_free"),
            reading.get("norm_ratio_max"), reading.get("grad_kd_norm_recent"),
        )
        decisions.append(decision)
        if decision.next_state == STATE_STOP:
            break
    return decisions
