"""Plateau detector using slope-vs-noise criterion and local rate criterion.
Shared between post-hoc analysis (detector1_slope_vs_noise) and online stopping
criterion (PlateauDetector, LocalRateDetector) — SINGLE IMPLEMENTATION: no module
can redefine this mathematics.

Content moved (not copied) from saturation analysis module: t_critical, _T_CRITICAL,
detector1_slope_vs_noise (same API, same behavior). PlateauDetector is the online
use case (one value at a time via push()) of the same statistics, with guard against
floor (G1: to avoid stopping on noise before the graft attaches, where floor guard
is not yet visible in post-hoc classification).

Local rate criterion: local_rate_ratio, detector_local_rate (post-hoc) and
LocalRateDetector (online, same interface as PlateauDetector). Compares the rate of
gain from the last L readings against the total gain from the first reading of the
series (a_0) — no noise window, no t-test: the denominator itself (a_k - a_0)
guards against floor and replaces the separate guard (no second guard for this
criterion).

No constants in units of metrics (same constraint as saturation_analysis.py):
_T_CRITICAL, FLOOR_GUARD_SIGMA, LOCAL_RATE_WINDOW, LOCAL_RATE_PHI,
LOCAL_RATE_PERSISTENCE are the ONLY numeric literals declared at module level
— _T_CRITICAL, LOCAL_RATE_WINDOW, LOCAL_RATE_PERSISTENCE are in readings
(dimensionless, like _T_CRITICAL), LOCAL_RATE_PHI is a dimensionless ratio
(like FLOOR_GUARD_SIGMA).
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# Student's t quantiles (two-tailed) for df = w-2 (w in {3,4,5,6,8} → df in
# {1,2,3,4,6}). Same table as in saturation_analysis (scipy not available).
# --------------------------------------------------------------------------

FLOOR_GUARD_SIGMA = 1  # Guard G1: in units of residual noise, not the metric itself

# Local rate criterion: L in readings, P in readings, phi dimensionless ratio.
# Frozen parameters (tuned on first iteration).
LOCAL_RATE_WINDOW = 4
LOCAL_RATE_PHI = 0.05
LOCAL_RATE_PERSISTENCE = 2

_T_CRITICAL: dict[tuple[int, float], float] = {
    (1, 0.90): 6.314, (1, 0.95): 12.706,
    (2, 0.90): 2.920, (2, 0.95): 4.303,
    (3, 0.90): 2.353, (3, 0.95): 3.182,
    (4, 0.90): 2.132, (4, 0.95): 2.776,
    (6, 0.90): 1.943, (6, 0.95): 2.447,
}


def t_critical(df: int, confidence: float) -> float:
    """Two-tailed Student's t quantile (tabulated, scipy not available). Raises
    ValueError for untabulated (df, confidence) — this module covers only df
    produced by w in {3,4,5,6,8} and confidence in {0.90, 0.95}, never silent
    extrapolation."""
    key = (df, confidence)
    if key not in _T_CRITICAL:
        raise ValueError(
            f"t_critical: (df={df}, confidence={confidence}) not tabulated -- this module covers "
            f"only {sorted(_T_CRITICAL)}"
        )
    return _T_CRITICAL[key]


def _ols_slope(window: "list[float]") -> float:
    """Slope of OLS regression on window (x-axis = 0..len(window)-1, uniform spacing
    — actual step spacing does not matter, only order). Extracted from _window_stat
    (single slope function, reused by local rate criterion without duplicating math)
    — same floating-point operation sequence as before extraction, so _window_stat
    remains bit-identical."""
    w = len(window)
    x = np.arange(w, dtype=float)
    x_mean = float(x.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    y = np.asarray(window, dtype=float)
    y_mean = float(y.mean())
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    return sxy / sxx


def _window_stat(window: "list[float]") -> "tuple[float, float]":
    """(t_stat, resid_std) of OLS regression on window (x-axis = 0..len(window)-1,
    uniform spacing — actual step spacing does not matter for the statistic, only
    order). resid_std = sqrt(sse/(w-2)). If se == 0 (perfectly linear window, no
    residual noise): t_stat = 0.0 if slope is also zero (perfectly flat line, a
    plateau by construction), else t_stat = inf (exact nonzero slope: no noise to
    judge it by, never a plateau) — same behavior as original detector."""
    w = len(window)
    df = w - 2
    x = np.arange(w, dtype=float)
    x_mean = float(x.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    y = np.asarray(window, dtype=float)
    y_mean = float(y.mean())
    slope = _ols_slope(window)
    pred = y_mean + slope * (x - x_mean)
    resid = y - pred
    sse = float(np.sum(resid ** 2))
    s2 = sse / df
    resid_std = math.sqrt(s2)
    se = math.sqrt(s2 / sxx)
    if se == 0.0:
        t_stat = 0.0 if slope == 0.0 else float("inf")
    else:
        t_stat = slope / se
    return float(t_stat), float(resid_std)


def detector1_slope_vs_noise(series: "list[float] | np.ndarray", w: int, confidence: float) -> "int | None":
    """First index i (0-based, in same indexing as series) where the slope of
    window series[i-w+1:i+1] is indistinguishable from zero at confidence level
    (two-tailed, df=w-2) — None if never happens. w=3 (df=1, degenerate test) is
    computable but should be excluded by caller. No floor guard here (that is for
    PlateauDetector, online use) — this detector serves post-hoc classification
    where floor is already visible in results."""
    if w < 3:
        raise ValueError(f"detector1_slope_vs_noise: w={w} < 3 (need at least df=1)")
    n = len(series)
    if n < w:
        return None
    df = w - 2
    tcrit = t_critical(df, confidence)
    for i in range(w - 1, n):
        window = list(series[i - w + 1:i + 1])
        t_stat, _resid_std = _window_stat(window)
        if abs(t_stat) < tcrit:
            return i
    return None


def floor_guard(window: "list[float]", first_value: float, resid_std: float) -> bool:
    """Guard G1: True if the candidate plateau is NOT a floor — the window mean
    exceeds first_value (first reading of the series, not the window) by more than
    FLOOR_GUARD_SIGMA * resid_std. If resid_std is 0 (perfectly constant or linear
    window), simply requires mean > first_value (no noise to multiply)."""
    mean_window = float(np.mean(window))
    if resid_std == 0.0:
        return mean_window > first_value
    return mean_window > first_value + FLOOR_GUARD_SIGMA * resid_std


class PlateauDetector:
    """Online use of detector1 (slope vs noise): push(step, value) one reading at
    a time — ignores value=None (same predicate as per_step_series, which drops
    records without the field), applies floor guard G1 before declaring plateau.
    State: n_skipped (None readings ignored), t_stat (of last live window evaluated,
    None if not enough readings yet), stopped_step (the step passed to push() when
    plateau was declared, None otherwise)."""

    def __init__(self, w: int, confidence: float):
        if w < 3:
            raise ValueError(f"PlateauDetector: w={w} < 3 (need at least df=1)")
        t_critical(w - 2, confidence)  # validate early (ValueError if not tabulated)
        self.w = w
        self.confidence = confidence
        self.steps: list[int] = []
        self.values: list[float] = []
        self.n_skipped = 0
        self.t_stat: "float | None" = None
        self.stopped_step: "int | None" = None

    def push(self, step: int, value: "float | None") -> bool:
        """Returns True the first time plateau fires (with floor guard G1 included)
        — after that, caller stops querying (stopped_step stays fixed at first stop)."""
        if value is None:
            self.n_skipped += 1
            return False
        self.steps.append(int(step))
        self.values.append(float(value))
        n = len(self.values)
        if n < self.w:
            return False
        window = self.values[-self.w:]
        t_stat, resid_std = _window_stat(window)
        self.t_stat = t_stat
        tcrit = t_critical(self.w - 2, self.confidence)
        if abs(t_stat) >= tcrit:
            return False
        first_value = self.values[0]  # first reading of the series, not the window
        if not floor_guard(window, first_value, resid_std):
            return False
        self.stopped_step = int(step)
        return True


# --------------------------------------------------------------------------
# Local rate criterion: local_rate_ratio, detector_local_rate (post-hoc),
# LocalRateDetector (online).
# --------------------------------------------------------------------------


def _validate_local_rate_params(L: int, phi: float, persistence: int) -> None:
    """Single validation function — used by LocalRateDetector constructor and
    detector_local_rate: they must not diverge."""
    if not isinstance(L, int) or L < 2:
        raise ValueError(f"local rate criterion: L={L!r} — must be integer >= 2")
    if not isinstance(phi, (int, float)) or isinstance(phi, bool) or not math.isfinite(phi) or phi <= 0:
        raise ValueError(f"local rate criterion: phi={phi!r} — must be finite real > 0")
    if not isinstance(persistence, int) or persistence < 1:
        raise ValueError(f"local rate criterion: persistence={persistence!r} — must be integer >= 1")


def local_rate_ratio(window: "list[float]", gain: float) -> "float | None":
    """r = len(window) * slope(window) / gain — None if gain <= 0 (floor guard:
    denominator itself prevents stopping before graft has gained something)."""
    if gain <= 0:
        return None
    return len(window) * _ols_slope(window) / gain


def detector_local_rate(series: "list[float]", L: int, phi: float, persistence: int) -> "int | None":
    """Post-hoc use: series is list of live readings only (no None — caller filters;
    raises ValueError if found). Returns first index i (0-based, in same indexing as
    series) where r_i < phi holds for persistence consecutive live readings — None
    if never happens. A reading with gain <= 0 (floor guard) does not count and
    resets persistence counter — same rule, same function local_rate_ratio, as
    LocalRateDetector.push: they must not diverge."""
    _validate_local_rate_params(L, phi, persistence)
    if any(v is None for v in series):
        raise ValueError("detector_local_rate: series contains None — caller must filter them")
    n = len(series)
    if n < L:
        return None
    a0 = series[0]
    consec = 0
    for i in range(L - 1, n):
        window = series[i - L + 1:i + 1]
        gain = series[i] - a0
        r = local_rate_ratio(window, gain)
        if r is None:
            consec = 0
            continue
        if r < phi:
            consec += 1
            if consec >= persistence:
                return i
        else:
            consec = 0
    return None


class LocalRateDetector:
    """Online use of local rate criterion: push(step, value) one reading at a time
    — ignores value=None (same predicate as PlateauDetector, B4), applies floor
    guard (gain <= 0) instead of G1 (no second guard for this criterion). State:
    n_skipped (None readings ignored), rate_ratio (r of last live reading evaluated,
    None until computable), a0 (first live reading of series), stopped_step (the
    step passed to push() when stop was declared, None otherwise)."""

    def __init__(self, L: int, phi: float, persistence: int):
        _validate_local_rate_params(L, phi, persistence)
        self.L = L
        self.phi = phi
        self.persistence = persistence
        self.steps: list[int] = []
        self.values: list[float] = []
        self.n_skipped = 0
        self.rate_ratio: "float | None" = None
        self.a0: "float | None" = None
        self.stopped_step: "int | None" = None
        self._consec = 0

    def push(self, step: int, value: "float | None") -> bool:
        """Returns True the first time stop fires — after that, caller stops querying
        (stopped_step stays fixed at first stop)."""
        if value is None:
            self.n_skipped += 1
            return False
        self.steps.append(int(step))
        self.values.append(float(value))
        if self.a0 is None:
            self.a0 = self.values[0]  # first live reading of series
        n = len(self.values)
        if n < self.L:
            return False
        window = self.values[-self.L:]
        gain = self.values[-1] - self.a0
        r = local_rate_ratio(window, gain)
        self.rate_ratio = r
        if r is None:
            self._consec = 0
            return False
        if r < self.phi:
            self._consec += 1
        else:
            self._consec = 0
        if self._consec >= self.persistence:
            self.stopped_step = int(step)
            return True
        return False
