"""Compute backend: device, weight store, residency policy, per-phase
precision configuration.

`PhaseDtypes`: numeric precision is a run configuration parameter, per
phase, not an architecture choice. Four fields (`prefix`, `grad`, `probe`,
`confirm`); int8 is allowed **only** in `probe`: the forward pass used for a
line-search/triage decision, never in the backward pass, never for a stop
probability, never for the final verdict. A config file's `dtypes` and the
CLI can override the default (all F32 = the original, bit-identical
behavior).

`Backend` also carries `weight_store` (`"ram_dequant"` = the plain
`weights.GgufWeights`, `"device_iq4"` = `weights.DeviceIQ4`) and
`resident_policy` (`"all"` or `{"working_set", byte_cap}`)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

VALID_DTYPES = ("f32", "bf16", "int8")
PHASES = ("prefix", "grad", "probe", "confirm")

_TORCH_DTYPE_BY_NAME = None  # populated lazily (import torch only when needed)


def _torch_dtype(name: str):
    global _TORCH_DTYPE_BY_NAME
    if _TORCH_DTYPE_BY_NAME is None:
        import torch

        _TORCH_DTYPE_BY_NAME = {"f32": torch.float32, "bf16": torch.bfloat16}
        # "int8" has no direct floating-point torch.dtype: a consumer of
        # dtypes.probe=="int8" implements its own quantization -- only the
        # configuration name is exposed here.
    return _TORCH_DTYPE_BY_NAME.get(name)


@dataclasses.dataclass(frozen=True)
class PhaseDtypes:
    prefix: str = "f32"
    grad: str = "f32"
    probe: str = "f32"
    confirm: str = "f32"

    def __post_init__(self) -> None:
        for field in PHASES:
            value = getattr(self, field)
            if value not in VALID_DTYPES:
                raise ValueError(
                    f"dtypes.{field}={value!r} not recognized (valid: {VALID_DTYPES})"
                )
            if value == "int8" and field != "probe":
                raise ValueError(
                    f"dtypes.{field}=int8 not allowed: int8 is allowed only in "
                    "dtypes.probe (the triage forward pass, never in the backward "
                    "pass, never for a stop probability, never for the final verdict)"
                )

    def torch_dtype(self, field: str):
        value = getattr(self, field)
        td = _torch_dtype(value)
        if td is None:
            raise ValueError(
                f"dtypes.{field}={value!r} has no direct torch.dtype (int8 is "
                "handled by the caller, e.g. a probe forward with its own "
                "quantization)"
            )
        return td

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> "PhaseDtypes":
        if not data:
            return cls()
        return cls(**{k: data[k] for k in PHASES if k in data})


@dataclasses.dataclass(frozen=True)
class ResidentPolicy:
    mode: str = "all"  # "all" | "working_set"
    working_set_bytes: int | None = None  # required if mode == "working_set"
    dequant_cache_bytes: int = 0  # budget for the dequantized-expert cache (weights.DeviceIQ4)

    def __post_init__(self) -> None:
        if self.mode not in ("all", "working_set"):
            raise ValueError(f"resident_policy.mode={self.mode!r} not recognized")
        if self.mode == "working_set" and self.working_set_bytes is None:
            raise ValueError("resident_policy working_set requires working_set_bytes")


@dataclasses.dataclass(frozen=True)
class Backend:
    device: str = "cpu"  # "cpu" | "cuda:N"
    weight_store: str = "ram_dequant"  # "ram_dequant" | "device_iq4"
    resident_policy: ResidentPolicy = dataclasses.field(default_factory=ResidentPolicy)
    dtypes: PhaseDtypes = dataclasses.field(default_factory=PhaseDtypes)
    moe_grouped: bool = False  # layers.moe_ffn(grouped=...) -- default = legacy per-expert loop path

    @classmethod
    def cpu_f32(cls) -> "Backend":
        """`Replica`'s default: plain CPU, F32 everywhere, no residency tricks."""
        return cls(device="cpu", weight_store="ram_dequant", resident_policy=ResidentPolicy("all"), dtypes=PhaseDtypes())

    @classmethod
    def device(cls, device: str, weight_store: str = "device_iq4", resident_policy: ResidentPolicy | None = None,
               dtypes: PhaseDtypes | None = None) -> "Backend":
        return cls(
            device=device, weight_store=weight_store,
            resident_policy=resident_policy or ResidentPolicy("all"),
            dtypes=dtypes or PhaseDtypes(),
        )

    def to_json(self) -> dict:
        return {
            "device": self.device,
            "weight_store": self.weight_store,
            "resident_policy": dataclasses.asdict(self.resident_policy),
            "dtypes": self.dtypes.to_json(),
            "moe_grouped": self.moe_grouped,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Backend":
        """Exact inverse of `to_json`."""
        rp = data.get("resident_policy") or {}
        return cls(
            device=data.get("device", "cpu"),
            weight_store=data.get("weight_store", "ram_dequant"),
            resident_policy=ResidentPolicy(
                mode=rp.get("mode", "all"),
                working_set_bytes=rp.get("working_set_bytes"),
                dequant_cache_bytes=rp.get("dequant_cache_bytes", 0),
            ),
            dtypes=PhaseDtypes.from_json(data.get("dtypes")),
            moe_grouped=data.get("moe_grouped", False),
        )

    @classmethod
    def from_graft_config(cls, cfg: dict, device: str = "cpu", weight_store: str = "ram_dequant") -> "Backend":
        """Reads `dtypes` from a run configuration dict; absent -> default F32."""
        dtypes = PhaseDtypes.from_json(cfg.get("dtypes"))
        return cls(device=device, weight_store=weight_store, dtypes=dtypes)


def backend_from_args(device: str | None, dtypes_arg: str | None, store: str | None,
                       graft_config_path: str | Path | None = None,
                       resident: str | None = None, working_set_bytes: int | None = None,
                       moe_grouped: bool | None = None, dequant_cache_bytes: int | None = None) -> Backend:
    """Builds a `Backend` from CLI flags + an optional run-config JSON file (the
    CLI wins over the file when both are given). `dtypes_arg`: a
    "prefix,grad,probe,confirm" string (e.g. "f32,bf16,int8,f32"), positional
    per phase in `PHASES` order.

    `resident`/`working_set_bytes`: "working_set" requires a byte cap
    (`ValueError` if absent or <= 0); "all"/`None` -> default.

    The run-config file, if present, stores the whole backend under a
    `"backend"` key (`Backend.to_json()`), not under a top-level
    `"resident_policy"` key -- `resident_policy`, `moe_grouped` and
    `dequant_cache_bytes` are therefore read from `cfg.get("backend", {})`;
    reading the top-level `cfg["dtypes"]` stays unchanged (written there too,
    for compatibility). `moe_grouped` and `dequant_cache_bytes`: `None` ->
    fall back to the file, then `False`/`0` -- a non-`None` CLI value always
    wins."""
    cfg = {}
    if graft_config_path is not None and Path(graft_config_path).exists():
        cfg = json.loads(Path(graft_config_path).read_text())
    backend_cfg = cfg.get("backend") or {}
    cfg_resident = backend_cfg.get("resident_policy") or {}

    dtypes = PhaseDtypes.from_json(cfg.get("dtypes"))
    if dtypes_arg:
        parts = [p.strip() for p in dtypes_arg.split(",")]
        if len(parts) != len(PHASES):
            raise ValueError(f"--dtypes requires {len(PHASES)} values ({PHASES}), got {parts}")
        dtypes = PhaseDtypes(**dict(zip(PHASES, parts)))

    if resident is None:
        mode = cfg_resident.get("mode", "all")
        wsb = cfg_resident.get("working_set_bytes")
    elif resident == "working_set":
        if not working_set_bytes or working_set_bytes <= 0:
            raise ValueError(
                "resident='working_set' requires working_set_bytes > 0 "
                "(a declared fallback, never implicit)"
            )
        mode, wsb = "working_set", working_set_bytes
    elif resident == "all":
        mode, wsb = "all", None
    else:
        raise ValueError(f"resident={resident!r} not recognized (valid: 'all', 'working_set')")

    dcb = cfg_resident.get("dequant_cache_bytes", 0) if dequant_cache_bytes is None else dequant_cache_bytes
    resident_policy = ResidentPolicy(mode=mode, working_set_bytes=wsb, dequant_cache_bytes=dcb or 0)

    mg = backend_cfg.get("moe_grouped", False) if moe_grouped is None else moe_grouped

    return Backend(
        device=device or "cpu",
        weight_store=store or "ram_dequant",
        resident_policy=resident_policy,
        dtypes=dtypes,
        moe_grouped=mg,
    )
