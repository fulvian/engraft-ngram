"""Pure torch dequantization.

One type per function, all element-wise operations (shift, and, gather):
no matmul, no TF32. Contract: in F32, byte-identical to
`gguf.quants.dequantize` on CPU; `dequant_*_bf16 == dequant_*_f32.to(bfloat16)`.

Input convention: `data` is a uint8 tensor of shape `[..., type_size]` (the
final axis carries the raw bytes of ONE block, as returned by `gguf` -- same
convention as dequantization at the call site, which already places raw data on
`[n_expert, ..., type_size]` before dequantizing). The output has shape
`[..., block_size]` in the requested dtype (default F32; rounding to a narrower
dtype happens *after* arithmetic in F32, never during).

Types covered: those found by format detection on the three shards and table
(F32, BF16, Q8_0, Q6_K, IQ4_NL, IQ4_XS, IQ3_S) -- closed list, not all ggml types.

References: `gguf/quants.py` for classes `BF16`, `Q8_0`, `Q6_K`, `IQ4_NL`,
`IQ4_XS`, `IQ3_S`; `ggml/src/ggml-common.h` for `KVALUES_IQ4NL` and the
IQ3_S grid (here decoded from `gguf.quants.IQ3_S.grid_hex`/`grid_map`,
constant format data -- not computation logic -- decoded here in torch,
not delegated to gguf)."""
from __future__ import annotations

import math

import torch

QK_K = 256

KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

TYPE_SIZES = {
    "F32": (1, 4),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q6_K": (256, 210),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "IQ3_S": (256, 110),
}


# --------------------------------------------------------------------------
# Module-level constants cache. Each call to `dequant_*` was recreating the
# same constants (shifts, kvalues, IQ3_S grid) on each invocation:
# `torch.tensor(..., device=device)` means small kernels and device transfers
# per call. Cache key is (str(device), name) -- `name` includes the dtype
# (e.g., "kvalues_i64", not "kvalues"): different dtypes for the same logical
# constant are distinct cache entries, never collide. No numeric change:
# same exact value, built once.
# --------------------------------------------------------------------------

_CONSTS: dict[tuple[str, str], torch.Tensor] = {}


def _const(device, name: str, constructor) -> torch.Tensor:
    key = (str(device), name)
    cached = _CONSTS.get(key)
    if cached is None:
        cached = constructor()
        _CONSTS[key] = cached
    return cached


def _shift_and(x: torch.Tensor, shifts: torch.Tensor, mask: int) -> torch.Tensor:
    """x: [...,1] uint8/uint16 broadcast against shifts: [...] -> [...,len(shifts)]."""
    shifted = x >> shifts
    return shifted & mask


def dequant_f32(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 4 and data.dtype == torch.uint8
    out = data.contiguous().view(torch.float32)
    return out.to(out_dtype)


def dequant_bf16(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of ggml_compute_fp32_to_bf16: bf16 -> f32 is <<16 on bits."""
    assert data.shape[-1] == 2 and data.dtype == torch.uint8
    bits16 = data.contiguous().view(torch.int16).to(torch.int32)
    bits32 = bits16 << 16
    out = bits32.view(torch.float32)
    return out.to(out_dtype)


def dequant_q8_0(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 34 and data.dtype == torch.uint8
    d_bytes, x_bytes = data[..., :2], data[..., 2:]
    d = d_bytes.contiguous().view(torch.float16).to(torch.float32)  # [...,1]
    x = x_bytes.contiguous().view(torch.int8).to(torch.float32)  # [...,32]
    out = x * d
    return out.to(out_dtype)


def dequant_q6_k(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 210 and data.dtype == torch.uint8
    lead = data.shape[:-1]
    n = math.prod(lead)
    b = data.reshape(n, 210)

    ql = b[:, : QK_K // 2]  # 128
    qh = b[:, QK_K // 2 : QK_K // 2 + QK_K // 4]  # 64
    scales_raw = b[:, QK_K // 2 + QK_K // 4 : QK_K // 2 + QK_K // 4 + QK_K // 16]  # 16
    d_bytes = b[:, QK_K // 2 + QK_K // 4 + QK_K // 16 :]  # 2

    device = data.device
    scales = scales_raw.view(torch.int8).to(torch.float32)  # [n,16]
    d = d_bytes.contiguous().view(torch.float16).to(torch.float32)  # [n,1]
    dl = (d * scales).reshape(n, QK_K // 16, 1)  # [n,16,1]

    shifts2 = _const(device, "shifts2_u8", lambda: torch.tensor([0, 4], dtype=torch.uint8, device=device))
    ql4 = ql.reshape(n, -1, 1, 64) >> shifts2.reshape(1, 1, 2, 1)
    ql4 = (ql4 & 0x0F).reshape(n, -1, 32)  # [n,4,32] -> reshape below

    shifts4 = _const(device, "shifts4_0246_u8", lambda: torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=device))
    qh4 = qh.reshape(n, -1, 1, 32) >> shifts4.reshape(1, 1, 4, 1)
    qh4 = (qh4 & 0x03).reshape(n, -1, 32)  # [n,4,32]

    q = (ql4.to(torch.int32) | (qh4.to(torch.int32) << 4)) - 32  # [n,4,32] int32, range matches int8
    q = q.reshape(n, QK_K // 16, -1).to(torch.float32)  # [n,16,16]

    out = (dl * q).reshape(n, QK_K)
    out = out.reshape(*lead, QK_K)
    return out.to(out_dtype)


def dequant_iq4_nl(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 18 and data.dtype == torch.uint8
    lead = data.shape[:-1]
    n = math.prod(lead)
    b = data.reshape(n, 18)
    d_bytes, qs_bytes = b[:, :2], b[:, 2:]  # qs: 16 bytes -> 32 nibbles
    device = data.device

    d = d_bytes.contiguous().view(torch.float16).to(torch.float32)  # [n,1]

    shifts2 = _const(device, "shifts2_u8", lambda: torch.tensor([0, 4], dtype=torch.uint8, device=device))
    nib = qs_bytes.contiguous().reshape(n, -1, 1, 16) >> shifts2.reshape(1, 1, 2, 1)  # [n,1,2,16]
    nib = (nib & 0x0F).reshape(n, -1).to(torch.int64)  # [n,32]

    kvalues = _const(device, "kvalues_iq4nl_i64",
                      lambda: torch.tensor(KVALUES_IQ4NL, dtype=torch.int64, device=device))  # [16]
    vals = kvalues[nib].to(torch.float32)  # [n,32] -- fancy indexing, same order as gguf

    out = d * vals  # [n,32]
    out = out.reshape(*lead, 32)
    return out.to(out_dtype)


def dequant_iq4_xs(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 136 and data.dtype == torch.uint8
    lead = data.shape[:-1]
    n = math.prod(lead)
    b = data.reshape(n, 136)
    device = data.device

    d_bytes = b[:, :2]
    scales_h_bytes = b[:, 2:4]
    scales_l_bytes = b[:, 4 : 4 + QK_K // 64]  # 4
    qs_bytes = b[:, 4 + QK_K // 64 :]  # 128

    d = d_bytes.contiguous().view(torch.float16).to(torch.float32)  # [n,1]
    # scales_h as a 16-bit little-endian word, reassembled manually (never `.view` on
    # a signed type: would avoid ambiguity of bit 15 in the shift below).
    scales_h = scales_h_bytes[:, 0].to(torch.int64) | (scales_h_bytes[:, 1].to(torch.int64) << 8)
    scales_h = scales_h.reshape(n, 1)  # [n,1]

    shifts2 = _const(device, "shifts2_u8", lambda: torch.tensor([0, 4], dtype=torch.uint8, device=device))
    scales_l = (scales_l_bytes.reshape(n, -1, 1) >> shifts2.reshape(1, 1, 2)).reshape(n, -1)  # [n,8]
    scales_l = (scales_l.to(torch.int64) & 0x0F)

    n_sub = QK_K // 32  # 8
    sh_shifts = _const(device, "sh_shifts_iq4xs_i64",
                        lambda: torch.tensor([2 * i for i in range(n_sub)], dtype=torch.int64, device=device))
    scales_h_sub = (scales_h.reshape(n, 1) >> sh_shifts.reshape(1, -1)) & 0x03  # [n,8]

    scales = (scales_l | (scales_h_sub << 4)).to(torch.int32) - 32  # [n,8]
    dl = (d * scales.to(torch.float32)).reshape(n, n_sub, 1)  # [n,8,1]

    nib_hi_lo = qs_bytes.contiguous().reshape(n, -1, 1, 16) >> shifts2.reshape(1, 1, 2, 1)  # [n,8,2,16]
    nib = (nib_hi_lo.reshape(n, n_sub, 32) & 0x0F).to(torch.int64)  # [n,8,32]

    kvalues = _const(device, "kvalues_iq4nl_i64",
                      lambda: torch.tensor(KVALUES_IQ4NL, dtype=torch.int64, device=device))  # [16]
    vals = kvalues[nib].to(torch.float32)  # [n,8,32] -- fancy indexing

    out = (dl * vals).reshape(n, QK_K)
    out = out.reshape(*lead, QK_K)
    return out.to(out_dtype)


def _iq3s_grid(device) -> torch.Tensor:
    """IQ3_S grid 512x4, decoded from the format's hexadecimal bytes (constant
    format data, not dequantization logic: same principle as the `KVALUES_IQ4NL`
    already hardcoded). Module-level cache by device string; the module can run
    on multiple devices in the same session."""

    def _build() -> torch.Tensor:
        from gguf.quants import IQ3_S as _GGUF_IQ3S  # constant format data, not computation

        grid_hex = _GGUF_IQ3S.grid_hex
        grid_map = _GGUF_IQ3S.grid_map  # (1,3,5,7,9,11,13,15)

        raw = torch.frombuffer(bytearray(grid_hex), dtype=torch.uint8).clone()  # ASCII hex chars
        raw = raw.reshape(-1, 2)
        hi = torch.where(raw[:, 0] > 0x40, raw[:, 0] + 9, raw[:, 0]) & 0x0F
        lo = torch.where(raw[:, 1] > 0x40, raw[:, 1] + 9, raw[:, 1]) & 0x0F
        byte_val = ((hi << 4) | lo).to(torch.int64)  # [n_bytes_decoded]

        bits_per_elem = 3  # ceil(log2(8))
        elems_per_byte = 8 // bits_per_elem  # 2
        shifts = torch.tensor([i * (8 // elems_per_byte) for i in range(elems_per_byte)], dtype=torch.int64)
        vals = (byte_val.reshape(-1, 1) >> shifts.reshape(1, -1)) & ((1 << bits_per_elem) - 1)
        vals = vals.reshape(-1)  # index into grid_map

        grid_map_t = torch.tensor(grid_map, dtype=torch.float32)
        grid = grid_map_t[vals].reshape(512, 4)  # flat: indexable directly from qs
        return grid.to(device)

    return _const(device, "iq3s_grid_f32", _build)


def dequant_iq3_s(data: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert data.shape[-1] == 110 and data.dtype == torch.uint8
    lead = data.shape[:-1]
    n = math.prod(lead)
    b = data.reshape(n, 110)
    device = data.device

    d_bytes = b[:, :2]
    qs_bytes = b[:, 2 : 2 + QK_K // 4]  # 64
    qh_bytes = b[:, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]  # 8
    signs_bytes = b[:, 2 + QK_K // 4 + QK_K // 32 : 2 + QK_K // 4 + QK_K // 32 + QK_K // 8]  # 32
    scales_bytes = b[:, 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 :]  # 4

    d = d_bytes.contiguous().view(torch.float16).to(torch.float32)  # [n,1]

    shifts4 = _const(device, "shifts2_u8", lambda: torch.tensor([0, 4], dtype=torch.uint8, device=device))
    scales = (scales_bytes.reshape(n, -1, 1) >> shifts4.reshape(1, 1, 2)) & 0x0F  # [n,4,2]
    scales = scales.reshape(n, -1).to(torch.float32)  # [n,8]
    db = d * (1 + 2 * scales)  # [n,8]
    db = db.reshape(n, -1, 1, 1)  # [n,8,1,1]

    shifts8 = _const(device, "shifts8_range_u8", lambda: torch.tensor(list(range(8)), dtype=torch.uint8, device=device))
    signs = (signs_bytes.reshape(n, -1, 1) >> shifts8.reshape(1, 1, 8)) & 0x01  # [n,32,8]
    one = _const(device, "one_f32", lambda: torch.tensor(1.0, device=device))
    neg_one = _const(device, "neg_one_f32", lambda: torch.tensor(-1.0, device=device))
    signs = torch.where(signs == 0, one, neg_one)
    signs = signs.reshape(n, -1, 4, 8)  # [n,8,4,8]

    qh_bits = (qh_bytes.reshape(n, -1, 1) >> shifts8.reshape(1, 1, 8)) & 0x01  # [n,8,8]
    qh_bits = qh_bits.to(torch.int64).reshape(n, -1)  # [n,64]
    qs = qs_bytes.to(torch.int64) | (qh_bits << 8)  # [n,64]

    grid = _iq3s_grid(device)  # [512,4]
    grid_sel = grid[qs]  # [n,64,4] -- direct indexing, same order as numpy gather
    grid_sel = grid_sel.reshape(n, -1, 4, 8)  # [n,8,4,8]

    out = (db * grid_sel * signs).reshape(n, -1)
    out = out.reshape(*lead, QK_K)
    return out.to(out_dtype)


DEQUANT_FNS = {
    "F32": dequant_f32,
    "BF16": dequant_bf16,
    "Q8_0": dequant_q8_0,
    "Q6_K": dequant_q6_k,
    "IQ4_NL": dequant_iq4_nl,
    "IQ4_XS": dequant_iq4_xs,
    "IQ3_S": dequant_iq3_s,
}
