"""Fused decode+matmul Triton kernel for quantized experts:
`y[T_e, n_out] = x[T_e, n_in] @ W[n_out, n_in]^T`, with `W` read directly from
the raw GGUF bytes (no intermediate torch dequantization, no materialized `W`).

Motivation: a plain torch dequantize-then-matmul is dominated by the
dequantization itself, and it does not shrink with sequence length -- a
sequence forward touches nearly every expert of every layer. Fusing decode
and matmul into one kernel (decoding each 32-column tile in registers, never
in global memory) removes that cost.

Byte layout per expert: `byte_tensor` has shape `[n_out, bytes_per_row]`, one
row per output row of `W`, contiguous.

Each quant format is decoded 32 columns of `n_in` at a time (32 is the common
minimum block granularity across the four formats used here: a full block for
Q8_0/IQ4_NL, a sub-block of a 256-wide block for IQ4_XS/IQ3_S). The same
device-side decode function serves both forward (tile over `n_out`, reduce
over `n_in`) and backward (tile over `n_in`, reduce over `n_out`) -- `W` is
read the same way in both directions, only which axis is the fixed tile and
which is the reduction axis changes.

Geometry assumption (true for the experts this repository targets: Q8_0/
IQ4_NL only appear on `ffn_down_exps` with `n_in=640`, IQ4_XS/IQ3_S only on
`ffn_gate_exps`/`ffn_up_exps` with `n_in=2560`): `n_in` is a multiple of 32
(Q8_0/IQ4_NL) or 256 (IQ4_XS/IQ3_S). Violating this is a caller error, not
handled here (no partial-block remainder).

This is the one Triton kernel published alongside the reference descent (see
the repository's publication notes): a single per-expert matmul, no grouped/
layer-wide kernel, no `torch.compile` fusion.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

QK_K = 256
KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

TYPE_SIZE = {"Q8_0": 34, "IQ4_NL": 18, "IQ4_XS": 136, "IQ3_S": 110}
BLOCK_SIZE = {"Q8_0": 32, "IQ4_NL": 32, "IQ4_XS": 256, "IQ3_S": 256}

_TABLE_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _kvalues_table(device) -> torch.Tensor:
    key = (str(device), "kvalues_iq4nl_f32")
    t = _TABLE_CACHE.get(key)
    if t is None:
        t = torch.tensor(KVALUES_IQ4NL, dtype=torch.float32, device=device)
        _TABLE_CACHE[key] = t
    return t


def iq3s_grid_flat(device) -> torch.Tensor:
    """Flattened IQ3_S grid `[512*4]` (row r -> `grid[4r:4r+4]`), same format
    data as `engraft.replica.weights` dequantization, flattened here for
    linear indexing (`qs_idx*4+m`) inside the kernel."""
    key = (str(device), "iq3s_grid_flat_f32")
    t = _TABLE_CACHE.get(key)
    if t is None:
        from engraft.replica.dequant import _iq3s_grid  # noqa: E402 -- format data, not computation

        grid = _iq3s_grid(device)  # [512,4]
        t = grid.reshape(-1).contiguous()
        _TABLE_CACHE[key] = t
    return t


# ==========================================================================
# Device decode functions: a [ROWS,32] tile of W for one 32-column block of
# n_in, from raw bytes. `rows`: int64 tensor [ROWS] (absolute row indices in
# [0,n_out)). `k`: starting column of the block (multiple of 32, runtime
# scalar). None of the four touch memory outside their own block.
# ==========================================================================


@triton.jit
def _decode_q8_0_tile32(byte_ptr, rows, bytes_per_row, k):
    row_base = rows.to(tl.int64) * bytes_per_row.to(tl.int64)  # [ROWS]
    kb = k // 32
    off_d = row_base + kb.to(tl.int64) * 34
    b0 = tl.load(byte_ptr + off_d).to(tl.int32)
    b1 = tl.load(byte_ptr + off_d + 1).to(tl.int32)
    bits16 = ((b0 | (b1 << 8)) & 0xFFFF).to(tl.int16)
    d = bits16.to(tl.float16, bitcast=True).to(tl.float32)  # [ROWS]

    cols = tl.arange(0, 32)
    off_qs = (off_d + 2)[:, None] + cols[None, :]  # [ROWS,32]
    qb = tl.load(byte_ptr + off_qs)
    qi8 = qb.to(tl.int8, bitcast=True).to(tl.float32)
    return d[:, None] * qi8


@triton.jit
def _decode_iq4_nl_tile32(byte_ptr, rows, bytes_per_row, k, tab_ptr):
    row_base = rows.to(tl.int64) * bytes_per_row.to(tl.int64)
    kb = k // 32
    off_d = row_base + kb.to(tl.int64) * 18
    b0 = tl.load(byte_ptr + off_d).to(tl.int32)
    b1 = tl.load(byte_ptr + off_d + 1).to(tl.int32)
    bits16 = ((b0 | (b1 << 8)) & 0xFFFF).to(tl.int16)
    d = bits16.to(tl.float16, bitcast=True).to(tl.float32)  # [ROWS]

    # Nibble order: [low(b0..b15), high(b0..b15)] -- NOT interleaved per byte
    # (the low-nibble block over 16 bytes comes before the high-nibble block,
    # same layout as IQ4_XS below).
    cols = tl.arange(0, 32)
    lo = cols < 16
    byte_idx = tl.where(lo, cols, cols - 16)
    shift = tl.where(lo, 0, 4)
    off_qs = (off_d + 2)[:, None] + byte_idx[None, :]  # [ROWS,32]
    qb = tl.load(byte_ptr + off_qs).to(tl.int32)
    nib = (qb >> shift[None, :]) & 0xF
    val = tl.load(tab_ptr + nib)
    return d[:, None] * val


@triton.jit
def _decode_iq4_xs_tile32(byte_ptr, rows, bytes_per_row, k, tab_ptr):
    row_base = rows.to(tl.int64) * bytes_per_row.to(tl.int64)
    blk = k // 256
    s = (k % 256) // 32
    off_blk = row_base + blk.to(tl.int64) * 136

    off_d = off_blk
    b0 = tl.load(byte_ptr + off_d).to(tl.int32)
    b1 = tl.load(byte_ptr + off_d + 1).to(tl.int32)
    d_bits = ((b0 | (b1 << 8)) & 0xFFFF).to(tl.int16)
    d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)  # [ROWS]

    hb0 = tl.load(byte_ptr + off_blk + 2).to(tl.int32)
    hb1 = tl.load(byte_ptr + off_blk + 3).to(tl.int32)
    scales_h = hb0 | (hb1 << 8)  # [ROWS], 16 bit

    sl_byte_idx = s // 2
    sl_shift = (s % 2) * 4
    sl_byte = tl.load(byte_ptr + off_blk + 4 + sl_byte_idx).to(tl.int32)
    scale_l = (sl_byte >> sl_shift) & 0xF
    scale_h = (scales_h >> (s * 2)) & 0x3
    scale = ((scale_l | (scale_h << 4)) - 32).to(tl.float32)  # [ROWS]
    dl = d * scale

    cols = tl.arange(0, 32)
    lo = cols < 16
    byte_idx = tl.where(lo, cols, cols - 16)
    shift = tl.where(lo, 0, 4)
    qs_off = off_blk + 8 + s.to(tl.int64) * 16
    off_qs = qs_off[:, None] + byte_idx[None, :]  # [ROWS,32]
    qb = tl.load(byte_ptr + off_qs).to(tl.int32)
    nib = (qb >> shift[None, :]) & 0xF
    val = tl.load(tab_ptr + nib)
    return dl[:, None] * val


@triton.jit
def _decode_iq3_s_tile32(byte_ptr, rows, bytes_per_row, k, grid_ptr):
    row_base = rows.to(tl.int64) * bytes_per_row.to(tl.int64)
    blk = k // 256
    s = (k % 256) // 32
    off_blk = row_base + blk.to(tl.int64) * 110

    off_d = off_blk
    b0 = tl.load(byte_ptr + off_d).to(tl.int32)
    b1 = tl.load(byte_ptr + off_d + 1).to(tl.int32)
    d_bits = ((b0 | (b1 << 8)) & 0xFFFF).to(tl.int16)
    d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)  # [ROWS]

    sc_byte_idx = s // 2
    sc_shift = (s % 2) * 4
    sc_byte = tl.load(byte_ptr + off_blk + 106 + sc_byte_idx).to(tl.int32)
    scale_nib = (sc_byte >> sc_shift) & 0xF
    db = d * (1.0 + 2.0 * scale_nib.to(tl.float32))  # [ROWS]

    cols = tl.arange(0, 32)  # local index j inside the sub-block
    q = cols // 8
    i = cols % 8
    g = 8 * s + 2 * q + i // 4  # [32], global qs/qh index in the block (0..63)
    m = i % 4  # [32]
    sign_byte_idx = 4 * s + q  # [32]
    sign_bit = i  # [32]

    qs_off = (off_blk + 2)[:, None] + g[None, :]  # [ROWS,32]
    qs_byte = tl.load(byte_ptr + qs_off).to(tl.int32)
    qh_off = (off_blk + 66)[:, None] + (g[None, :] // 8)
    qh_byte = tl.load(byte_ptr + qh_off).to(tl.int32)
    qh_bit = (qh_byte >> (g[None, :] % 8)) & 1
    qs_idx = qs_byte | (qh_bit << 8)  # [ROWS,32], 0..511

    grid_val = tl.load(grid_ptr + qs_idx * 4 + m[None, :])  # [ROWS,32]

    signs_off = (off_blk + 74)[:, None] + sign_byte_idx[None, :]
    sign_byte = tl.load(byte_ptr + signs_off).to(tl.int32)
    sign_bit_val = (sign_byte >> sign_bit[None, :]) & 1
    sign = tl.where(sign_bit_val == 0, 1.0, -1.0)

    return db[:, None] * grid_val * sign


@triton.jit
def _dot_wdot(a_bf16, w_f32_t, WDOT: tl.constexpr):
    """`a_bf16 @ w_f32_t` (already in the right orientation), with the
    precision of the W side chosen by `WDOT`: `bf16` rounds the decoded W
    tile to bf16 before the dot (fastest, ~1.5e-3 relative error measured on
    Q8_0); `f32` is exact but 6-9x slower; `split` (default) sums two bf16
    dots over the high part and the residual of `w_f32_t`, keeping `x`
    strictly bf16 (no range constraint) while recovering near-F32 precision
    at about 2x the cost of one bf16 dot."""
    if WDOT == "bf16":
        return tl.dot(a_bf16, w_f32_t.to(tl.bfloat16), out_dtype=tl.float32)
    if WDOT == "f32":
        return tl.dot(a_bf16.to(tl.float32), w_f32_t, out_dtype=tl.float32)
    # "split": w = w_hi + w_lo, both bf16-representable (w_hi truncates,
    # w_lo is the residual -- summing the two dots reconstructs ~F32
    # precision).
    w_hi = w_f32_t.to(tl.bfloat16)
    w_lo = (w_f32_t - w_hi.to(tl.float32)).to(tl.bfloat16)
    return (tl.dot(a_bf16, w_hi, out_dtype=tl.float32)
            + tl.dot(a_bf16, w_lo, out_dtype=tl.float32))


@triton.jit
def _dot_wdot_bwd(a_f32, w_f32, WDOT: tl.constexpr):
    """Like `_dot_wdot`, but for backward: unlike `x` in forward, `a` (the
    `grad_y` tile) is not already bf16 by construction -- it comes from the
    F32 islands of the reference engine (loss, norms, ...), so forcing it to
    bf16 unconditionally loses real precision. With `WDOT="split"` both
    operands are split (3 dots; the lo*lo cross term is dropped as
    negligible, order 2^-16 of the value)."""
    if WDOT == "bf16":
        return tl.dot(a_f32.to(tl.bfloat16), w_f32.to(tl.bfloat16), out_dtype=tl.float32)
    if WDOT == "f32":
        return tl.dot(a_f32, w_f32, out_dtype=tl.float32)
    a_hi = a_f32.to(tl.bfloat16)
    a_lo = (a_f32 - a_hi.to(tl.float32)).to(tl.bfloat16)
    w_hi = w_f32.to(tl.bfloat16)
    w_lo = (w_f32 - w_hi.to(tl.float32)).to(tl.bfloat16)
    return (tl.dot(a_hi, w_hi, out_dtype=tl.float32)
            + tl.dot(a_hi, w_lo, out_dtype=tl.float32)
            + tl.dot(a_lo, w_hi, out_dtype=tl.float32))


def _make_forward_kernel(decode_fn, extra_args: int):
    if extra_args == 0:
        @triton.jit
        def kernel(x_ptr, byte_ptr, y_ptr, t_e, n_in, n_out, bytes_per_row,
                   sxm, sxk, sym, syn,
                   BM: tl.constexpr, BN: tl.constexpr, WDOT: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            rm = pid_m * BM + tl.arange(0, BM)
            rn = pid_n * BN + tl.arange(0, BN)
            m_mask = rm < t_e
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, n_in, 32):
                cols = k + tl.arange(0, 32)
                x_tile = tl.load(x_ptr + rm[:, None] * sxm + cols[None, :] * sxk,
                                  mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
                w_tile = decode_fn(byte_ptr, rn, bytes_per_row, k)  # [BN,32] f32
                acc += _dot_wdot(x_tile, tl.trans(w_tile), WDOT)
            tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, acc, mask=m_mask[:, None])
        return kernel

    @triton.jit
    def kernel_extra(x_ptr, byte_ptr, y_ptr, t_e, n_in, n_out, bytes_per_row,
                      sxm, sxk, sym, syn, tab_ptr,
                      BM: tl.constexpr, BN: tl.constexpr, WDOT: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        m_mask = rm < t_e
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, n_in, 32):
            cols = k + tl.arange(0, 32)
            x_tile = tl.load(x_ptr + rm[:, None] * sxm + cols[None, :] * sxk,
                              mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
            w_tile = decode_fn(byte_ptr, rn, bytes_per_row, k, tab_ptr)  # [BN,32] f32
            acc += _dot_wdot(x_tile, tl.trans(w_tile), WDOT)
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, acc, mask=m_mask[:, None])
    return kernel_extra


_FWD_KERNELS = {
    "Q8_0": _make_forward_kernel(_decode_q8_0_tile32, 0),
    "IQ4_NL": _make_forward_kernel(_decode_iq4_nl_tile32, 1),
    "IQ4_XS": _make_forward_kernel(_decode_iq4_xs_tile32, 1),
    "IQ3_S": _make_forward_kernel(_decode_iq3_s_tile32, 1),
}


def _extra_table(qtype: str, device):
    if qtype in ("IQ4_NL", "IQ4_XS"):
        return _kvalues_table(device)
    if qtype == "IQ3_S":
        return iq3s_grid_flat(device)
    return None


def expert_matmul_fwd(x: torch.Tensor, byte_tensor: torch.Tensor, qtype: str,
                       n_in: int, n_out: int, BM: int = 32, BN: int = 32,
                       wdot: str = "split") -> torch.Tensor:
    """`y[T_e,n_out] = x[T_e,n_in] @ W[n_out,n_in]^T`, `W` decoded from format
    `qtype` in `byte_tensor` `[n_out, bytes_per_row]`. `x`: any float dtype
    (converted to bf16 inside the kernel); output is F32. `n_in` must be a
    multiple of 32 (Q8_0/IQ4_NL) or 256 (IQ4_XS/IQ3_S) -- true for the real
    shapes this repository targets; no remainder handling here (caller
    invariant). `wdot`: precision of the W side in the `tl.dot` -- "bf16"
    (fastest, out of a 1e-3 relative-error budget on Q8_0), "f32" (exact,
    6-9x slower), "split" (default)."""
    assert byte_tensor.dtype == torch.uint8 and byte_tensor.is_contiguous()
    assert wdot in ("bf16", "f32", "split")
    t_e = x.shape[0]
    device = x.device
    bytes_per_row = byte_tensor.shape[1]
    y = torch.empty(t_e, n_out, dtype=torch.float32, device=device)
    grid = (triton.cdiv(t_e, BM), triton.cdiv(n_out, BN))
    kernel = _FWD_KERNELS[qtype]
    extra = _extra_table(qtype, device)
    args = [x, byte_tensor, y, t_e, n_in, n_out, bytes_per_row,
            x.stride(0), x.stride(1), y.stride(0), y.stride(1)]
    if extra is not None:
        args.append(extra)
    kernel[grid](*args, BM=BM, BN=BN, WDOT=wdot)
    return y


def _make_backward_kernel(decode_fn, extra_args: int):
    if extra_args == 0:
        @triton.jit
        def kernel(gy_ptr, byte_ptr, gx_ptr, t_e, n_in, n_out, bytes_per_row,
                   sgym, sgyn, sgxm, sgxk,
                   BM: tl.constexpr, BKOUT: tl.constexpr, WDOT: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_k = tl.program_id(1)  # block of 32 n_in columns, fixed per program
            rm = pid_m * BM + tl.arange(0, BM)
            k0 = pid_k * 32
            m_mask = rm < t_e
            acc = tl.zeros((BM, 32), dtype=tl.float32)
            for ro in range(0, n_out, BKOUT):
                rows = ro + tl.arange(0, BKOUT)
                gy_tile = tl.load(gy_ptr + rm[:, None] * sgym + rows[None, :] * sgyn,
                                   mask=m_mask[:, None], other=0.0).to(tl.float32)
                w_tile = decode_fn(byte_ptr, rows, bytes_per_row, k0)  # [BKOUT,32] f32
                acc += _dot_wdot_bwd(gy_tile, w_tile, WDOT)
            cols = k0 + tl.arange(0, 32)
            tl.store(gx_ptr + rm[:, None] * sgxm + cols[None, :] * sgxk, acc, mask=m_mask[:, None])
        return kernel

    @triton.jit
    def kernel_extra(gy_ptr, byte_ptr, gx_ptr, t_e, n_in, n_out, bytes_per_row,
                      sgym, sgyn, sgxm, sgxk, tab_ptr,
                      BM: tl.constexpr, BKOUT: tl.constexpr, WDOT: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        k0 = pid_k * 32
        m_mask = rm < t_e
        acc = tl.zeros((BM, 32), dtype=tl.float32)
        for ro in range(0, n_out, BKOUT):
            rows = ro + tl.arange(0, BKOUT)
            gy_tile = tl.load(gy_ptr + rm[:, None] * sgym + rows[None, :] * sgyn,
                               mask=m_mask[:, None], other=0.0).to(tl.float32)
            w_tile = decode_fn(byte_ptr, rows, bytes_per_row, k0, tab_ptr)  # [BKOUT,32] f32
            acc += _dot_wdot_bwd(gy_tile, w_tile, WDOT)
        cols = k0 + tl.arange(0, 32)
        tl.store(gx_ptr + rm[:, None] * sgxm + cols[None, :] * sgxk, acc, mask=m_mask[:, None])
    return kernel_extra


_BWD_KERNELS = {
    "Q8_0": _make_backward_kernel(_decode_q8_0_tile32, 0),
    "IQ4_NL": _make_backward_kernel(_decode_iq4_nl_tile32, 1),
    "IQ4_XS": _make_backward_kernel(_decode_iq4_xs_tile32, 1),
    "IQ3_S": _make_backward_kernel(_decode_iq3_s_tile32, 1),
}


def expert_matmul_bwd(grad_y: torch.Tensor, byte_tensor: torch.Tensor, qtype: str,
                       n_in: int, n_out: int, BM: int = 32, BKOUT: int = 32,
                       wdot: str = "split") -> torch.Tensor:
    """`gx[T_e,n_in] = grad_y[T_e,n_out] @ W[n_out,n_in]`. `BKOUT` must divide
    `n_out` exactly (caller invariant, true for the real shapes here).
    `wdot`: see `expert_matmul_fwd` -- here the reduction is over `n_out`
    (up to 2560 for Q8_0/IQ4_NL), so bf16 rounding error is worse than in
    forward for the same reason (more summed terms, each with its own
    rounding)."""
    assert byte_tensor.dtype == torch.uint8 and byte_tensor.is_contiguous()
    assert wdot in ("bf16", "f32", "split")
    t_e = grad_y.shape[0]
    device = grad_y.device
    bytes_per_row = byte_tensor.shape[1]
    gx = torch.empty(t_e, n_in, dtype=torch.float32, device=device)
    grid = (triton.cdiv(t_e, BM), n_in // 32)
    kernel = _BWD_KERNELS[qtype]
    extra = _extra_table(qtype, device)
    args = [grad_y, byte_tensor, gx, t_e, n_in, n_out, bytes_per_row,
            grad_y.stride(0), grad_y.stride(1), gx.stride(0), gx.stride(1)]
    if extra is not None:
        args.append(extra)
    kernel[grid](*args, BM=BM, BKOUT=BKOUT, WDOT=wdot)
    return gx


class TritonExpertMatmul(torch.autograd.Function):
    """`y = x @ W.T`, `W` read and decoded from the quantized `byte_tensor` by
    the fused kernels above -- no torch dequantization in the path, `W` never
    materialized (neither in forward nor in backward). No gradient with
    respect to `byte_tensor` (frozen weights): `byte_tensor`/`qtype`/`n_in`/
    `n_out`/`wdot` are saved as non-tensor attributes. `wdot="split"` by
    default (`bf16` is measured out of a 1e-3 relative-error budget, worse in
    backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, byte_tensor: torch.Tensor, qtype: str,
                n_in: int, n_out: int, wdot: str = "split") -> torch.Tensor:
        # No `save_for_backward(x)`: backward no longer needs `x` (it does not
        # need it for a downcast decision -- see the note in `backward`), so
        # the activation is not retained in the graph.
        ctx.byte_tensor = byte_tensor
        ctx.qtype = qtype
        ctx.n_in = n_in
        ctx.n_out = n_out
        ctx.wdot = wdot
        return expert_matmul_fwd(x, byte_tensor, qtype, n_in, n_out, wdot=wdot)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # No downcast of the gradient to `x.dtype` here: `x` is bf16 for
        # activation-memory reasons, but the gradient lives in the F32
        # islands of the reference engine (loss, norms, ...) and stays there
        # -- the downstream consumer (the overlay embedding) expects F32.
        # Downcasting here would throw away the precision the kernel just
        # recovered.
        grad_x = expert_matmul_bwd(grad_out.contiguous(), ctx.byte_tensor, ctx.qtype, ctx.n_in, ctx.n_out,
                                    wdot=ctx.wdot)
        return grad_x, None, None, None, None, None
