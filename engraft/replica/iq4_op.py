"""Per-expert matmul without retaining the dequantized weight matrix in the
autograd graph.

A sequence forward touches nearly all experts of every layer. If autograd
saved each dequantized F32 expert matrix for backward, peak memory would
explode (hundreds of experts per layer, tens of layers). `ExpertMatmul` saves
only the (small) activation `x` and calls `dequant_fn()` a second time in
backward: the expert matrix lives only for the duration of a single call
(forward or backward), never retained in between.

`dequant_fn` is a closure (not a tensor), so autograd does not try to track it
as a differentiable input -- `torch.autograd.Function` automatically ignores
non-tensor `forward` parameters for gradient purposes (we return `None`
explicitly for that position in `backward`, for clarity).
"""
from __future__ import annotations

import torch


class ExpertMatmul(torch.autograd.Function):
    """`y = x @ W.T` with `W = dequant_fn()`, called in both `forward` and
    `backward`. `ctx` saves only `x` -- never `W` -- so the dequantized expert
    matrix does not stay alive beyond the single call.

    `dequant_fn` is typically a closure over a cached dequantization store: at
    warm cache, the second call in backward is nearly free (cache hit, no
    re-dequantization); at cold cache it dequantizes twice per call."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, dequant_fn) -> torch.Tensor:
        w = dequant_fn()
        ctx.save_for_backward(x)
        ctx.dequant_fn = dequant_fn
        return x @ w.T

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        w = ctx.dequant_fn()
        grad_x = grad_out @ w
        return grad_x, None
