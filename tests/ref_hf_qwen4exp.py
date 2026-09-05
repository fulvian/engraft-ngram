"""Verbatim copies of the HF reference functions for the Qwen4Exp architecture.

Source: `transformers` 5.16.1,
`transformers/models/qwen4_exp/modeling_qwen4_exp.py`. No `import transformers`:
the functions are pasted here, with the source line number noted in each
docstring, so this module carries no runtime dependency on the `transformers`
package (see NOTICE for attribution).

Usage: only in `tests/test_replica_layers.py`, to compare
`engraft/replica/layers.py` against the HF reference. The replica follows the
llama.cpp fork's conventions (GGUF tensors are already in that form); where
the conventions diverge (tiled vs. grouped V-head permutation, q scaling, the
space g lives in, L2 norm) the four adaptations are applied in the test, not
here: these functions remain the unmodified HF originals.
"""
from __future__ import annotations

import torch


def rotate_half(x):
    """modeling_qwen4_exp.py:566-570, unchanged."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
    """modeling_qwen4_exp.py:573-600, unchanged."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)

    if k is not None:
        k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
        k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
        k_rotated = torch.cat([k_rope, k_nope], dim=-1)
        return q_rotated, k_rotated
    else:
        return q_rotated


def apply_interleaved_mrope(freqs, mrope_section):
    """modeling_qwen4_exp.py:140-156 (a method of Qwen4ExpTextRotaryEmbedding), ported
    to a free function: `freqs` is cloned here instead of being mutated in place
    (`freqs_t = freqs[0]` shares memory with `freqs[0]` in torch; the HF original
    overwrites it by reference) so the caller's argument is never touched -- the
    only deviation, purely about memory management; the formula is identical.

    freqs: (3, ..., head_dim//2). mrope_section: (3,).
    """
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """modeling_qwen4_exp.py:259-262, unchanged."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """modeling_qwen4_exp.py:348-390, unchanged (only the kernel-hub decorators removed).

    Four adaptations applied in the *test* (not here) to compare against
    `layers.gated_delta_net_recurrence` (the fork's conventions):
      (i)   inverse tiled -> grouped permutation on v, z, beta, alpha, ssm_a,
            dt_bias, ssm_out columns, the V part of the conv;
      (ii)  `g` passed in log space (this function exponentiates it inside, like
            the fork);
      (iii) q not prescaled (this function scales by 1/sqrt(d) inside, like the
            fork);
      (iv)  use_qk_l2norm_in_kernel=False and L2 norm applied outside with the
            model's eps (not this function's fixed 1e-6).
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
