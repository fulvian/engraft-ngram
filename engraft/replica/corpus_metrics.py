"""Pure counters for the descent's telemetry.

Receives masks already computed by the runner. Does not build a loss,
does not import the replica runtime, and does not estimate token counts from
step counts or nominal capacity."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

COUNTER_FIELDS = (
    "pack_tokens",
    "forward_positions",
    "loss_nll",
    "loss_kl",
    "fragments",
)


def count_pack(
    pack_tokens: int,
    fragments: int,
    loss_mask: Sequence[bool | float],
    answer_mask: Sequence[bool],
    *,
    answer_nll: bool,
) -> dict[str, int]:
    """Counts one update from its pack's actual masks.

    `loss_mask` and `answer_mask` are already aligned to the `pack_tokens - 1`
    forward positions. For KD every active loss position is KL; for lm-base
    the active answers are NLL and the other active loss positions are KL. An
    answer excluded from the loss does not count in either branch."""
    if pack_tokens < 1:
        raise ValueError("pack_tokens must be >= 1")
    if fragments < 0:
        raise ValueError("fragments must be >= 0")
    forward_positions = pack_tokens - 1
    if len(loss_mask) != forward_positions or len(answer_mask) != forward_positions:
        raise ValueError("masks must have pack_tokens - 1 elements")

    active_loss = [float(value) > 0.0 for value in loss_mask]
    active_answer = [loss and bool(answer) for loss, answer in zip(active_loss, answer_mask)]
    loss_nll = sum(active_answer) if answer_nll else 0
    loss_kl = sum(active_loss) - loss_nll
    return {
        "pack_tokens": pack_tokens,
        "forward_positions": forward_positions,
        "loss_nll": loss_nll,
        "loss_kl": loss_kl,
        "fragments": fragments,
    }


def add_counters(*counters: Mapping[str, int]) -> dict[str, int]:
    """Sums homogeneous counters without introducing per-pass estimates."""
    return {
        field: sum(int(counter[field]) for counter in counters)
        for field in COUNTER_FIELDS
    }
