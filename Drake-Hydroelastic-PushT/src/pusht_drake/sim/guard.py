"""The guard chain every command source runs: square clamp -> fence -> leash.

Demonstration collection and evaluation both guard their candidate commands
through this module, so the two are constrained identically. numpy only: the
chain is pure geometry, and keeping pydrake out lets the checks exercise it
without a simulator.

Layer order is square -> fence everywhere, including the re-application after
a leash rescale. The square is certified inside the fence offline, so on the
normal path the fence projection is the identity after the square clamp; the
fence stays as the final safety layer for profiles with no square, and for
the case where that certification is wrong.

The "moved" predicates use np.allclose(..., atol=1e-12). The default rtol=1e-5
gives a ~5 um deadband at these magnitudes, so changing it would shift the
recorded teleop tick statistics with no other visible symptom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LEASH_M", "GuardTicks", "guard_command"]

# The leash length. In teleop this bounds ||stored target - measured pusher||
# every tick: the 1 kHz-safe form of the measured-pose rebase in the upstream
# Drake station (Michaelszeng/diffusion-policy-drake; see NOTICE.md). In
# evaluation the chunk's waypoints are guarded in a chain at each replan, the
# first anchor against the measured pusher and each later one against its
# guarded predecessor, so the bound is ||anchor_k - measured|| <= k * LEASH_M
# per chunk. The continuous command between anchors can then transiently
# exceed one leash from the pusher in adversarial regimes; DiffIK's demand cap
# and the output containment clamp bound that case.
LEASH_M = 0.05


@dataclass(frozen=True)
class GuardTicks:
    """Which layers moved the candidate this call: 0/1 per layer.

    The re-application after a leash rescale counts toward square and fence
    too, so boundary contact is reported even when the leash triggered it.
    """

    square: int
    fence: int
    leash: int


def _clamp_then_project(candidate, square, workspace):
    """Square clamp then fence projection; reports which layer moved the point."""
    square_moved = False
    if square is not None:
        clamped = square.clamp(candidate)
        square_moved = not np.allclose(clamped, candidate, atol=1e-12)
        candidate = clamped
    projected = workspace.project(candidate)
    fence_moved = not np.allclose(projected, candidate, atol=1e-12)
    return projected, square_moved, fence_moved


def guard_command(
    candidate,
    measured=None,
    *,
    square,
    workspace,
    leash_m: float | None = LEASH_M,
) -> tuple[np.ndarray, GuardTicks]:
    """Constrain one candidate command; return the stored-safe target and ticks.

    square may be None (fence-only profiles). leash_m = None disables the leash
    (first-latch and reset targets have no meaningful measured pose to leash
    against); otherwise measured is required. After a leash rescale both layers
    run again, so the returned target is inside both the square and the fence;
    storing anything else reintroduces upstream's windup bug.
    """
    candidate = np.asarray(candidate, dtype=float).reshape(2)
    candidate, square_moved, fence_moved = _clamp_then_project(candidate, square, workspace)

    leashed = False
    if leash_m is not None:
        measured = np.asarray(measured, dtype=float).reshape(2)
        offset_from_measured = candidate - measured
        distance = float(np.linalg.norm(offset_from_measured))
        if distance > leash_m:
            leashed = True
            candidate = measured + offset_from_measured * (leash_m / distance)
            candidate, square_again, fence_again = _clamp_then_project(candidate, square, workspace)
            square_moved = square_moved or square_again
            fence_moved = fence_moved or fence_again

    return candidate, GuardTicks(
        square=int(square_moved), fence=int(fence_moved), leash=int(leashed)
    )
