"""Continuous cubic B-spline execution of a 10 Hz waypoint chunk.

Adapted from the official B-spline Policy implementation
(https://github.com/B-spline-policy/bspline-policy @ 61ed5f42, MIT, (c) 2026
Haoyu Xiong). Their runtime executes each predicted segment as one
scipy.interpolate.BSpline with knots in data-sample units, queried by phase
(t_now - t_origin) * data_freq. This module keeps that representation and
timing model but builds the spline by interpolation through already-guarded
absolute waypoints: the policy here predicts 10 Hz waypoints, not spline
parameters, so there is nothing to fit or compress (their
ScipyBSplineCompression is a training-time tool).

Deviations from the official runtime:

- interpolation through anchors (make_interp_spline with k=3 passed
  explicitly, since their fit path omits k and therefore always fits cubics)
  instead of an episode-level LSQ fit;
- past-domain queries clamp to the endpoint (hold) instead of raising: the
  only past-domain reads here are settle-time holds, where holding the last
  anchor is exactly the pre-spline behavior;
- chunk-to-chunk continuity by construction, the next chunk starting at the
  previous chunk's position and velocity at the splice node (C^1 splice),
  instead of their phase-matched hard switch. Under receding horizon
  (T_exec < T_p) that node is the commitment boundary rather than the plan's
  end, which is what splice_state exists for.

Lookahead vs commitment. n_spans is the geometric domain: every anchor the
policy predicted, all of which shape the cubic. n_commit is how many of those
spans may ever reach the command, and eval clamps to it. Under open loop the
two coincide. Under receding horizon they must not, or a held or exhausted plan
would walk the command through waypoints the controller never committed to and
the policy has already replaced.

Units: positions are absolute world meters (SI). The knot axis is in data
samples (one knot span = one policy period), so spline-derivative values are
meters per knot span; multiply by data_freq for m/s. v_start and end_state()
stay in phase units because the C^1 splice consumes them as-is; mixing the two
unit systems is the 10x whip bug the unit tests pin against.

numpy + scipy only; no pydrake (the check suite pins it), so the math is
testable in the fast tier without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

__all__ = ["ChunkSpline", "build_chunk_spline"]


@dataclass(frozen=True)
class ChunkSpline:
    """One executing chunk: a cubic B-spline over n_spans policy periods.

    eval maps absolute sim time to the commanded xy; the phase is clamped to
    [0, n_commit], so before the origin the spline holds its start and past the
    commitment boundary it holds the last committed anchor, never an
    uncommitted lookahead waypoint.
    """

    bspline: BSpline
    t_origin: float  # absolute sim time of knot 0 (the replan event)
    n_spans: int  # = number of anchors H; anchor j is attained at phase j+1... see eval
    data_freq: float  # knot spans per second (10.0)
    n_commit: int  # spans that may reach the command; == n_spans under open loop

    def eval(self, t_abs: float) -> np.ndarray:
        phase = (float(t_abs) - self.t_origin) * self.data_freq
        phase = min(max(phase, 0.0), float(self.n_commit))
        return np.asarray(self.bspline(phase), dtype=float).reshape(2)

    def end_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity-in-phase-units) at the final node, exactly.

        Evaluated at the integer node in phase space, which is exact by
        construction and immune to absolute-time ULP wobble, so chaining chunks
        through this is deterministic.
        """
        return self._state_at_phase(float(self.n_spans))

    def state_at(self, t_abs: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity-in-phase-units) at an arbitrary absolute time.

        The mid-plan half of the plan-replacement seam: a replan that arrives
        before exhaustion passes state_at(t_splice) (instead of end_state())
        plus the new waypoint chunk to build_chunk_spline, and the new plan is
        C^1 with the trajectory already being executed. The phase is clamped to
        the domain like eval, so past-domain queries return the held endpoint
        state; the velocity of that clamped hold is the boundary derivative, and
        callers splicing after exhaustion should use end_state(), which this
        equals at the final node.
        """
        phase = (float(t_abs) - self.t_origin) * self.data_freq
        return self._state_at_phase(min(max(phase, 0.0), float(self.n_spans)))

    def splice_state(self, t_abs: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity-in-phase-units) at the node nearest t_abs.

        The receding-horizon splice. A replan fires on the policy grid, so the
        cut always lands on an integer node, phase n_commit of the outgoing
        plan. Snapping to that node instead of recomputing the phase from
        absolute time preserves the exactness end_state was written for:
        absolute-time arithmetic is off by an ULP or so, and this value is the
        next plan's C^1 boundary condition, so the wobble would compound across
        replans.

        Clamps to [0, n_commit], the committed end rather than the geometric
        one. A splice can only start from where the command actually was, and
        eval never took the command past n_commit; a plan that was held past its
        commitment must resume from the held anchor, not from lookahead it never
        executed. The clamp also stops the reset path's degenerate one-span
        spline from extrapolating a velocity out of a parked hold (a BSpline
        extrapolates past its domain by default). Equals end_state() at
        exhaustion under open loop.
        """
        phase = (float(t_abs) - self.t_origin) * self.data_freq
        return self._state_at_phase(float(min(max(round(phase), 0), self.n_commit)))

    def _state_at_phase(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(self.bspline(phase), dtype=float).reshape(2)
        v = np.asarray(self.bspline.derivative(1)(phase), dtype=float).reshape(2)
        return p, v

    @classmethod
    def constant(cls, xy, t_origin: float, data_freq: float) -> ChunkSpline:
        """A degenerate hold at xy (parked mailbox, episode start)."""
        xy = np.asarray(xy, dtype=float).reshape(2)
        return build_chunk_spline(
            p_start=xy,
            v_start_phase=np.zeros(2),
            anchors=xy[None, :],
            t_origin=t_origin,
            data_freq=data_freq,
            n_commit=1,
        )


def build_chunk_spline(
    p_start,
    v_start_phase,
    anchors,
    t_origin: float,
    data_freq: float,
    n_commit: int | None = None,
) -> ChunkSpline:
    """Interpolating cubic through [p_start, anchors...] on nodes 0..H.

    p_start/v_start_phase are the outgoing chunk's state at the splice node:
    end_state() under open loop, splice_state(now) under receding horizon, or
    the parked position and zero after a reset. That is the boundary condition
    which makes consecutive chunks C^1. The end condition is natural (zero
    second derivative), matching an unconstrained approach into the final
    anchor. Anchor j (0-based) is attained exactly at phase j + 1, i.e. one
    policy period after the event that scheduled it, the same attainment timing
    the staircase mode has.

    anchors is the lookahead: every predicted waypoint, all of which shape the
    cubic. n_commit (default: all of them) is how many spans may reach the
    command. The full lookahead is passed so that the far anchors act as the
    natural boundary condition for the committed span; pushing the
    zero-acceleration end condition out of the way measured better than
    truncating the interpolation at the commitment.
    """
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 1:
        raise ValueError(f"anchors must be (H >= 1, 2), got {anchors.shape}")
    p_start = np.asarray(p_start, dtype=float).reshape(2)
    v_start_phase = np.asarray(v_start_phase, dtype=float).reshape(2)
    n_spans = len(anchors)
    n_commit = n_spans if n_commit is None else int(n_commit)
    if not 1 <= n_commit <= n_spans:
        raise ValueError(f"n_commit must be in [1, {n_spans}], got {n_commit}")
    nodes = np.arange(n_spans + 1, dtype=float)
    values = np.vstack([p_start[None, :], anchors])
    bspline = make_interp_spline(
        nodes,
        values,
        k=3,
        bc_type=([(1, v_start_phase)], [(2, np.zeros(2))]),
    )
    return ChunkSpline(
        bspline=bspline,
        t_origin=float(t_origin),
        n_spans=n_spans,
        data_freq=float(data_freq),
        n_commit=n_commit,
    )
