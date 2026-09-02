"""Fixed-step ODE integration of a velocity field.

A solver takes the current state, the current flow time, the step size, and a
callable, and nothing about what produced the velocity: no observation, no
condition, no neural network class, no Push-T, no previous chunk. That boundary
lets a policy cache an expensive condition, or a real-time chunking wrapper
correct a velocity, without either reaching in here.

Integration runs from tau=0 to tau=1 with positive dt, per pusht_drake.fm.path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, TypeAlias

import torch

# (x_t (B, H, A), tau (B, 1)) -> velocity (B, H, A)
VelocityFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# (x_t, tau, velocity) -> corrected velocity, same shapes. The wrapper contract
# reserved for real-time chunking and guidance; the solver never sees what the
# correction does.
VelocityCorrection: TypeAlias = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

Solver: TypeAlias = Literal["euler"]

# Velocity evaluations the solver spends per step: an NFE of n means exactly n
# forward passes.
SOLVER_EVALUATIONS: dict[str, int] = {"euler": 1}


def _validate_steps(num_steps: int) -> None:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")


def _velocity_at(
    velocity_fn: VelocityFn,
    state: torch.Tensor,
    tau_value: float,
) -> torch.Tensor:
    """Evaluate the field once, with the time broadcast to one value per row."""

    tau = state.new_full((state.shape[0], 1), tau_value)
    velocity = velocity_fn(state, tau)
    if velocity.shape != state.shape:
        raise ValueError(
            f"Velocity function returned {tuple(velocity.shape)}; expected {tuple(state.shape)}."
        )
    return velocity


def euler_integrate(
    velocity_fn: VelocityFn,
    initial_state: torch.Tensor,
    *,
    num_steps: int = 10,
) -> torch.Tensor:
    """Integrate with explicit Euler, the baseline recipe's solver at 10 steps.

    Uniform steps from tau=0 to tau=1, field evaluated at
    0, 1/N, ..., (N-1)/N. Costs exactly num_steps velocity evaluations.
    """

    _validate_steps(num_steps)
    step_size = 1.0 / num_steps
    state = initial_state
    for step in range(num_steps):
        state = state + step_size * _velocity_at(velocity_fn, state, step / num_steps)
    return state


SOLVERS: dict[str, Callable[..., torch.Tensor]] = {
    "euler": euler_integrate,
}


class CoupledState(NamedTuple):
    """The result of a horizon-coupled integration.

    velocity is the field at the last evaluation, returned because the
    persistent rollout needs it for its endpoint estimate and must not pay a
    further forward to get it.
    """

    state: torch.Tensor  # (B, H, A)
    tau: torch.Tensor  # (B, H, 1), equal to tau_end up to float error
    velocity: torch.Tensor  # (B, H, A)


def coupled_euler_integrate(
    velocity_fn: VelocityFn,
    initial_state: torch.Tensor,
    tau_start: torch.Tensor,
    tau_end: torch.Tensor,
    *,
    num_steps: int = 10,
) -> CoupledState:
    """Advance every horizon position from its own tau_start to its own tau_end.

    The horizon-coupled counterpart of euler_integrate. Positions travel
    different amounts of flow time (a position due for execution runs to tau=1,
    a distant one stops short) but they travel together: one shared progress
    coordinate is stepped num_steps times, and each position's local step is
    scaled by its own interval.

    Heterogeneous local times therefore change the step each row takes, not the
    number of model calls, which is what makes the method affordable. This
    costs exactly num_steps velocity evaluations, the same as the synchronized
    solver, so an NFE axis means the same thing for every arm.

    delta is computed once, before the loop, so after num_steps steps each
    position has advanced by exactly its own interval. Recomputing it against a
    moving tau inside the loop would decay geometrically and never arrive.
    """

    _validate_steps(num_steps)
    if tau_start.shape != tau_end.shape:
        raise ValueError(
            "tau_start and tau_end must have identical shapes, got "
            f"{tuple(tau_start.shape)} and {tuple(tau_end.shape)}."
        )
    expected = (initial_state.shape[0], initial_state.shape[1], 1)
    if tuple(tau_start.shape) != expected:
        raise ValueError(f"Expected tau shape {expected}, got {tuple(tau_start.shape)}.")
    if torch.any(tau_end < tau_start):
        raise ValueError(
            "tau_end must be >= tau_start at every position; a negative interval "
            "would integrate the flow backwards across a replan."
        )

    delta = tau_end - tau_start
    step_size = 1.0 / num_steps
    state = initial_state
    tau = tau_start
    velocity = None
    for _ in range(num_steps):
        velocity = velocity_fn(state, tau)
        if velocity.shape != state.shape:
            raise ValueError(
                f"Velocity function returned {tuple(velocity.shape)}; "
                f"expected {tuple(state.shape)}."
            )
        state = state + step_size * delta * velocity
        tau = tau + step_size * delta
    return CoupledState(state=state, tau=tau, velocity=velocity)
