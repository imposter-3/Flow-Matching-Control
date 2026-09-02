"""The four methods, the frozen protocol, and the scoring rules.

Two kinds of constant live here:

1. The methods (METHODS): the only axes allowed to vary. A Method carries only
   the source construction and, through source_type, the flow-time schedule
   and the cross-replan reuse rule. It has no field for width, depth, learning
   rate, step budget, horizon or episode count, so an arm cannot be given more
   capacity or more training than its competitors, even by accident.
2. The evaluation protocol: seeds, blocks, workers, thresholds. Every
   published number depends on these; none is sampled at run time.

Everything a method may not vary lives in pusht_drake.fm.recipe: the frozen
optimizer recipe and the shared control contract (H_p, H_e).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceMode = Literal["vanilla", "warm_preview", "forecast_weight", "coupled"]

#: Sources whose tau=0 state is centred on a forecast weighted by lambda_k.
FORECAST_SOURCE_TYPES = ("forecast_weight", "coupled")


@dataclass(frozen=True)
class Method:
    """One row of the table.

    source_type decides where transport starts and whether the partially
    generated chunk survives a replan. "coupled" is the only persistent mode,
    so the illegal combination (a persistent flow with no forecast weight to
    re-anchor against) cannot be written down at all.
    """

    key: str
    name: str
    source_type: SourceMode

    #: Number of leading positions warmed. warm_preview only.
    warm_depth: int | None = None
    #: Shared horizon-decay rate. Forecast-weighted sources only. It sets the
    #: training source profile and the flow-time schedule, so each value needs
    #: its own trained checkpoint.
    alpha: float | None = None

    def __post_init__(self) -> None:
        warm = self.source_type == "warm_preview"
        if warm and self.warm_depth is None:
            raise ValueError(f"{self.key}: a warm arm needs a warm_depth.")
        if not warm and self.warm_depth is not None:
            raise ValueError(
                f"{self.key}: warm_depth is meaningful only for 'warm_preview', got "
                f"source_type={self.source_type!r}."
            )
        forecast = self.source_type in FORECAST_SOURCE_TYPES
        if forecast and self.alpha is None:
            raise ValueError(f"{self.key}: a forecast-weighted arm needs an alpha.")
        if not forecast and self.alpha is not None:
            raise ValueError(
                f"{self.key}: alpha is meaningful only for a forecast-weighted source, "
                f"got source_type={self.source_type!r}."
            )

    @property
    def flow_mode(self) -> str:
        return "persistent" if self.source_type == "coupled" else "restart"

    @property
    def warm_sigma(self) -> float | None:
        """Residual scale of the warm source; fixed at 1 for the warm arm."""

        return 1.0 if self.source_type == "warm_preview" else None


#: The four methods of the comparison, in table order. Display names follow
#: the paper; keys follow the training artifacts.
METHODS: dict[str, Method] = {
    method.key: method
    for method in (
        Method(key="cfm_restart", name="Vanilla CFM", source_type="vanilla"),
        Method(
            key="warm2",
            name="WarmPrior (H_e warm)",
            source_type="warm_preview",
            warm_depth=2,
        ),
        Method(
            key="forecast_weight_a2",
            name="Forecast Weight (this work)",
            source_type="forecast_weight",
            alpha=2.0,
        ),
        Method(
            key="coupled_a2",
            name="Fully Coupled (this work)",
            source_type="coupled",
            alpha=2.0,
        ),
    )
}


# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------

#: Training seeds. Every method is trained once per seed.
TRAIN_SEEDS = (0, 1, 2)

#: Scene seeds: episode i draws its initial poses from
#: numpy.random.default_rng(SeedSequence((ENV_SEED_BASE, i))). The base is held
#: out of every demonstration export.
ENV_SEED_BASE = 1000
NUM_EVAL_EPISODES = 300

#: Action-noise blocks: episode i of a block uses action seed block + i, fed
#: to a fresh per-episode torch.Generator. Three blocks over 300 scenes and 3
#: training seeds gives 2700 paired episodes per table cell.
ACTION_SEED_BLOCKS = (1000, 2000, 3000)

#: Network evaluations per replan. The table's operating point; the sole
#: budget this package evaluates at.
NFE = 1

#: In-cell evaluation workers. Part of the protocol, not a tuning knob: each
#: worker reuses one simulator rig across its stripe of episodes, and the
#: reset settles to a tolerance, so a different worker count moves outcomes
#: at the third decimal. Every published episode ran at 24.
WORKERS = 24

#: Success threshold: an episode succeeds when its maximum coverage exceeds
#: 0.90 (strict). Termination stays at the environment's native 0.95 and is
#: never touched, so trajectories are unaffected by scoring.
SCORE_TAU = 0.90
TERMINATE_COVERAGE = 0.95

#: Episode cap, in 10 Hz control steps (30 seconds of robot time).
MAX_EPISODE_STEPS = 300
