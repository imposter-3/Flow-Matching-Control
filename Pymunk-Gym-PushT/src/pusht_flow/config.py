"""Constants: the task, the shared training recipe, the methods, the protocol.

Three groups of constant live here, kept separate:

1. Push-T facts: the replay, the action box, the env id, the scoring threshold.
2. The shared recipe (RECIPE): the settings all four methods hold identical.
3. The methods (METHODS): the axes allowed to vary.

MethodConfig carries only the source construction, the flow-time schedule and
the cross-replan reuse rule. It has no field for width, depth, learning rate,
step budget or episode count, so a method cannot be given more capacity or more
training than the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------
# 1. Push-T facts
# --------------------------------------------------------------------------

#: Raw state is agent_x, agent_y, block_x, block_y, block_angle.
STATE_DIM = 5

#: The action space is the 512 x 512 pixel frame. These bounds also clip a
#: decoded forecast, so caching absolute waypoints never needs gym.
ACTION_LOW = 0.0
ACTION_HIGH = 512.0

ENV_ID = "gym_pusht/PushT-v0"

PUSHT_DATASET_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
PUSHT_ZARR_RELPATH = "pusht/pusht_cchi_v7_replay.zarr"

#: Integrity pin for the downloaded archive.
PUSHT_ZIP_SHA256 = "63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9"

#: Scoring threshold for reward and success. The environment's native 0.95
#: stays in charge of termination and is never touched, so trajectories are
#: unaffected; only how an episode is scored changes.
#:
#: 0.90 rather than 0.95 because the demonstrations top out there: across the
#: 206 human episodes (25,650 frames) coverage peaks at 0.9014, no frame
#: exceeds 0.95 and only two exceed 0.90. A 0.95 bar would score imitation of
#: behaviour the data never contains.
SUCCESS_THRESHOLD = 0.90

#: Also emitted per episode so the stricter reading stays auditable. It is a
#: CSV column only; no table in this package is scored at 0.95.
AUXILIARY_THRESHOLD = 0.95

#: Episode cap, the registered TimeLimit. Stated here because every published
#: number depends on it.
MAX_EPISODE_STEPS = 300


def agent_position(state):
    """Return the agent (x, y): the first two state components.

    This is the anchor a relative action chunk is decoded against. It accepts
    numpy arrays and torch tensors alike, and every anchoring site calls it.
    """

    return state[..., 0:2]


# --------------------------------------------------------------------------
# 2. The shared recipe: identical for every method, by construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """Everything the four methods must share. No method may override any of it."""

    # Horizons. H - H_e = 14 positions can carry a temporally aligned forecast.
    chunk_size: int = 16
    execution_horizon: int = 2

    # Architecture (144,394 parameters).
    d_model: int = 64
    num_blocks: int = 3
    num_heads: int = 4
    ffn_dim: int = 88

    # Optimization.
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    cosine_start_fraction: float = 0.5
    grad_clip_norm: float = 1.0
    max_train_steps: int = 17_600


RECIPE = Recipe()


# --------------------------------------------------------------------------
# 3. The methods: the only axes allowed to vary
# --------------------------------------------------------------------------

SourceMode = Literal["gaussian", "warmprior_binary", "forecast_weight"]
FlowMode = Literal["restart", "persistent"]


@dataclass(frozen=True)
class MethodConfig:
    """One row of the main table.

    source_mode decides where transport begins, flow_mode decides whether the
    partially generated state survives a replan, and alpha sets both horizon
    schedules at once when the source is the continuous weight.
    """

    key: str
    name: str
    source_mode: SourceMode

    #: Number of leading positions warmed, for warmprior_binary only.
    warm_count: int = 0

    flow_mode: FlowMode = "restart"

    #: Shared horizon-decay rate. Used by forecast_weight sources and by the
    #: maturity schedule of a persistent policy; inert otherwise.
    alpha: float = 2.0

    #: Residual scale of the WarmPrior source: a0 = a1 + sigma * eps on the
    #: warmed positions.
    warmprior_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.source_mode == "warmprior_binary":
            if self.warm_count <= 0:
                raise ValueError(f"{self.key}: a binary warm arm needs warm_count > 0.")
        elif self.warm_count != 0:
            raise ValueError(
                f"{self.key}: warm_count is meaningful only for "
                f"'warmprior_binary', got source_mode={self.source_mode!r}."
            )
        if self.flow_mode == "persistent" and self.source_mode != "forecast_weight":
            # Persistence reuses the partial state, and the reweighting that
            # makes the carried state consistent is written in terms of
            # lambda_k. A binary or context-free source has no lambda to
            # reweight with, so the combination is undefined, not just untested.
            raise ValueError(
                f"{self.key}: flow_mode='persistent' requires "
                f"source_mode='forecast_weight', got {self.source_mode!r}."
            )

    @property
    def uses_forecast(self) -> bool:
        """Whether inference needs the previous replan's endpoint forecast."""

        return self.source_mode in ("warmprior_binary", "forecast_weight")


#: The four methods of the comparison, in table order.
METHODS: dict[str, MethodConfig] = {
    method.key: method
    for method in (
        MethodConfig(
            key="cfm_restart",
            name="CFM Restart",
            source_mode="gaussian",
        ),
        MethodConfig(
            key="warm2",
            name="WarmPrior (Warm2)",
            source_mode="warmprior_binary",
            warm_count=2,
        ),
        MethodConfig(
            key="forecast_weight_a2",
            name="Forecast Weight (alpha=2)",
            source_mode="forecast_weight",
            alpha=2.0,
        ),
        MethodConfig(
            key="coupled_a2",
            name="Full Coupled (alpha=2)",
            source_mode="forecast_weight",
            flow_mode="persistent",
            alpha=2.0,
        ),
    )
}


# --------------------------------------------------------------------------
# 4. Evaluation protocol
# --------------------------------------------------------------------------

#: Training seeds. Every method is trained once per seed.
TRAIN_SEEDS = (0, 1, 2)

#: Held-out environment seeds: env_seed = ENV_SEED_START + i.
ENV_SEED_START = 1000
NUM_EVAL_EPISODES = 300

#: Action-noise blocks: action_seed = block + i. Three blocks over 300
#: environment seeds and 3 training seeds gives 2700 paired episodes per cell.
ACTION_SEED_BLOCKS = (1000, 2000, 3000)

#: Network evaluations per replan. The table's operating point; the sole
#: budget this package evaluates at.
NFE = 1
