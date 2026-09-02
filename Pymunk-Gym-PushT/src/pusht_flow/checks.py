"""Isolation and wiring checks. Run these before spending compute on real runs.

    uv run python -m pusht_flow.checks

Two kinds of check live here. The first kind asserts that each method is the
method it claims to be: that Warm2 warms exactly two positions, that the
restart arms never touch a cached flow state, that the persistent arm resumes
where the schedule says it should. A variant that was never really isolated
produces a full results table that looks fine and means nothing.

The second kind guards mistakes that are invisible to ordinary testing:

* The NFE counter. NFE = n must be n network evaluations per replan, and the
  endpoint forecast must add none. Counted, not assumed.
* The loss reduction. Summed over (H, A); an elementwise mean is 32x smaller,
  which is equivalent to retuning the learning rate.
* Draw order. Residual before flow time.
* The frame correction. Tested with a moving agent. A probe that holds the
  agent still multiplies the whole correction by zero and passes whether the
  term is present or absent, which is exactly how such a term goes missing.

Every check prints PASS or raises. Add to them rather than loosening them.
"""

from __future__ import annotations

import numpy as np
import torch

from pusht_flow.checkpoint import build_payload, parse_payload
from pusht_flow.config import (
    ACTION_SEED_BLOCKS,
    ENV_SEED_START,
    MAX_EPISODE_STEPS,
    METHODS,
    NFE,
    NUM_EVAL_EPISODES,
    RECIPE,
    TRAIN_SEEDS,
    MethodConfig,
    agent_position,
)
from pusht_flow.data import Normalizer
from pusht_flow.env import make_env
from pusht_flow.flow import (
    HorizonProfiles,
    build_training_source,
    flow_matching_loss,
    sample_training_flow_times,
    training_step_tensors,
)
from pusht_flow.model import build_model, count_parameters
from pusht_flow.rollout import (
    CallCounter,
    build_policy,
    reframe_state,
    reframe_weight,
)
from pusht_flow.schedules import (
    flow_interval,
    forecast_weight,
    target_flow_time,
    validate_schedules,
    warm_mask,
)

H = RECIPE.chunk_size
HE = RECIPE.execution_horizon
OVERLAP = H - HE
ALPHA = 2.0
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAILED {name}: {detail}")
    PASSED.append(name)
    print(f"  PASS  {name}")


def _dummy_normalizer() -> Normalizer:
    return Normalizer(
        feature_mean=np.zeros(5, dtype=np.float32),
        feature_std=np.ones(5, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.full(2, 8.0, dtype=np.float32),
        chunk_size=H,
    )


def _policy(method: MethodConfig, counter: CallCounter | None = None):
    torch.manual_seed(0)
    normalizer = _dummy_normalizer()
    model = build_model(condition_dim=5, action_dim=2, recipe=RECIPE)
    model.eval()
    return build_policy(
        model, normalizer, method, RECIPE, device="cpu", counter=counter
    )


# ---------------------------------------------------------------- schedules


def check_schedules() -> None:
    print("schedules")
    validate_schedules(H, HE, ALPHA)
    lam = forecast_weight(H, HE, ALPHA)
    tau_out = target_flow_time(H, HE, ALPHA)
    tau_in, _ = flow_interval(H, HE, ALPHA)

    check("horizons are H=16, H_e=2", (H, HE) == (16, 2), f"got {(H, HE)}")
    check(
        "lambda is zero on the tail",
        lam[OVERLAP] == 0.0 and lam[H - 1] == 0.0,
        f"lambda tail = {lam[OVERLAP:]}",
    )
    check(
        "tau' saturates over the execution window",
        tau_out[0] == 1.0 and tau_out[1] == 1.0,
        f"{tau_out[:HE]}",
    )
    endpoint = float(np.exp(-ALPHA))
    check(
        "lambda at position H-He-1 == tau' at H-1 == exp(-alpha)",
        abs(lam[OVERLAP - 1] - endpoint) < 1e-12
        and abs(tau_out[H - 1] - endpoint) < 1e-12,
        f"{lam[OVERLAP - 1]} vs {tau_out[H - 1]} vs {endpoint}",
    )
    check(
        "tau_in equals tau' shifted by He with the zero extension",
        np.allclose(tau_in[:OVERLAP], tau_out[HE:]) and np.all(tau_in[OVERLAP:] == 0),
        f"{tau_in}",
    )
    ratios = tau_out[HE:OVERLAP] / tau_out[HE + HE : OVERLAP + HE]
    check(
        "maturity ratio is constant = exp(alpha*He/(H-He))",
        np.allclose(ratios, np.exp(ALPHA * HE / OVERLAP), atol=1e-12),
        f"{np.unique(np.round(ratios, 12))}",
    )
    check("warm2 mask sums to 2", warm_mask(H, HE, 2).sum() == 2)
    refused = False
    try:
        warm_mask(H, HE, OVERLAP + 1)
    except ValueError:
        refused = True
    check("warming a tail position is refused", refused)


# ------------------------------------------------------------ method isolation


def check_method_isolation() -> None:
    print("method isolation")
    reference = METHODS["cfm_restart"]
    allowed = {"key", "name", "source_mode", "warm_count", "flow_mode"}
    for method in METHODS.values():
        differing = {
            field
            for field in vars(method)
            if getattr(method, field) != getattr(reference, field)
        }
        check(
            f"{method.key} varies only along the allowed axes",
            differing <= allowed,
            f"also differs in {differing - allowed}",
        )

    torch.manual_seed(0)
    data = torch.randn(4, H, 2)
    residual = torch.randn(4, H, 2)

    warm2 = METHODS["warm2"]
    profiles = HorizonProfiles.build(warm2, chunk_size=H, execution_horizon=HE)
    check(
        "warm2 profile is exactly the two-position mask",
        np.array_equal(
            profiles.warm.reshape(-1).numpy(),
            warm_mask(H, HE, 2).astype(np.float32),
        ),
    )
    source = build_training_source(data, residual, warm2, profiles)
    sigma = warm2.warmprior_sigma
    check(
        "warm2 source is data + sigma*eps on the two warm positions",
        torch.allclose(source[:, :2], data[:, :2] + sigma * residual[:, :2]),
    )
    check(
        "warm2 source is pure eps everywhere else",
        torch.equal(source[:, 2:], residual[:, 2:]),
    )

    coupled = METHODS["coupled_a2"]
    profiles = HorizonProfiles.build(coupled, chunk_size=H, execution_horizon=HE)
    forecast_source = build_training_source(data, residual, coupled, profiles)
    lam = torch.from_numpy(forecast_weight(H, HE, ALPHA)).float().reshape(1, -1, 1)
    check(
        "forecast-weight source is lambda*a1 + eps with unit residual",
        torch.allclose(forecast_source, lam * data + residual, atol=1e-6),
    )
    check(
        "both context sources leave the tail context-free",
        torch.allclose(source[:, H - 1], residual[:, H - 1])
        and torch.allclose(forecast_source[:, H - 1], residual[:, H - 1]),
    )
    gaussian = build_training_source(data, residual, METHODS["cfm_restart"], profiles)
    check("gaussian source is exactly eps", torch.equal(gaussian, residual))

    refused = False
    try:
        MethodConfig(
            key="bad",
            name="bad",
            source_mode="gaussian",
            flow_mode="persistent",
        )
    except ValueError:
        refused = True
    check("persistence without a forecast weight is refused", refused)
    refused = False
    try:
        MethodConfig(key="bad", name="bad", source_mode="gaussian", warm_count=3)
    except ValueError:
        refused = True
    check("a warm count on a context-free source is refused", refused)


def check_warm2_inference_substitution() -> None:
    """The inference source is the training source with the forecast swapped in.

    Warm2 must use the previous forecast on exactly its two warm positions and
    ignore it everywhere else, even though the aligned forecast covers 14.
    """

    print("warm2 inference substitution")
    policy = _policy(METHODS["warm2"])
    policy.reset()
    observation = np.array([100.0, 120.0, 256.0, 256.0, 1.0], dtype=np.float32)
    rng = np.random.default_rng(1)
    previous = rng.uniform(100.0, 400.0, size=(H, 2)).astype(np.float32)
    policy.previous_absolute = previous.copy()

    source = policy.build_source(observation, torch.Generator().manual_seed(11))
    # Replay the generator to recover the residual the call consumed.
    residual = policy._noise(torch.Generator().manual_seed(11))
    # Rebuild the aligned mean exactly as build_source does.
    aligned = np.zeros((H, 2), dtype=np.float32)
    aligned[:OVERLAP] = previous[HE:]
    mean = policy.normalizer.encode_action(
        aligned[None], agent_position(observation)[None]
    )
    mean_t = torch.from_numpy(mean).float()
    sigma = policy.method.warmprior_sigma

    check(
        "warm positions use the forecast mean plus sigma*eps",
        torch.allclose(source[:, :2], mean_t[:, :2] + sigma * residual[:, :2]),
    )
    check(
        "all other positions are pure eps despite an available forecast",
        torch.equal(source[:, 2:], residual[:, 2:]),
    )
    policy.reset()
    check(
        "reset forgets the previous forecast",
        policy.previous_absolute is None,
    )


def check_training_flow_times() -> None:
    print("training flow times")
    restart = METHODS["forecast_weight_a2"]
    coupled = METHODS["coupled_a2"]
    for method, expected in ((restart, (8, 1)), (coupled, (8, H, 1))):
        profiles = HorizonProfiles.build(method, chunk_size=H, execution_horizon=HE)
        torch.manual_seed(0)
        tau = sample_training_flow_times(8, method, profiles, device="cpu")
        check(
            f"{method.key} trains on tau of shape {expected}",
            tuple(tau.shape) == expected,
            f"got {tuple(tau.shape)}",
        )
    profiles = HorizonProfiles.build(coupled, chunk_size=H, execution_horizon=HE)
    tau_in, tau_out = flow_interval(H, HE, ALPHA)
    for xi, label, target in (
        (0.0, "xi=0 lands on tau_in", tau_in),
        (1.0, "xi=1 lands on tau_out", tau_out),
    ):
        sampled = (
            (profiles.tau_in + xi * (profiles.tau_out - profiles.tau_in))
            .reshape(-1)
            .numpy()
        )
        check(label, np.allclose(sampled, target, atol=1e-6))


def check_draw_order_and_loss() -> None:
    print("draw order and loss reduction")
    method = METHODS["cfm_restart"]
    profiles = HorizonProfiles.build(method, chunk_size=H, execution_horizon=HE)
    condition = torch.randn(4, 5)
    data = torch.randn(4, H, 2)

    torch.manual_seed(7)
    source, tau = training_step_tensors(data, method, profiles)
    # Replay the stream by hand in the documented order.
    torch.manual_seed(7)
    residual_first = torch.randn(data.shape)
    tau_second = torch.rand(4, 1)
    check(
        "residual is drawn before the flow time",
        torch.allclose(source, residual_first) and torch.allclose(tau, tau_second),
    )

    model = build_model(condition_dim=5, action_dim=2, recipe=RECIPE)
    loss = flow_matching_loss(model, condition, data, source, tau)
    tau_path = tau.reshape(-1, 1, 1)
    noisy = (1 - tau_path) * source + tau_path * data
    squared = (model(condition, noisy, tau) - (data - source)) ** 2
    check(
        "loss sums over (H, A) and means over the batch",
        torch.allclose(loss, squared.sum(dim=(1, 2)).mean(), atol=1e-5),
    )
    check(
        f"an elementwise mean would be {H * 2}x smaller",
        abs(loss.item() / squared.mean().item() - H * 2) < 1e-3,
    )


# ------------------------------------------------------------------ inference


def check_nfe_counting() -> None:
    print("NFE accounting")
    observation = np.array([100.0, 120.0, 256.0, 256.0, 1.0], dtype=np.float32)
    # The protocol evaluates at NFE = 1 only; the budgets above it are here to
    # prove the counter itself, since a counter that always reads 1 would pass
    # a single-point probe.
    for key in METHODS:
        counter = CallCounter()
        policy = _policy(METHODS[key], counter)
        generator = torch.Generator().manual_seed(0)
        for nfe in (1, 2, 5):
            counter.reset()
            policy.reset()
            policy.predict(observation, nfe=nfe, generator=generator)
            check(
                f"{key}: one replan at NFE={nfe} costs exactly {nfe} forwards",
                counter.forwards == nfe,
                f"counted {counter.forwards}",
            )


def check_restart_has_no_memory() -> None:
    print("restart arms keep no flow state")
    restart_keys = [m.key for m in METHODS.values() if m.flow_mode == "restart"]
    check(
        "three of the four methods restart",
        restart_keys == ["cfm_restart", "warm2", "forecast_weight_a2"],
        f"got {restart_keys}",
    )
    for key in restart_keys:
        policy = _policy(METHODS[key])
        check(
            f"{key}: no persistent flow-state attribute",
            not hasattr(policy, "state"),
            "a restart policy must not carry a partial state",
        )
    policy = _policy(METHODS["cfm_restart"])
    policy.reset()
    generator = torch.Generator().manual_seed(0)
    observation = np.array([100.0, 120.0, 256.0, 256.0, 1.0], dtype=np.float32)
    policy.predict(observation, nfe=5, generator=generator)
    check(
        "CFM Restart stores no forecast after a replan",
        not METHODS["cfm_restart"].uses_forecast and policy.previous_absolute is None,
    )


def check_persistent_resumes_on_schedule() -> None:
    print("persistent arm resumes where the schedule says")
    tau_in, tau_out = flow_interval(H, HE, ALPHA)
    policy = _policy(METHODS["coupled_a2"])
    policy.reset()
    generator = torch.Generator().manual_seed(0)
    observation = np.array([100.0, 120.0, 256.0, 256.0, 1.0], dtype=np.float32)

    check("first query starts with no cached state", policy.state is None)
    policy.predict(observation, nfe=5, generator=generator)
    check(
        "after one replan the carried tau equals tau' shifted by He",
        np.allclose(policy.tau[:OVERLAP], tau_out[HE:], atol=1e-6),
        f"{policy.tau[:OVERLAP]}",
    )
    check(
        "tail positions are carried at flow time zero",
        np.all(policy.tau[OVERLAP:] == 0.0),
    )
    check(
        "the carried tau is exactly the trained tau_in",
        np.allclose(policy.tau, tau_in, atol=1e-6),
    )
    check(
        "tail positions carry no forecast weight",
        np.all(policy.anchor_lambda[OVERLAP:] == 0.0),
    )


def check_frame_correction() -> None:
    """Exercise the frame correction with an agent that actually moves."""

    print("frame correction (moving agent)")
    rng = np.random.default_rng(0)
    tau = np.linspace(0.0, 0.9, H)
    lam = forecast_weight(H, HE, ALPHA)
    residual = torch.from_numpy(rng.normal(size=(1, H, 2))).float()
    forecast = torch.from_numpy(rng.normal(size=(1, H, 2))).float()
    endpoint = torch.from_numpy(rng.normal(size=(1, H, 2))).float()
    delta = np.array([0.7, -0.4], dtype=np.float32)

    def state_from(mean: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        tau_c = torch.from_numpy(tau.reshape(1, -1, 1)).float()
        lam_c = torch.from_numpy(lam.reshape(1, -1, 1)).float()
        return (1 - tau_c) * (lam_c * mean + residual) + tau_c * data

    delta_t = torch.from_numpy(delta).reshape(1, 1, -1)
    old = state_from(forecast, endpoint)
    # The same physical state, written in a frame displaced by delta: every
    # action-space point moves by -delta, the residual does not move at all.
    expected = state_from(forecast - delta_t, endpoint - delta_t)
    corrected = reframe_state(old, tau, lam, delta, device=torch.device("cpu"))
    check(
        "reframe reproduces the state written in the moved frame",
        torch.allclose(corrected, expected, atol=1e-6),
        f"max diff = {(corrected - expected).abs().max().item():.3e}",
    )
    check(
        "the correction is non-trivial (a dropped term would be caught)",
        (old - expected).abs().max().item() > 1e-2,
        "delta was too small for this test to have teeth",
    )
    check(
        "weight is tau at lambda=0 and 1 at tau=1",
        np.allclose(reframe_weight(tau, np.zeros_like(lam)), tau)
        and np.isclose(reframe_weight(np.array([1.0]), np.array([0.3]))[0], 1.0),
    )


def check_determinism() -> None:
    print("determinism")
    observation = np.array([100.0, 120.0, 256.0, 256.0, 1.0], dtype=np.float32)
    outputs = []
    for _ in range(2):
        policy = _policy(METHODS["coupled_a2"])
        policy.reset()
        generator = torch.Generator().manual_seed(3)
        for _ in range(3):
            chunk = policy.predict(observation, nfe=5, generator=generator)
        outputs.append(chunk)
    check(
        "same seed reproduces the same chunk bitwise",
        np.array_equal(outputs[0], outputs[1]),
        f"max diff = {np.abs(outputs[0] - outputs[1]).max():.3e}",
    )


# ------------------------------------------------------- model and checkpoint


def check_model_contract() -> None:
    print("model contract")
    torch.manual_seed(0)
    model = build_model(condition_dim=5, action_dim=2, recipe=RECIPE)
    check(
        "parameter count is exactly 144,394",
        count_parameters(model) == 144_394,
        f"got {count_parameters(model)}",
    )
    check(
        "state dict holds exactly 45 tensors",
        len(model.state_dict()) == 45,
        f"got {len(model.state_dict())}",
    )
    torch.manual_seed(0)
    again = build_model(condition_dim=5, action_dim=2, recipe=RECIPE)
    check(
        "initialization is a pure function of the seed",
        all(
            torch.equal(model.state_dict()[key], again.state_dict()[key])
            for key in model.state_dict()
        ),
        "the construction order inside VelocityModel must stay frozen",
    )


def check_checkpoint_roundtrip() -> None:
    print("checkpoint payload roundtrip")
    torch.manual_seed(0)
    normalizer = _dummy_normalizer()
    model = build_model(condition_dim=5, action_dim=2, recipe=RECIPE)
    method = METHODS["coupled_a2"]
    payload = build_payload(
        model=model,
        normalizer=normalizer,
        method=method,
        recipe=RECIPE,
        step=17,
        seed=1,
    )
    restored = parse_payload(payload)
    check(
        "roundtrip restores every weight bitwise",
        all(
            torch.equal(model.state_dict()[key], restored.model.state_dict()[key])
            for key in model.state_dict()
        ),
    )
    check(
        "roundtrip restores method, recipe, step and seed",
        restored.method is method
        and restored.recipe == RECIPE
        and (restored.step, restored.seed) == (17, 1),
    )
    check(
        "roundtrip restores the normalizer statistics exactly",
        np.array_equal(restored.normalizer.action_std, normalizer.action_std)
        and np.array_equal(restored.normalizer.feature_mean, normalizer.feature_mean),
    )

    tampered = dict(payload)
    tampered["method"] = dict(payload["method"], warm_count=3)
    refused = False
    try:
        parse_payload(tampered)
    except ValueError:
        refused = True
    check("a payload that contradicts its method key is refused", refused)

    foreign = dict(payload)
    foreign["artifact_type"] = "experiment_policy"
    refused = False
    try:
        parse_payload(foreign)
    except ValueError:
        refused = True
    check("a payload in the earlier checkpoint format is refused", refused)


# ------------------------------------------------------------------- protocol


def check_eval_protocol() -> None:
    print("evaluation protocol")
    check("300 held-out environment seeds", NUM_EVAL_EPISODES == 300)
    check("environment seeds start at 1000", ENV_SEED_START == 1000)
    check("three action blocks", ACTION_SEED_BLOCKS == (1000, 2000, 3000))
    check("three training seeds", TRAIN_SEEDS == (0, 1, 2))
    check("the table's operating point is one evaluation per replan", NFE == 1)
    check("four methods", len(METHODS) == 4)
    check(
        "the sweep is 12 cells of 900 episodes",
        len(METHODS) * len(TRAIN_SEEDS) == 12
        and NUM_EVAL_EPISODES * len(ACTION_SEED_BLOCKS) == 900,
    )
    check(
        "2700 paired episodes per method",
        NUM_EVAL_EPISODES * len(ACTION_SEED_BLOCKS) * len(TRAIN_SEEDS) == 2700,
    )
    env = make_env()
    registered = env.spec.max_episode_steps if env.spec is not None else None
    env.close()
    check(
        "the registered TimeLimit is MAX_EPISODE_STEPS",
        registered == MAX_EPISODE_STEPS,
        f"registered {registered}, config states {MAX_EPISODE_STEPS}",
    )


def main() -> None:
    check_schedules()
    check_method_isolation()
    check_warm2_inference_substitution()
    check_training_flow_times()
    check_draw_order_and_loss()
    check_nfe_counting()
    check_restart_has_no_memory()
    check_persistent_resumes_on_schedule()
    check_frame_correction()
    check_determinism()
    check_model_contract()
    check_checkpoint_roundtrip()
    check_eval_protocol()
    print(f"\n{len(PASSED)} checks passed.")


if __name__ == "__main__":
    main()
