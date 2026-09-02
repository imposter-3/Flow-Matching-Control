"""Isolation and wiring checks. Run these before spending compute on real runs.

    uv run python -m pusht_drake.checks

Every check is simulator-free and finishes in seconds; the Drake rig itself is
exercised by a short evaluation run instead. One group pins each method to
its own definition: the warm arm warms exactly H_e positions, an invalid
context reduces every warm source bitwise to the vanilla one, the persistent
arm resumes where the schedule says it should. The other group covers what
ordinary testing does not reach: the NFE counter, the loss reduction, the RNG
pairing across arms, the moving-frame correction, and the checkpoint payload
contract. Each check either prints a PASS line or raises AssertionError.
"""

from __future__ import annotations

import numpy as np
import torch

from pusht_drake.config import (
    ACTION_SEED_BLOCKS,
    ENV_SEED_BASE,
    MAX_EPISODE_STEPS,
    METHODS,
    NFE,
    NUM_EVAL_EPISODES,
    SCORE_TAU,
    TERMINATE_COVERAGE,
    TRAIN_SEEDS,
    WORKERS,
    Method,
)
from pusht_drake.fm.adapter import CoupledRolloutPolicy, FMRolloutPolicy
from pusht_drake.fm.checkpoint import (
    HorizonSpec,
    build_payload,
    build_source,
    build_time_profile,
    build_velocity_model,
    parse_payload,
)
from pusht_drake.fm.flow_matching import FlowMatchingPolicy
from pusht_drake.fm.path import LinearPath
from pusht_drake.fm.representation import Representation
from pusht_drake.fm.schedules import (
    flow_interval,
    forecast_weight,
    target_flow_time,
    validate_schedules,
    warm_mask,
)
from pusht_drake.fm.sources import WarmContext

H = 16
HE = 2
OVERLAP = H - HE
ALPHA = 2.0
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAILED {name}: {detail}")
    PASSED.append(name)
    print(f"  PASS  {name}")


def _dummy_representation() -> Representation:
    return Representation.from_metadata(
        {
            "metadata_version": 1,
            "observation_representation": "raw_pose",
            "observation_horizon": 1,
            "action_representation": "relative",
            "prediction_horizon": H,
            "condition_dim": 5,
            "feature_mean": [0.5, 0.0, 0.5, 0.0, 3.14],
            "feature_std": [0.1, 0.1, 0.06, 0.06, 2.6],
            "action_mean": [0.0, 0.0],
            "action_std": [0.05, 0.05],
            "action_low": [0.26, -0.24],
            "action_high": [0.74, 0.24],
        }
    )


def _payload_for(method: Method) -> dict:
    torch.manual_seed(0)
    model = build_velocity_model(condition_dim=5, action_dim=2, chunk_size=H)
    policy = FlowMatchingPolicy(
        velocity_model=model,
        action_dim=2,
        chunk_size=H,
        num_integration_steps=NFE,
        source=build_source(method, HorizonSpec(H, HE)),
        solver="euler",
        time_sampling="uniform",
        time_profile=build_time_profile(method, HorizonSpec(H, HE)),
    )
    return build_payload(
        state_dict=policy.state_dict(),
        representation=_dummy_representation(),
        method=method,
        horizons=HorizonSpec(H, HE),
        step=17,
        seed=1,
    )


def _rollout_policy(key: str):
    payload = parse_payload(_payload_for(METHODS[key]))
    cls = CoupledRolloutPolicy if METHODS[key].flow_mode == "persistent" else FMRolloutPolicy
    return cls(payload, device="cpu")


OBSERVATION = np.array([0.42, -0.05, 0.5, 0.02, 3.0], dtype=np.float32)


def _following(policy) -> np.ndarray:
    """An observation whose anchor sits where the last plan put the pusher.

    The adapters guard against a teleported anchor, since an anchor further
    than 2 cm from the last committed waypoint means the carried plan previews
    nothing that happened. A synthetic warm replan therefore has to move the
    observation the way a real executor would.
    """

    last = policy._last_chunk if hasattr(policy, "_last_chunk") else policy._previous_chunk
    observation = OBSERVATION.copy()
    observation[:2] = np.asarray(last[HE - 1], dtype=np.float32)
    return observation


# ---------------------------------------------------------------- schedules


def check_schedules() -> None:
    print("schedules")
    validate_schedules(H, HE, ALPHA)
    lam = forecast_weight(H, HE, ALPHA)
    tau_out = target_flow_time(H, HE, ALPHA)
    tau_in, _ = flow_interval(H, HE, ALPHA)

    check("horizons are H=16, H_e=2", (H, HE) == (16, 2))
    positions = np.arange(OVERLAP, dtype=np.float64)
    check(
        "lambda matches its closed form on the reusable region",
        np.allclose(lam[:OVERLAP], np.exp(-ALPHA * ((positions + 1.0) / OVERLAP) ** 2), atol=1e-15),
    )
    check("lambda is zero on the tail", np.all(lam[OVERLAP:] == 0.0))
    check("tau' saturates over the execution window", np.all(tau_out[:HE] == 1.0))
    tail = np.arange(HE, H, dtype=np.float64)
    check(
        "tau' matches its closed form beyond the window",
        np.allclose(tau_out[HE:], np.exp(-ALPHA * (tail - HE + 1.0) / OVERLAP), atol=1e-15),
    )
    endpoint = float(np.exp(-ALPHA))
    check(
        "lambda at position H-He-1 == tau' at H-1 == exp(-alpha)",
        abs(lam[OVERLAP - 1] - endpoint) < 1e-12 and abs(tau_out[H - 1] - endpoint) < 1e-12,
    )
    check(
        "tau_in equals tau' shifted by He with the zero extension",
        np.allclose(tau_in[:OVERLAP], tau_out[HE:]) and np.all(tau_in[OVERLAP:] == 0),
    )
    ratios = tau_out[HE:OVERLAP] / tau_out[HE + HE : OVERLAP + HE]
    check(
        "maturity ratio is constant = exp(alpha*He/(H-He))",
        np.allclose(ratios, np.exp(ALPHA * HE / OVERLAP), atol=1e-12),
    )
    check("schedules are float64", lam.dtype == np.float64 and tau_out.dtype == np.float64)
    check("the warm mask warms exactly He positions", warm_mask(H, HE, HE).sum() == HE)
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
    allowed = {"key", "name", "source_type", "warm_depth", "alpha"}
    for method in METHODS.values():
        differing = {
            field for field in vars(method) if getattr(method, field) != getattr(reference, field)
        }
        check(
            f"{method.key} varies only along the allowed axes",
            differing <= allowed,
            f"also differs in {differing - allowed}",
        )
    check("four methods, three restart, one persistent", len(METHODS) == 4)
    check(
        "the one persistent method is coupled_a2",
        [m.key for m in METHODS.values() if m.flow_mode == "persistent"] == ["coupled_a2"],
    )

    horizons = HorizonSpec(H, HE)
    torch.manual_seed(0)
    condition = torch.randn(4, 5)
    target = torch.randn(4, H, 2)
    warm_method = METHODS["warm2"]
    warm_source = build_source(warm_method, horizons)
    check("the flow source carries no weights", len(warm_source.state_dict()) == 0)

    context = WarmContext(
        mean=target[:, : warm_source.context_length],
        valid=torch.ones(4, 1),
    )
    generator = torch.Generator().manual_seed(5)
    sample = warm_source.sample(condition, (4, H, 2), context=context, generator=generator)
    eps = torch.randn((4, H, 2), generator=torch.Generator().manual_seed(5))
    check(
        "warm source is mean + sigma*eps on exactly the warm positions",
        torch.allclose(sample.value[:, :HE], target[:, :HE] + 1.0 * eps[:, :HE]),
    )
    check(
        "warm source is pure eps everywhere else",
        torch.equal(sample.value[:, HE:], eps[:, HE:]),
    )

    invalid = WarmContext(mean=torch.full_like(context.mean, 99.0), valid=torch.zeros(4, 1))
    generator = torch.Generator().manual_seed(5)
    degenerate = warm_source.sample(condition, (4, H, 2), context=invalid, generator=generator)
    check(
        "an invalid context reduces the warm source bitwise to vanilla",
        torch.equal(degenerate.value, eps),
    )

    forecast_source = build_source(METHODS["coupled_a2"], horizons)
    check(
        "the forecast-weight context is the whole overlap",
        forecast_source.context_length == OVERLAP,
    )
    context = WarmContext(mean=target[:, :OVERLAP], valid=torch.ones(4, 1))
    generator = torch.Generator().manual_seed(5)
    sample = forecast_source.sample(condition, (4, H, 2), context=context, generator=generator)
    lam = torch.from_numpy(forecast_weight(H, HE, ALPHA)[:OVERLAP]).float().reshape(1, -1, 1)
    check(
        "forecast-weight source is lambda*a1 + eps with unit residual",
        torch.allclose(sample.value[:, :OVERLAP], lam * target[:, :OVERLAP] + eps[:, :OVERLAP])
        and torch.equal(sample.value[:, OVERLAP:], eps[:, OVERLAP:]),
    )

    refused = False
    try:
        Method(key="bad", name="bad", source_type="vanilla", warm_depth=3)
    except ValueError:
        refused = True
    check("a warm depth on a context-free source is refused", refused)
    refused = False
    try:
        Method(key="bad", name="bad", source_type="coupled")
    except ValueError:
        refused = True
    check("a forecast-weighted method without alpha is refused", refused)


# ----------------------------------------------- training draws and the loss


def check_training_draws_and_loss() -> None:
    print("training draws and loss reduction")
    horizons = HorizonSpec(H, HE)
    torch.manual_seed(0)
    condition = torch.randn(8, 5)
    target = torch.randn(8, H, 2)

    def fresh_policy(key: str) -> FlowMatchingPolicy:
        method = METHODS[key]
        torch.manual_seed(0)
        return FlowMatchingPolicy(
            velocity_model=build_velocity_model(condition_dim=5, action_dim=2, chunk_size=H),
            action_dim=2,
            chunk_size=H,
            num_integration_steps=NFE,
            source=build_source(method, horizons),
            solver="euler",
            time_sampling="uniform",
            time_profile=build_time_profile(method, horizons),
        )

    # Draw order: epsilon first, tau second, both from the ambient stream.
    vanilla = fresh_policy("cfm_restart")
    torch.manual_seed(11)
    losses = vanilla.compute_losses(condition, target, None)
    torch.manual_seed(11)
    eps = torch.randn((8, H, 2))
    tau = torch.rand((8, 1))
    path = LinearPath()
    resample = path.sample(eps, target, tau)
    with torch.no_grad():
        predicted = vanilla.velocity_model(condition, resample.x_t, tau)
    expected = (predicted - resample.velocity).pow(2).flatten(start_dim=1).sum(dim=1).mean()
    check(
        "epsilon is drawn before the flow time, and the loss matches a replay",
        torch.allclose(losses.total.detach(), expected, atol=1e-5),
        f"{float(losses.total.detach()):.6f} vs {float(expected):.6f}",
    )
    squared = (predicted - resample.velocity).pow(2)
    check(
        f"the loss sums over (H, A); an element mean would be {H * 2}x smaller",
        abs(float(losses.total.detach() / squared.mean()) - H * 2) < 1e-3,
    )

    # RNG pairing: a coupled arm and a restart arm consume identical randomness.
    coupled = fresh_policy("coupled_a2")
    context = WarmContext(mean=target[:, :OVERLAP], valid=torch.ones(8, 1))
    torch.manual_seed(13)
    vanilla.compute_losses(condition, target, None)
    probe_after_vanilla = torch.rand(3)
    torch.manual_seed(13)
    coupled.compute_losses(condition, target, context)
    probe_after_coupled = torch.rand(3)
    check(
        "a coupled step and a restart step consume identical randomness",
        torch.equal(probe_after_vanilla, probe_after_coupled),
    )

    # The coupled arm trains over the flow interval: one shared xi per row.
    profile = build_time_profile(METHODS["coupled_a2"], horizons)
    tau_in, tau_out = flow_interval(H, HE, ALPHA)
    xi = torch.zeros(4, 1)
    check(
        "xi=0 lands on tau_in",
        np.allclose(profile.map(xi).numpy().reshape(4, H), np.tile(tau_in, (4, 1)), atol=1e-6),
    )
    xi = torch.ones(4, 1)
    check(
        "xi=1 lands on tau_out",
        np.allclose(profile.map(xi).numpy().reshape(4, H), np.tile(tau_out, (4, 1)), atol=1e-6),
    )


# ------------------------------------------------------------------ inference


def check_nfe_accounting() -> None:
    print("NFE accounting")
    for key in METHODS:
        policy = _rollout_policy(key)
        calls = 0
        inner = policy.policy.velocity_model
        original_forward = inner.forward

        def counted(condition, chunk, tau, history=None, _forward=original_forward):
            nonlocal calls
            calls += 1
            return _forward(condition, chunk, tau, history)

        inner.forward = counted
        for nfe in (1, 2, 5):
            if isinstance(policy, CoupledRolloutPolicy):
                policy.num_integration_steps = nfe
            else:
                policy.policy.num_integration_steps = nfe
            policy.reset(0)
            calls = 0
            policy.predict_chunk(OBSERVATION)
            check(
                f"{key}: a cold replan at NFE={nfe} costs exactly {nfe} forwards",
                calls == nfe,
                f"counted {calls}",
            )
            policy.notify_committed(HE)
            calls = 0
            policy.predict_chunk(_following(policy))
            check(
                f"{key}: a warm replan at NFE={nfe} costs exactly {nfe} forwards",
                calls == nfe,
                f"counted {calls}",
            )


def check_warm_inference_substitution() -> None:
    """The inference source is the training source with the forecast swapped in."""

    print("warm inference substitution")
    policy = _rollout_policy("warm2")
    policy.reset(0)
    rng = np.random.default_rng(1)
    previous = rng.uniform(0.3, 0.7, size=(H, 2)).astype(np.float32)
    anchor = previous[HE - 1]
    observation = OBSERVATION.copy()
    observation[:2] = anchor
    policy._previous_chunk = previous.copy()
    policy.notify_committed(HE)

    context = policy._warm_context(anchor)
    check("a committed previous chunk yields a valid context", float(context.valid) == 1.0)
    expected_mean = policy.representation.encode_action(previous[HE : HE + HE], anchor)
    check(
        "the warm mean is the re-anchored tail of the previous chunk",
        np.allclose(context.mean.numpy()[0], expected_mean, atol=1e-7),
    )

    source = policy.policy.source
    condition = torch.zeros(1, 5)
    generator = torch.Generator().manual_seed(9)
    sample = source.sample(condition, (1, H, 2), context=context, generator=generator)
    eps = torch.randn((1, H, 2), generator=torch.Generator().manual_seed(9))
    check(
        "warm positions use the forecast mean plus sigma*eps",
        torch.allclose(sample.value[:, :HE], context.mean + 1.0 * eps[:, :HE], atol=1e-6),
    )
    check(
        "all other positions are pure eps despite an available forecast",
        torch.equal(sample.value[:, HE:], eps[:, HE:]),
    )

    policy.reset(0)
    check("reset forgets the previous chunk", policy._previous_chunk is None)
    check(
        "a missing commitment falls back to the invalid context",
        float((policy._warm_context(anchor)).valid) == 0.0,
    )


def check_persistent_semantics() -> None:
    print("persistent rollout semantics")
    tau_in, tau_out = flow_interval(H, HE, ALPHA)

    policy = _rollout_policy("coupled_a2")
    check(
        "the coupled method dispatches to the persistent class",
        isinstance(policy, CoupledRolloutPolicy),
    )
    policy.reset(0)
    check("first query starts with no carried state", policy._state is None)
    policy.predict_chunk(OBSERVATION)
    check("the first replan of an episode is cold", policy.cold_fallbacks == 1)
    check(
        "the carried tau after a replan is the schedule's tau_out",
        np.allclose(policy._tau, tau_out, atol=1e-6),
    )
    check(
        "shifting the carried tau by He reproduces the trained tau_in",
        np.allclose(np.concatenate([policy._tau[HE:], np.zeros(HE)]), tau_in, atol=1e-6),
    )
    check(
        "the executed prefix reaches flow time 1",
        np.allclose(policy._tau[:HE], 1.0, atol=1e-6),
    )
    check(
        "a cold start carries no forecast weight",
        np.all(policy._anchor_lambda == 0.0),
    )

    policy.notify_committed(HE)
    policy.predict_chunk(_following(policy))
    check("a committed replan resumes warm", policy.cold_fallbacks == 1)
    check(
        "after a warm replan the anchor weight is the forecast schedule",
        np.allclose(policy._anchor_lambda, forecast_weight(H, HE, ALPHA)),
    )

    policy.notify_committed(HE - 1)
    policy.predict_chunk(_following(policy))
    check("a commitment other than He restarts cold", policy.cold_fallbacks == 2)

    policy.notify_committed(HE)
    far = _following(policy)
    far[:2] += 0.5
    policy.predict_chunk(far)
    check("a teleported anchor restarts cold", policy.cold_fallbacks == 3)

    policy.notify_committed(HE)
    policy.invalidate_warm_cache()
    policy.predict_chunk(OBSERVATION)
    check("an invalidated cache restarts cold", policy.cold_fallbacks == 4)

    policy.reset(7)
    policy.predict_chunk(OBSERVATION)
    first_episode_first = policy.cold_fallbacks
    check("a new episode starts cold again", first_episode_first == 1)

    chunks = []
    for _ in range(2):
        replay = _rollout_policy("coupled_a2")
        replay.reset(3)
        replay.predict_chunk(OBSERVATION)
        replay.notify_committed(HE)
        chunk = replay.predict_chunk(_following(replay))
        chunks.append(chunk)
    check(
        "same seed reproduces the same chunks bitwise",
        np.array_equal(chunks[0], chunks[1]),
    )


def check_frame_correction() -> None:
    """Exercise the frame correction with a moving agent.

    A probe that holds the agent still sets the displacement to zero and
    multiplies the whole correction away, so it would pass with the term
    dropped.
    """

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

    # The same physical state, written in a frame displaced by delta: every
    # action-space point moves by -delta, the residual does not move at all.
    old = state_from(forecast, endpoint)
    delta_t = torch.from_numpy(delta).reshape(1, 1, -1)
    expected = state_from(forecast - delta_t, endpoint - delta_t)
    weight = torch.from_numpy((tau + (1.0 - tau) * lam).reshape(1, -1, 1)).float()
    corrected = old - weight * delta_t
    check(
        "the reframe weight tau + (1-tau)*lambda reproduces the moved frame",
        torch.allclose(corrected, expected, atol=1e-6),
        f"max diff = {(corrected - expected).abs().max().item():.3e}",
    )
    check(
        "the correction is non-trivial (a dropped term would be caught)",
        (old - expected).abs().max().item() > 1e-2,
    )


# ------------------------------------------------------- model and checkpoint


def check_model_contract() -> None:
    print("model contract")
    torch.manual_seed(0)
    model = build_velocity_model(condition_dim=5, action_dim=2, chunk_size=H)
    total = sum(p.numel() for p in model.parameters())
    check("parameter count is exactly 144,394", total == 144_394, f"got {total}")
    check(
        "state dict holds exactly 45 tensors",
        len(model.state_dict()) == 45,
        f"got {len(model.state_dict())}",
    )
    torch.manual_seed(0)
    again = build_velocity_model(condition_dim=5, action_dim=2, chunk_size=H)
    check(
        "initialization is a pure function of the seed",
        all(
            torch.equal(model.state_dict()[key], again.state_dict()[key])
            for key in model.state_dict()
        ),
        "the construction order inside the model must stay frozen",
    )
    condition = torch.randn(3, 5)
    chunk = torch.randn(3, H, 2)
    with torch.no_grad():
        scalar = model(condition, chunk, torch.rand(3, 1))
        profile = model(condition, chunk, torch.rand(3, H, 1))
    check(
        "the model accepts both the scalar and the per-position time",
        scalar.shape == (3, H, 2) and profile.shape == (3, H, 2),
    )


def check_checkpoint_roundtrip() -> None:
    print("checkpoint payload roundtrip")
    method = METHODS["coupled_a2"]
    payload = _payload_for(method)
    restored = parse_payload(payload)
    check(
        "roundtrip restores every weight bitwise",
        all(
            torch.equal(payload["state_dict"][key], restored.state_dict[key])
            for key in payload["state_dict"]
        ),
    )
    check(
        "roundtrip restores method, horizons, step and seed",
        restored.model is method
        and (restored.horizons.prediction_horizon, restored.horizons.execution_horizon) == (H, HE)
        and (restored.step, restored.seed) == (17, 1),
    )
    check(
        "loading injects the package's operating point as the budget",
        restored.horizons.num_integration_steps == NFE,
    )

    tampered = dict(payload)
    tampered["method"] = dict(payload["method"], warm_depth=3)
    refused = False
    try:
        parse_payload(tampered)
    except ValueError:
        refused = True
    check("a payload that contradicts its method key is refused", refused)

    foreign = dict(payload)
    foreign["artifact_type"] = "pusht_fm_baseline"
    refused = False
    try:
        parse_payload(foreign)
    except ValueError:
        refused = True
    check("an unconverted research-repository payload is refused", refused)

    incomplete = {k: v for k, v in payload.items() if k != "horizons"}
    refused = False
    try:
        parse_payload(incomplete)
    except ValueError:
        refused = True
    check("a payload missing a top-level key is refused", refused)


# ------------------------------------------------------------------- protocol


def check_eval_protocol() -> None:
    print("evaluation protocol")
    check("300 held-out scene indices", NUM_EVAL_EPISODES == 300)
    check("the scene seed base is 1000", ENV_SEED_BASE == 1000)
    check("three action blocks", ACTION_SEED_BLOCKS == (1000, 2000, 3000))
    check("three training seeds", TRAIN_SEEDS == (0, 1, 2))
    check("the table's operating point is one evaluation per replan", NFE == 1)
    check("24 in-cell workers are the protocol", WORKERS == 24)
    check(
        "scoring at 0.90, termination native at 0.95",
        (SCORE_TAU, TERMINATE_COVERAGE) == (0.90, 0.95),
    )
    check("episodes cap at 300 control steps", MAX_EPISODE_STEPS == 300)
    check(
        "the campaign is 36 artifacts of 300 episodes, 2700 per table cell",
        len(METHODS) * len(TRAIN_SEEDS) * len(ACTION_SEED_BLOCKS) == 36
        and NUM_EVAL_EPISODES * len(ACTION_SEED_BLOCKS) * len(TRAIN_SEEDS) == 2700,
    )
    probe = np.random.default_rng(np.random.SeedSequence([ENV_SEED_BASE, 7])).uniform()
    again = np.random.default_rng(np.random.SeedSequence([ENV_SEED_BASE, 7])).uniform()
    check("the scene derivation is a pure function of the episode index", probe == again)


def check_replay_html_hygiene() -> None:
    """The saved Meshcat replay must open inside a sandboxed viewer.

    Drake's meshcat.js checks the gamepad property (a test that passes even
    when the feature is blocked) and then calls it unguarded from the render
    loop; under a restrictive embedding the call throws, the loop dies after
    one frame, and the replay is a blank pane. Pure string surgery here, so
    the checks stay simulator-free.
    """

    print("replay html portability")
    from pusht_drake.sim.recording import make_portable

    # The part that matters: the flag declaration meshcat gates its
    # render-loop gamepad call on, exactly as Drake 1.48 emits it.
    minimal = (
        "<!DOCTYPE html>\n<html>\n<head>\n  <title>Drake MeshCat</title>\n"
        "</head>\n<body>\n<script>\n"
        "const gamepads_supported = !!navigator.getGamepads;\n"
        "function animate() { viewer.animate(); if (gamepads_supported) "
        "{ handle_gamepads(); } requestAnimationFrame(animate); }\n"
        "</script>\n</body>\n</html>"
    )

    patched = make_portable(minimal)
    check(
        "the render loop never calls the gamepad api",
        "const gamepads_supported = false;" in patched
        and "!!navigator.getGamepads" not in patched
        and "requestAnimationFrame(animate)" in patched,
    )
    check(
        "the shim is injected into head before the viewer",
        "navigator.getGamepads = function" in patched
        and patched.index("navigator.getGamepads = function") < patched.index("<body>"),
    )
    check(
        "the shim swallows a throwing gamepad api and defers to a working one",
        "try { return native(); } catch (error) { return []; }" in patched
        and "if (!native) { return; }" in patched,
    )
    check(
        "the raf fallback watches continuously and yields to native ticks",
        "raf fallback" in patched
        and patched.index("raf fallback") < patched.index("<body>")
        and "lastNativeTick > 500" in patched
        and "native(heartbeat)" in patched,
    )
    check("patching is idempotent", make_portable(patched) == patched)
    check(
        "original content survives",
        "<title>Drake MeshCat</title>" in patched and patched.endswith("</html>"),
    )
    refused = False
    try:
        make_portable("<html><body>no head here</body></html>")
    except ValueError:
        refused = True
    check("html without a head is refused loudly", refused)
    refused = False
    try:
        make_portable("<html><head></head><body>viewer without the flag</body></html>")
    except ValueError:
        refused = True
    check("a moved gamepad flag is refused rather than silently shipped", refused)


def check_torch_boundary() -> None:
    """The sim tier must be importable without torch entering the process.

    interface.py states that boundary and the harness relies on it: torch
    reaches an evaluation worker only through the policy factory spec, never
    through a static import. A probe in the current process would prove
    nothing once the torch suites above have run, so the import happens in a
    fresh interpreter.
    """

    print("sim tier torch boundary")
    import subprocess
    import sys

    probe = (
        "import sys; "
        "import pusht_drake.sim.harness, pusht_drake.sim.rollout, "
        "pusht_drake.sim.coverage; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    detail = result.stderr.strip()[-300:] if result.stderr else "torch was imported"
    check("importing the sim tier pulls no torch", result.returncode == 0, detail)


def main() -> None:
    check_schedules()
    check_method_isolation()
    check_training_draws_and_loss()
    check_nfe_accounting()
    check_warm_inference_substitution()
    check_persistent_semantics()
    check_frame_correction()
    check_model_contract()
    check_checkpoint_roundtrip()
    check_eval_protocol()
    check_replay_html_hygiene()
    check_torch_boundary()
    print(f"\n{len(PASSED)} checks passed.")


if __name__ == "__main__":
    main()
