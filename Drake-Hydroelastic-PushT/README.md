# Drake-Hydroelastic-PushT

Flow matching action-chunk policies on Drake hydroelastic Push-T: a 7-DoF
KUKA iiwa14 pushes a 120 mm T block across a table under distributed contact,
driven at 10 Hz through a guarded differential-IK stack. Four methods are
trained under one shared recipe on a human demonstration corpus that ships with
this repository, and evaluated under one fixed protocol. The package reproduces
the main comparison at one network evaluation per replan (Table 1) from 12
provided checkpoints; retraining from scratch is also supported.

## The four methods

All methods share the same network (144,394 parameters), corpus, optimizer
and derived step budget. They differ only in where transport starts (the
source), which flow times training samples, and whether anything survives a
replan. The chunk length is H_p = 16, the execution horizon is H_e = 2, and
alpha = 2 throughout.

| Key | Name | Source a0 | Training flow time | Across replans |
|---|---|---|---|---|
| cfm_restart | Vanilla CFM | eps | one scalar, U(0,1) | nothing reused |
| warm2 | WarmPrior (H_e warm) | forecast + sigma * eps on the first 2 positions, eps elsewhere | one scalar, U(0,1) | forecast reused |
| forecast_weight_a2 | Forecast Weight (this work) | lambda_k * forecast + eps | one scalar, U(0,1) | forecast reused |
| coupled_a2 | Fully Coupled (this work) | lambda_k * forecast + eps | per-position interval, shared progress xi | partial flow state carried |

The two horizon schedules, with k the position and eps standard normal noise
(at training time the target chunk's own leading positions stand in for the
forecast, gated by whether a previous replan existed):

~~~
lambda_k = exp(-alpha * ((k+1) / (H_p - H_e))^2)     for k < H_p - H_e, else 0
tau'_k   = 1                                          for k < H_e
tau'_k   = exp(-alpha * (k - H_e + 1) / (H_p - H_e))  for H_e <= k < H_p
~~~

The persistent method trains each position over the interval from
tau_in_k = tau'_(k+H_e) to tau'_k, exactly the interval the receding horizon
produces at deployment. Its inference carries the partially generated chunk
across replans: shift by H_e, re-express the carried state in the new action
frame (actions are relative to the measured pusher, so the frame moves with
it), re-anchor the source mean, refresh the tail with noise, and continue.
The frame correction weights the displacement by tau + (1 - tau) * lambda and
is exercised by the checks with a moving agent, because a probe that holds
the agent still multiplies the whole term by zero.

## Layout

~~~
pyproject.toml         dependencies and build metadata; uv.lock is the record
config/                the Drake environment description (one YAML)
checkpoints/           12 provided policies: <key>-s<seed>.pt
data/demos_all.zarr    the human demonstration corpus (342 episodes)
src/pusht_drake/
  config.py            the four methods, the protocol, the scoring rules
  paths.py             project-root anchored default locations
  observation.py       the 5-D observation frame (shared numpy tier)
  store.py             corpus reader and its loud validator
  train.py             training CLI (one method, one seed)
  evaluate.py          evaluation CLI (one method, one seed, one block)
  replay.py            replay episodes as HTML animations, or watch live
  checks.py            the pre-flight check suite
  fm/                  torch tier: model, sources, schedules, training,
                       checkpoint contract, rollout adapters
  sim/                 pydrake tier: the scene, the station diagram and its
                       leaf systems, the rig, rollout, harness, guards,
                       coverage, replay recording
  assets/              the 120 mm T geometry and the scene's model files
results/rollouts/      one JSON artifact per cell, created by evaluate.py
~~~

The fm tier may import torch and never pydrake; the sim tier may import
pydrake and never torch. The policy crosses that boundary only as a factory
spec resolved inside each evaluation worker.

## Installation

### Prerequisites

- Python 3.12
- uv (https://docs.astral.sh/uv/)

No GPU is needed to reproduce the table; a GPU helps only for retraining.

1. **Sync the environment**, from this directory. The drake wheel is several
   hundred MB, so the first sync takes a while:

~~~
uv sync
~~~

2. **Run the pre-flight checks**:

~~~
uv run python -m pusht_drake.checks
~~~

They take under a minute, are simulator-free, and validate the schedules, the
method isolation, the NFE accounting, the warm and persistent rollout
semantics, the moving-frame correction and the checkpoint format before any
compute is spent.

## Reproducing Table 1 from the provided checkpoints

Table 1 is 36 cells: 4 methods x 3 training seeds x 3 action blocks, 300
episodes each. One cell at a time:

~~~
uv run python -m pusht_drake.evaluate --method coupled_a2 --seed 0 --block 1000
~~~

All 36, on CPU:

~~~
for m in cfm_restart warm2 forecast_weight_a2 coupled_a2; do
  for s in 0 1 2; do
    for b in 1000 2000 3000; do
      uv run python -m pusht_drake.evaluate --method $m --seed $s --block $b
    done
  done
done
~~~

On the reference machine (28 logical cores) the full set takes about 7 hours.
Each cell runs its 300 episodes across 24 simulator workers and prints its own
mean max coverage and success rate. A cell that already has an artifact is
skipped unless `--force` is passed, so an interrupted run can be restarted.

The worker count is part of the protocol rather than a tuning knob: each worker
reuses one simulator rig across its stripe of episodes, and the reset settles
only to a tolerance, so episode k inherits a microscopic residue of episode
k-1. The same worker count reproduces itself exactly; a different one moves
outcomes at the third decimal. Every published episode ran at 24 workers, on
CPU.

The data of record is one JSON artifact per cell at
results/rollouts/<method>-s<seed>-b<block>.json, holding 300 per-episode
records (env seed, action seed, max coverage, success, episode length,
termination reason, guard and tracking diagnostics) plus their aggregate
metrics. A table cell is a two-stage reduction over those artifacts: the three
300-episode blocks of each training seed are pooled and averaged, then the
three seed means are reduced to mean +/- sample standard deviation
(denominator n-1).

## Watching an episode

Any episode can be replayed as a self-contained HTML animation with a
playback timeline (play, pause, scrub, speed), saved under results/replays/
and openable in any browser:

~~~
uv run python -m pusht_drake.replay --method coupled_a2 --seed 0 --episode 0
~~~

Episode i of a block always reproduces the scene and sampling noise the
campaign evaluated at that index. The replay includes the reset transient
(the arm driving to its push-start posture and the T dropping into place)
before the policy takes over; a full episode is roughly a 7 to 12 MB file,
and the browser console warning about gamepads is expected in every saved
replay.

Once a cell's artifact exists, --pick selects episodes by their recorded
score, which is the quick way to look at failures:

~~~
uv run python -m pusht_drake.replay --method coupled_a2 --seed 0 --pick worst --count 3
~~~

A plain replay runs on a fresh simulator, so a borderline episode can end
differently from its record: a campaign worker hands each episode the
previous one's microscopic reset residue, and contact dynamics amplify that
over 300 steps. Add --exact to reproduce the recorded trajectory exactly; it
re-runs the episode's worker-stripe predecessors first inside the same
recording and prints where on the timeline the picked episode begins.

To watch a rollout as it happens instead, add --live: the simulation runs at
wall-clock speed and the tool prints a local URL to open.

## Expected results

Table 1: the main comparison at NFE = 1. Mean +/- sample std across the
three training seeds, 2700 episodes per cell; higher is better, bold marks a
lead exceeding the root-sum-square of the two seed standard deviations.

| Method | Max Coverage | Success@0.90 |
|---|---|---|
| Vanilla CFM | 0.556 +/- 0.039 | 0.222 +/- 0.041 |
| WarmPrior (H_e warm) | 0.551 +/- 0.016 | 0.230 +/- 0.046 |
| Forecast Weight (this work) | 0.810 +/- 0.036 | 0.580 +/- 0.094 |
| Fully Coupled (this work) | **0.850 +/- 0.006** | **0.796 +/- 0.009** |

Per-seed values behind Table 1, for checking a partial run:

| Method | Metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| Vanilla CFM | Max Coverage | 0.556 | 0.516 | 0.595 |
| Vanilla CFM | Success@0.90 | 0.197 | 0.200 | 0.270 |
| WarmPrior (H_e warm) | Max Coverage | 0.541 | 0.542 | 0.570 |
| WarmPrior (H_e warm) | Success@0.90 | 0.184 | 0.229 | 0.277 |
| Forecast Weight (this work) | Max Coverage | 0.838 | 0.770 | 0.821 |
| Forecast Weight (this work) | Success@0.90 | 0.630 | 0.471 | 0.639 |
| Fully Coupled (this work) | Max Coverage | 0.852 | 0.843 | 0.855 |
| Fully Coupled (this work) | Success@0.90 | 0.801 | 0.786 | 0.800 |

## Seeds and determinism

Nothing in the protocol is drawn at run time; every seed is a deterministic
function of its episode index, so every method sees identical scenes and
identical sampling noise.

| Axis | Values | Set by |
|---|---|---|
| Training seed | 0, 1, 2 | one seeding call per run, before any module is built |
| Scene of episode i | base 1000 | numpy default_rng over SeedSequence (1000, i) |
| Action noise of episode i | block + i for blocks 1000, 2000, 3000 | a fresh per-episode torch.Generator |

The scene derivation draws the T pose first and then the pusher position,
resampling only the pusher until it clears the block. The per-episode
generator is device bound, so CPU and CUDA draw different streams; every
published episode ran on CPU, and the tools default to it. The 24-worker
rule is explained above.

## Retraining from scratch

The corpus ships in the repository (data/demos_all.zarr: 342 human
demonstrations, 59,581 frames at 10 Hz, SI units, validated on load), so this
needs no download. 301 of the episodes are free-form gamepad teleoperation and
41 were recorded as corrections from start states an earlier policy failed on,
so the initial-state distribution of that last group is policy-derived rather
than sampled:

~~~
for method in cfm_restart warm2 forecast_weight_a2 coupled_a2; do
  for seed in 0 1 2; do
    uv run python -m pusht_drake.train --method $method --seed $seed
  done
done
~~~

The step budget is derived, never typed: 400 epochs at the realized 106
updates per epoch = 42,400 steps at batch 512 on the shipped corpus. Each run
takes about 90 seconds on the reference GPU, compiles the loss boundary with
eager-matching RNG, and writes checkpoints/<method>-s<seed>.pt plus a small
summary next to it, overwriting the provided checkpoint of the same name.
Final training losses of the runs behind the table (the mean of the last 25
logged steps), as a wiring check:

| Method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Vanilla CFM | 7.268 | 7.194 | 7.156 |
| WarmPrior (H_e warm) | 7.245 | 7.141 | 7.046 |
| Forecast Weight (this work) | 5.958 | 5.927 | 5.798 |
| Fully Coupled (this work) | 3.967 | 4.072 | 4.015 |

Losses differ between methods because a source nearer the data gives a
lower-variance regression target; only the within-method spread would
indicate a wiring fault. On hardware and versions matching the reference
environment, retraining reproduces the provided checkpoints; elsewhere
expect statistically equivalent rather than bit-identical results.

## Scoring and aggregation

Coverage is the exact polygon intersection of the T with its goal pose
(clipped convex pieces, never rasterized), divided by the T's area. Per
episode the maximum over control steps is recorded; Max Coverage is its mean
over episodes, and Success@0.90 is the fraction of episodes whose maximum
exceeds 0.90 strictly. The threshold is calibrated against the corpus: the
human demonstrator clears 0.90 in 87.4 percent of episodes, while a 0.95 bar
would fail two thirds of the demonstrations. Termination stays the
environment's native rule at 0.95 coverage with a 300-step cap, untouched by
scoring, so trajectories are identical either way.

Aggregation is two-stage: each training seed is reduced to its own mean over
900 pooled episodes, then the table reports mean +/- sample std (denominator
n-1) across the three seeds. The error bar is disagreement between
independently trained policies, not episode spread.

## Reproducibility notes

- The environment of record is Python 3.12 with the packages pinned in
  uv.lock (drake 1.48.0, manipulation 2025.10.20, torch 2.13.0, numpy
  2.5.2). Contact under distributed pressure fields amplifies floating point
  differences, so other platforms drift slightly, well inside the printed
  +/- values; on the reference environment the reproduction is exact.
- CPU and 24 in-cell workers are part of the protocol; see above.
- Three training seeds means three samples behind every +/- value; read the
  error bars accordingly.
- The provided checkpoints are the original trained weights, converted field
  for field into this package's payload with every tensor verified bit for
  bit. Training determinism is pinned down to the RNG draw order, and the
  short-run training path was verified to reproduce the original
  implementation exactly.
- Each evaluation worker starts one local meshcat websocket server that
  nobody opens; it is reused for the worker's whole stripe.

## Third-party code

The scene's model files (src/pusht_drake/assets/models/) and the design of the
Drake simulation stack are adapted from MIT-licensed upstream projects, and the
B-spline execution follows a published design. NOTICE.md records the
provenance, what changed, and the papers to cite.
