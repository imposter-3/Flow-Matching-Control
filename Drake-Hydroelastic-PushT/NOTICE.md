# Third-party code

This package derives from MIT-licensed work by others. Upstream copyrights are
retained by their authors; this file records what was taken, from where, and at
which commit. No upstream source file is redistributed here.

## Michaelszeng/diffusion-policy-drake

- Repository: https://github.com/Michaelszeng/diffusion-policy-drake
- Commit: 2bfd13944a641ae66004076d225f16bf4fccf778 (2026-08-18)
- License: MIT
- Authors: Michael Zeng and contributors. Derived in turn from work by Adam
  Wei, Abhinav Agarwal and Bernhard Paus Græsdal (sim-and-real-cotraining and
  bernhardpg planning-through-contact).
- Adapted into: src/pusht_drake/assets/models/ and src/pusht_drake/sim/

Taken:

- The scene assets. src/pusht_drake/assets/models/pusher.sdf, table.urdf and
  pedestal.sdf carry the geometry, inertia, friction and hydroelastic
  parameters of the upstream pusher_floating_hydroelastic.sdf,
  small_table_hydroelastic.urdf and iiwa_pedestal.sdf. arm.yaml and scene.yaml
  carry the same model directives as the upstream scenario's
  "workspace-optimized" section, with the `package://` prefix retargeted to
  `pusht_drake`. Attribution for those descriptions belongs to the upstream
  authors (Bernhard Paus Græsdal, Adam Wei, Abhinav Agarwal).
- The scene arrangement: the iiwa base placement, the welds, and the
  `pusher_end` frame.
- The control architecture: an absolute planar position command tracked by
  differential IK on an iiwa, rate-limited, driven through SimIiwaDriver into a
  1 kHz hydroelastic SAP plant.
- The data convention `action[t] = state[t+1][:2]`, and the reset pattern that
  teleports directly to the push-start configuration.

Not taken: the upstream Python. src/pusht_drake/sim/ is a separate
implementation of the station, scene construction, differential IK, rate
limiting, reset and visualization. It reproduces the upstream construction
closely enough that an 8-episode evaluation is bit-for-bit identical to the
same evaluation run against the upstream stack. Where the shape of the control
chain, the joint-rate limiter's evaluation timing or the reset's teleport order
is upstream's design rather than this project's, the module that implements it
says so.

Changed against upstream, in the assets: the table URDF's top is widened in y
from 0.761 to 0.810 m so the model's own legs sit under it. That is physically
inert here, because the slider's worst-case reach is |y| = 0.261 m, so nothing
ever approaches the edge that moved.

Changed against upstream, in the reimplementation: differential IK tracks a
yaw-relaxed 5-D task, because the pusher is axisymmetric and fixing its yaw
drives joint 7 into saturation; the slider SDF is generated into a
content-addressed cache instead of being rewritten in place on every load; and
the state machine, port switch, disturbance systems and cameras are absent,
being unreachable in this package.

Papers to cite when using this environment:

- A. Wei, A. Agarwal, B. Chen, R. Bosworth, N. Pfaff, R. Tedrake. Empirical
  Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar
  Pushing from Pixels. arXiv:2503.22634, 2025.
- M. Zeng, A. Agarwal, A. Bati, B. Lee, S. Ancha, R. Tedrake. Revisiting
  Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing
  Policies. arXiv:2608.15938, 2026.

## Flow matching implementation

src/pusht_drake/fm/ is written for this study: the task-free algorithm modules
(path, solvers, sources, flow_matching), the velocity model, the representation
and windowing math, the training recipe, and the checkpoint format.

fm/schedules.py is logic-identical to the schedules module of the sibling
Pymunk-Gym-PushT study in this repository, so the two studies define lambda_k
and tau'_k the same way. Their velocity models are the same size and shape but
separate implementations: they differ in the adaLN nonlinearity (SiLU here,
GELU there) and in submodule construction order, so their initialization RNG
streams are not interchangeable and each study's checkpoints belong to its own
code.

The coverage score in src/pusht_drake/sim/coverage.py reproduces the exact
polygon-intersection semantics of the gym-pusht reward natively.

## B-spline-policy/bspline-policy

- Repository: https://github.com/B-spline-policy/bspline-policy
- Commit: 61ed5f42fced971d50a89b46417493790876ccd1 (2026-07-22)
- License: MIT
- Authors: Haoyu Xiong and contributors (paper: Han, Xiong, Chen, Liu,
  Torralba, Zhu, Du, B-spline Policy: Accelerating Manipulation Policies via
  B-spline Action Representations, arXiv:2607.09648)
- Adapted into: src/pusht_drake/sim/bspline_chunk.py (no files vendored
  verbatim)

Taken as design and math, not code: the segment-wise runtime representation,
one cubic scipy BSpline per action chunk with knots in data-sample units,
queried by phase and domain-clamped rather than extrapolated. Not taken:
their training stack, episode fitting, replay-buffer code, and robot
infrastructure. Deviations here: interpolation through guarded anchors, a
clamp-high hold, and C1 position-velocity chunk splicing in place of
phase-matched switching.

## Diffusion Policy (the Push-T task)

The Push-T task, its 512-px geometry and the coverage reward originate with
Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action
Diffusion*, arXiv:2303.04137, 2023. The environment here is a millimetre-for-
pixel replica of that task in world metres.
