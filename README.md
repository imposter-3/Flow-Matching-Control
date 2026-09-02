# Flow-Matching-Control

For teleoperation, DAgger, training, evaluation and visualization on this task,
please refer to [PushT-suite](https://github.com/imposter-3/PushT-suite).

Two self-contained studies compare flow matching action-chunk policies for
receding-horizon control on the Push-T task, under one fixed protocol. In each
study four methods share a network, a training recipe and a compute budget, and
differ in three places: how the source distribution is built, which flow times
training samples, and whether anything is carried across a replan. Each study
reproduces its Table 1, the main comparison at one network evaluation per
replan, from twelve provided checkpoints. Retraining from scratch is supported
and documented in both.

| Study | Simulator |
|---|---|
| [Pymunk-Gym-PushT](Pymunk-Gym-PushT/README.md) | gym-pusht, 2-D pymunk physics, state-based |
| [Drake-Hydroelastic-PushT](Drake-Hydroelastic-PushT/README.md) | Drake hydroelastic contact, KUKA iiwa14 |

Each subproject carries its own locked environment, checkpoints and data.
From inside either directory:

~~~
uv sync
uv run python -m <package>.checks    # pusht_flow or pusht_drake
~~~

The Pymunk study evaluates all 12 of its cells with `python -m pusht_flow.sweep`.
The Drake study runs one cell at a time with `python -m pusht_drake.evaluate`;
its README gives the loop over all 36. Either way the per-episode artifacts land
under results/rollouts/, and each README lists the expected values, aggregate
and per-seed.

The same four methods run in both studies, under the same keys: cfm_restart,
warm2, forecast_weight_a2 and coupled_a2. The two implementations are separate.
The networks match in size (144,394 parameters) but differ in the adaLN
nonlinearity and in module construction order, so checkpoints are not
interchangeable, and each README describes its own study.

## Third-party code

The Drake study adapts its simulation stack from the codebases of the two
papers below; Drake-Hydroelastic-PushT/NOTICE.md records the exact commits,
licenses and modifications. These are the papers to cite when using that
environment:

~~~bibtex
@misc{wei2025empiricalanalysissimandrealcotraining,
      title={Empirical Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar Pushing from Pixels},
      author={Adam Wei and Abhinav Agarwal and Boyuan Chen and Rohan Bosworth and Nicholas Pfaff and Russ Tedrake},
      year={2025},
      eprint={2503.22634},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.22634},
}
@misc{zeng2026revisitingopenloopexecutionrobotics,
      title={Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies},
      author={Michael Zeng and Abhinav Agarwal and Ajay Bati and Brian Lee and Siddharth Ancha and Russ Tedrake},
      year={2026},
      eprint={2608.15938},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.15938},
}
~~~

The B-spline chunk execution in the Drake study follows the design of
B-spline Policy (arXiv:2607.09648); NOTICE.md records that adaptation too.

## License

MIT, see [LICENSE](LICENSE). Third-party components keep their authors' terms,
recorded per study in
[Drake-Hydroelastic-PushT/NOTICE.md](Drake-Hydroelastic-PushT/NOTICE.md) and
[Pymunk-Gym-PushT/NOTICE.md](Pymunk-Gym-PushT/NOTICE.md).
