# Third-party code and attribution

This study is MIT licensed (see the repository LICENSE). It depends on
third-party work whose copyrights are retained by their authors.

## gym-pusht (Hugging Face / LeRobot)

- Package: https://pypi.org/project/gym-pusht/ (Apache-2.0, Rémi Cadène and
  contributors), itself derived from the pymunk Push-T environment of the
  Diffusion Policy paper.
- Used as a plain PyPI dependency, never vendored or patched. The environment's
  dynamics, PD controller, termination rule and rendering are entirely theirs.
  This study adds only a `gym.Wrapper` that rescales the reported reward; the
  dynamics and the termination threshold are untouched, so an episode run here
  is step for step the episode the unmodified environment produces.

## The Push-T demonstration replay (Diffusion Policy)

- `pusht_cchi_v7_replay.zarr` (206 episodes, 25,650 frames), downloaded from
  https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip on the first
  training run and verified against a pinned sha256 in `src/pusht_flow/config.py`.
- Collected by the Diffusion Policy authors and used unmodified as the training
  corpus.

## Paper to cite for the task and the data

- C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song.
  *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.*
  arXiv:2303.04137, 2023.

```bibtex
@misc{chi2023diffusionpolicy,
      title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
      author={Cheng Chi and Siyuan Feng and Yilun Du and Zhenjia Xu and Eric Cousineau and Benjamin Burchfiel and Shuran Song},
      year={2023},
      eprint={2303.04137},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2303.04137},
}
```

## The flow matching implementation

`src/pusht_flow/` is written for this study. The velocity field is an adaLN
action expert over one token per horizon position; the sources, schedules,
training loop and inference paths are documented in the module docstrings and
pinned by `src/pusht_flow/checks.py`.
