# flow-planning

## Conventions

- Never write READMEs, docstrings, or comments. I will write those myself later.
  And yes, I really mean this.
- **No leading-underscore function or method names.** Name everything publicly
  (e.g. `setup_ik`, not `_setup_ik`); dunders like `__init__` are fine.
- Lint/format with `pixi run pre-commit run --files <files>` (ruff + ty).
- Tests: `pixi run pytest`.

## Running on the cluster

- Prefer the `flow` cluster over the local GPU for any real compute — training,
  eval, sweeps, recording. First check `sky status flow`: if nothing is running
  (cluster idle), this is the preferred method. Put the command in `run:` of
  `configs/sky.yaml` and launch with `pixi run launch flow`. `outputs/` is
  gitignored, so the cluster uses its own checkpoints, not the local ones.
