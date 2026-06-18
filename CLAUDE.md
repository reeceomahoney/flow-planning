# flow-planning

## Conventions

- **Keep comments short.** Avoid long or multi-line comments; add one only where
  the code is genuinely unclear, and explain *why*, not *what*. Prefer clear
  code and names over commentary. Same for docstrings — concise.
- **No leading-underscore function or method names.** Name everything publicly
  (e.g. `setup_ik`, not `_setup_ik`); dunders like `__init__` are fine.
- Lint/format with `uv run pre-commit run --files <files>` (ruff + ty).
- Tests: `uv run pytest`.

## Running on the cluster

- Prefer the `flow` cluster over the local GPU for any real compute — training,
  eval, sweeps, recording. First check `sky status flow`: if nothing is running
  (cluster idle), this is the preferred method. Put the command in `run:` of
  `configs/sky.yaml` and launch with `./scripts/launch_sky.sh flow` (sweeps live
  in `scripts/sweep.sh`, which `run:` can call). `outputs/` is gitignored, so
  the cluster uses its own checkpoints, not the local ones.
