# flow-planning

## Conventions

- Never write READMEs, docstrings, or comments. I will write those myself later.
  And yes, I really mean this.
- **No leading-underscore function or method names.** Name everything publicly
  (e.g. `setup_ik`, not `_setup_ik`); dunders like `__init__` are fine.
- Lint/format with `pixi run pre-commit run --files <files>` (ruff + ty).
- Tests: `pixi run pytest`.

## Running on the cluster

- All real compute — training, eval, sweeps, recording, even quick diagnostics —
  goes remote, in this order: `civo` first, `flow` second, local GPU last
  resort.
- civo (A100 slurm): `pixi run slurm run --cluster civo --command "..."`, or
  `scripts/slurm_suite.sh` for per-task pipelines. Logs land in
  `civo:~/flow-planning/slurm/slurm-<id>.out`. QOS caps 4 concurrent GPUs.
- flow (SkyPilot): check `sky status flow` first; put the command in `run:` of
  `configs/sky.yaml` and launch with `pixi run launch flow`. `outputs/` is
  gitignored, so each cluster uses its own checkpoints, not the local ones.
