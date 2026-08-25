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
  goes remote, in this order: Vast.ai first, `flow` second, local GPU third,
  `civo` only if the others are unavailable.
- Vast.ai (SkyPilot): put the command in `run:` of `configs/vast.yaml`, launch a
  cluster with `pixi run vast` (`pixi run vast exec` to reuse a running one, or
  `sky launch configs/vast.yaml -c vast2 ...` for a parallel cluster). Set idle
  autostop (`sky autostop <name> -i 20 --down`); a fresh instance spends ~15 min
  on setup (pixi env) before the job runs.
- flow (SkyPilot, two 4090 slots sharing GPU 1): check `sky status flow` first;
  put the command in `run:` of `configs/sky.yaml` and launch with
  `pixi run launch flow`.
- civo (A100 slurm): `command:` block of `configs/slurm.yaml`, launched with
  `pixi run slurm run --cluster civo`; often queue-blocked, QOS caps 4 GPUs.
- `outputs/` is gitignored, so each cluster uses its own checkpoints, not the
  local ones.
