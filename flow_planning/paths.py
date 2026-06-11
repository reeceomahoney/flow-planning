"""Output directory helpers: runs live under <base>/<env>/<date>/<time>."""

from datetime import datetime
from pathlib import Path


def make_run_dir(base: str, env_name: str) -> Path:
    return Path(base) / env_name / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def latest_run_dir(base: str, env_name: str) -> Path:
    runs = sorted(p for p in (Path(base) / env_name).glob("*/*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}/{env_name}")
    return runs[-1]
