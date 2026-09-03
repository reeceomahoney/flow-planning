"""Fine-tune the state-only flow policy on nominal and recovery windows."""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from flow_planning.policy import (
    FlowMatchingPolicy,
    FlowTransformer,
    make_flow_matching_pre_post_processors,
)
from flow_planning.utils import hf_column


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = [parameter.detach().clone() for parameter in model.parameters()]
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for shadow, parameter in zip(
            self.shadow, model.parameters(), strict=True
        ):
            shadow.lerp_(parameter.detach(), 1.0 - self.decay)

    def store(self, model):
        self.backup = [parameter.detach().clone() for parameter in model.parameters()]
        for shadow, parameter in zip(
            self.shadow, model.parameters(), strict=True
        ):
            parameter.data.copy_(shadow)

    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA restore called without store")
        for backup, parameter in zip(
            self.backup, model.parameters(), strict=True
        ):
            parameter.data.copy_(backup)
        self.backup = None


class RecoveryMixture:
    """Balanced nominal/recovery H-step minibatches."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        bank_path: Path,
        horizon: int,
        recovery_fraction: float,
    ):
        if not 0.0 < recovery_fraction < 1.0:
            raise ValueError("recovery_fraction must be strictly between zero and one")
        self.recovery_fraction = recovery_fraction
        hf = dataset.hf_dataset
        self.nominal_observation = torch.from_numpy(hf_column(hf, OBS_STATE))
        self.nominal_action = torch.from_numpy(hf_column(hf, ACTION))
        episode = torch.as_tensor(
            hf_column(hf, "episode_index"), dtype=torch.long
        )
        last = torch.zeros(int(episode.max()) + 1, dtype=torch.long)
        last[episode] = torch.arange(len(episode))
        self.episode_end = last[episode]
        self.offsets = torch.arange(horizon)

        with np.load(bank_path.resolve()) as bank:
            self.recovery_observation = torch.from_numpy(
                bank["observation"].copy()
            )
            self.recovery_action = torch.from_numpy(bank["action"].copy())
            metadata = bank["metadata"].copy()
        if self.recovery_observation.shape[1] != horizon:
            raise ValueError(
                f"bank horizon {self.recovery_observation.shape[1]} != {horizon}"
            )
        # Uniform over (direction, radius), rather than raw accepted windows.
        # Large radii are rejected by IK more often and otherwise get starved.
        key = metadata[:, [2, 3]]
        self.recovery_groups = [
            torch.from_numpy(
                np.flatnonzero(np.all(np.isclose(key, value), axis=1))
            )
            for value in np.unique(key, axis=0)
        ]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        recovery_count = round(batch_size * self.recovery_fraction)
        nominal_count = batch_size - recovery_count
        index = torch.randint(0, len(self.nominal_observation), (nominal_count,))
        rows = torch.minimum(
            index[:, None] + self.offsets,
            self.episode_end[index][:, None],
        )
        observation = self.nominal_observation[rows]
        action = self.nominal_action[rows]

        selected_group = torch.randint(
            0, len(self.recovery_groups), (recovery_count,)
        )
        recovery_index = torch.empty(recovery_count, dtype=torch.long)
        for group_index in selected_group.unique():
            mask = selected_group == group_index
            members = self.recovery_groups[int(group_index)]
            recovery_index[mask] = members[
                torch.randint(0, len(members), (int(mask.sum()),))
            ]
        observation = torch.cat(
            [observation, self.recovery_observation[recovery_index]], dim=0
        )
        action = torch.cat(
            [action, self.recovery_action[recovery_index]], dim=0
        )
        return observation, action


def save_policy(policy, preprocessor, postprocessor, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    compiled = policy.model
    original = getattr(compiled, "_orig_mod", compiled)
    policy.model = cast(FlowTransformer, original)
    policy.save_pretrained(output)
    policy.model = compiled
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path(
            "slop/goal/outputs/spatial_task2_h50_s10_50k/checkpoints/final"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("slop/goal/data/spatial_task2_base"),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("bidirectional-contract/artifacts/recovery_windows.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bidirectional-contract/artifacts/recovery_policy"),
    )
    parser.add_argument("--repo-id", default="local/safelibero-spatial-task2-base")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recovery-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = args.device if torch.cuda.is_available() else "cpu"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root.resolve())
    policy = FlowMatchingPolicy.from_pretrained(args.resume.resolve())
    policy.config.device = device
    policy.config.n_action_steps = 10
    if policy.config.horizon != 50:
        raise ValueError(f"Expected H=50 checkpoint, got {policy.config.horizon}")
    policy.to(device).train()
    if args.compile:
        policy.model = cast(FlowTransformer, torch.compile(policy.model))
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config, dataset_stats=dataset.meta.stats
    )
    mixture = RecoveryMixture(
        dataset, args.bank, policy.config.horizon, args.recovery_fraction
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=policy.config.optimizer_lr,
        weight_decay=policy.config.optimizer_weight_decay,
        fused=device == "cuda",
    )

    def learning_rate_scale(step: int):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(
            1, args.steps - args.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_scale
    )
    ema = EMA(policy.model)
    use_amp = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )
    metric_path = output / "metrics.jsonl"
    metric_path.unlink(missing_ok=True)
    tick = time.perf_counter()

    for step in range(1, args.steps + 1):
        observation, action = mixture.sample(args.batch_size)
        batch = preprocessor({OBS_STATE: observation, ACTION: action})
        with amp:
            loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update(policy.model)

        if step == 1 or step % 100 == 0:
            elapsed = time.perf_counter() - tick
            tick = time.perf_counter()
            with metric_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.item()),
                            "steps_per_second": 0.0 if step == 1 else 100 / elapsed,
                        }
                    )
                    + "\n"
                )
        if step == 1 or step % 1_000 == 0:
            print(f"step {step:5d}/{args.steps} loss={loss.item():.5f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            ema.store(policy.model)
            save_policy(
                policy,
                preprocessor,
                postprocessor,
                output / "checkpoints" / f"step_{step:06d}",
            )
            ema.restore(policy.model)

    ema.store(policy.model)
    save_policy(policy, preprocessor, postprocessor, output / "checkpoints" / "final")
    print(f"Saved {output / 'checkpoints' / 'final'}")


if __name__ == "__main__":
    main()
