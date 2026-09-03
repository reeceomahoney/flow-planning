"""Parallel evaluation on the fixed SafeLIBERO Spatial task-2 scene.

This is deliberately the only rollout entry point in the PR folder.  It runs
the nominal action chunker, AEGIS, or the bidirectional-contract controller on
Level I, initialization 0, with independent policy seeds.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("base", "aegis", "ours"), required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialization", type=int, default=0)
    parser.add_argument("--level", choices=("I", "II"), default="I")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--safety-margin", type=float, default=None)
    parser.add_argument("--repo-id", default="local/safelibero-spatial-task2-base")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("slop/goal/data/spatial_task2_base"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("slop/goal/outputs/spatial_task2_h50_s10/checkpoints/final"),
    )
    parser.add_argument(
        "--recovery-checkpoint",
        type=Path,
        default=Path(
            "slop/goal/outputs/spatial_task2_h50_s10_ramp_hold_recovery/"
            "checkpoints/step_010000"
        ),
    )
    parser.add_argument(
        "--recovery-bank",
        type=Path,
        default=Path("bidirectional-contract/artifacts/recovery_windows.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("bidirectional-contract/results/reproduction")
    )
    return parser.parse_args()


def _worker(worker_id: int, gpu: str, jobs, results, config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import newton.viewer
        import numpy as np
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import OBS_STATE

        from bidirectional_contract.aegis import (
            AegisCBF,
            AegisLiberoEnv,
            JointTargetToOSC,
            minimum_volume_enclosing_ellipsoid,
            obstacle_points,
        )
        from bidirectional_contract.controller import RecoveryBankController
        from flow_planning.envs.libero import LiberoConfig, LiberoEnv
        from flow_planning.policy import (
            FlowMatchingPolicy,
            make_flow_matching_pre_post_processors,
        )

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("high")
        dataset = LeRobotDataset(config["repo_id"], root=config["dataset_root"])
        checkpoint = (
            config["recovery_checkpoint"]
            if config["method"] == "ours"
            else config["base_checkpoint"]
        )
        policy = FlowMatchingPolicy.from_pretrained(checkpoint)
        policy.config.device = "cuda"
        policy.config.n_action_steps = config["execution_horizon"]
        if policy.config.horizon != 50:
            raise ValueError(f"Expected H=50, got H={policy.config.horizon}")
        policy.to("cuda").eval()
        preprocessor, postprocessor = make_flow_matching_pre_post_processors(
            policy.config, dataset_stats=dataset.meta.stats
        )

        env_type = AegisLiberoEnv if config["method"] == "aegis" else LiberoEnv
        env = env_type(
            LiberoConfig(
                suite="safelibero_spatial",
                task_id=2,
                level=config["level"],
                world_count=1,
                obstacle=True,
                n_action_steps=config["execution_horizon"],
                render=False,
            ),
            newton.viewer.ViewerNull(),
        )
        env.max_frames = config["max_frames"]

        contract = None
        if config["method"] == "ours":
            contract = RecoveryBankController(
                dataset,
                Path(config["recovery_bank"]),
                env.robot_base_pos,
                device="cuda",
                horizon=50,
                execution_horizon=config["execution_horizon"],
                safety_margin=config["safety_margin"],
            )
    except Exception:
        results.put({"fatal": traceback.format_exc(), "worker": worker_id})
        return

    while True:
        job = jobs.get()
        if job is None:
            return
        started = time.perf_counter()
        try:
            episode = job["episode"]
            policy_seed = job["policy_seed"]
            torch.manual_seed(policy_seed)
            np.random.seed(policy_seed)
            env.episode_idx = config["initialization"]
            env.reset()
            policy.reset()
            if contract is not None:
                contract.reset()

            adapter = cbf = None
            if config["method"] == "aegis":
                adapter = JointTargetToOSC(env)
                center, rotation, axes = minimum_volume_enclosing_ellipsoid(
                    obstacle_points(env)
                )
                cbf = AegisCBF(
                    center,
                    rotation,
                    axes,
                    env.obs[0],
                    safety_margin=config["safety_margin"],
                )

            frames = 0
            completion = None
            first_collision = None
            intervention_steps = 0
            intervention_cycles = 0
            infeasible_cycles = 0
            reasons: dict[str, int] = {}
            minimum_clearance = float("inf")
            minimum_h = float("inf")

            while frames < env.max_frames and completion is None:
                if config["method"] == "ours":
                    state = env.get_obs()[0]
                    normalized = policy.predict_action_chunk(
                        preprocessor({OBS_STATE: torch.from_numpy(state[None])})
                    )
                    nominal = postprocessor(normalized).cpu().numpy()[0]
                    decision = contract.plan(state, nominal, env.obstacle_boxes()[0])
                    intervention_cycles += int(decision.intervened)
                    infeasible_cycles += int(not decision.feasible)
                    reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
                    minimum_clearance = min(
                        minimum_clearance, decision.selected_clearance_m
                    )
                    actions = decision.action_chunk[: config["execution_horizon"]]
                    for action in actions:
                        env.apply_action(action[None])
                        frames += 1
                        if bool(env.failure()[0]) and first_collision is None:
                            first_collision = frames
                        if bool(env.success()[0]):
                            completion = frames
                            break
                        if frames >= env.max_frames:
                            break
                else:
                    observation = torch.from_numpy(env.get_obs())
                    action = policy.select_action(
                        preprocessor({OBS_STATE: observation})
                    )
                    joint_action = postprocessor(action).numpy().astype(np.float32)
                    if config["method"] == "aegis":
                        filtered = cbf.filter(adapter(joint_action[0]), env.obs[0])
                        env.apply_osc_action(filtered.action)
                        intervention_steps += int(filtered.intervened)
                        minimum_h = min(minimum_h, filtered.h)
                    else:
                        env.apply_action(joint_action)
                    frames += 1
                    if bool(env.failure()[0]) and first_collision is None:
                        first_collision = frames
                    if bool(env.success()[0]):
                        completion = frames

            results.put(
                {
                    "worker": worker_id,
                    "gpu": gpu,
                    "episode": episode,
                    "policy_seed": policy_seed,
                    "method": config["method"],
                    "success": int(completion is not None),
                    "completion_frame": completion if completion is not None else -1,
                    "collision": int(first_collision is not None),
                    "first_collision_frame": (
                        first_collision if first_collision is not None else -1
                    ),
                    "safe_success": int(
                        completion is not None and first_collision is None
                    ),
                    "frames": frames,
                    "intervention_steps": intervention_steps,
                    "intervention_cycles": intervention_cycles,
                    "infeasible_cycles": infeasible_cycles,
                    "minimum_clearance_m": (
                        minimum_clearance if minimum_clearance != float("inf") else ""
                    ),
                    "minimum_aegis_h": minimum_h if minimum_h != float("inf") else "",
                    "decision_reasons": json.dumps(reasons, sort_keys=True),
                    "seconds": time.perf_counter() - started,
                }
            )
        except Exception:
            results.put(
                {
                    "error": traceback.format_exc(),
                    "episode": job["episode"],
                    "worker": worker_id,
                }
            )


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.safety_margin is None:
        args.safety_margin = 0.01 if args.method == "ours" else 0.0
    if args.safety_margin < 0.0:
        raise ValueError("--safety-margin must be non-negative")

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must name at least one GPU")
    worker_count = min(args.workers, args.episodes)
    if worker_count <= 0:
        raise ValueError("--workers must be positive")
    jobs_spec = [
        {"episode": episode, "policy_seed": args.seed + episode}
        for episode in range(args.episodes)
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method,
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "recovery_checkpoint": str(args.recovery_checkpoint.resolve()),
        "recovery_bank": str(args.recovery_bank.resolve()),
        "initialization": args.initialization,
        "level": args.level,
        "max_frames": args.max_frames,
        "execution_horizon": args.execution_horizon,
        "safety_margin": args.safety_margin,
    }

    context = mp.get_context("spawn")
    jobs = context.Queue()
    results = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(index, gpu_ids[index % len(gpu_ids)], jobs, results, config),
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    for job in jobs_spec:
        jobs.put(job)
    for _ in workers:
        jobs.put(None)

    started = time.perf_counter()
    rows = []
    fatal_count = 0
    while len(rows) < len(jobs_spec):
        try:
            result = results.get(timeout=600)
        except queue.Empty as error:
            raise RuntimeError("Timed out waiting for rollout workers") from error
        if "fatal" in result:
            fatal_count += 1
            if fatal_count == len(workers):
                raise RuntimeError("Every worker failed:\n" + result["fatal"])
            continue
        if "error" in result:
            raise RuntimeError(
                f"Episode {result['episode']} failed:\n{result['error']}"
            )
        rows.append(result)
        print(
            f"{args.method} seed={result['policy_seed']:02d}: "
            f"success={bool(result['success'])} collision={bool(result['collision'])} "
            f"frames={result['frames']} infeasible={result['infeasible_cycles']}",
            flush=True,
        )

    for worker in workers:
        worker.join()
        if worker.exitcode:
            raise RuntimeError(f"Worker {worker.pid} exited with {worker.exitcode}")

    rows.sort(key=lambda row: row["episode"])
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completion_frames = [
        row["completion_frame"] for row in rows if row["success"]
    ]
    summary = {
        "method": args.method,
        "scene": {
            "suite": "safelibero_spatial",
            "task_id": 2,
            "level": args.level,
            "initialization": args.initialization,
            "obstacle": True,
        },
        "policy": {
            "horizon": 50,
            "execution_horizon": args.execution_horizon,
            "checkpoint": (
                str(args.recovery_checkpoint)
                if args.method == "ours"
                else str(args.base_checkpoint)
            ),
        },
        "safety_margin_m": args.safety_margin,
        "seed_protocol": "seed + episode",
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "collisions": sum(row["collision"] for row in rows),
        "safe_successes": sum(row["safe_success"] for row in rows),
        "episodes_with_infeasible_cycle": sum(
            row["infeasible_cycles"] > 0 for row in rows
        ),
        "mean_completion_frame": (
            statistics.fmean(completion_frames) if completion_frames else None
        ),
        "workers": worker_count,
        "gpus": gpu_ids,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
