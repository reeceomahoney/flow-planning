from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import draccus
import numpy as np
import pytorch_kinematics as pk
import torch
from augment import DEV, Config, libero_candidates, libero_demos, libero_solve
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

from flow_planning.envs.libero import LiberoConfig, LiberoEnv
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision

GREY, GREEN, RED = (160, 160, 160), (0, 220, 0), (0, 0, 255)


@dataclass
class ViewConfig(Config):
    env: LiberoConfig = field(default_factory=LiberoConfig)
    out: str = "outputs/view_aug"
    n_demos: int = 2
    per_demo: int = 4
    size: int = 512


def project(sim, pts, size):
    mat = get_camera_transform_matrix(sim, "agentview", size, size)
    px = project_points_from_world_to_camera(np.asarray(pts), mat, size, size)
    return np.stack([size - 1 - px[..., 1], px[..., 0]], -1)


def draw(frame, pts, colour, r=2):
    for u, v in pts:
        cv2.circle(frame, (int(u), int(v)), r, colour, -1)


def record(env, cfg, x, c, path, writer):
    env.reset()
    env.obs[0] = env.envs[0].set_init_state(x["state0"])
    env.succ[:] = False
    sim = env.envs[0].sim
    orig = project(sim, x["ee"], cfg.size)
    plan = project(sim, c["ee_t"], cfg.size)
    for t, a in enumerate(c["act"]):
        env.apply_action(a[None])
        f = np.ascontiguousarray(env.obs[0]["agentview_image"][::-1, ::-1])
        draw(f, orig, GREY, 1)
        draw(f, plan, GREEN, 1)
        draw(f, plan[t : t + 1], RED, 5)
        cv2.putText(f, path.stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        writer.write(f[..., ::-1])
    for _ in range(cfg.hold):
        env.apply_action(c["act"][-1][None])
    cv2.putText(
        f,
        f"success={int(env.succ[0])}",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    for _ in range(cfg.env.fps):
        writer.write(f[..., ::-1])
    return bool(env.succ[0])


@draccus.wrap()
def main(cfg: ViewConfig):
    rng = np.random.default_rng(cfg.seed)
    env = LiberoEnv(
        replace(
            cfg.env, world_count=1, obstacle=False, render=True, render_size=cfg.size
        )
    )
    base = np.asarray(env.robot_base_pos, np.float32)
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    demos = libero_demos(cfg, env, chain, base, cfg.n_demos)
    cands = libero_candidates(cfg, demos, rng, 4 * cfg.per_demo)
    libero_solve(cfg, cands, demos, chain, base, fcol)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.env.suite}_{cfg.env.task_id}.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), cfg.env.fps, (cfg.size, cfg.size)
    )
    for d, x in enumerate(demos):
        orig = dict(ee_t=x["ee"], act=x["act"], combo=None)
        succ = record(env, cfg, x, orig, Path(f"demo{d}_original"), writer)
        print(f"demo {d} original success={succ}")
        shown = 0
        for c in cands:
            if c["d"] != d or not c["ok"] or shown >= cfg.per_demo:
                continue
            label = " ".join(
                f"app{pa:.2f}/{ta:.1f} tr{pt:.2f}/{tt:.1f} y{np.degrees(psi):+.0f}"
                for pa, ta, pt, tt, psi in c["combo"]
            )
            succ = record(env, cfg, x, c, Path(f"demo{d} {label}"), writer)
            print(f"demo {d} [{label}] success={succ}")
            shown += 1
    writer.release()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
