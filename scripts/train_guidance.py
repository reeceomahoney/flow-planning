"""Train the learned classifier guidance (GuidanceNet).

Auto-labels (demo window, random box) pairs collision-free/not via FK + box_sdf,
then trains a transformer to score trajectories. No new demos, no rollouts: the
labels come from the same geometry the sim penalizes. Output feeds eval.py's
`learned_guidance_ckpt`.
"""

import time
from dataclasses import dataclass, field

import draccus
import newton.viewer
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_planning.envs import EnvConfig, FrankaConfig, ParticleConfig, make_env
from flow_planning.guidance import box_sdf
from flow_planning.guidance_net import (
    FrankaCollision,
    GuidanceNet,
    normalize_box,
    sample_boxes,
    sample_boxes_2d,
)
from flow_planning.policy import (
    FlowMatchingConfig,
    make_flow_matching_pre_post_processors,
)

CUBE_IDX = 18  # cube xyz start in the obs vector (see FrankaEnv.get_obs)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka"
    device: str = "cuda"
    n_pairs: int = 200_000  # (window, box) pairs to label
    label_stride: int = 3  # subsample timesteps when checking collision
    margin: float = 0.02  # collision-label inflation (thin safety buffer, not a band)
    batch_size: int = 512
    num_iters: int = 6_000  # more steps: per-step labels are far more numerous
    pos_weight: float = 0.0  # BCE positive-class weight; 0 = auto (neg/pos ratio)
    feasible_boxes: bool = False  # resample boxes off the cube/goal (hurt: see memory)
    out: str = "outputs/guidance_net.pt"
    seed: int = 0


@draccus.wrap()
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    dataset = LeRobotDataset(cfg.repo_id)
    features = dataset_to_policy_features(dataset.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    input_features = {k: v for k, v in features.items() if k not in output_features}
    policy_cfg = FlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        goal_dim=cfg.env.goal_dim,
    )
    policy_cfg.horizon = max(dataset.meta.episodes["length"])
    delta = {
        "action": [i / dataset.fps for i in policy_cfg.action_delta_indices],
        "observation.state": [
            i / dataset.fps for i in policy_cfg.observation_delta_indices
        ],
    }
    dataset = LeRobotDataset(cfg.repo_id, delta_timestamps=delta)
    stats = dataset.meta.stats
    assert stats is not None

    particle = isinstance(cfg.env, ParticleConfig)
    preprocessor, _ = make_flow_matching_pre_post_processors(policy_cfg, stats)

    # materialize windows in VRAM (dataset is tiny)
    loader = DataLoader(dataset, batch_size=512, num_workers=8, shuffle=False)
    loader = tqdm(loader, desc="loading windows")
    obs_l, act_l = zip(*[(b[OBS_STATE], b[ACTION]) for b in loader])
    obs_all = torch.cat(obs_l).to(device)
    act_all = torch.cat(act_l).to(device)
    n_windows = obs_all.shape[0]
    state_dim = obs_all.shape[-1] - cfg.env.goal_dim
    traj_dim = state_dim + act_all.shape[-1]
    print(f"{n_windows} windows, traj_dim {traj_dim}, horizon {policy_cfg.horizon}")

    # sample and label (window, box) pairs, per timestep
    rng = np.random.default_rng(cfg.seed)
    idx = torch.randint(0, n_windows, (cfg.n_pairs,), device=device)
    ts = slice(None, None, cfg.label_stride)
    if particle:
        # commanded target xy (the action, which drives the ball) inside the
        # inflated 2D bar — label the action so guidance moves the executed path
        box = sample_boxes_2d(rng, cfg.n_pairs, device)  # (P, 4)
        box_dim, pos_dims = 4, 2
        pos = act_all[idx][:, ts, :2]
        c, h = box[:, None, :2], box[:, None, 2:]
        hit = box_sdf(pos, c, h) < cfg.env.ball_radius + cfg.margin
    else:
        print("building env + labeler", flush=True)
        env = make_env(cfg.env, newton.viewer.ViewerNull())
        geom = env.obstacle_geometry
        labeler = FrankaCollision(device, list(env.robot_base_pos), env.cube_size)
        box = sample_boxes(
            rng, cfg.n_pairs, geom["center"], geom["half_extents"], device
        )
        if cfg.feasible_boxes:
            # keep only task-feasible boxes (eval walls never bury the cube/goal):
            # cuts grasp repulsion but also removes avoidance supervision where
            # collisions live — net loss in eval, kept for recombination
            cube_start = obs_all[idx, 0, CUBE_IDX : CUBE_IDX + 3]
            goal = obs_all[idx, 0, -3:]
            feas = 0.5 * env.cube_size + 0.05  # fingers + label margin clearance
            for _ in range(50):
                c, h = box[:, None, :3], box[:, None, 3:]
                pts = torch.stack([cube_start, goal], dim=1)
                bad = (box_sdf(pts, c, h) < feas).any(dim=1)
                if not bad.any():
                    break
                box[bad] = sample_boxes(
                    rng, int(bad.sum()), geom["center"], geom["half_extents"], device
                )
            print(f"feasibility filter: {int(bad.sum())} infeasible boxes left")
        # net input = FK arm points, box-relative — matches FKLearnedGuidance;
        # labels are arm-only (the hand offset points stand in for a held cube)
        box_dim = 3
        traj_dim = 3 * labeler.arm_points(torch.zeros(1, 7, device=device)).shape[1]
        n_steps = len(range(*ts.indices(act_all.shape[1])))
        hit = torch.empty(cfg.n_pairs, n_steps, dtype=torch.bool, device=device)
        C = 4096
        for i in tqdm(range(0, cfg.n_pairs, C), desc="labeling pairs via FK"):
            j = act_all[idx[i : i + C]][:, ts, :7].contiguous()
            hit[i : i + C] = labeler.collisions(
                j, None, box[i : i + C], cfg.margin, chunk=C
            )
    p_hit = hit.float().mean().item()
    print(f"labels: {p_hit:.3f} of steps collide", flush=True)
    pos_weight = torch.tensor(
        cfg.pos_weight or (1 - p_hit) / max(p_hit, 1e-4), device=device
    )

    if particle:
        f32 = {"dtype": torch.float32, "device": device}
        pos_mean = torch.as_tensor(stats[OBS_STATE]["mean"][:pos_dims], **f32)
        pos_std = torch.as_tensor(stats[OBS_STATE]["std"][:pos_dims], **f32)

    net = GuidanceNet(traj_dim, box_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    net.train()
    t0 = time.perf_counter()
    for it in range(cfg.num_iters):
        mb = torch.randint(0, cfg.n_pairs, (cfg.batch_size,), device=device)
        if particle:
            b = preprocessor({OBS_STATE: obs_all[idx[mb]], ACTION: act_all[idx[mb]]})
            traj = torch.cat([b[OBS_STATE][..., :state_dim], b[ACTION]], dim=-1).float()
            inp, bx = traj[:, ts], normalize_box(box[mb], pos_mean, pos_std)
        else:
            q = act_all[idx[mb]][:, ts, :7]  # physical joints (B, T, 7)
            m, t = q.shape[:2]
            with torch.no_grad():
                pts = labeler.arm_points(q.reshape(-1, 7)).reshape(m, t, -1, 3)
                inp = (pts + labeler.base_pos - box[mb][:, None, None, :3]).flatten(-2)
            bx = box[mb][:, 3:]
        logit = net(inp, bx)
        loss = F.binary_cross_entropy_with_logits(
            logit, hit[mb].float(), pos_weight=pos_weight
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            pred = logit > 0
            tp = (pred & hit[mb]).sum().item()
            recall = tp / max(hit[mb].sum().item(), 1)
            prec = tp / max(pred.sum().item(), 1)
            print(
                f"step {it}/{cfg.num_iters}  loss {loss.item():.4f}  "
                f"recall {recall:.3f}  prec {prec:.3f}",
                flush=True,
            )

    torch.save(
        {"state_dict": net.state_dict(), "traj_dim": traj_dim, "box_dim": box_dim},
        cfg.out,
    )
    print(f"Saved guidance net to {cfg.out}  ({time.perf_counter() - t0:.0f}s train)")


if __name__ == "__main__":
    main()
