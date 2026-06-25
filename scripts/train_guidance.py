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
import torch
import torch.nn.functional as F
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import DataLoader

from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.guidance_net import (
    FrankaCollision,
    GuidanceNet,
    normalize_box,
    sample_boxes,
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
    batch_size: int = 512
    num_iters: int = 4_000  # tiny net on 200k pairs; acc plateaus early
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
        spacing_uniform=cfg.env.spacing_uniform,
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

    print("building env + labeler (newton sim, ~1 min)...", flush=True)
    env = make_env(cfg.env, newton.viewer.ViewerNull())
    geom = env.obstacle_geometry
    labeler = FrankaCollision(device, list(env.robot_base_pos), env.cube_size)
    preprocessor, _ = make_flow_matching_pre_post_processors(policy_cfg, stats)

    # materialize windows in VRAM (dataset is tiny)
    loader = DataLoader(dataset, batch_size=512, num_workers=8, shuffle=False)
    obs_l, act_l = zip(*[(b[OBS_STATE], b[ACTION]) for b in loader])
    obs_all = torch.cat(obs_l).to(device)
    act_all = torch.cat(act_l).to(device)
    n_windows = obs_all.shape[0]
    state_dim = obs_all.shape[-1] - cfg.env.goal_dim
    traj_dim = state_dim + act_all.shape[-1]
    print(f"{n_windows} windows, traj_dim {traj_dim}, horizon {policy_cfg.horizon}")

    # sample and label (window, box) pairs
    rng = __import__("numpy").random.default_rng(cfg.seed)
    idx = torch.randint(0, n_windows, (cfg.n_pairs,), device=device)
    box = sample_boxes(rng, cfg.n_pairs, geom["center"], geom["half_extents"], device)
    free = torch.empty(cfg.n_pairs, dtype=torch.bool, device=device)
    C = 4096
    n_chunks = (cfg.n_pairs + C - 1) // C
    print(f"labeling {cfg.n_pairs} (window, box) pairs via FK...", flush=True)
    for k, i in enumerate(range(0, cfg.n_pairs, C)):
        ci = idx[i : i + C]
        j = act_all[ci][:, :: cfg.label_stride, :7].contiguous()
        cu = obs_all[ci][:, :: cfg.label_stride, CUBE_IDX : CUBE_IDX + 3].contiguous()
        free[i : i + C] = labeler.collision_free(j, cu, box[i : i + C], chunk=C)
        if k % 10 == 0:
            print(f"  labeled chunk {k}/{n_chunks}", flush=True)
    print(f"labels: {free.float().mean():.3f} collision-free", flush=True)

    ee = env.ee_state_index
    f32 = {"dtype": torch.float32, "device": device}
    cube_mean = torch.as_tensor(stats[OBS_STATE]["mean"][ee:], **f32)[:3]
    cube_std = torch.as_tensor(stats[OBS_STATE]["std"][ee:], **f32)[:3]

    net = GuidanceNet(traj_dim, policy_cfg.horizon).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    net.train()
    t0 = time.perf_counter()
    for it in range(cfg.num_iters):
        mb = torch.randint(0, cfg.n_pairs, (cfg.batch_size,), device=device)
        b = preprocessor({OBS_STATE: obs_all[idx[mb]], ACTION: act_all[idx[mb]]})
        traj = torch.cat([b[OBS_STATE][..., :state_dim], b[ACTION]], dim=-1).float()
        logit = net(traj, normalize_box(box[mb], cube_mean, cube_std))
        loss = F.binary_cross_entropy_with_logits(logit, free[mb].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            acc = ((logit > 0) == free[mb]).float().mean().item()
            print(
                f"step {it}/{cfg.num_iters}  loss {loss.item():.4f}  acc {acc:.3f}",
                flush=True,
            )

    torch.save(
        {
            "state_dict": net.state_dict(),
            "traj_dim": traj_dim,
            "max_len": policy_cfg.horizon,
        },
        cfg.out,
    )
    print(f"Saved guidance net to {cfg.out}  ({time.perf_counter() - t0:.0f}s train)")


if __name__ == "__main__":
    main()
