"""Roll out the trained policy: headless batched eval or interactive viewer."""

import enum
import itertools
from collections import Counter
from dataclasses import dataclass, field

import cv2
import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from tqdm import tqdm

from flow_planning.bend import LABEL_DIM
from flow_planning.cbf import EllipsoidCBF
from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.envs.franka import subsample_cloud
from flow_planning.kinematics import build_franka_chain, ee_positions
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)
from flow_planning.selector import AnalyticSelector, FrankaCollision
from flow_planning.utils import hf_column, latest_run_dir, make_viewer

np.set_printoptions(linewidth=100000)


np.set_printoptions(linewidth=100000)


class Cond(enum.StrEnum):
    NULL = "null"  # the learned CFG null token
    ZERO = "zero"  # the label the unbent originals carry
    RANDOM = "random"  # one uniform latent per world, redrawn each episode
    SEARCH = "search"  # n_cond uniform candidates, selector-scored and latched
    SAMPLE = "sample"  # null token, n_cond noise draws, selector picks each replan
    ORACLE = "oracle"  # true wall [height, width] as the label (DemoGen arm)
    FIXED = "fixed"  # the label given by --fixed (comma-separated), no search
    CLOUD = "cloud"  # scene point cloud into the checkpoint's point encoder


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    episodes: int = 1  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 0.0  # 0 = train.py's 1.5 * env.episode_frames
    viewer: str = "none"  # "none", "rerun", or "opengl"
    video: str = ""  # non-empty: render headless to this mp4
    video_failures: bool = False  # video: keep only episodes that collide or time out
    seed: int = 0
    n_action_steps: int = 0  # >0 overrides the checkpoint replan interval
    num_inference_steps: int = 0  # >0 overrides the checkpoint ODE step count
    cond: Cond = Cond.SEARCH
    n_cond: int = 8  # search: uniform candidates drawn at startup
    fixed: str = ""  # cond=fixed: comma-separated label values
    collision: str = "box"  # selector geometry: "box" (ground truth) or "pointcloud"
    resample: bool = False  # search: fresh candidates + re-selection every replan
    cbf: bool = False  # AEGIS-style EE-ellipsoid safety filter on executed actions
    ensemble: int = 1  # >1: temporal ensembling over this many overlapping chunks
    ensemble_m: float = 0.0  # chunk weight exp(-m * age), m>0 favours newer chunks
    cbf_alpha: float = 10.0
    cbf_geoms: bool = False  # one ellipsoid per obstacle geom instead of one MVEE
    cbf_margin: float = 0.0  # barrier acts on h - margin
    cbf_init: int = 0  # >0: filter-only steps at reset until h >= margin
    cbf_hold: float = 0.0  # >0: held-object sphere radius joins the barrier
    cbf_chunk: bool = False  # project the whole predicted chunk through the barrier
    cbf_axes: str = ""  # hand ellipsoid semi-axes "x,y,z" (default paper values)
    cbf_arm: str = ""  # comma-separated fr3 link indices to add to the barrier
    cbf_offset: str = ""  # hand ellipsoid centre offset in the TCP frame "x,y,z"


def sample_cond(bend: np.ndarray, k: int, rng, device) -> torch.Tensor:
    if bend.shape[1] >= LABEL_DIM:
        rows, counts = np.unique(
            bend[np.abs(bend).sum(1) > 0], axis=0, return_counts=True
        )
        assert len(rows), "dataset has no augmented episodes"
        pick = rng.choice(len(rows), k, p=counts / counts.sum())
        return torch.tensor(rows[pick], dtype=torch.float32, device=device)
    arc = bend[:, :2]
    r = np.linalg.norm(arc, axis=1)
    nz = r > 0
    assert nz.any(), "dataset has no bent episodes; every bend label is zero"
    ang = np.arctan2(arc[nz, 1], arc[nz, 0])
    phi = rng.uniform(r[nz].min(), r.max(), k)
    theta = rng.uniform(ang.min(), ang.max(), k)
    cols = [phi * np.cos(theta), phi * np.sin(theta)]
    cols += [rng.choice(np.unique(bend[:, j]), k) for j in range(2, bend.shape[1])]
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32, device=device)


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    checkpoint = cfg.checkpoint or latest_run_dir("outputs", cfg.env.type)
    print(f"Loading checkpoint: {checkpoint}")
    policy = FlowMatchingPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    if cfg.n_action_steps > 0:
        policy.config.n_action_steps = cfg.n_action_steps
    if cfg.num_inference_steps > 0:
        policy.config.num_inference_steps = cfg.num_inference_steps
    policy.ensemble, policy.ensemble_m = cfg.ensemble, cfg.ensemble_m
    policy.to(device)

    dataset = LeRobotDataset(cfg.repo_id)
    stats = dataset.meta.stats
    assert stats is not None
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config, dataset_stats=stats
    )

    # action stats for unnormalizing the planned path when visualizing
    act_mean = np.asarray(stats[ACTION]["mean"])
    act_std = np.asarray(stats[ACTION]["std"])

    live = cfg.viewer != "none" or bool(cfg.video)
    own_frames = getattr(cfg.env, "render", False)
    viewer = make_viewer(
        "opengl" if cfg.video and not own_frames else cfg.viewer,
        headless=bool(cfg.video),
    )

    env = make_env(cfg.env, viewer)
    env.rng = np.random.default_rng(cfg.seed)  # same instances as train.py's evaluate

    rng = np.random.default_rng(cfg.seed)
    searching = cfg.cond is Cond.SEARCH
    sel = None
    if cfg.env.obstacle and hasattr(env, "ee_state_index"):
        geom = env.obstacle_geometry
        cloud = env.scene_pointcloud() if cfg.collision == "pointcloud" else None
        if cloud is not None:
            print(f"scene pointcloud: {len(cloud)} points")
        sel = AnalyticSelector(
            geom["center"] + geom["half_extents"],
            list(env.robot_base_pos),
            stats[ACTION],
            stats[OBS_STATE],
            joint_start=policy.state_dim,
            device=device,
            cube_size=env.cube_size,
            cube_index=getattr(env, "cube_index", 18),
            pointcloud=cloud,
        )
        joints = hf_column(dataset.hf_dataset, "observation.state")[::50, :7]
        sel.fc.calibrate_self(torch.as_tensor(joints, device=device))
        if cfg.cond in (Cond.SEARCH, Cond.SAMPLE) or (
            cfg.cond in (Cond.ORACLE, Cond.FIXED, Cond.CLOUD) and cfg.n_cond > 1
        ):
            policy.selector_fn = sel.score
    if cfg.cond is Cond.CLOUD:
        n_pts = getattr(policy.config, "cloud_points", 0)
        assert n_pts, "checkpoint has no point encoder"
        pts = env.scene_pointcloud() if cfg.env.obstacle else np.zeros((0, 3))
        cloud = subsample_cloud(pts, n_pts, rng)
        policy.cloud = torch.as_tensor(cloud, device=device)[None]
        print(f"cloud cond: {len(pts)} scene points -> {n_pts}")
        if policy.selector_fn is not None:
            policy.n_samples = cfg.n_cond
    if cfg.cond is Cond.SAMPLE:
        assert policy.selector_fn is not None, "sample needs --env.obstacle"
        policy.n_samples = cfg.n_cond
        print(f"null-token best-of-{cfg.n_cond} over noise")
    bend = None
    if getattr(policy.config, "cond_dim", 0):
        bend = hf_column(dataset.hf_dataset, "bend")
        if cfg.cond is Cond.ZERO:
            policy.cond = torch.zeros(1, policy.config.cond_dim, device=device)
        elif cfg.cond is Cond.FIXED:
            vals = [float(v) for v in cfg.fixed.split(",")]
            vals += [0.0] * (policy.config.cond_dim - len(vals))
            policy.cond = torch.tensor([vals], dtype=torch.float32, device=device)
            if policy.selector_fn is not None:
                policy.n_samples = cfg.n_cond
        elif cfg.cond is Cond.ORACLE:
            assert isinstance(cfg.env, FrankaConfig) and cfg.env.obstacle
            policy.cond = torch.tensor(
                [[cfg.env.obstacle_height, cfg.env.obstacle_width]], device=device
            )
            if policy.selector_fn is not None:
                policy.n_samples = cfg.n_cond
                print(f"oracle cond best-of-{cfg.n_cond} over noise")
        elif cfg.cond is Cond.SEARCH:  # candidates scored + latched via selector
            assert policy.selector_fn is not None, "search needs --env.obstacle"
            cand = sample_cond(bend, cfg.n_cond, rng, device)
            policy.cond_candidates = cand
            policy.n_samples = cfg.n_cond
            print(f"search candidates ({len(cand)}):\n{cand.cpu().numpy().round(2)}")
            if cfg.resample:
                policy.cond_candidates = lambda: sample_cond(
                    bend, cfg.n_cond, rng, device
                )
                print(f"resampling {cfg.n_cond} candidates at every replan")

    cbf = None
    if cfg.cbf:
        assert cfg.env.obstacle and hasattr(env, "obstacle_points")
        kw: dict = {}
        if cfg.cbf_axes:
            kw["axes"] = [float(v) for v in cfg.cbf_axes.split(",")]
        if cfg.cbf_offset:
            kw["offset"] = [float(v) for v in cfg.cbf_offset.split(",")]
        cbf = EllipsoidCBF(
            device, env.robot_base_pos, alpha=cfg.cbf_alpha, dt=1 / cfg.env.fps, **kw
        )
        cbf.margin = cfg.cbf_margin
        if cfg.cbf_arm:
            assert cfg.cbf_geoms, "arm barrier needs --cbf_geoms"
            cbf.arm_links = tuple(int(v) for v in cfg.cbf_arm.split(","))
            cbf.fc = FrankaCollision(device, env.robot_base_pos, 0.0)
        print(f"cbf filter on, alpha {cfg.cbf_alpha} margin {cfg.cbf_margin}")
        if cfg.cbf_chunk:
            am = torch.as_tensor(act_mean[:7], dtype=torch.float32, device=device)
            asd = torch.as_tensor(act_std[:7], dtype=torch.float32, device=device)

            def project(acts):
                assert cbf is not None
                q0 = last_obs[:, :7].to(device)
                tgt = acts[..., :7] * asd + am
                acts = acts.clone()
                acts[..., :7] = (cbf.project_chunk(q0, tgt) - am) / asd
                return acts

            policy.chunk_fn = project

    arm = hasattr(env, "ee_state_index")  # franka: planned path needs EE FK
    if arm:
        chain = build_franka_chain(device)[0]
        base_pos = np.asarray(env.robot_base_pos, np.float32)  # base frame -> world

    n = cfg.env.world_count
    frames = (
        round(cfg.episode_seconds * cfg.env.fps)
        if cfg.episode_seconds > 0
        else getattr(env, "max_frames", int(1.5 * env.episode_frames))
    )
    writer = None
    clip: list = []
    ei = getattr(env, "ee_state_index", 0)
    total_succ = total_fail = total_task = total = 0
    stages = getattr(env, "STAGES", None)  # franka pick-place stage ladder
    nz = len(stages) if stages else 0
    stage_hist = np.zeros(nz, int)
    stuck_hist = {k: np.zeros(nz, int) for k in ("timeout", "failure")}
    contacts = Counter()  # what hit the wall, pooled over batches
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        if sel is not None and hasattr(env, "obstacle_boxes"):
            sel.set_boxes(env.obstacle_boxes())
        if cbf is not None:
            if cfg.cbf_geoms:
                cbf.set_boxes(env.obstacle_own_boxes())
            else:
                cbf.set_obstacles(env.obstacle_points())
            h_min = np.full(n, np.inf)
            cbf_active = np.zeros(n)
            for _ in range(cfg.cbf_init):
                obs0 = torch.from_numpy(env.get_obs())
                q0 = obs0[:, :7].to(device)
                if bool((cbf.value(q0) >= cfg.cbf_margin).all()):
                    break
                dq0, _ = cbf.filter(q0, torch.zeros_like(q0))
                a0 = np.concatenate([(q0 + dq0).cpu().numpy(), obs0[:, 7:8].numpy()], 1)
                env.apply_action(a0.astype(np.float32))
        if cfg.cond is Cond.RANDOM and bend is not None:
            policy.cond = sample_cond(bend, n, rng, device)
        if cfg.cond is Cond.SEARCH and bend is not None and not cfg.resample:
            policy.cond_candidates = sample_cond(bend, cfg.n_cond, rng, device)
        succ = np.zeros(n, dtype=bool)
        max_stage = np.zeros(n, int)
        frame = 0
        pbar = tqdm(total=frames, desc=f"batch {ep}", leave=False)
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = torch.from_numpy(env.get_obs())
                last_obs = obs
                action = policy.select_action(preprocessor({OBS_STATE: obs}))
                phys = postprocessor(action).numpy().astype(np.float32)
                if cbf is not None:
                    q = obs[:, :7].to(device)
                    nom = torch.as_tensor(phys[:, :7], device=device) - q
                    attach = None
                    if cfg.cbf_hold > 0:
                        held = torch.as_tensor(obs[:, 18:21], device=device)
                        near = (held - obs[:, ei : ei + 3].to(device)).norm(
                            dim=1
                        ) < 0.12
                        closed = torch.as_tensor(phys[:, 7] < 0.02, device=device)
                        off = cbf.attach_offset(q, held)
                        far = torch.full_like(off, 1e3)
                        attach = (
                            torch.where((near & closed)[:, None], off, far),
                            cfg.cbf_hold,
                        )
                    dq, h = cbf.filter(q, nom, attach)
                    phys[:, :7] = (q + dq).cpu().numpy()
                    h_min = np.minimum(h_min, h.cpu().numpy())
                    cbf_active += ((dq - nom).abs().amax(1) > 1e-6).cpu().numpy()
                env.apply_action(phys)
                if live and arm and policy.last_chunk is not None:
                    chunk = policy.last_chunk[0].cpu().numpy() * act_std + act_mean
                    q = torch.from_numpy(chunk[:, :7]).float().to(device)
                    env.set_predicted_path(
                        ee_positions(chain, q).cpu().numpy() + base_pos
                    )
                frame += 1
                pbar.update(1)
                succ |= env.success()
                if stages is not None:
                    max_stage = np.maximum(max_stage, env.stage())
                if not cfg.video and succ.all():
                    break
            if live:
                env.render()
            if cfg.video:
                f = env.get_frame() if own_frames else viewer.get_frame().numpy()
                f = np.ascontiguousarray(f)
                tag = f"ep {ep} t {frame}"
                if cbf is not None:
                    tag += f" h {h_min[0]:+.3f}"
                if env.failure()[0]:
                    tag += f" COLLIDED {env.contact_labels()[0]}"
                elif succ[0]:
                    tag += " success"
                cv2.putText(
                    f, tag, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        cfg.video,
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        cfg.env.fps,
                        (f.shape[1], f.shape[0]),
                    )
                clip.append(f[..., ::-1])
        pbar.close()
        if not viewer.is_running():
            break
        fail = env.failure()
        if writer is not None and (not cfg.video_failures or not succ[0] or fail[0]):
            for f in clip:
                writer.write(f)
        clip = []
        total_task += int(succ.sum())
        succ &= ~fail
        stuck = ~succ & ~fail
        total_succ += int(succ.sum())
        total_fail += int(fail.sum())
        total += n
        print(
            f"batch {ep}: success {succ.mean():.3f} "
            f"failure {fail.mean():.3f} timeout {stuck.mean():.3f}"
        )
        if cbf is not None:
            print(
                f"cbf: min h {h_min.round(3)} active frac "
                f"{(cbf_active / max(frame, 1)).round(2)} fail {fail.astype(int)}"
            )
        if hasattr(env, "contact_labels") and fail.any():
            contacts.update(env.contact_labels()[fail].tolist())
        if stages is not None:
            stage_hist += np.bincount(max_stage, minlength=nz)
            for name, m in (("timeout", stuck), ("failure", fail)):
                if m.any():
                    stuck_hist[name] += np.bincount(max_stage[m], minlength=nz)

    if total:
        print(
            f"overall: success {total_succ / total:.3f} "
            f"failure {total_fail / total:.3f} "
            f"timeout {(total - total_succ - total_fail) / total:.3f} "
            f"({total} episodes, cond {cfg.cond})"
        )
        if total_fail:
            print(
                f"overall task success incl. collisions {total_task / total:.3f} "
                f"collision avoidance {1 - total_fail / total:.3f}"
            )
    if stages is not None and total:
        reward = sum(i * c for i, c in enumerate(stage_hist)) / total
        print(f"overall reward (avg furthest stage reached): {reward:.2f}")
        print("overall stage (furthest reached, fraction of episodes):")
        print(
            "  "
            + "  ".join(
                f"{name} {c / total:.3f}" for name, c in zip(stages, stage_hist)
            )
        )
        for k, h in stuck_hist.items():
            if h.sum():
                print(
                    f"overall stage|{k} (n={h.sum()}): "
                    + "  ".join(f"{s} {v / h.sum():.3f}" for s, v in zip(stages, h))
                )
    if contacts:
        hits = "  ".join(f"{k}:{v}" for k, v in contacts.most_common())
        print(f"wall contacts (n={sum(contacts.values())}): {hits}")
    lc = policy.latched_cond
    if searching and lc is not None:
        lc = lc.cpu().numpy()
        print(f"latched cond mean {lc.mean(0).round(3)} std {lc.std(0).round(3)}")
    if writer is not None:
        writer.release()
        print(f"Saved video to {cfg.video}")
    viewer.close()


if __name__ == "__main__":
    main()
