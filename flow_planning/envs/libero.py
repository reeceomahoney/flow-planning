import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flow_planning.envs.env import EnvConfig
from flow_planning.utils import quat_to_rot6d

HF_DEMOS = "yifengzhu-hf/LIBERO-datasets"
HF_SAFE = "THURCSCT/SafeLIBERO"
SAFE_ROOT = "safelibero/libero/libero"

TASKS = {
    "safelibero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "safelibero_object": [
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    ],
    "safelibero_goal": [
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_cream_cheese_in_the_bowl",
    ],
    "safelibero_long": [
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    ],
}
SOURCE_SUITE = {
    "safelibero_spatial": "libero_spatial",
    "safelibero_object": "libero_object",
    "safelibero_goal": "libero_goal",
    "safelibero_long": "libero_10",
}
MAX_STEPS = {
    "safelibero_spatial": 220,
    "safelibero_object": 280,
    "safelibero_goal": 300,
    "safelibero_long": 520,
}
DUMMY_ACTION = np.array([0.0] * 7 + [-1.0], np.float32)
WORKSPACE = 0.5
GRIPPER_OPEN = 0.04


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoConfig(EnvConfig):
    suite: str = "safelibero_goal"
    task_id: int = 0
    level: str = "I"
    world_count: int = 5
    fps: int = 20
    goal_dim: int = 0
    n_action_steps: int = 30
    kp: float = 400.0
    delta_max: float = 0.2
    settle_steps: int = 10
    collision_displacement: float = 0.001
    render: bool = False
    render_size: int = 256


def task_name(cfg: LiberoConfig) -> str:
    names = TASKS[cfg.suite]
    i = cfg.task_id
    if cfg.suite == "safelibero_goal" and cfg.level == "II" and i == 3:
        i = 4
    return names[i]


def libero_package_root() -> Path:
    spec = importlib.util.find_spec("libero")
    assert spec is not None and spec.submodule_search_locations
    return Path(list(spec.submodule_search_locations)[0]) / "libero"


def ensure_libero_config():
    cfg_dir = Path(
        os.environ.setdefault(
            "LIBERO_CONFIG_PATH", str(Path.home() / ".cache/flow_planning/libero")
        )
    )
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return
    root = libero_package_root()
    paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
        "assets": root / "assets",
    }
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))


def patch_robosuite():
    import mujoco
    import robosuite.controllers.base_controller as bc

    if getattr(bc, "flow_planning_patched", False):
        return

    class Shim:
        data = None

        def __getattr__(self, k):
            return getattr(mujoco, k)

        def mj_fullM(self, m, dst, qM):
            mujoco.mj_fullM(m, self.data, dst)

    shim = Shim()
    orig = bc.Controller.update

    def update(self, force=False):
        shim.data = self.sim.data._data
        return orig(self, force)

    setattr(bc, "mujoco", shim)
    bc.Controller.update = update
    setattr(bc, "flow_planning_patched", True)


def safelibero_root() -> Path:
    from huggingface_hub import snapshot_download

    p = snapshot_download(
        HF_SAFE,
        repo_type="dataset",
        allow_patterns=[
            f"{SAFE_ROOT}/assets/obstacle_objects/*",
            f"{SAFE_ROOT}/bddl_files/safelibero_*/*",
            f"{SAFE_ROOT}/init_files/safelibero_*/*",
        ],
    )
    return Path(p) / SAFE_ROOT


def register_obstacles(root: Path):
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object
    from robosuite.models.objects import MujocoXMLObject

    for d in sorted((root / "assets/obstacle_objects").iterdir()):
        snake = d.name
        if snake in OBJECTS_DICT:
            continue
        camel = "".join(p.capitalize() for p in snake.split("_"))
        xml = str(d / f"{snake}.xml")

        def init(self, name=snake, xml=xml, snake=snake):
            MujocoXMLObject.__init__(
                self,
                xml,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = snake
            self.rotation = (np.pi / 2, np.pi / 2)
            self.rotation_axis = "x"
            self.object_properties = {"vis_site_names": {}}

        register_object(type(camel, (MujocoXMLObject,), {"__init__": init}))


def disable_robosuite_file_log():
    spec = importlib.util.find_spec("robosuite")
    assert spec is not None and spec.submodule_search_locations
    f = Path(list(spec.submodule_search_locations)[0]) / "macros_private.py"
    if not f.exists():
        f.write_text("FILE_LOGGING_LEVEL = None\n")


def libero_setup() -> Path:
    ensure_libero_config()
    disable_robosuite_file_log()
    patch_robosuite()
    root = safelibero_root()
    register_obstacles(root)
    return root


def task_files(cfg: LiberoConfig, safe_root: Path) -> tuple[Path, Path]:
    from libero.libero import get_libero_path

    name = task_name(cfg)
    if cfg.obstacle:
        bddl = safe_root / "bddl_files" / cfg.suite / f"{name}.bddl"
        init = (
            safe_root
            / "init_files"
            / cfg.suite
            / f"{name}_level_{cfg.level}.pruned_init"
        )
    else:
        src = SOURCE_SUITE[cfg.suite]
        bddl = Path(get_libero_path("bddl_files")) / src / f"{name}.bddl"
        init = Path(get_libero_path("init_states")) / src / f"{name}.pruned_init"
    return bddl, init


def geom_boxes(e, name: str) -> np.ndarray:
    m, d = e.sim.model._model, e.sim.data._data
    b = m.body(f"{name}_main").id
    signs = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T
    boxes = []
    for g in range(m.ngeom):
        if m.body_rootid[m.geom_bodyid[g]] != b:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        local = m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]
        pts = d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T
        lo, hi = pts.min(0), pts.max(0)
        boxes.append(np.concatenate([(lo + hi) / 2, (hi - lo) / 2]))
    return np.stack(boxes)


def union_box(boxes: np.ndarray) -> np.ndarray:
    lo = (boxes[:, :3] - boxes[:, 3:]).min(0)
    hi = (boxes[:, :3] + boxes[:, 3:]).max(0)
    return np.concatenate([(lo + hi) / 2, (hi - lo) / 2])


def body_aabb(e, name: str) -> np.ndarray:
    return union_box(geom_boxes(e, name))


def obstacle_geom_boxes(e, name: str) -> np.ndarray:
    boxes = geom_boxes(e, name)
    ob = union_box(boxes)
    for other in e.env.objects_dict:
        if "base" not in other or other == name:
            continue
        bb = body_aabb(e, other)
        if np.all(np.abs(bb[:2] - ob[:2]) < bb[3:5] + ob[3:5]):
            boxes = np.concatenate([boxes, geom_boxes(e, other)])
    return boxes


def demo_to_episode(env, g) -> tuple[np.ndarray, np.ndarray]:
    states, acts = g["states"][:], g["actions"][:]
    env.envs[0].reset()
    obs = []
    for s in states:
        env.set_state(0, s)
        obs.append(env.get_obs()[0])
    obs = np.stack(obs)
    joints = np.concatenate([obs[1:, :7], obs[-1:, :7]])
    grip = GRIPPER_OPEN * (acts[:, 6:7] < 0)
    return obs, np.concatenate([joints, grip], axis=1).astype(np.float32)


class LiberoEnv:
    ee_state_index = 9
    cube_size = 0.0
    cube_index = None

    def __init__(self, cfg: LiberoConfig, viewer=None):
        import mujoco

        if cfg.render:
            os.environ.setdefault("MUJOCO_GL", "egl")
        safe_root = libero_setup()

        self.cfg = cfg
        self.viewer = viewer
        self.rng = np.random.default_rng(0)
        self.name = task_name(cfg)
        self.language = " ".join(
            re.sub(r"^[A-Z_0-9]+SCENE\d+_", "", self.name).split("_")
        )
        bddl, init = task_files(cfg, safe_root)
        self.init_states = np.asarray(torch.load(init, weights_only=False))
        self.envs = [self.make_env(cfg, bddl, i) for i in range(cfg.world_count)]
        for e in self.envs:
            e.seed(0)
            c = e.env.robot_configs[0]["controller_config"]
            c["kp"] = cfg.kp
            c["output_max"] = cfg.delta_max
            c["output_min"] = -cfg.delta_max
        sim = self.envs[0].sim
        self.robot_base_pos = sim.data.get_body_xpos("robot0_base").copy()
        self.objects = sorted(self.envs[0].obj_of_interest)
        m = sim.model._model
        joints = [
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_qposadr[j])
            for j in range(m.njnt)
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.fixture_qpos = [
            a for n, a in sorted(joints) if not n.startswith(("robot0", "gripper0"))
        ]
        self.max_frames = MAX_STEPS[cfg.suite]
        self.episode_frames = self.max_frames
        self.episode_idx = 0
        self.reset()
        self.objects = [n for n in self.objects if f"{n}_pos" in self.obs[0]]
        self.episode_idx = 0

    def make_env(self, cfg: LiberoConfig, bddl: Path, i: int):
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.errors import RandomizationError

        for attempt in range(10):
            try:
                return ControlEnv(
                    bddl_file_name=str(bddl),
                    controller="JOINT_POSITION",
                    use_camera_obs=cfg.render and i == 0,
                    has_offscreen_renderer=cfg.render and i == 0,
                    has_renderer=False,
                    camera_names=["agentview"] if cfg.render and i == 0 else [],
                    camera_heights=cfg.render_size,
                    camera_widths=cfg.render_size,
                )
            except RandomizationError:
                np.random.seed(attempt + 1)
        raise RuntimeError(f"could not build {bddl}")

    def reset(self):
        n = len(self.envs)
        self.obs: list = [None] * n
        self.obstacle: list = [None] * n
        self.obstacle_pos0 = np.zeros((n, 3), np.float32)
        self.collided = np.zeros(n, dtype=bool)
        self.succ = np.zeros(n, dtype=bool)
        self.contact = [""] * n
        self.step = 0
        self.path = None
        self.trace: list = []
        for i, e in enumerate(self.envs):
            e.reset()
            s = self.init_states[self.episode_idx % len(self.init_states)]
            self.episode_idx += 1
            self.obs[i] = e.set_init_state(s)
            for _ in range(self.cfg.settle_steps):
                self.obs[i] = e.step(DUMMY_ACTION)[0]
            if not self.cfg.obstacle:
                continue
            for name in e.env.objects_dict:
                p = self.obs[i].get(f"{name}_pos")
                if "obstacle" not in name or p is None:
                    continue
                if p[2] > 0 and abs(p[0]) < WORKSPACE and abs(p[1]) < WORKSPACE:
                    self.obstacle[i] = name
                    self.obstacle_pos0[i] = p
                    break

    def set_state(self, i: int, state):
        self.obs[i] = self.envs[i].regenerate_obs_from_state(state)

    def get_obs(self):
        rows = []
        for i, e in enumerate(self.envs):
            o = self.obs[i]
            assert o is not None
            quats = np.stack(
                [o["robot0_eef_quat"]] + [o[f"{n}_quat"] for n in self.objects]
            )
            r6 = quat_to_rot6d(quats)
            parts = [
                o["robot0_joint_pos"],
                o["robot0_gripper_qpos"],
                o["robot0_eef_pos"],
                r6[0],
            ]
            for k, n in enumerate(self.objects):
                parts += [o[f"{n}_pos"], r6[k + 1]]
            parts.append(e.sim.data.qpos[self.fixture_qpos])
            rows.append(np.concatenate(parts))
        return np.stack(rows).astype(np.float32)

    def apply_action(self, action):
        n = len(self.envs)
        a = np.asarray(action, np.float32).reshape(n, 8)
        self.step += 1
        for i, e in enumerate(self.envs):
            q = e.robots[0]._joint_positions
            delta = np.clip((a[i, :7] - q) / self.cfg.delta_max, -1.0, 1.0)
            grip = 1.0 if a[i, 7] < 0.5 * GRIPPER_OPEN else -1.0
            self.obs[i] = e.step(np.concatenate([delta, [grip]]))[0]
            self.succ[i] |= bool(e.check_success())
            name = self.obstacle[i]
            if name is not None and not self.collided[i]:
                moved = np.abs(self.obs[i][f"{name}_pos"] - self.obstacle_pos0[i]).sum()
                self.collided[i] = moved > self.cfg.collision_displacement
                if self.collided[i]:
                    self.contact[i] = self.contact_body(i, name, grip > 0)

    def contact_body(self, i: int, name: str, grasped: bool) -> str:
        e = self.envs[i]
        m, d = e.sim.model._model, e.sim.data._data
        b = m.body(f"{name}_main").id
        hits = []
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if m.body_rootid[b1] == b and m.body_rootid[b2] != b:
                hits.append(m.body(b2).name)
            elif m.body_rootid[b2] == b and m.body_rootid[b1] != b:
                hits.append(m.body(b1).name)
        skip = ("table", "floor", "main_table", "box_small_base", "box_base")
        hits = [h for h in hits if not h.startswith(skip)]
        other = hits[0] if hits else "none"
        phase = "post" if grasped else "pre"
        return f"{name}/{other}/{phase}-grasp/t{self.step // 20 * 20}"

    def success(self):
        return self.succ.copy()

    def failure(self):
        return self.collided.copy()

    def contact_labels(self):
        return np.array(self.contact, dtype=object)

    def goals(self):
        return [g for g in self.envs[0].env.parsed_problem["goal_state"] if len(g) == 3]

    def obstacle_boxes(self):
        far = np.array([1e3, 1e3, 1e3, 0.01, 0.01, 0.01], np.float32)
        per = [
            obstacle_geom_boxes(e, self.obstacle[i])
            if self.obstacle[i] is not None
            else far[None]
            for i, e in enumerate(self.envs)
        ]
        k = max(len(b) for b in per)
        out = np.tile(far, (len(per), k, 1)).astype(np.float32)
        for i, b in enumerate(per):
            out[i, : len(b)] = b
        return out

    @property
    def obstacle_geometry(self):
        box = union_box(self.obstacle_boxes()[0])
        return {"center": box[:3].tolist(), "half_extents": box[3:].tolist()}

    def project(self, pts):
        from robosuite.utils.camera_utils import (
            get_camera_transform_matrix,
            project_points_from_world_to_camera,
        )

        n = self.cfg.render_size
        mat = get_camera_transform_matrix(self.envs[0].sim, "agentview", n, n)
        px = project_points_from_world_to_camera(np.asarray(pts), mat, n, n)
        return np.stack([n - 1 - px[..., 1], px[..., 0]], -1)

    def get_frame(self):
        import cv2

        f = np.ascontiguousarray(self.obs[0]["agentview_image"][::-1, ::-1])
        self.trace.append(self.obs[0]["robot0_eef_pos"].copy())
        for pts, colour, r in (
            (self.path, (0, 220, 0), 2),
            (np.stack(self.trace), (255, 200, 0), 2),
        ):
            if pts is not None and len(pts):
                for u, v in self.project(pts):
                    cv2.circle(f, (int(u), int(v)), r, colour, -1)
        return f

    def set_predicted_path(self, positions):
        self.path = np.asarray(positions)

    def render(self):
        pass
