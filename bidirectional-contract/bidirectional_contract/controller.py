"""Predictive safety controller constrained to the trained recovery support."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from flow_planning.bend import ARM, EE, ROT, fk, grasp_segments, held_object, obj_slice
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision, box_sdf
from flow_planning.utils import hf_column

from .interventions import (
    local_normal_frame,
    normal_direction,
    ramp_profile,
    solve_recovery_ik,
)


@dataclass
class ControllerDecision:
    action_chunk: np.ndarray
    eef_path: np.ndarray
    intervened: bool
    feasible: bool
    reason: str
    reference_demo: int
    reference_frame: int
    anchor: int | None
    direction: int | None
    target_radius_m: float
    current_radius_m: float
    raw_clearance_m: float
    raw_prefix_clearance_m: float
    selected_clearance_m: float
    first_unsafe_step: int | None
    candidate_count: int

    def json(self) -> dict:
        return {
            "intervened": self.intervened,
            "feasible": self.feasible,
            "reason": self.reason,
            "reference_demo": self.reference_demo,
            "reference_frame": self.reference_frame,
            "anchor": self.anchor,
            "direction": self.direction,
            "target_radius_m": self.target_radius_m,
            "current_radius_m": self.current_radius_m,
            "raw_clearance_m": self.raw_clearance_m,
            "raw_prefix_clearance_m": self.raw_prefix_clearance_m,
            "selected_clearance_m": self.selected_clearance_m,
            "first_unsafe_step": self.first_unsafe_step,
            "candidate_count": self.candidate_count,
        }


class RecoveryBankController:
    """Discrete normal-ray selector implementing the bidirectional contract.

    The controller may move outward only through cells accepted into the
    recovery bank. It reasons over H actions, executes s, keeps its branch and
    last safe tail, and releases once the recovery policy is safe on its own.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        recovery_bank: Path,
        base_pos: np.ndarray,
        device: str = "cuda",
        horizon: int = 50,
        execution_horizon: int = 10,
        safety_margin: float = 0.01,
        anchor_tolerance: int = 12,
        release_radius: float = 0.015,
        held_object_radius: float = 0.055,
        ik_iters: int = 15,
        ik_damping: float = 0.05,
        posture_gain: float = 0.1,
        ik_tolerance: float = 0.015,
        max_joint_step: float = 0.18,
        direction_count: int = 5,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.base = np.asarray(base_pos, np.float32)
        self.horizon = horizon
        self.execution_horizon = execution_horizon
        self.safety_margin = safety_margin
        self.anchor_tolerance = anchor_tolerance
        self.release_radius = release_radius
        self.held_object_radius = held_object_radius
        self.ik_iters = ik_iters
        self.ik_damping = ik_damping
        self.posture_gain = posture_gain
        self.ik_tolerance = ik_tolerance
        self.max_joint_step = max_joint_step
        self.direction_count = direction_count

        hf = dataset.hf_dataset
        observation = np.asarray(
            hf_column(hf, "observation.state"), dtype=np.float32
        )
        action = np.asarray(hf_column(hf, "action"), dtype=np.float32)
        episode = np.asarray(hf_column(hf, "episode_index"), dtype=np.int64)
        object_count = max(0, (observation.shape[1] - 20) // 9)
        self.demo_obs: dict[int, np.ndarray] = {}
        self.held_object: dict[int, int | None] = {}
        for demo_id in np.unique(episode):
            selected = episode == demo_id
            demo = int(demo_id)
            self.demo_obs[demo] = observation[selected]
            segments = grasp_segments(action[selected, 7])
            held = None
            if segments:
                close, opened = max(
                    segments, key=lambda segment: segment[1] - segment[0]
                )
                held = held_object(
                    observation[selected], close, opened, object_count
                )
            self.held_object[demo] = held

        root_chain, _, _ = build_franka_chain("cpu")
        self.chain = pk.SerialChain(root_chain, EE_FRAME).to(
            dtype=torch.float32, device=self.device
        )
        self.reference_fk: dict[int, np.ndarray] = {}
        self.reference_side: dict[int, np.ndarray] = {}
        self.reference_vertical: dict[int, np.ndarray] = {}
        for demo_id, demo_obs in self.demo_obs.items():
            position, _ = fk(self.chain, demo_obs[:, ARM], self.base)
            side, vertical = local_normal_frame(position)
            self.reference_fk[demo_id] = position
            self.reference_side[demo_id] = side
            self.reference_vertical[demo_id] = vertical

        with np.load(Path(recovery_bank).resolve()) as bank:
            metadata = bank["metadata"].copy()
        self.support: dict[tuple[int, int, int], list[float]] = {}
        self.anchor_phase: dict[tuple[int, int], int] = {}
        for row in metadata:
            demo_id, anchor, direction = (int(row[index]) for index in (0, 1, 2))
            amplitude, phase = float(row[3]), int(row[5])
            self.support.setdefault((demo_id, anchor, direction), []).append(
                amplitude
            )
            self.anchor_phase[(demo_id, anchor)] = phase
        for key, values in self.support.items():
            self.support[key] = sorted(set(round(value, 6) for value in values))

        scale = observation.std(axis=0).clip(min=1e-3)
        weight = np.zeros(observation.shape[1], np.float32)
        weight[ARM] = 0.12
        weight[7:9] = 2.0
        weight[EE] = 0.4
        weight[ROT] = 0.08
        for start in range(18, observation.shape[1] - 2, 9):
            weight[start : start + 3] = 1.0
            weight[start + 3 : start + 9] = 0.08
        weight[-2:] = 0.2
        self.match_scale = scale
        self.match_weight = weight
        self.collision = FrankaCollision(self.device, self.base, cube_size=0.0)
        self.reset()

    def reset(self):
        self.active_direction: int | None = None
        self.active_anchor: int | None = None
        self.active_radius = 0.0
        self.cached_action_chunk: np.ndarray | None = None
        self.matched_demo: int | None = None
        self.matched_frame: int | None = None

    def _distance(self, reference: np.ndarray, state: np.ndarray) -> np.ndarray:
        squared = ((reference - state) / self.match_scale) ** 2
        return (squared * self.match_weight).sum(axis=1) / self.match_weight.sum()

    def match_phase(self, state: np.ndarray) -> tuple[int, int]:
        closed = float(np.mean(state[7:9])) < 0.025
        best = (float("inf"), 0, 0)
        items = (
            self.demo_obs.items()
            if self.matched_demo is None
            else [(self.matched_demo, self.demo_obs[self.matched_demo])]
        )
        for demo_id, reference in items:
            valid = (reference[:, 7:9].mean(axis=1) < 0.025) == closed
            if self.matched_demo == demo_id and self.matched_frame is not None:
                time = np.arange(len(reference))
                valid &= time >= self.matched_frame
                valid &= time <= min(len(reference) - 1, self.matched_frame + 30)
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            distance = self._distance(reference[indices], state)
            local = int(distance.argmin())
            candidate = (float(distance[local]), int(demo_id), int(indices[local]))
            if candidate < best:
                best = candidate
        _, self.matched_demo, self.matched_frame = best
        return self.matched_demo, self.matched_frame

    def nearest_supported_anchor(
        self, demo_id: int, frame: int, phase: int
    ) -> int | None:
        anchors = [
            anchor
            for (candidate_demo, anchor), candidate_phase in self.anchor_phase.items()
            if candidate_demo == demo_id and candidate_phase == phase
        ]
        if not anchors:
            return None
        upcoming = [anchor for anchor in anchors if anchor >= frame]
        if upcoming and min(upcoming) - frame <= self.anchor_tolerance:
            anchor = min(upcoming)
        else:
            anchor = min(anchors, key=lambda value: abs(value - frame))
        return anchor if abs(anchor - frame) <= self.anchor_tolerance else None

    def direction(
        self, demo_id: int, indices: np.ndarray, direction_index: int
    ) -> np.ndarray:
        return normal_direction(
            self.reference_side[demo_id][indices],
            self.reference_vertical[demo_id][indices],
            direction_index,
            self.direction_count,
        )

    def _clearance_path(
        self,
        joints: np.ndarray,
        boxes: np.ndarray,
        object_positions: np.ndarray | None,
    ) -> np.ndarray:
        if joints.ndim == 2:
            joints = joints[None]
        batch, steps = joints.shape[:2]
        q = torch.as_tensor(
            joints.reshape(-1, 7), dtype=torch.float32, device=self.device
        )
        points = self.collision.arm_points(q) + self.collision.base_pos
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        distance = box_sdf(
            points[:, None, :, :],
            boxes_t[None, :, None, :3],
            boxes_t[None, :, None, 3:],
        )
        clearance = (
            distance.amin(dim=(1, 2)) - self.collision.radius
        ).reshape(batch, steps)
        if object_positions is not None:
            if object_positions.ndim == 2:
                object_positions = object_positions[None]
            obj = torch.as_tensor(
                object_positions.reshape(-1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            object_clearance = box_sdf(
                obj[:, None, :], boxes_t[None, :, :3], boxes_t[None, :, 3:]
            ).amin(dim=1)
            object_clearance = (
                object_clearance.reshape(batch, steps) - self.held_object_radius
            )
            clearance = torch.minimum(clearance, object_clearance)
        return clearance.cpu().numpy()

    def _object_path(
        self,
        state: np.ndarray,
        eef_path: np.ndarray,
        quaternion_path: np.ndarray,
        carrying: bool,
        held_index: int | None,
        current_eef: np.ndarray,
        current_quaternion: np.ndarray,
    ) -> np.ndarray | None:
        if not carrying or held_index is None:
            return None
        relative_world = state[obj_slice(held_index)] - current_eef
        relative_local = Rotation.from_quat(current_quaternion).inv().apply(
            relative_world
        )
        relative_path = Rotation.from_quat(quaternion_path).apply(
            np.broadcast_to(relative_local, eef_path.shape)
        )
        return eef_path + relative_path

    def _path_clearance(
        self,
        state: np.ndarray,
        joint_path: np.ndarray,
        boxes: np.ndarray,
        carrying: bool,
        held_index: int | None,
        current_eef: np.ndarray,
        current_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eef, quaternion = fk(self.chain, joint_path, self.base)
        object_path = self._object_path(
            state,
            eef,
            quaternion,
            carrying,
            held_index,
            current_eef,
            current_quaternion,
        )
        return eef, quaternion, self._clearance_path(
            joint_path, boxes, object_path
        )[0]

    def _shifted_cached_plan(
        self,
        state: np.ndarray,
        boxes: np.ndarray,
        carrying: bool,
        held_index: int | None,
        current_eef: np.ndarray,
        current_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if self.cached_action_chunk is None:
            return None
        tail = self.cached_action_chunk[self.execution_horizon :]
        padding = np.repeat(
            self.cached_action_chunk[-1:], self.execution_horizon, axis=0
        )
        chunk = np.concatenate([tail, padding], axis=0)
        if np.abs(chunk[0, ARM] - state[ARM]).max() > self.max_joint_step:
            return None
        eef, _, clearance = self._path_clearance(
            state,
            chunk[:, ARM],
            boxes,
            carrying,
            held_index,
            current_eef,
            current_quaternion,
        )
        minimum = float(clearance.min())
        if minimum < self.safety_margin:
            return None
        return chunk, eef, minimum

    def _decision(
        self,
        chunk: np.ndarray,
        eef_path: np.ndarray,
        *,
        reason: str,
        demo_id: int,
        frame: int,
        raw_clearance: float,
        raw_prefix_clearance: float,
        selected_clearance: float,
        first_unsafe_step: int | None,
        intervened: bool,
        feasible: bool,
        anchor: int | None,
        direction: int | None,
        target_radius: float,
        current_radius: float,
        candidate_count: int,
        advance_reference: bool = False,
    ) -> ControllerDecision:
        decision = ControllerDecision(
            action_chunk=chunk,
            eef_path=eef_path,
            intervened=intervened,
            feasible=feasible,
            reason=reason,
            reference_demo=demo_id,
            reference_frame=frame,
            anchor=anchor,
            direction=direction,
            target_radius_m=target_radius,
            current_radius_m=current_radius,
            raw_clearance_m=raw_clearance,
            raw_prefix_clearance_m=raw_prefix_clearance,
            selected_clearance_m=selected_clearance,
            first_unsafe_step=first_unsafe_step,
            candidate_count=candidate_count,
        )
        if advance_reference:
            self.matched_frame = min(
                frame + self.execution_horizon,
                len(self.demo_obs[demo_id]) - 1,
            )
        return decision

    def _hold_decision(
        self,
        state: np.ndarray,
        policy_chunk: np.ndarray,
        boxes: np.ndarray,
        current_eef: np.ndarray,
        current_quaternion: np.ndarray,
        carrying: bool,
        held_index: int | None,
        *,
        reason: str,
        demo_id: int,
        frame: int,
        anchor: int | None,
        current_radius: float,
        raw_clearance: float,
        raw_prefix_clearance: float,
        first_unsafe_step: int | None,
        candidate_count: int,
    ) -> ControllerDecision:
        hold = policy_chunk.copy()
        hold[:, ARM] = state[ARM]
        hold_eef = np.repeat(current_eef[None], self.horizon, axis=0)
        hold_quaternion = np.repeat(
            current_quaternion[None], self.horizon, axis=0
        )
        hold_object = self._object_path(
            state,
            hold_eef,
            hold_quaternion,
            carrying,
            held_index,
            current_eef,
            current_quaternion,
        )
        hold_clearance = float(
            self._clearance_path(hold[:, ARM], boxes, hold_object).min()
        )
        return self._decision(
            hold,
            hold_eef,
            reason=reason,
            demo_id=demo_id,
            frame=frame,
            raw_clearance=raw_clearance,
            raw_prefix_clearance=raw_prefix_clearance,
            selected_clearance=hold_clearance,
            first_unsafe_step=first_unsafe_step,
            intervened=True,
            feasible=False,
            anchor=anchor,
            direction=self.active_direction,
            target_radius=0.0,
            current_radius=current_radius,
            candidate_count=candidate_count,
        )

    @torch.no_grad()
    def plan(
        self,
        state: np.ndarray,
        policy_chunk: np.ndarray,
        obstacle_boxes: np.ndarray,
    ) -> ControllerDecision:
        state = np.asarray(state, np.float32)
        policy_chunk = np.asarray(policy_chunk, np.float32)
        if policy_chunk.shape != (self.horizon, 8):
            raise ValueError(
                f"Expected policy chunk {(self.horizon, 8)}, got {policy_chunk.shape}"
            )
        boxes = np.asarray(obstacle_boxes, np.float32).reshape(-1, 6)
        demo_id, frame = self.match_phase(state)
        reference = self.demo_obs[demo_id]
        carrying = float(np.mean(state[7:9])) < 0.025
        phase = int(carrying)
        held_index = self.held_object[demo_id]
        indices = np.minimum(
            frame + np.arange(1, self.horizon + 1), len(reference) - 1
        )
        current_eef_path, current_quaternion_path = fk(
            self.chain, state[ARM][None], self.base
        )
        current_eef = current_eef_path[0]
        current_quaternion = current_quaternion_path[0]

        raw_q = policy_chunk[:, ARM]
        raw_eef, raw_quaternion, raw_clearance_path = self._path_clearance(
            state,
            raw_q,
            boxes,
            carrying,
            held_index,
            current_eef,
            current_quaternion,
        )
        raw_clearance = float(raw_clearance_path.min())
        raw_prefix_clearance = float(
            raw_clearance_path[: self.execution_horizon].min()
        )
        unsafe = np.flatnonzero(raw_clearance_path < self.safety_margin)
        first_unsafe_step = int(unsafe[0] + 1) if len(unsafe) else None
        current_radius = 0.0
        if self.active_direction is not None:
            direction_now = self.direction(
                demo_id, np.asarray([frame]), self.active_direction
            )[0]
            current_radius = float(
                np.dot(
                    current_eef - self.reference_fk[demo_id][frame], direction_now
                )
            )

        common = {
            "demo_id": demo_id,
            "frame": frame,
            "raw_clearance": raw_clearance,
            "raw_prefix_clearance": raw_prefix_clearance,
            "first_unsafe_step": first_unsafe_step,
        }
        if raw_clearance >= self.safety_margin:
            self.cached_action_chunk = None
            reason = "raw_policy_safe"
            if self.active_direction is not None:
                reason = "released_to_recovery_policy"
                if current_radius <= self.release_radius:
                    self.active_direction = None
                    self.active_anchor = None
                    self.active_radius = 0.0
                    reason = "recovery_complete"
            return self._decision(
                policy_chunk,
                raw_eef,
                reason=reason,
                selected_clearance=raw_clearance,
                intervened=False,
                feasible=True,
                anchor=None,
                direction=self.active_direction,
                target_radius=0.0,
                current_radius=current_radius,
                candidate_count=0,
                **common,
            )

        cached = self._shifted_cached_plan(
            state,
            boxes,
            carrying,
            held_index,
            current_eef,
            current_quaternion,
        )
        anchor = self.nearest_supported_anchor(demo_id, frame, phase)
        if anchor is None:
            if cached is not None:
                chunk, eef, clearance = cached
                self.cached_action_chunk = chunk
                return self._decision(
                    chunk,
                    eef,
                    reason="latched_safe_plan",
                    selected_clearance=clearance,
                    intervened=True,
                    feasible=True,
                    anchor=self.active_anchor,
                    direction=self.active_direction,
                    target_radius=self.active_radius,
                    current_radius=current_radius,
                    candidate_count=0,
                    advance_reference=True,
                    **common,
                )
            if raw_prefix_clearance >= self.safety_margin:
                return self._decision(
                    policy_chunk,
                    raw_eef,
                    reason="safe_prefix_toward_supported_anchor",
                    selected_clearance=raw_prefix_clearance,
                    intervened=False,
                    feasible=True,
                    anchor=None,
                    direction=self.active_direction,
                    target_radius=0.0,
                    current_radius=current_radius,
                    candidate_count=0,
                    **common,
                )
            return self._hold_decision(
                state,
                policy_chunk,
                boxes,
                current_eef,
                current_quaternion,
                carrying,
                held_index,
                reason="no_supported_anchor_hold",
                anchor=None,
                current_radius=current_radius,
                candidate_count=0,
                **common,
            )

        directions = list(range(self.direction_count))
        if self.active_direction is not None:
            directions.remove(self.active_direction)
            directions.insert(0, self.active_direction)
        blend = ramp_profile(self.horizon, self.execution_horizon)
        reference_position = self.reference_fk[demo_id][indices]
        candidates: list[tuple[int, float, np.ndarray]] = []
        for direction_index in directions:
            direction_path = self.direction(demo_id, indices, direction_index)
            direction_now = self.direction(
                demo_id, np.asarray([frame]), direction_index
            )[0]
            deviation = current_eef - self.reference_fk[demo_id][frame]
            radius_now = float(np.dot(deviation, direction_now))
            residual = deviation - radius_now * direction_now
            for amplitude in self.support.get(
                (demo_id, anchor, direction_index), []
            ):
                if amplitude + 0.01 < max(radius_now, 0.0):
                    continue
                radius = radius_now + blend * (amplitude - radius_now)
                target = (
                    reference_position
                    + radius[:, None] * direction_path
                    + (1.0 - blend[:, None]) * residual
                )
                candidates.append((direction_index, amplitude, target))

        if not candidates:
            if cached is not None:
                chunk, eef, clearance = cached
                self.cached_action_chunk = chunk
                return self._decision(
                    chunk,
                    eef,
                    reason="latched_safe_plan",
                    selected_clearance=clearance,
                    intervened=True,
                    feasible=True,
                    anchor=self.active_anchor,
                    direction=self.active_direction,
                    target_radius=self.active_radius,
                    current_radius=current_radius,
                    candidate_count=0,
                    advance_reference=True,
                    **common,
                )
            return self._hold_decision(
                state,
                policy_chunk,
                boxes,
                current_eef,
                current_quaternion,
                carrying,
                held_index,
                reason="outside_validated_radius_hold",
                anchor=anchor,
                current_radius=current_radius,
                candidate_count=0,
                **common,
            )

        target_position = np.stack([candidate[2] for candidate in candidates], axis=1)
        target_quaternion = np.repeat(
            raw_quaternion[:, None, :], len(candidates), axis=1
        )
        reference_joint = np.repeat(
            raw_q[:, None, :], len(candidates), axis=1
        )
        seed = np.repeat(state[ARM][None], len(candidates), axis=0)
        solved = solve_recovery_ik(
            self.chain,
            target_position,
            target_quaternion,
            reference_joint,
            seed,
            self.ik_iters,
            self.ik_damping,
            self.posture_gain,
            self.base,
        ).transpose(1, 0, 2)
        solved_eef, solved_quaternion = fk(
            self.chain, solved.reshape(-1, 7), self.base
        )
        solved_eef = solved_eef.reshape(len(candidates), self.horizon, 3)
        solved_quaternion = solved_quaternion.reshape(
            len(candidates), self.horizon, 4
        )
        target_bh = target_position.transpose(1, 0, 2)
        ik_error = np.linalg.norm(solved_eef - target_bh, axis=2)
        execution_ik_error = ik_error[:, : self.execution_horizon].max(axis=1)
        previous = np.concatenate(
            [np.repeat(state[None, None, ARM], len(candidates), axis=0), solved],
            axis=1,
        )
        joint_step = np.abs(np.diff(previous, axis=1)).max(axis=2)
        execution_velocity = joint_step[:, : self.execution_horizon].max(axis=1)

        object_path = None
        if carrying and held_index is not None:
            object_path = np.stack(
                [
                    self._object_path(
                        state,
                        solved_eef[index],
                        solved_quaternion[index],
                        carrying,
                        held_index,
                        current_eef,
                        current_quaternion,
                    )
                    for index in range(len(candidates))
                ]
            )
        clearance = self._clearance_path(solved, boxes, object_path).min(axis=1)
        valid = (execution_ik_error <= self.ik_tolerance) & (
            execution_velocity <= self.max_joint_step
        )
        safe = valid & (clearance >= self.safety_margin)

        if safe.any():
            costs = []
            for index in np.flatnonzero(safe):
                direction_index, amplitude, _ = candidates[index]
                branch_switch = int(
                    self.active_direction is not None
                    and direction_index != self.active_direction
                )
                joint_cost = float(np.mean((solved[index] - raw_q) ** 2))
                costs.append(
                    (
                        branch_switch,
                        amplitude,
                        -float(clearance[index]),
                        joint_cost,
                        index,
                    )
                )
            chosen = min(costs)[-1]
            direction_index, amplitude, _ = candidates[chosen]
            selected = policy_chunk.copy()
            selected[:, ARM] = solved[chosen]
            self.active_direction = direction_index
            self.active_anchor = anchor
            self.active_radius = float(amplitude)
            self.cached_action_chunk = selected.copy()
            return self._decision(
                selected,
                solved_eef[chosen],
                reason="bank_supported_intervention",
                selected_clearance=float(clearance[chosen]),
                intervened=True,
                feasible=True,
                anchor=anchor,
                direction=direction_index,
                target_radius=float(amplitude),
                current_radius=current_radius,
                candidate_count=len(candidates),
                advance_reference=True,
                **common,
            )

        if cached is not None:
            chunk, eef, cached_clearance = cached
            self.cached_action_chunk = chunk
            return self._decision(
                chunk,
                eef,
                reason="latched_safe_plan",
                selected_clearance=cached_clearance,
                intervened=True,
                feasible=True,
                anchor=self.active_anchor,
                direction=self.active_direction,
                target_radius=self.active_radius,
                current_radius=current_radius,
                candidate_count=len(candidates),
                advance_reference=True,
                **common,
            )
        if raw_prefix_clearance >= self.safety_margin:
            return self._decision(
                policy_chunk,
                raw_eef,
                reason="no_full_horizon_candidate_safe_prefix",
                selected_clearance=raw_prefix_clearance,
                intervened=False,
                feasible=True,
                anchor=anchor,
                direction=self.active_direction,
                target_radius=0.0,
                current_radius=current_radius,
                candidate_count=len(candidates),
                **common,
            )
        return self._hold_decision(
            state,
            policy_chunk,
            boxes,
            current_eef,
            current_quaternion,
            carrying,
            held_index,
            reason="no_safe_bank_candidate_hold",
            anchor=anchor,
            current_radius=current_radius,
            candidate_count=len(candidates),
            **common,
        )
