"""Optional teleop command sinks for printing, MuJoCo, and hardware."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from midas_hand_retargeter.paths import default_mjcf_path
from midas_hand_retargeter.retargeter import RetargetingResult


class TeleopBackend(Protocol):
    def send(self, result: RetargetingResult) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class PrintBackend:
    interval_s: float = 0.5

    def __post_init__(self) -> None:
        self._last_print = 0.0

    def send(self, result: RetargetingResult) -> None:
        now = time.monotonic()
        if now - self._last_print < self.interval_s:
            return
        self._last_print = now
        compact = {
            name: round(value, 3)
            for name, value in result.active_joint_positions.items()
        }
        print(compact)

    def close(self) -> None:
        pass


class MujocoBackend:
    """Send active joint targets to the MIDAS MuJoCo model."""

    def __init__(
        self,
        *,
        xml_path: str | None = None,
        mujoco_repo: str | None = None,
        render: bool = False,
        steps_per_frame: int = 30,
        floating_wrist: bool = False,
        palm_step_limit: float = 0.008,
        finger_ctrl_alpha: float = 0.62,
        finger_ctrl_deadzone: float = 0.012,
        wrist_xy_scale: float = 0.36,
        wrist_y_scale: float = 0.20,
        wrist_z_base: float = 0.08,
        wrist_z_scale: float = 0.07,
        palm_pos_deadzone: float = 0.008,
        palm_quat_alpha: float = 0.22,
        landmark_still_threshold: float = 0.007,
    ):
        import mujoco

        self._mujoco = mujoco
        xml = str(xml_path or default_mjcf_path(mujoco_repo))
        self.floating_wrist = bool(floating_wrist)
        self.wrist_xy_scale = float(wrist_xy_scale)
        self.wrist_y_scale = float(wrist_y_scale)
        self.wrist_z_base = float(wrist_z_base)
        self.wrist_z_scale = float(wrist_z_scale)

        if self.floating_wrist:
            spec = mujoco.MjSpec.from_file(xml)
            palm = None
            for body in spec.worldbody.bodies:
                if body.name == "palm_base":
                    palm = body
                    break
            if palm is None:
                raise RuntimeError("palm_base body not found in MJCF")
            has_freejoint = any(
                j.name == "palm_floating" for j in palm.joints
            )
            if not has_freejoint:
                palm.add_freejoint(name="palm_floating")
            self._set_gravcomp_subtree(palm)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(xml)

        self.data = mujoco.MjData(self.model)
        self.steps_per_frame = int(steps_per_frame)
        if self.floating_wrist:
            self.steps_per_frame = max(self.steps_per_frame, 35)
        self.palm_step_limit = float(palm_step_limit)
        self.finger_ctrl_alpha = float(np.clip(finger_ctrl_alpha, 0.05, 1.0))
        self.finger_ctrl_deadzone = float(finger_ctrl_deadzone)
        self.palm_pos_deadzone = float(palm_pos_deadzone)
        self.palm_quat_alpha = float(np.clip(palm_quat_alpha, 0.05, 1.0))
        self.landmark_still_threshold = float(landmark_still_threshold)
        self._actuator_id = {
            joint_name: self._find_actuator_for_joint(joint_name)
            for joint_name in self._active_actuated_joint_names()
        }

        self._palm_qpos_adr = -1
        if self.floating_wrist:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "palm_floating"
            )
            if jid < 0:
                raise RuntimeError("palm_floating freejoint not found")
            self._palm_qpos_adr = int(self.model.jnt_qposadr[jid])
            self._palm_dof_adr = int(self.model.jnt_dofadr[jid])
            self.data.qpos[self._palm_qpos_adr: self._palm_qpos_adr + 3] = [
                0.0, 0.15, self.wrist_z_base,
            ]
            self.data.qpos[self._palm_qpos_adr + 3: self._palm_qpos_adr + 7] = [
                1.0, 0.0, 0.0, 0.0,
            ]
            self._neutral_palm_rotation: np.ndarray | None = None
            self._smoothed_palm_pos: np.ndarray | None = None
            self._smoothed_palm_quat: np.ndarray | None = None
            self._raw_palm_target_pos: np.ndarray | None = None
            self._last_landmarks: np.ndarray | None = None
            self._frozen_joint_ctrl: dict[int, float] | None = None
            self._hand_contact_geom_ids = {
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_group[geom_id]) == 2
                and int(self.model.geom_contype[geom_id]) > 0
            }
        self.viewer = None
        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def send(self, result: RetargetingResult, *, hand_frame=None) -> None:
        still = self._landmarks_are_still(hand_frame)

        if self.floating_wrist and hand_frame is not None and not still:
            self._update_floating_wrist(hand_frame)

        if still and self._frozen_joint_ctrl is not None:
            for actuator_id, value in self._frozen_joint_ctrl.items():
                self.data.ctrl[actuator_id] = value
        else:
            alpha = self.finger_ctrl_alpha if not still else 0.0
            for joint_name, value in result.active_joint_positions.items():
                actuator_id = self._actuator_id.get(joint_name)
                if actuator_id is None:
                    continue
                lower, upper = self.model.actuator_ctrlrange[actuator_id]
                target = float(np.clip(value, lower, upper))
                previous = float(self.data.ctrl[actuator_id])
                deadzone = self._finger_actuator_deadzone(joint_name)
                if abs(target - previous) < deadzone:
                    continue
                if alpha >= 1.0:
                    self.data.ctrl[actuator_id] = target
                elif alpha > 0.0:
                    self.data.ctrl[actuator_id] = (1.0 - alpha) * previous + alpha * target
            self._frozen_joint_ctrl = {
                int(actuator_id): float(self.data.ctrl[actuator_id])
                for actuator_id in self._actuator_id.values()
            }

        substeps = self.steps_per_frame
        if self.floating_wrist and self._palm_qpos_adr >= 0:
            self._anchor_floating_palm()
            for _ in range(substeps):
                self._mujoco.mj_step(self.model, self.data)
            if not still:
                self._relax_finger_controls_in_contact()
                for _ in range(2):
                    self._correct_palm_penetration()
            self._anchor_floating_palm()
        else:
            for _ in range(substeps):
                self._mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def _update_floating_wrist(self, hand_frame) -> None:
        """Set palm freejoint position and orientation from webcam landmarks.

        Three independent axes from the camera image (intuitive teleop):
          - wrist X  -> MuJoCo X  (left / right)
          - wrist Y  -> MuJoCo Y  (lower in image = reach forward toward objects)
          - hand size -> MuJoCo Z  (bigger in frame = closer to camera = lower / toward table)
        Rot: full 3D palm basis from 21 landmarks (pitch / yaw / roll relative to calib).
        """

        if self._palm_qpos_adr < 0 or hand_frame.image_landmarks is None:
            return
        if hand_frame.landmarks is None:
            return

        lm = hand_frame.image_landmarks
        wrist = lm[0]
        mid_mcp = lm[9]
        idx_mcp = lm[5]
        pinky_mcp = lm[17]

        wx = float(np.clip(wrist.x, 0.0, 1.0))
        wy = float(np.clip(wrist.y, 0.0, 1.0))
        mx, my = float(mid_mcp.x), float(mid_mcp.y)

        # Map the central 80% of the image to the full desk workspace.
        wx_norm = (wx - 0.1) / 0.8
        wy_norm = (wy - 0.1) / 0.8
        wx_norm = float(np.clip(wx_norm, 0.0, 1.0))
        wy_norm = float(np.clip(wy_norm, 0.0, 1.0))

        hand_len = math.hypot(mx - wx, my - wy)
        pw = math.hypot(
            float(idx_mcp.x) - float(pinky_mcp.x),
            float(idx_mcp.y) - float(pinky_mcp.y),
        )
        size = (hand_len + pw) * 0.5
        size_near = 0.34
        size_far = 0.11
        t = (size - size_far) / (size_near - size_far)
        t = max(0.0, min(1.0, t))

        # X: 左右
        x = (0.5 - wx_norm) * self.wrist_xy_scale

        # Y: 前后 — 手在画面下方 = 往桌子伸（指尖朝 +Y）
        y = 0.09 + wy_norm * self.wrist_y_scale
        y = max(0.08, min(0.24, y))

        # Z: 手靠近摄像头 = 下降靠近桌面
        z = self.wrist_z_base - t * self.wrist_z_scale
        z = max(0.02, min(0.12, z))

        target_pos = np.array([x, y, z], dtype=np.float64)

        update_position = True
        if self._raw_palm_target_pos is not None:
            if (
                float(np.linalg.norm(target_pos - self._raw_palm_target_pos))
                < self.palm_pos_deadzone
            ):
                update_position = False
        if update_position:
            self._raw_palm_target_pos = target_pos.copy()
            pos_alpha = 0.18
            if self._smoothed_palm_pos is None:
                self._smoothed_palm_pos = target_pos
            else:
                self._smoothed_palm_pos = (
                    (1.0 - pos_alpha) * self._smoothed_palm_pos + pos_alpha * target_pos
                )

        palm_rot = self._palm_rotation_from_landmarks(
            hand_frame.landmarks,
            fallback=np.asarray(hand_frame.wrist_rotation, dtype=np.float64),
        )
        if self._neutral_palm_rotation is None:
            self._neutral_palm_rotation = palm_rot.copy()
        relative_rot = palm_rot @ self._neutral_palm_rotation.T
        target_quat = self._mat_to_quat(relative_rot)

        update_rotation = True
        if self._smoothed_palm_quat is not None:
            if self._quat_angle(self._smoothed_palm_quat, target_quat) < 0.012:
                update_rotation = False
        if update_rotation:
            quat_alpha = self.palm_quat_alpha
            if self._smoothed_palm_quat is None:
                self._smoothed_palm_quat = target_quat
            else:
                if float(np.dot(self._smoothed_palm_quat, target_quat)) < 0.0:
                    target_quat = -target_quat
                self._smoothed_palm_quat = (
                    (1.0 - quat_alpha) * self._smoothed_palm_quat
                    + quat_alpha * target_quat
                )
                self._smoothed_palm_quat /= (
                    np.linalg.norm(self._smoothed_palm_quat) + 1e-12
                )

    def _landmarks_are_still(self, hand_frame) -> bool:
        """True when 3D landmarks barely change — freeze teleop to stop twitching."""

        if hand_frame is None or hand_frame.landmarks is None:
            return False

        pts = np.asarray(hand_frame.landmarks, dtype=np.float32)
        if self._last_landmarks is None:
            self._last_landmarks = pts.copy()
            return False

        delta = float(np.max(np.abs(pts - self._last_landmarks)))
        if delta < self.landmark_still_threshold:
            return True

        self._last_landmarks = pts.copy()
        return False

    @staticmethod
    def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
        """Geodesic angle (rad) between two unit quaternions."""

        dot = abs(float(np.dot(q1, q2)))
        dot = min(1.0, dot)
        return 2.0 * math.acos(dot)

    @staticmethod
    def _finger_actuator_deadzone(joint_name: str) -> float:
        """Curl joints need a smaller deadzone so a fist closes fully."""

        if joint_name.endswith((
            "_mcp_pitch_joint",
            "_pip_joint",
            "thumb_mcp_joint",
            "thumb_dip_joint",
        )):
            return 0.008
        return 0.018

    @staticmethod
    def _palm_rotation_from_landmarks(
        landmarks: np.ndarray,
        *,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build palm orientation from MCP layout; fallback when fingertips overlap (fist)."""

        from midas_hand_retargeter.human import estimate_frame_from_hand_points

        pts = np.asarray(landmarks, dtype=np.float64)
        if MujocoBackend._is_fist_pose(pts):
            return np.asarray(
                estimate_frame_from_hand_points(pts),
                dtype=np.float64,
            )

        wrist = pts[0]
        forward = pts[9] - wrist
        across = pts[5] - pts[17]
        forward_norm = float(np.linalg.norm(forward))
        across_norm = float(np.linalg.norm(across))
        if forward_norm < 1e-6 or across_norm < 1e-6:
            if fallback is not None:
                return np.asarray(fallback, dtype=np.float64)
            return np.eye(3, dtype=np.float64)
        forward /= forward_norm
        across /= across_norm
        cos_angle = abs(float(np.dot(forward, across)))
        if cos_angle > 0.92 and fallback is not None:
            return np.asarray(fallback, dtype=np.float64)
        normal = np.cross(forward, across)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-6:
            if fallback is not None:
                return np.asarray(fallback, dtype=np.float64)
            return np.eye(3, dtype=np.float64)
        normal /= normal_norm
        across = np.cross(normal, forward)
        across /= np.linalg.norm(across) + 1e-12
        return np.stack([across, forward, normal], axis=1)

    @staticmethod
    def _is_fist_pose(landmarks: np.ndarray) -> bool:
        """Heuristic: fingertips are bunched close to the palm center."""

        pts = np.asarray(landmarks, dtype=np.float64)
        palm = np.mean(pts[[5, 9, 13]], axis=0)
        tip_ids = (4, 8, 12, 16)
        spread = float(
            np.mean([np.linalg.norm(pts[i] - palm) for i in tip_ids])
        )
        return spread < 0.055

    @staticmethod
    def _set_gravcomp_subtree(body) -> None:
        """Cancel gravity on the floating hand so it does not sag between teleop updates."""

        body.gravcomp = 1.0
        for child in body.bodies:
            MujocoBackend._set_gravcomp_subtree(child)

    def _anchor_floating_palm(self) -> None:
        """Lock palm pose after physics so gravity/contact cannot drift the wrist."""

        if self._palm_qpos_adr < 0 or self._smoothed_palm_pos is None:
            return

        pos = self._palm_qpos_adr
        self.data.qpos[pos: pos + 3] = self._smoothed_palm_pos
        if self._smoothed_palm_quat is not None:
            self.data.qpos[pos + 3: pos + 7] = self._smoothed_palm_quat
        self.data.qvel[self._palm_dof_adr: self._palm_dof_adr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _correct_palm_penetration(self) -> None:
        """Push the floating palm out of penetrations found after stepping."""

        if self._palm_qpos_adr < 0:
            return

        self._mujoco.mj_forward(self.model, self.data)
        correction = np.zeros(3, dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            dist = float(contact.dist)
            if dist >= -1e-5:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            hand_is_1 = geom1 in self._hand_contact_geom_ids
            hand_is_2 = geom2 in self._hand_contact_geom_ids
            if not hand_is_1 and not hand_is_2:
                continue

            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            normal = frame[:, 0]
            if hand_is_2 and not hand_is_1:
                normal = -normal
            correction += normal * (-dist)

        if float(np.linalg.norm(correction)) < 1e-9:
            return

        pos = self._palm_qpos_adr
        self.data.qpos[pos: pos + 3] += correction
        self.data.qvel[self._palm_dof_adr: self._palm_dof_adr + 3] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _relax_finger_controls_in_contact(self) -> None:
        """Ease finger position targets when fingertips are penetrating."""

        if not self._hand_contact_geom_ids:
            return

        penetrating = False
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if float(contact.dist) >= -5e-4:
                continue
            if int(contact.geom1) in self._hand_contact_geom_ids or int(
                contact.geom2
            ) in self._hand_contact_geom_ids:
                penetrating = True
                break
        if not penetrating:
            return

        relax = 0.82
        for joint_name, actuator_id in self._actuator_id.items():
            if joint_name.endswith(("_mcp_pitch_joint", "_pip_joint")):
                self.data.ctrl[actuator_id] *= 0.95
            else:
                self.data.ctrl[actuator_id] *= relax

    @staticmethod
    def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a MuJoCo [w, x, y, z] quaternion."""

        m = np.asarray(rot, dtype=np.float64)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (m[2, 1] - m[1, 2]) / s,
                    (m[0, 2] - m[2, 0]) / s,
                    (m[1, 0] - m[0, 1]) / s,
                ],
                dtype=np.float64,
            )
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            quat = np.array(
                [
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            quat = np.array(
                [
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            quat = np.array(
                [
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
        return quat / (np.linalg.norm(quat) + 1e-12)

    def _active_actuated_joint_names(self) -> list[str]:
        names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            names.append(self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        return names

    def _find_actuator_for_joint(self, joint_name: str) -> int:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Could not find MuJoCo joint {joint_name!r}")
        for actuator_id in range(self.model.nu):
            if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id:
                return actuator_id
        raise RuntimeError(f"No MuJoCo actuator found for joint {joint_name!r}")


class HardwareBackend:
    """Send 13 active motor positions to ``midas_hand_api``.

    Vision/retargeting frames can arrive with irregular timing, so hardware
    commands are sent from a fixed-rate loop. ``send`` only updates the latest
    desired target; the command loop interpolates from the previous command
    toward that target at ``update_rate_hz``.
    """

    def __init__(
        self,
        *,
        configure: bool = False,
        autoconnect: bool = True,
        config_path: str | None = None,
        port: str | None = None,
        baudrate: int | None = None,
        current_limit_ma: int | None = None,
        command_scale: float = 1.0,
        max_step_rad: float = 0.05,
        update_rate_hz: float = 50.0,
        interpolation_alpha: float = 0.35,
    ):
        from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand

        self.command_scale = float(command_scale)
        self.max_step_rad = float(max_step_rad)
        self.update_rate_hz = float(update_rate_hz)
        self.interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        self._lock = threading.Lock()
        self._closed = False
        self._loop_error: Exception | None = None
        self._command_thread: threading.Thread | None = None
        config = self._load_config(HandConfig, DEFAULT_CONFIG_PATH, config_path)
        updates = {}
        if port is not None:
            updates["port"] = port
        if baudrate is not None:
            updates["baudrate"] = int(baudrate)
        if current_limit_ma is not None:
            updates["goal_current_limit"] = int(current_limit_ma)
        if updates:
            config = replace(config, **updates)

        self.hand = MidasHand(config=config, autoconnect=autoconnect)
        if configure:
            self.hand.configure(enable_torque=False)
        self._last_command = self._read_start_positions()
        self._target_command = self._last_command.copy()
        self.hand.set_positions(self._last_command, clip=True)
        if configure:
            self.hand.enable_torque()
        if self.update_rate_hz > 0:
            self._command_thread = threading.Thread(
                target=self._run_command_loop,
                name="midas-hardware-command-loop",
                daemon=True,
            )
            self._command_thread.start()

    def send(self, result: RetargetingResult) -> None:
        if self._loop_error is not None:
            raise RuntimeError("Hardware command loop failed") from self._loop_error
        target = self._prepare_target(result)
        if self.update_rate_hz <= 0:
            self._send_interpolated_command(target)
            return
        with self._lock:
            self._target_command = target

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)
        self.hand.shutdown()

    def _prepare_target(self, result: RetargetingResult) -> np.ndarray:
        target = np.asarray(result.hardware_motor_positions, dtype=np.float64)
        target *= self.command_scale
        return self.hand.clip_positions(target)

    def _run_command_loop(self) -> None:
        period_s = 1.0 / max(self.update_rate_hz, 1e-6)
        next_tick = time.monotonic()
        while True:
            with self._lock:
                if self._closed:
                    return
                target = self._target_command.copy()
            try:
                self._send_interpolated_command(target)
            except Exception as exc:
                self._loop_error = exc
                print(f"Hardware command loop stopped: {exc}")
                return

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def _send_interpolated_command(self, target: np.ndarray) -> None:
        delta = target - self._last_command
        if self.interpolation_alpha < 1.0:
            delta *= self.interpolation_alpha
        if self.max_step_rad > 0:
            delta = np.clip(delta, -self.max_step_rad, self.max_step_rad)
        command = self._last_command + delta
        self.hand.set_positions(command, clip=True)
        self._last_command = command

    @staticmethod
    def _load_config(HandConfig, default_config_path, config_path: str | None):
        if config_path is not None:
            return HandConfig.load(config_path)
        if Path(default_config_path).exists():
            return HandConfig.load(default_config_path)
        print(
            f"No saved MIDAS hand config at {default_config_path}; "
            "using default zero calibration. Run homing first for hardware use."
        )
        return HandConfig.xm335_t323()

    def _read_start_positions(self) -> np.ndarray:
        try:
            return self.hand.clip_positions(self.hand.read_pos())
        except Exception as exc:
            print(f"Could not read current motor positions; starting from zeros: {exc}")
            return np.zeros(len(self.hand.motor_ids), dtype=np.float64)
