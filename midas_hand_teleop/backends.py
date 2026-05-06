"""Optional teleop command sinks for printing, MuJoCo, and hardware."""

from __future__ import annotations

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
    ):
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or default_mjcf_path(mujoco_repo)))
        self.data = mujoco.MjData(self.model)
        self.steps_per_frame = int(steps_per_frame)
        self._actuator_id = {
            joint_name: self._find_actuator_for_joint(joint_name)
            for joint_name in self._active_actuated_joint_names()
        }
        self.viewer = None
        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def send(self, result: RetargetingResult) -> None:
        for joint_name, value in result.active_joint_positions.items():
            actuator_id = self._actuator_id.get(joint_name)
            if actuator_id is None:
                continue
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = float(np.clip(value, lower, upper))
        for _ in range(self.steps_per_frame):
            self._mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

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

    The backend starts from the measured motor positions and rate-limits toward
    each retargeted target. This avoids a large first-frame jump when torque is
    enabled on the real hand.
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
    ):
        from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand

        self.command_scale = float(command_scale)
        self.max_step_rad = float(max_step_rad)
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
        self.hand.set_positions(self._last_command, clip=True)
        if configure:
            self.hand.enable_torque()

    def send(self, result: RetargetingResult) -> None:
        target = np.asarray(result.hardware_motor_positions, dtype=np.float64)
        target *= self.command_scale
        target = self.hand.clip_positions(target)
        if self.max_step_rad > 0:
            delta = np.clip(
                target - self._last_command,
                -self.max_step_rad,
                self.max_step_rad,
            )
            target = self._last_command + delta
        self.hand.set_positions(target, clip=True)
        self._last_command = target

    def close(self) -> None:
        self.hand.shutdown()

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
