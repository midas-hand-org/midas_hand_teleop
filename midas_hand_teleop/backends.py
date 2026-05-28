"""Optional teleop command sinks for printing, MuJoCo, and hardware."""

from __future__ import annotations

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
        self._io_lock = threading.Lock()
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

    def read_motor_positions(self) -> np.ndarray:
        """Return current hardware motor positions in HandConfig motor order."""

        with self._io_lock:
            return self.hand.clip_positions(self.hand.read_pos())

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
        with self._io_lock:
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
