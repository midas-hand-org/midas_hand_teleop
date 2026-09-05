"""Optional teleop command sinks for printing, MuJoCo, and hardware."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES
from midas_hand_retargeter.paths import default_mjcf_path
from midas_hand_retargeter.retargeter import RetargetingResult


class TeleopBackend(Protocol):
    def send(self, result: RetargetingResult) -> None:
        ...

    def measured(self) -> dict[str, float]:
        """Latest measured joint positions, keyed by JOINT NAME.

        Keyed by name rather than index on purpose: the hardware reports 13
        values in HARDWARE_MOTOR_JOINT_NAMES order (thumb first) while the
        retargeter emits ACTIVE_JOINT_NAMES order (index first), so an
        index-keyed comparison would silently pair up the wrong joints.
        Returns an empty dict when the backend cannot measure anything.
        """

    def close(self) -> None:
        ...


@dataclass
class PrintBackend:
    interval_s: float = 0.5

    def __post_init__(self) -> None:
        self._last_print = 0.0

    def measured(self) -> dict[str, float]:
        return {}

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

    def measured(self) -> dict[str, float]:
        """Actual simulated joint angles, so the UI can show command vs reality."""

        measured: dict[str, float] = {}
        for joint_name in self._actuator_id:
            joint_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                continue
            measured[joint_name] = float(
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
            )
        return measured

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
        start_armed: bool = True,
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
        self._measured: dict[str, float] = {}
        self._measured_current_ma: dict[str, float] = {}
        self._measured_seq = 0
        self._measured_monotonic = 0.0
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
        # configure(enable_torque=False) is the ONLY call that disables torque,
        # selects current-based position mode, and writes the goal-current cap.
        # Skipping it leaves motors from a crashed run live with whatever cap
        # they had, so it runs even when starting disarmed.
        if configure:
            self.hand.configure(enable_torque=False)
        self._last_command = self._read_start_positions()
        self._target_command = self._last_command.copy()
        # The first commanded pose is the measured pose, so arming cannot jump.
        self.hand.set_positions(self._last_command, clip=True)
        self._armed = False
        if start_armed:
            self.arm()
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
            if self._armed:
                self._send_interpolated_command(target)
            return
        with self._lock:
            self._target_command = target

    @property
    def is_armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        """Enable torque and start tracking, from the current measured pose.

        Re-seeds the command from what the hand is actually doing first, so
        arming after a pause does not snap the fingers to a stale target.
        """

        with self._lock:
            if self._armed:
                return
            measured = self._read_start_positions()
            self._last_command = measured
            self._target_command = measured.copy()
        self.hand.set_positions(self._last_command, clip=True)
        self.hand.enable_torque()
        with self._lock:
            self._armed = True

    def disarm(self) -> None:
        """Stop commanding and drop torque. Safe to call repeatedly."""

        with self._lock:
            self._armed = False
        try:
            self.hand.disable_torque()
        except Exception:
            # Never let a teardown failure mask the caller's own error path.
            pass

    def measured(self) -> dict[str, float]:
        """Snapshot of the last positions read by the command thread.

        Deliberately does NOT touch the serial bus: the command thread already
        owns it at update_rate_hz, and a second reader from the control loop
        would contend on a DynamixelClient with no documented thread safety.
        """

        with self._lock:
            return dict(self._measured)

    def measured_current_ma(self) -> dict[str, float]:
        with self._lock:
            return dict(self._measured_current_ma)

    def measurement_age_s(self) -> float:
        """Seconds since the last successful read, or inf if never read.

        DynamixelClient returns cached last-good values when a packet drops, so
        a frozen read is indistinguishable from a stationary hand. Anything
        that trips on measured current must treat a stale sample as a fault and
        fail closed, not simply never fire.
        """

        with self._lock:
            if self._measured_seq == 0:
                return float("inf")
            return time.monotonic() - self._measured_monotonic

    def _sample_state(self) -> None:
        try:
            positions, currents = self.hand.read_pos(), self.hand.read_cur()
        except Exception:
            return  # keep the previous snapshot; staleness is reported by age
        # Index by the hand's actual motor ids, not the full 13-name list:
        # HandConfig supports motor subsets, and pairing a short reading against
        # the full name list would silently attribute values to the wrong joints.
        names = [HARDWARE_MOTOR_JOINT_NAMES[i] for i in self.hand.motor_ids]
        with self._lock:
            self._measured = {
                name: float(value)
                for name, value in zip(names, positions, strict=True)
            }
            self._measured_current_ma = {
                name: float(value)
                for name, value in zip(names, currents, strict=True)
            }
            self._measured_seq += 1
            self._measured_monotonic = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._armed = False
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
                armed = self._armed
                target = self._target_command.copy()
            try:
                if armed:
                    self._send_interpolated_command(target)
                self._sample_state()
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
