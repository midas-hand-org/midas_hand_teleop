"""Optional teleop command sinks for printing, MuJoCo, and hardware."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES
from midas_hand_retargeter.model import MIDAS_RIGHT_HAND
from midas_hand_retargeter.paths import default_mjcf_path
from midas_hand_retargeter.retargeter import RetargetingResult


class TeleopBackend(Protocol):
    """What every command sink must provide.

    Annotated on ``backend_cli.build_backend``, so this is the contract a new
    backend has to satisfy. ``arm``/``disarm``/``is_armed`` are deliberately
    NOT part of it: only a sink that can energise something has them, and
    callers gate on their presence (see ``TunerLoop.hardware_available``).
    """

    def send(self, result: RetargetingResult) -> None: ...

    def measured(self) -> dict[str, float]:
        """Latest measured joint positions, keyed by JOINT NAME.

        Keyed by name rather than index on purpose: the hardware reports 13
        values in HARDWARE_MOTOR_JOINT_NAMES order (thumb first) while the
        retargeter emits ACTIVE_JOINT_NAMES order (index first), so an
        index-keyed comparison would silently pair up the wrong joints.
        Returns an empty dict when the backend cannot measure anything.
        """

    def close(self) -> None: ...


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
        compact = {name: round(value, 3) for name, value in result.active_joint_positions.items()}
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
            measured[joint_name] = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return measured

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def _active_actuated_joint_names(self) -> list[str]:
        names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            names.append(
                self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            )
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
        #: False, so constructing a backend never energises a motor. The only
        #: caller passes this explicitly from --start-armed; a default of True
        #: meant `HardwareBackend()` in a REPL enabled torque on the spot.
        start_armed: bool = False,
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
        self._rejected_frames = 0
        self._armed_request: bool | None = None
        self._arm_error: Exception | None = None
        self._arm_applied = threading.Event()
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
        self._limits = self._resolve_limits(config)
        # configure(enable_torque=False) is the ONLY call that disables torque,
        # selects current-based position mode, and writes the goal-current cap.
        # Skipping it leaves motors from a crashed run live with whatever cap
        # they had, so it runs even when starting disarmed.
        if configure:
            self.hand.configure(enable_torque=False)
        # Before the command thread starts, so this thread still owns the bus.
        self._last_command = self._read_start_positions()
        self._target_command = self._last_command.copy()
        # The first commanded pose is the measured pose, so arming cannot jump.
        self.hand.set_positions(self._last_command, clip=True)
        self._armed = False
        if start_armed:
            # Inline: the command thread does not exist yet, so this thread is
            # still the bus owner and _request_armed would have nobody to wait
            # for. Ordering matters -- the thread is started below.
            self._apply_armed(True)
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
        try:
            target = self._prepare_target(result)
        except ValueError as exc:
            # Drop the frame and keep the previous target. Raising here would
            # tear down the run on a single bad solve, which on hardware means
            # losing torque mid-grasp for a fault that lasts one frame.
            self._rejected_frames += 1
            if self._rejected_frames % 60 == 1:
                print(f"Dropping frame: {exc}")
            return
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

        The bus work is handed to the command thread rather than done here.
        ``measured()`` already documents why -- that thread owns the bus at
        update_rate_hz and DynamixelClient documents no thread safety, while
        ``check_connected`` force-clears the SDK's own busy interlock on the
        stated assumption of a single thread. Arming is exactly when the two
        would collide, and a corrupted read here is the dangerous one.
        """

        self._request_armed(True)

    def disarm(self) -> None:
        """Stop commanding and drop torque. Safe to call repeatedly."""

        # Stop commanding immediately, whatever happens to the bus work: this
        # half needs no transaction and must not wait for a thread.
        with self._lock:
            self._armed = False
        self._request_armed(False)

    def _request_armed(self, armed: bool) -> None:
        """Ask the command thread to change torque state, and wait for it."""

        with self._lock:
            # NOT while holding the lock: _apply_armed takes it too, and
            # self._lock is a plain non-reentrant Lock.
            inline = self._command_thread is None
            if not inline:
                self._armed_request = armed
                self._arm_error = None
        if inline:
            # No command thread (update_rate_hz <= 0): this IS the bus owner.
            self._apply_armed(armed)
            return
        if not self._arm_applied.wait(timeout=self._ARM_TIMEOUT_S):
            raise RuntimeError("Timed out waiting for the hardware command loop to arm")
        self._arm_applied.clear()
        with self._lock:
            error = self._arm_error
        if error is not None:
            raise error

    def _apply_armed(self, armed: bool) -> None:
        """Perform the torque transition. Command thread only."""

        if armed:
            measured = self._read_start_positions()
            with self._lock:
                self._last_command = measured
                self._target_command = measured.copy()
            self.hand.set_positions(measured, clip=True)
            self.hand.enable_torque()
            with self._lock:
                self._armed = True
            return
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
                name: float(value) for name, value in zip(names, positions, strict=True)
            }
            self._measured_current_ma = {
                name: float(value) for name, value in zip(names, currents, strict=True)
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

    #: Per-motor (lower, upper) from the URDF, in motor order. The hand's own
    #: clip_positions reads limits out of ~/.midas_hand/config.yaml, and homing
    #: writes those as +/-pi -- so on a real, correctly homed hand it clips
    #: nothing at all, and this is the only bound left.
    _MODEL_LIMITS = np.array(
        [MIDAS_RIGHT_HAND.limits(name) for name in HARDWARE_MOTOR_JOINT_NAMES],
        dtype=np.float64,
    )

    #: Back off this far from a computed hard stop. Deliberately small: the
    #: stall comes from commanding PAST a stop, which the intersection below
    #: already prevents, so this only covers the stop position being derived
    #: from CAD offsets rather than measured per hand. Larger values cost real
    #: travel -- the thumb's CMC roll stop IS its rest pose, so every radian of
    #: margin there is a thumb that never fully returns to neutral.
    _STOP_MARGIN_RAD = 0.005

    def _prepare_target(self, result: RetargetingResult) -> np.ndarray:
        # np.array, not np.asarray: asarray returns the SAME object when the
        # dtype already matches, and the *= below would then scale the caller's
        # result in place. Today's results are float32 so it copies by luck.
        target = np.array(result.hardware_motor_positions, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            # A non-finite goal position reaches the servo as garbage, and
            # clip_positions cannot catch it: np.clip propagates NaN. The
            # optimizer can emit one when nlopt fails on a degenerate frame, so
            # this is a real path, not a theoretical one. Hold the last command.
            raise ValueError(
                "Refusing to command a non-finite joint target: "
                f"{dict(zip(HARDWARE_MOTOR_JOINT_NAMES, target, strict=False))}"
            )
        target *= self.command_scale
        # Model limits FIRST, because command_scale is applied above and a
        # scale > 1 can push a joint that the retargeter had bounded correctly
        # straight into a mechanical stop.
        limits = getattr(self, "_limits", self._MODEL_LIMITS)
        if len(target) == len(limits):
            target = np.clip(target, limits[:, 0], limits[:, 1])
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
                self._service_arm_request()
                with self._lock:
                    armed = self._armed
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

    def _service_arm_request(self) -> None:
        """Apply a pending arm/disarm. Command thread only -- see arm()."""

        with self._lock:
            wanted, self._armed_request = self._armed_request, None
        if wanted is None:
            return
        try:
            self._apply_armed(wanted)
        except Exception as exc:  # noqa: BLE001 - reported back to the caller
            with self._lock:
                self._arm_error = exc
                self._armed = False
        finally:
            self._arm_applied.set()

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

    @classmethod
    def _resolve_limits(cls, config) -> np.ndarray:
        """Intersect the URDF limits with the hand's measured hard stops.

        Homing drives each motor onto a physical stop and records the offset
        that puts URDF zero at the right place, which also tells us where that
        stop is: at motor position ``-cad_offset * joint_sign``, bounding the
        side it was driven toward. Two URDF limits reach past it on this hand --
        thumb MCP by 0.125 rad and thumb CMC side by 0.015 -- and both are
        commanded by ordinary input, so without this the motor stalls against
        its stop drawing the goal-current cap continuously.

        Falls back to the URDF alone if the homing tables cannot be read; that
        is the status quo, not a new risk.
        """

        limits = cls._MODEL_LIMITS.copy()
        try:
            from midas_hand_api.homing import FINGER_HOMING_TABLE, THUMB_HOMING_TABLE
        except ImportError:  # pragma: no cover - hardware extra not installed
            return limits

        signs = dict(zip(config.motor_ids, config.joint_signs, strict=False))
        for motor_id, _name, cad_offset, direction in (*THUMB_HOMING_TABLE, *FINGER_HOMING_TABLE):
            if motor_id >= len(limits):
                continue
            sign = float(signs.get(motor_id, 1.0))
            stop = -float(cad_offset) * sign
            if direction * sign < 0:  # driven toward the lower bound
                limits[motor_id, 0] = max(limits[motor_id, 0], stop + cls._STOP_MARGIN_RAD)
            else:  # driven toward the upper bound
                limits[motor_id, 1] = min(limits[motor_id, 1], stop - cls._STOP_MARGIN_RAD)
        # A margin must never invert a bound on a joint with almost no travel.
        crossed = limits[:, 0] > limits[:, 1]
        if np.any(crossed):
            midpoint = (limits[crossed, 0] + limits[crossed, 1]) / 2.0
            limits[crossed, 0] = limits[crossed, 1] = midpoint
        return limits

    #: How many times to retry a start-position read before refusing to arm.
    _START_READ_ATTEMPTS = 5

    #: How long arm()/disarm() wait for the command thread to do the bus work.
    #: Generous: a retried start-position read is several serial round trips.
    _ARM_TIMEOUT_S = 5.0

    def _read_start_positions(self) -> np.ndarray:
        """Read where the hand actually is, or refuse to say.

        This value is written straight to the servos, unslewed, and then torque
        is enabled onto it -- so it is the one read that must not be wrong.

        An exception is the easy case. The dangerous one is silent: a failed
        sync read makes DynamixelReader return its CACHE, which starts as
        zeros, and raw 0 counts map to ``(0 - home_offset) * sign`` -- about
        -3.09 rad on twelve of these thirteen motors. Commanding that is a
        ~177 degree move at whatever speed the motor can manage, since homing
        leaves Profile Velocity at 0 (unlimited) and the current cap is the
        only thing in the way. ``last_read_ok`` exists to detect exactly this.
        """

        last_error: Exception | None = None
        for attempt in range(self._START_READ_ATTEMPTS):
            try:
                positions = self.hand.read_pos()
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_error = exc
                continue
            if getattr(self.hand, "last_read_ok", True):
                return self.hand.clip_positions(positions)
            print(
                f"Start-position read {attempt + 1}/{self._START_READ_ATTEMPTS} "
                "returned cached data; retrying."
            )
        raise RuntimeError(
            "Could not read the hand's current position reliably after "
            f"{self._START_READ_ATTEMPTS} attempts. Refusing to command or arm: "
            "a stale read reports roughly -3.1 rad on every motor, and arming "
            "onto that would drive the whole hand about 177 degrees into its "
            "stops. Check the bus, the power supply, and the baud rate."
        ) from last_error
