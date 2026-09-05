"""Shared backend construction and CLI arguments.

Extracted from ``webcam_demo`` so the glove driver gets the same hardware
controls — slew limiting, current cap, command scale — by sharing them rather
than by growing a second, subtly different copy.

Safety defaults live here too, in one place:

* the default backend is ``print``. A bare invocation must not energise motors.
* ``--backend hardware`` is refused when the hand has never been homed, because
  without ``~/.midas_hand/config.yaml`` the motor zero has no defined relation
  to the URDF zero and ``MidasHand.clip_positions`` is a no-op.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .backends import HardwareBackend, MujocoBackend, PrintBackend

logger = logging.getLogger(__name__)

BACKEND_CHOICES = ("print", "mujoco", "hardware")
DEFAULT_BACKEND = "print"

DEFAULT_HARDWARE_PORT = "/dev/ttyUSB0"
DEFAULT_HARDWARE_CURRENT_LIMIT = 350
DEFAULT_HARDWARE_COMMAND_SCALE = 1.0
DEFAULT_HARDWARE_MAX_STEP_RAD = 0.15
DEFAULT_HARDWARE_RATE_HZ = 50.0
DEFAULT_HARDWARE_INTERPOLATION_ALPHA = 0.4


def add_backend_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_BACKEND,
    include_mujoco: bool = True,
) -> None:
    """Add the shared backend/hardware flags.

    ``include_mujoco=False`` for callers that already define their own MuJoCo
    flags (``manus_teleop`` does), so this can be adopted without renaming
    options people already type.
    """

    group = parser.add_argument_group("backend")
    group.add_argument(
        "--backend", choices=BACKEND_CHOICES, default=default,
        help=f"Where joint commands go (default: {default}). 'hardware' moves "
             "the real hand.",
    )
    if include_mujoco:
        group.add_argument("--mujoco-xml", default=None, help="Override the MJCF path.")
        group.add_argument("--mujoco-repo", default=None,
                           help="Path to midas_hand_mujoco if not auto-discovered.")
        group.add_argument("--mujoco-viewer", action="store_true",
                           help="Open the MuJoCo passive viewer.")
        group.add_argument("--mujoco-steps", type=int, default=None,
                           help="Physics steps per frame (default: derived from the "
                                "control rate so the sim runs near real time).")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument(
        "--configure-hardware", action=argparse.BooleanOptionalAction, default=True,
        help="Write operating mode, gains and the current cap at startup. This "
             "is also the only call that disables torque first, so turning it "
             "off leaves motors from a crashed run live.",
    )
    hardware.add_argument("--hardware-config", default=None,
                          help="Calibration config (default: ~/.midas_hand/config.yaml).")
    hardware.add_argument("--hardware-port", default=DEFAULT_HARDWARE_PORT)
    hardware.add_argument("--hardware-baudrate", type=int, default=None)
    hardware.add_argument("--hardware-current-limit", type=int,
                          default=DEFAULT_HARDWARE_CURRENT_LIMIT,
                          help="Goal current cap in mA. Start low on first bring-up.")
    hardware.add_argument("--hardware-command-scale", type=float,
                          default=DEFAULT_HARDWARE_COMMAND_SCALE,
                          help="Scale commands around calibrated zero; 0.3-0.5 for bring-up.")
    hardware.add_argument("--hardware-max-step-rad", type=float,
                          default=DEFAULT_HARDWARE_MAX_STEP_RAD,
                          help="Slew limit per hardware tick, radians. 0 disables it.")
    hardware.add_argument("--hardware-rate-hz", type=float, default=DEFAULT_HARDWARE_RATE_HZ)
    hardware.add_argument("--hardware-interpolation-alpha", type=float,
                          default=DEFAULT_HARDWARE_INTERPOLATION_ALPHA)
    hardware.add_argument(
        "--allow-unhomed", action="store_true",
        help="Permit --backend hardware with no saved homing calibration. "
             "Unsafe: joint limits are unenforced and the zero is undefined.",
    )
    hardware.add_argument(
        "--start-armed", action="store_true",
        help="Enable torque immediately instead of waiting to be armed.",
    )


def hardware_is_homed(config_path: str | None = None) -> bool:
    """Whether a saved homing calibration exists."""

    try:
        from midas_hand_api import DEFAULT_CONFIG_PATH
    except ImportError:
        return False
    return Path(config_path or DEFAULT_CONFIG_PATH).expanduser().exists()


def check_hardware_preconditions(args) -> None:
    """Refuse obviously unsafe hardware runs before anything is energised."""

    if args.backend != "hardware":
        return
    if hardware_is_homed(getattr(args, "hardware_config", None)) or getattr(
        args, "allow_unhomed", False
    ):
        return
    from midas_hand_api import DEFAULT_CONFIG_PATH

    raise SystemExit(
        f"Refusing --backend hardware: no homing calibration at {DEFAULT_CONFIG_PATH}.\n"
        "Without it the motor zero has no defined relationship to the URDF "
        "zero, and the API's joint-limit clamp is a no-op, so commanded angles "
        "are not bounded by anything.\n"
        "Run homing first (see midas_hand_api), or pass --allow-unhomed if you "
        "know what you are doing and the current limit is low."
    )


def build_backend(args, *, control_hz: float | None = None):
    """Construct the backend named by ``args.backend``."""

    if args.backend == "print":
        return PrintBackend()
    if args.backend == "mujoco":
        steps = getattr(args, "mujoco_steps", None) or getattr(args, "steps_per_frame", None)
        if steps is None:
            # Keep the sim near real time rather than simulating a second of
            # physics per frame, which the old fixed default of 30 did.
            timestep = 0.002
            steps = max(1, round((1.0 / (control_hz or 60.0)) / timestep))
        return MujocoBackend(
            xml_path=getattr(args, "mujoco_xml", None) or getattr(args, "xml_path", None),
            mujoco_repo=getattr(args, "mujoco_repo", None),
            render=getattr(args, "mujoco_viewer", False),
            steps_per_frame=steps,
        )
    if args.backend == "hardware":
        check_hardware_preconditions(args)
        return HardwareBackend(
            configure=args.configure_hardware,
            config_path=args.hardware_config,
            port=args.hardware_port,
            baudrate=args.hardware_baudrate,
            current_limit_ma=args.hardware_current_limit,
            command_scale=args.hardware_command_scale,
            max_step_rad=args.hardware_max_step_rad,
            update_rate_hz=args.hardware_rate_hz,
            interpolation_alpha=args.hardware_interpolation_alpha,
            start_armed=getattr(args, "start_armed", False),
        )
    raise ValueError(f"Unsupported backend: {args.backend}")
