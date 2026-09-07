"""Backend selection, safety gates, and the glove -> hardware plumbing.

None of these touch real hardware: they exercise the decisions made *before*
anything is energised, which is exactly where the unsafe defaults lived.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
from midas_hand_retargeter.constants import (
    ACTIVE_JOINT_NAMES,
    HARDWARE_MOTOR_JOINT_NAMES,
)

from midas_hand_teleop import backend_cli
from midas_hand_teleop.manus_glove.manus_teleop import _Control


def _args(**overrides):
    base = dict(
        backend="hardware",
        hardware_config=None,
        allow_unhomed=False,
        start_armed=False,
        mujoco_xml=None,
        mujoco_repo=None,
        mujoco_viewer=False,
        mujoco_steps=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_backend_does_not_energise_motors():
    """A bare invocation must never move a real hand."""

    assert backend_cli.DEFAULT_BACKEND == "print"

    from midas_hand_teleop import webcam_demo

    assert webcam_demo.DEFAULT_BACKEND == "print"


def test_hardware_is_refused_without_a_homing_calibration(monkeypatch):
    monkeypatch.setattr(backend_cli, "hardware_is_homed", lambda *a, **k: False)
    with pytest.raises(SystemExit) as excinfo:
        backend_cli.check_hardware_preconditions(_args())
    message = str(excinfo.value)
    assert "no homing calibration" in message
    assert "--allow-unhomed" in message, "the escape hatch must be discoverable"


def test_allow_unhomed_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(backend_cli, "hardware_is_homed", lambda *a, **k: False)
    backend_cli.check_hardware_preconditions(_args(allow_unhomed=True))


def test_homed_hand_passes_the_gate(monkeypatch):
    monkeypatch.setattr(backend_cli, "hardware_is_homed", lambda *a, **k: True)
    backend_cli.check_hardware_preconditions(_args())


def test_non_hardware_backends_skip_the_gate(monkeypatch):
    monkeypatch.setattr(backend_cli, "hardware_is_homed", lambda *a, **k: False)
    for backend in ("print", "mujoco"):
        backend_cli.check_hardware_preconditions(_args(backend=backend))


def test_control_exposes_hardware_motor_positions_in_motor_order():
    """The geometric path used to raise AttributeError on any hardware send."""

    values = {name: float(index) for index, name in enumerate(ACTIVE_JOINT_NAMES)}
    control = _Control(values)

    motors = control.hardware_motor_positions
    assert motors.shape == (13,)
    expected = [values[name] for name in HARDWARE_MOTOR_JOINT_NAMES]
    np.testing.assert_allclose(motors, expected)

    # Motor order is thumb-first and genuinely differs from ACTIVE order, which
    # is why a positional comparison between the two would be silently wrong.
    assert HARDWARE_MOTOR_JOINT_NAMES != ACTIVE_JOINT_NAMES
    assert not np.allclose(motors, [values[n] for n in ACTIVE_JOINT_NAMES])


def test_left_hand_is_refused_rather_than_warned():
    """Mirroring does not fix left-hand support, so proceeding is never right."""

    from midas_hand_teleop.manus_glove import manus_teleop

    args = argparse.Namespace(
        side="left",
        retarget="full",
        calibrate_delay=0.0,
        preset=None,
        profile="glove",
        filter_alpha=None,
        host="localhost",
    )
    for name in (
        "finger_curl_gain",
        "finger_abad_gain",
        "finger_smoothing_alpha",
        "thumb_cmc_gain",
        "thumb_cmc_side_gain",
        "thumb_cmc_roll_gain",
        "thumb_flexion_gain",
        "thumb_smoothing_alpha",
    ):
        setattr(args, name, None)
    with pytest.raises(SystemExit, match="Left-hand teleop is not implemented"):
        manus_teleop.run(args)


def test_dead_cli_flags_are_gone():
    """~20 flags were parsed and silently discarded; they misled tuning work."""

    import inspect

    from midas_hand_teleop import webcam_demo

    source = inspect.getsource(webcam_demo)
    for flag in (
        "--thumb-pinch-gain",
        "--thumb-cmc-roll-signed",
        "--finger-abad-limit",
        "--thumb-cmc-side-oppose",
    ):
        assert flag not in source, f"{flag} names behaviour that does not exist"
    assert "argparse.SUPPRESS" not in source


def test_every_entry_point_can_actually_arm_hardware():
    """webcam_demo had drifted into its own hardware group with no
    --start-armed and no --allow-unhomed, so `midas-hand-teleop --backend
    hardware` connected, configured, and could then never energise the hand:
    build_backend reads start_armed with a getattr default of False and nothing
    else in that path calls arm()."""

    from midas_hand_teleop.manus_glove.manus_teleop import (
        build_parser as manus_parser,
    )
    from midas_hand_teleop.tuner.cli import build_parser as tuner_parser
    from midas_hand_teleop.webcam_demo import build_parser as webcam_parser

    for name, parser in (
        ("midas-hand-teleop", webcam_parser()),
        ("midas-manus-teleop", manus_parser()),
        ("midas-hand-tune", tuner_parser()),
    ):
        args = parser.parse_args([])
        for flag in (
            "start_armed",
            "allow_unhomed",
            "hardware_current_limit",
            "hardware_max_step_rad",
            "configure_hardware",
        ):
            assert hasattr(args, flag), f"{name} is missing --{flag.replace('_', '-')}"
        assert args.start_armed is False, f"{name} arms by default"
        assert "hardware" in parser.parse_args(["--backend", "hardware"]).backend


def test_no_entry_point_defaults_to_hardware():
    """A bare invocation must never energise a motor."""

    from midas_hand_teleop.manus_glove.manus_teleop import (
        build_parser as manus_parser,
    )
    from midas_hand_teleop.tuner.cli import build_parser as tuner_parser
    from midas_hand_teleop.webcam_demo import build_parser as webcam_parser

    for parser in (webcam_parser(), manus_parser(), tuner_parser()):
        assert parser.parse_args([]).backend != "hardware"


def test_no_entry_point_silently_accepts_a_left_glove():
    """The tuner used to retarget a left glove onto the right-hand model
    without a word. The two layers fail differently -- analytic is
    reflection-invariant so mirroring changes nothing, while the Cartesian
    modes produce a genuinely mirrored solve -- so there is no reading of it
    that works, and the check is shared to keep them in step."""

    from midas_hand_teleop.backend_cli import check_side_supported

    check_side_supported("right")  # must not raise
    with pytest.raises(SystemExit, match="Left-hand teleop is not implemented"):
        check_side_supported("left")

    from midas_hand_teleop.tuner import cli as tuner_cli

    with pytest.raises(SystemExit, match="Left-hand teleop is not implemented"):
        tuner_cli.main(["--side", "left", "--duration", "0.1"])


def test_the_glove_path_does_not_need_opencv_or_mediapipe():
    """They are 252 MB between them and only the webcam path uses them, so
    they are the [webcam] extra. A module-level import anywhere on the glove
    path would quietly make them required again."""

    import subprocess
    import sys

    # A fresh interpreter, so nothing another test imported can mask this.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "import midas_hand_teleop.tuner.cli;"
            "import midas_hand_teleop.manus_glove.manus_teleop;"
            "import midas_hand_teleop.backend_cli;"
            "leaked = [m for m in ('cv2', 'mediapipe') if m in sys.modules];"
            "print(leaked)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "[]", (
        f"the glove path imported {result.stdout.strip()} at module scope"
    )


def test_the_webcam_entry_point_names_the_extra_if_it_is_missing():
    """Rather than an ImportError traceback on a 252 MB dependency."""

    import inspect

    from midas_hand_teleop import webcam_demo

    source = inspect.getsource(webcam_demo.main)
    assert "import cv2" in source, "cv2 must be imported lazily inside main()"
    assert "[webcam]" in source
