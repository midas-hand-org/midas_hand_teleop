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
        backend="hardware", hardware_config=None, allow_unhomed=False,
        start_armed=False, mujoco_xml=None, mujoco_repo=None,
        mujoco_viewer=False, mujoco_steps=None,
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

    args = argparse.Namespace(side="left", retarget="geometric", calibrate_delay=0.0,
                              profile="glove", filter_alpha=None, host="localhost")
    for name in ("finger_curl_gain", "finger_abad_gain", "finger_smoothing_alpha",
                 "thumb_cmc_gain", "thumb_cmc_side_gain", "thumb_cmc_roll_gain",
                 "thumb_flexion_gain", "thumb_smoothing_alpha"):
        setattr(args, name, None)
    with pytest.raises(SystemExit, match="Left-hand teleop is not implemented"):
        manus_teleop.run(args)


def test_dead_cli_flags_are_gone():
    """~20 flags were parsed and silently discarded; they misled tuning work."""

    import inspect

    from midas_hand_teleop import webcam_demo

    source = inspect.getsource(webcam_demo)
    for flag in ("--thumb-pinch-gain", "--thumb-cmc-roll-signed",
                 "--finger-abad-limit", "--thumb-cmc-side-oppose"):
        assert flag not in source, f"{flag} names behaviour that does not exist"
    assert "argparse.SUPPRESS" not in source
