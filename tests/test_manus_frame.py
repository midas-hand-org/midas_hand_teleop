"""Tests for the glove -> MediaPipe-world frame transform in manus_bridge."""

from __future__ import annotations

import numpy as np
import pytest

from midas_hand_teleop.manus_glove.manus_bridge import (
    DEFAULT_GLOVE_FRAME,
    GLOVE_FRAME_PRESETS,
    NODE_TO_MEDIAPIPE,
    _positions_to_mediapipe,
    resolve_glove_frame,
)


def _raw_positions(n: int = 25) -> np.ndarray:
    return np.arange(n * 3, dtype=np.float64).reshape(n, 3) * 0.001


def test_default_reflects_to_match_the_robot_handedness():
    """The published skeleton must match the robot's chirality.

    Measured on live glove data: as published, curling a finger moves its tip
    +40 mm along the palm normal while the robot's fingers curl -27 mm along
    it, so the raw right-hand skeleton is mirrored relative to the MIDAS hand.

    This test previously asserted the opposite (det > 0, "chirality
    preserving"), which encoded the bug: the analytic map is provably
    reflection-invariant, so a mirrored frame is invisible to it, while the
    DexPilot optimizer is asked to bend the fingers backwards and holds them
    extended instead.
    """

    name, transform = resolve_glove_frame(None)
    assert name == DEFAULT_GLOVE_FRAME
    assert np.linalg.det(transform) < 0, "the default must be a reflection"


def test_all_presets_are_orthogonal_axis_maps():
    for name, transform in GLOVE_FRAME_PRESETS.items():
        assert transform.shape == (3, 3)
        assert abs(abs(np.linalg.det(transform)) - 1.0) < 1e-9, name


def test_env_override(monkeypatch):
    monkeypatch.setenv("MIDAS_GLOVE_FRAME", "flip_z")
    name, transform = resolve_glove_frame(None)
    assert name == "flip_z"
    assert np.linalg.det(transform) < 0


def test_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MIDAS_GLOVE_FRAME", "flip_z")
    name, _ = resolve_glove_frame("identity")
    assert name == "identity"


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_glove_frame("sideways")


def test_identity_maps_nodes_by_table():
    pos = _raw_positions()
    kp = _positions_to_mediapipe(pos, GLOVE_FRAME_PRESETS["identity"])
    assert kp.shape == (21, 3)
    for node_idx, mp_idx in NODE_TO_MEDIAPIPE.items():
        assert np.allclose(kp[mp_idx], pos[node_idx])


def test_flip_y_reproduces_legacy_behavior():
    pos = _raw_positions()
    kp_identity = _positions_to_mediapipe(pos, GLOVE_FRAME_PRESETS["identity"])
    kp_flip_y = _positions_to_mediapipe(pos, GLOVE_FRAME_PRESETS["flip_y"])
    # The legacy transform was `keypoints[:, 1] *= -1`.
    expected = kp_identity * np.array([1.0, -1.0, 1.0])
    assert np.allclose(kp_flip_y, expected)


def test_returns_none_when_too_few_nodes():
    assert _positions_to_mediapipe(np.zeros((5, 3)), GLOVE_FRAME_PRESETS["identity"]) is None
