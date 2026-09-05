"""Tests for the diagnostics harness (canonical poses, analysis, invariance)."""

from __future__ import annotations

import numpy as np
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.tuning import vision_tuning

from midas_hand_teleop.diagnostics import (
    CANONICAL_POSES,
    analyze_frame,
    build_hand,
    frame_handedness,
)


def _targets(kp):
    info = analyze_frame(kp, vision_tuning())
    return np.array([info["targets"][n] for n in ACTIVE_JOINT_NAMES])


def test_flat_pose_maps_to_zero_targets():
    # The builder is in the correct convention, so a flat hand is exact MIDAS zero.
    info = analyze_frame(build_hand(), vision_tuning())
    for name in ACTIVE_JOINT_NAMES:
        assert abs(info["targets"][name]) < 1e-6, name
    # Flat hand => palm basis is the canonical axis frame.
    assert np.allclose(info["palm_forward"], [0, 1, 0], atol=1e-6)
    assert np.allclose(info["palm_lateral"], [1, 0, 0], atol=1e-6)
    assert np.allclose(info["palm_normal"], [0, 0, 1], atol=1e-6)


def test_fist_curls_fingers_negative():
    flat = _targets(build_hand())
    fist = _targets(build_hand(finger_curl=1.0, thumb_curl=0.8))
    for finger in ("index", "middle", "ring"):
        i_pitch = ACTIVE_JOINT_NAMES.index(f"{finger}_mcp_pitch_joint")
        i_pip = ACTIVE_JOINT_NAMES.index(f"{finger}_pip_joint")
        assert fist[i_pitch] < flat[i_pitch] - 0.5
        assert fist[i_pip] < flat[i_pip] - 0.5


def test_canonical_poses_are_valid_frames():
    for name, (kp, signature) in CANONICAL_POSES.items():
        assert kp.shape == (21, 3), name
        assert isinstance(signature, str)
        analyze_frame(kp, vision_tuning())  # must not raise


def test_analyze_is_frame_invariant():
    # Same property the retargeter pins, checked through the harness path.
    kp = build_hand(
        finger_curl={"index": 0.7, "middle": 0.3, "ring": 0.9},
        finger_splay={"index": -0.3, "middle": 0.0, "ring": 0.4},
        thumb_curl=0.5,
        thumb_side=0.3,
        thumb_oppose=0.8,
    )
    base = _targets(kp)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # arbitrary rotation/reflection
    moved = (kp @ q.T) + np.array([0.5, -0.4, 0.2])
    assert np.allclose(_targets(moved), base, atol=1e-5)


def test_compare_detects_thumb_mapping_error():
    # Swapping the thumb IP<->tip nodes (a plausible mapping bug) must show up as
    # a large error on the thumb targets, not the fingers.
    good = CANONICAL_POSES["pinch"][0].copy()
    bad = good.copy()
    bad[[3, 4]] = bad[[4, 3]]
    g = analyze_frame(good, vision_tuning())["targets"]
    b = analyze_frame(bad, vision_tuning())["targets"]
    thumb_delta = abs(b["thumb_dip_joint"] - g["thumb_dip_joint"])
    finger_delta = max(
        abs(b[j] - g[j]) for j in ACTIVE_JOINT_NAMES if j.startswith(("index", "middle", "ring"))
    )
    assert thumb_delta > 0.5
    assert finger_delta < 1e-6


def test_frame_handedness_reads_curled_hand():
    fist = build_hand(finger_curl=1.0)
    evidence, confidence = frame_handedness(fist)
    assert confidence > 0.005  # curled enough to read
    assert evidence < 0  # correct convention => dorsal normal
    # A mirrored copy flips the sign.
    mirrored = fist @ np.diag([1.0, -1.0, 1.0])
    ev_m, _ = frame_handedness(mirrored)
    assert ev_m > 0
