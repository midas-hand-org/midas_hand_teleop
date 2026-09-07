"""Tests for the manus_teleop tuning-profile and smoothing resolution."""

from __future__ import annotations

import argparse

from midas_hand_retargeter.tuning import glove_tuning, vision_tuning

from midas_hand_teleop.manus_glove.manus_teleop import (
    build_tuning,
    resolve_filter_alpha,
)


def _args(**kw):
    base = dict(
        preset=None,
        scaling_factor=None,
        profile="glove",
        filter_alpha=None,
        finger_curl_gain=None,
        finger_abad_gain=None,
        thumb_cmc_side_gain=None,
        thumb_cmc_roll_gain=None,
        thumb_flexion_gain=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_profile_selects_base_tuning():
    # build_tuning returns (tuning, neutral_offsets): a preset carries the zero
    # pose it was saved with, and --profile has none.
    assert build_tuning(_args(profile="glove")) == (glove_tuning(), {})
    assert build_tuning(_args(profile="vision")) == (vision_tuning(), {})


def test_gain_flag_overrides_only_that_field():
    tuned, _ = build_tuning(_args(profile="glove", thumb_cmc_side_gain=3.3))
    assert tuned.thumb_cmc_side_gain == 3.3
    # Everything else still comes from the glove profile.
    base = glove_tuning()
    assert tuned.finger_abad_gain == base.finger_abad_gain
    assert tuned.finger_smoothing_alpha == base.finger_smoothing_alpha


def test_filter_alpha_defaults_off():
    """The retargeter smooths internally from the profile, so a second
    glove-side EMA stage would double-smooth."""

    assert resolve_filter_alpha(_args()) == 1.0


def test_explicit_filter_alpha_wins():
    assert resolve_filter_alpha(_args(filter_alpha=0.5)) == 0.5


def test_nothing_is_commanded_before_the_first_glove_frame():
    """last_control is seeded to every joint = 0.0 and last_data_time to the
    start, so for the first --stale-timeout seconds the loop used to command
    the all-zeros pose having received no glove frame at all.

    On hardware that is uncommanded motion at power-on: arm() seeds from the
    measured pose, the next tick asks for URDF zero, and the slew limiter walks
    the whole hand there in ~0.3 s -- inside the deadman window, so the deadman
    fires only after the motion has finished.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "midas_hand_teleop.manus_glove.manus_teleop",
         "--backend", "print", "--retarget", "full",
         "--duration", "2", "--no-proxy"],
        capture_output=True, text=True, timeout=120,
    )
    combined = result.stdout + result.stderr
    assert "total solves=0" in combined, combined[-2000:]
    # PrintBackend prints the joint dict on every send. No send, no dict.
    assert "index_mcp_pitch_joint" not in combined, (
        "commanded a pose with no glove frame:\n" + combined[-2000:]
    )


def test_preset_supplies_the_dexpilot_section(tmp_path):
    """The only way a tuned DexPilot session reaches this CLI.

    --profile produces a legacy flat RetargeterTuning, which
    RetargetProfile.from_legacy_tuning converts without ever populating
    `dexpilot=` -- so every solver knob fell back to a library default while
    the startup log printed a scaling_factor that was not in force.
    """

    from midas_hand_retargeter import presets
    from midas_hand_retargeter.params import RetargetProfile

    saved = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.42, "dexpilot.abduction_limit": 0.1}
    )
    presets.save(tmp_path / "tuned.json", saved, neutral_offsets={"index_pip_joint": -0.2})

    profile, neutral = build_tuning(_args(preset=str(tmp_path / "tuned.json")))
    assert profile.dexpilot.scaling_factor == 1.42
    assert profile.dexpilot.abduction_limit == 0.1
    assert neutral == {"index_pip_joint": -0.2}


def test_scaling_factor_flag_overrides_the_preset(tmp_path):
    from midas_hand_retargeter import presets
    from midas_hand_retargeter.params import RetargetProfile

    presets.save(
        tmp_path / "t.json",
        RetargetProfile().with_values({"dexpilot.scaling_factor": 1.42}),
    )
    profile, _ = build_tuning(
        _args(preset=str(tmp_path / "t.json"), scaling_factor=1.05)
    )
    assert profile.dexpilot.scaling_factor == 1.05


def test_the_cartesian_modes_default_to_the_real_four_bar_coupling():
    """The CLI used to hardcode fixed_passive, which overrode the
    mode-dependent default and left the solver aiming at a fingertip up to
    59 mm from where the linkage actually puts it."""

    from midas_hand_teleop.manus_glove.manus_teleop import build_parser

    assert build_parser().parse_args([]).coupling_mode is None
