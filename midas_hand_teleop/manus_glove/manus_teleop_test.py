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
    assert build_tuning(_args(profile="glove")) == glove_tuning()
    assert build_tuning(_args(profile="vision")) == vision_tuning()


def test_gain_flag_overrides_only_that_field():
    tuned = build_tuning(_args(profile="glove", thumb_cmc_side_gain=3.3))
    assert tuned.thumb_cmc_side_gain == 3.3
    # Everything else still comes from the glove profile.
    base = glove_tuning()
    assert tuned.finger_abad_gain == base.finger_abad_gain
    assert tuned.finger_smoothing_alpha == base.finger_smoothing_alpha


def test_filter_alpha_full_mode_defaults_off():
    # Full retargeter smooths internally -> no extra glove-side EMA by default.
    assert resolve_filter_alpha(_args(), full_mode=True, tuning=glove_tuning()) == 1.0


def test_filter_alpha_geometric_mode_uses_profile():
    g = glove_tuning()
    assert resolve_filter_alpha(_args(), full_mode=False, tuning=g) == g.finger_smoothing_alpha


def test_explicit_filter_alpha_wins():
    assert resolve_filter_alpha(_args(filter_alpha=0.5), full_mode=True, tuning=glove_tuning()) == 0.5
    assert resolve_filter_alpha(_args(filter_alpha=0.5), full_mode=False, tuning=glove_tuning()) == 0.5
