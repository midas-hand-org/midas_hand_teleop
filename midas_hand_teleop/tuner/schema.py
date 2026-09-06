"""Describe the tuning parameters so the UI can build itself.

The form is generated from the dataclass fields rather than hand-written in
JavaScript, so adding a parameter cannot leave the UI silently out of date —
which is exactly how ~20 dead CLI flags accumulated in webcam_demo.

Slider tracks for the ``*_range`` fields come from the robot's own joint
limits, so the UI physically cannot present a value the hand cannot reach, and
the unreachable part of the historical default is visible as unused track.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

from midas_hand_retargeter.model import MIDAS_RIGHT_HAND, HandModel
from midas_hand_retargeter.params import (
    MODE_SECTIONS,
    DexPilotParams,
    FingerParams,
    RetargetProfile,
    ThumbParams,
)

#: UI slider bounds for scalar knobs that are not joint ranges.
#: (minimum, maximum, step). Chosen to bracket useful tuning, not to be limits.
SCALAR_BOUNDS: dict[str, tuple[float, float, float]] = {
    "curl_gain": (0.1, 3.0, 0.01),
    "curl_max_bend": (0.3, 3.0, 0.01),
    "curl_pip_weight": (0.0, 1.0, 0.01),
    "splay_gain": (0.0, 4.0, 0.01),
    "splay_deadzone": (0.0, 0.4, 0.005),
    "splay_limit": (0.0, 0.79, 0.005),
    "splay_curl_damping": (0.0, 1.0, 0.01),
    "smoothing_alpha": (0.02, 1.0, 0.01),
    "flexion_gain": (0.1, 3.0, 0.01),
    "mcp_max_bend": (0.3, 3.0, 0.01),
    "dip_max_bend": (0.3, 3.0, 0.01),
    "dip_follows_mcp": (0.0, 1.0, 0.01),
    "cmc_side_gain": (0.0, 4.0, 0.01),
    "cmc_side_neutral_angle": (-1.2, 1.2, 0.01),
    "cmc_side_deadzone": (0.0, 0.4, 0.005),
    "cmc_side_open": (-0.785, 0.9, 0.005),
    "cmc_roll_gain": (0.0, 3.0, 0.01),
    "cmc_roll_span": (0.05, 1.5, 0.01),
    "cmc_roll_deadzone": (0.0, 0.4, 0.005),
    # DexPilot solver knobs.
    "scaling_factor": (0.5, 2.5, 0.01),
    "huber_delta": (0.005, 0.15, 0.005),
    "norm_delta": (0.0, 0.05, 0.0005),
    "project_dist": (0.0, 0.10, 0.001),   # 0 disables pinch snapping entirely
    "escape_dist": (0.01, 0.15, 0.001),
    "eta1": (0.0, 0.05, 0.0005),
    "eta2": (0.0, 0.10, 0.001),
    "low_pass_alpha": (0.05, 1.0, 0.01),
    "thumb_vector_scale": (0.8, 1.6, 0.01),
    "spread_scale": (0.8, 1.6, 0.01),
}

#: Which joint's limits bound each ``*_range`` field. ``{finger}`` is filled in.
RANGE_JOINTS: dict[str, str] = {
    "mcp_pitch_range": "{finger}_mcp_pitch_joint",
    "pip_range": "{finger}_pip_joint",
    "mcp_range": "thumb_mcp_joint",
    "dip_range": "thumb_dip_joint",
    "cmc_side_range": "thumb_cmc_side_joint",
    "cmc_roll_range": "thumb_cmc_roll_joint",
}

#: Fields shown before the "advanced" fold, in this order.
BASIC_FIELDS = {
    "curl_gain",
    "curl_max_bend",
    "mcp_pitch_range",
    "pip_range",
    "splay_gain",
    "smoothing_alpha",
    "enabled",
    "flexion_gain",
    "mcp_range",
    "dip_range",
    "cmc_side_gain",
    "cmc_roll_gain",
    "scaling_factor",
    "spread_scale",
    "project_dist",
    "eta1",
}

#: Short help shown under each control.
HELP: dict[str, str] = {
    "curl_gain": "Multiplies measured bend. Raise if the finger closes too late.",
    "curl_max_bend": "Human bend (rad) meaning fully closed. Raise if the finger "
    "saturates before your hand is actually shut.",
    "curl_pip_weight": "Share of the blend taken from the PIP bend; the rest "
    "comes from the DIP bend.",
    "mcp_pitch_range": "Commanded open/closed angles. Widening reclaims travel "
    "the historical default never reached.",
    "pip_range": "Commanded open/closed angles for the PIP joint.",
    "splay_gain": "Multiplies sideways finger spread.",
    "splay_deadzone": "Lateral angle ignored around neutral, to reject jitter.",
    "splay_limit": "Hard clamp on the abduction command.",
    "splay_curl_damping": "How much a closed finger suppresses splay; splay "
    "tracking gets unreliable as the finger curls.",
    "smoothing_alpha": "1.0 = instant, smaller = smoother but laggier.",
    "enabled": "Off freezes this digit at its last command (it does not open).",
    "flexion_gain": "Multiplies thumb MCP/DIP bend.",
    "mcp_max_bend": "Thumb bend (rad) meaning fully flexed at the MCP.",
    "dip_max_bend": "Thumb bend (rad) meaning fully flexed at the tip.",
    "mcp_range": "Commanded thumb MCP open/closed angles.",
    "dip_range": "Commanded thumb tip open/closed angles.",
    "dip_follows_mcp": "Floor tying tip curl to MCP curl, for when the IP bend "
    "is poorly seen.",
    "cmc_side_gain": "Multiplies the thumb's in-plane side sweep.",
    "cmc_side_range": "Commanded limits for thumb side sweep.",
    "cmc_side_neutral_angle": "Measured angle treated as the thumb's rest pose.",
    "cmc_side_deadzone": "Side angle ignored around neutral.",
    "cmc_side_open": "Commanded value at the neutral angle.",
    "cmc_roll_gain": "Multiplies thumb opposition (rolling across the palm).",
    "cmc_roll_range": "Commanded limits for thumb opposition.",
    "cmc_roll_span": "Out-of-plane angle spanning neutral to full opposition.",
    "cmc_roll_deadzone": "Opposition angle ignored around neutral.",
    # DexPilot.
    "scaling_factor": "Your hand size relative to the robot's. THE key knob in "
    "this mode — the analytic map ignores hand size entirely, this one does not. "
    "Too small and the fingers over-close; too large and they never reach.",
    "huber_delta": "Error width (m) below which tracking is quadratic. Smaller "
    "chases small errors harder but is jitterier.",
    "norm_delta": "Temporal regularizer: how strongly each solve is anchored to "
    "the previous one. Larger is smoother but laggier.",
    "project_dist": "Fingertip gap (m) at which a pair is treated as trying to "
    "touch, snapping it to eta. This is what makes pinches land instead of "
    "hover — and also what makes fingertips STICK together: once snapped, a "
    "pair only releases past escape_dist, so the robot holds a 1 mm gap while "
    "your own fingers open to 48 mm. Set to 0 to disable snapping entirely.",
    "escape_dist": "Gap (m) at which a snapped pair releases. Must exceed "
    "project_dist; the difference is hysteresis against chatter.",
    "eta1": "Target gap (m) for thumb-to-finger pairs once snapped.",
    "eta2": "Target gap (m) for finger-to-finger pairs once snapped.",
    "low_pass_alpha": "Solver-side low-pass. 1.0 = off.",
    "spread_scale": "How far apart the fingers are, independently of how far "
    "they reach. The MIDAS fingertips span 61 mm at rest against ~48 mm for a "
    "scaled human hand, so without this the solver swings each finger sideways "
    "to reach targets inside its own knuckle spacing. Set by 'Calibrate hand "
    "size'; 1.0 = off.",
    "thumb_vector_scale": "Extra reach given to the thumb's own targets. 1.0 = "
    "off, and off is the default: the thumb-root rebase already removes most of "
    "the MIDAS thumb's proportional excess. Raise toward ~1.15 for a straighter "
    "thumb, at a measured cost of 1-3 mm of fingertip accuracy.",
}

SECTIONS = ("thumb", "index", "middle", "ring", "dexpilot")

SECTION_PARAMS: dict[str, type] = {
    "thumb": ThumbParams,
    "index": FingerParams,
    "middle": FingerParams,
    "ring": FingerParams,
    "dexpilot": DexPilotParams,
}

#: Shown when a mode reads none of the sections the UI can edit.
_MODE_NOTES = {
    "vector": "mode=vector is the pure palm-rooted optimizer. It exposes no "
              "tunable parameters here — switch to dexpilot to tune the solver, "
              "or analytic to tune per-finger response.",
}


def _describe_field(section: str, field, model: HandModel) -> dict:
    name = field.name
    entry: dict = {
        "name": name,
        "path": f"{section}.{name}",
        "label": name.replace("_", " "),
        "help": HELP.get(name, ""),
        "advanced": name not in BASIC_FIELDS,
    }
    if field.type == "bool" or isinstance(field.default, bool):
        entry["kind"] = "bool"
        return entry
    if name in RANGE_JOINTS:
        joint = RANGE_JOINTS[name].format(finger=section)
        lower, upper = model.limits(joint)
        entry.update(kind="range", joint=joint, min=lower, max=upper, step=0.005)
        return entry
    low, high, step = SCALAR_BOUNDS.get(name, (0.0, 2.0, 0.01))
    entry.update(kind="scalar", min=low, max=high, step=step)
    return entry


def build_schema(
    model: HandModel = MIDAS_RIGHT_HAND, mode: str = "analytic"
) -> dict:
    """UI description for one retargeting mode.

    Only the sections the mode actually reads are returned. A slider that
    silently does nothing is the worst thing a tuning tool can offer — it is
    how ~20 dead CLI flags accumulated in webcam_demo — so the per-finger
    controls are simply absent in dexpilot mode, and vice versa.
    """

    defaults = RetargetProfile()
    active = MODE_SECTIONS.get(mode, MODE_SECTIONS["analytic"])
    sections = []
    for section in SECTIONS:
        if section not in active:
            continue
        controls = [
            _describe_field(section, field, model)
            for field in dataclass_fields(SECTION_PARAMS[section])
        ]
        label = "Solver" if section == "dexpilot" else section.capitalize()
        sections.append({"name": section, "label": label, "controls": controls})

    return {
        "mode": mode,
        "note": _MODE_NOTES.get(mode, ""),
        "sections": sections,
        "defaults": defaults.to_flat_dict(),
        "joints": [
            {"name": name, "lower": float(low), "upper": float(high)}
            for name, (low, high) in (
                (n, model.limits(n)) for n in model.joint_names
            )
        ],
    }
