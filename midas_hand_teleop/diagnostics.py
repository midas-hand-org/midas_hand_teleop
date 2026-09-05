"""diagnostics.py — inspect and compare hand-tracking sources through the retargeter.

The MIDAS retargeter was tuned against the MediaPipe *vision* source and works
poorly on the Manus glove. It is tempting to blame the input coordinate frame,
but the default (``--retarget full``, postprocess ON) path is provably
**invariant to any rotation, reflection, or translation** of the input: every
finger/thumb target is built from angles and dot products against a palm-local
basis derived from the landmarks themselves, and thumb roll takes ``abs()`` of
the opposition angle. ``test_diagnostics.py`` pins this invariance. So the frame
is NOT why the glove looks wrong.

What actually differs between sources is the **relative keypoint geometry** (how
the 21 points sit relative to each other), which comes from:
  1. the Manus node -> MediaPipe mapping (esp. the thumb — the pinch complaint),
  2. the glove skeleton's proportions / rest pose (measured bend ROM), and
  3. the neutral offset (vision captures a neutral pose; the glove defaults off).

This harness measures that geometry instead of tuning blind:

* ``poses``   — print the 13 targets the retargeter produces for canonical poses
                built in the correct convention: the ground-truth signature each
                physical pose should reproduce.
* ``analyze`` — load a captured ``(21,3)`` frame (``.npy``) and dump the palm
                basis, per-finger curl/splay, thumb angles, and 13 targets.
* ``compare`` — score a captured frame against a named canonical pose (L1 over
                the 13 targets), so you can see WHICH joints are off and by how
                much — that points at the mapping / ROM / neutral cause.
* ``presets`` — apply every glove frame preset to a captured frame; it will show
                the targets are identical across presets (the invariance above),
                confirming the frame is a red herring for the postprocess path.
                (Frame choice only matters under ``--no-*-postprocess``.)
* ``capture`` — subscribe to a live glove topic, freeze one frame, save to
                ``.npy`` for offline ``analyze`` / ``compare`` / ``presets``.
* ``live``    — subscribe, freeze the latest frame, and analyze it.

Run:
    midas-hand-diag poses
    midas-hand-diag capture --side right --out /tmp/pinch.npy    # hold a pinch
    midas-hand-diag compare --file /tmp/pinch.npy --expect pinch
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np

from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.postprocess import (
    FINGER_LANDMARKS,
    THUMB_LANDMARKS,
    _palm_basis,
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)
from midas_hand_retargeter.tuning import RetargeterTuning, tuning_for_source
from midas_hand_teleop.manus_glove.manus_bridge import GLOVE_FRAME_PRESETS

logger = logging.getLogger("midas_hand_diag")

# ─── Canonical pose builder (correct/vision convention) ────────────────────
# Right hand, wrist at origin: +Y distal (wrist -> fingertips), +X toward the
# pinky (index MCP -> ring MCP), +Z dorsal (back of hand). Fingers curl toward
# -Z (palmar). This matches the convention pinned by the retargeter's postprocess
# tests and ``fake_glove_publisher.synthetic_hand``.
_SEG = 0.035  # phalanx length (m)
_MCP_Y = 0.045  # finger-base distance from the wrist along +Y
# Symmetric about the middle finger so the "flat" pose has palm_forward == +Y and
# palm_lateral == +X, i.e. zero splay at rest.
_FINGER_X = {"index": -0.020, "middle": 0.0, "ring": 0.020}
_PALMAR = np.array([0.0, 0.0, -1.0])


def _finger_points(base: np.ndarray, curl: float, splay: float) -> np.ndarray:
    """Four points (MCP, PIP, DIP, tip) for one finger.

    ``curl`` in [0, 1] flexes the chain toward -Z (palmar). ``splay`` (rad)
    abducts the finger about the palm normal (+Z): positive rotates the heading
    toward +X (pinky side).
    """

    mcp_flex, pip_flex, dip_flex = 0.5 * curl, 1.2 * curl, 0.8 * curl
    heading = np.array([math.sin(splay), math.cos(splay), 0.0])
    pts = [np.asarray(base, dtype=np.float64)]
    angle = mcp_flex
    for extra in (pip_flex, dip_flex, 0.0):
        direction = math.cos(angle) * heading + math.sin(angle) * _PALMAR
        pts.append(pts[-1] + _SEG * direction)
        angle += extra
    return np.array(pts)


def _thumb_points(curl: float, side: float, oppose: float) -> np.ndarray:
    """Four points (CMC, MCP, IP, tip) for the thumb.

    ``curl`` flexes the thumb chain, ``side`` (rad) sweeps it in the palm plane
    (positive toward the fingers / +Y), ``oppose`` (rad) rolls it toward the
    palm (-Z).
    """

    cmc = np.array([-0.035, 0.010, 0.005])
    # Radial base heading swept in-plane by ``side`` and tilted out of plane
    # toward palmar by ``oppose``. The resting angle is chosen so a relaxed thumb
    # sits near the vision neutral (THUMB_CMC_SIDE_NEUTRAL_ANGLE), i.e. flat ~0.
    base = np.array([-0.31, 0.95, 0.0])
    base = base / np.linalg.norm(base)
    rot = np.array(
        [
            base[0] * math.cos(side) - base[1] * math.sin(side),
            base[0] * math.sin(side) + base[1] * math.cos(side),
            0.0,
        ]
    )
    heading = math.cos(oppose) * rot + math.sin(oppose) * _PALMAR
    heading = heading / np.linalg.norm(heading)
    # Four points (CMC, MCP, IP, tip): three segments accumulating flex.
    pts = [cmc]
    angle = 0.6 * curl
    for extra in (0.9 * curl, 0.6 * curl, 0.0):
        direction = math.cos(angle) * heading + math.sin(angle) * _PALMAR
        pts.append(pts[-1] + _SEG * direction)
        angle += extra
    return np.array(pts)


def build_hand(
    *,
    finger_curl: dict[str, float] | float = 0.0,
    finger_splay: dict[str, float] | float = 0.0,
    thumb_curl: float = 0.0,
    thumb_side: float = 0.0,
    thumb_oppose: float = 0.0,
) -> np.ndarray:
    """Build a canonical right-hand ``(21, 3)`` pose in the correct convention."""

    def _per_finger(value: dict[str, float] | float, finger: str) -> float:
        return float(value[finger]) if isinstance(value, dict) else float(value)

    kp = np.zeros((21, 3), dtype=np.float64)
    for finger, (mcp_i, pip_i, dip_i, tip_i) in FINGER_LANDMARKS.items():
        base = np.array([_FINGER_X[finger], _MCP_Y, 0.0])
        chain = _finger_points(
            base,
            _per_finger(finger_curl, finger),
            _per_finger(finger_splay, finger),
        )
        kp[mcp_i], kp[pip_i], kp[dip_i], kp[tip_i] = chain
    thumb = _thumb_points(thumb_curl, thumb_side, thumb_oppose)
    for point, index in zip(thumb, THUMB_LANDMARKS):
        kp[index] = point
    return kp


# Canonical poses + a one-line signature of what their targets should show.
CANONICAL_POSES: dict[str, tuple[np.ndarray, str]] = {
    "flat": (build_hand(), "all targets ~0 (open/relaxed hand)"),
    "fist": (
        build_hand(finger_curl=1.0, thumb_curl=0.8, thumb_oppose=0.6),
        "finger mcp_pitch/pip strongly negative; thumb mcp/dip negative",
    ),
    "spread": (
        build_hand(finger_splay={"index": -0.5, "middle": 0.0, "ring": 0.5}),
        "index and ring mcp_abad have OPPOSITE signs (fingers splay apart), middle ~0",
    ),
    "pinch": (
        build_hand(
            finger_curl={"index": 0.5, "middle": 0.0, "ring": 0.0},
            thumb_curl=0.4,
            thumb_side=0.5,
            thumb_oppose=0.9,
        ),
        "thumb_cmc_roll > 0 (opposition), index curled; middle/ring open",
    ),
    "thumb_oppose": (
        build_hand(thumb_oppose=1.0),
        "thumb_cmc_roll strongly > 0; fingers ~0",
    ),
}


# ─── Analysis ──────────────────────────────────────────────────────────────


def analyze_frame(kp: np.ndarray, tuning: RetargeterTuning) -> dict:
    """Compute palm basis, per-finger/thumb intermediates, and the 13 targets."""

    kp = np.asarray(kp, dtype=np.float64)
    forward, lateral, normal = _palm_basis(kp)
    targets = {}
    targets.update(finger_joint_targets_from_landmarks(kp, tuning))
    targets.update(thumb_joint_targets_from_landmarks(kp, tuning))

    per_finger = {}
    for finger, (mcp_i, pip_i, dip_i, tip_i) in FINGER_LANDMARKS.items():
        proximal = kp[pip_i] - kp[mcp_i]
        distal = kp[tip_i] - kp[dip_i]
        per_finger[finger] = {
            "pitch": targets[f"{finger}_mcp_pitch_joint"],
            "pip": targets[f"{finger}_pip_joint"],
            "abad": targets[f"{finger}_mcp_abad_joint"],
            "tip_out_of_plane": float(np.dot(kp[tip_i] - kp[mcp_i], normal)),
        }
    return {
        "palm_forward": forward,
        "palm_lateral": lateral,
        "palm_normal": normal,
        "targets": targets,
        "per_finger": per_finger,
    }


def frame_handedness(kp: np.ndarray) -> tuple[float, float]:
    """Report the input frame's handedness from finger-curl direction.

    Returns ``(evidence, confidence)``. ``evidence < 0`` means the palm normal
    ``cross(lateral, forward)`` points dorsally (a right-handed embedding of the
    hand); ``evidence > 0`` means the frame is mirrored (left-handed embedding).
    The signal only exists when fingers are curled, so ``confidence`` (mean
    absolute out-of-plane reach) must be non-trivial — capture a fist or pinch.

    NOTE: this does NOT predict whether the postprocess targets are right. The
    postprocess is handedness-invariant (see the module docstring); this is only
    a sanity check for the optimizer (``--no-*-postprocess``) path and for
    interpreting the palm-basis dump.
    """

    kp = np.asarray(kp, dtype=np.float64)
    _, _, normal = _palm_basis(kp)
    reaches = []
    for mcp_i, _pip_i, _dip_i, tip_i in FINGER_LANDMARKS.values():
        reaches.append(float(np.dot(kp[tip_i] - kp[mcp_i], normal)))
    reaches = np.array(reaches)
    return float(reaches.mean()), float(np.abs(reaches).mean())


def frame_handedness_note(kp: np.ndarray, *, min_confidence: float = 0.005) -> str:
    evidence, confidence = frame_handedness(kp)
    if confidence < min_confidence:
        return f"flat (confidence={confidence:.4f}; curl the hand to read handedness)"
    hand = "right-handed embedding" if evidence < 0 else "left-handed (mirrored) embedding"
    return f"{hand} (evidence={evidence:+.4f}); does NOT affect postprocess targets"


# ─── Presentation ──────────────────────────────────────────────────────────


def _print_analysis(label: str, kp: np.ndarray, tuning: RetargeterTuning) -> None:
    info = analyze_frame(kp, tuning)
    print(f"── {label} ──")
    print(
        f"  palm  forward={np.round(info['palm_forward'], 3)} "
        f"lateral={np.round(info['palm_lateral'], 3)} "
        f"normal={np.round(info['palm_normal'], 3)}"
    )
    print(f"  frame: {frame_handedness_note(kp)}")
    for name in ACTIVE_JOINT_NAMES:
        print(f"    {name:24s} {info['targets'][name]:+.3f}")


def _resolve_expect(name: str) -> np.ndarray:
    if name not in CANONICAL_POSES:
        raise SystemExit(f"--expect must be one of {sorted(CANONICAL_POSES)}, got {name!r}")
    return CANONICAL_POSES[name][0]


def cmd_poses(args: argparse.Namespace) -> None:
    tuning = tuning_for_source(args.profile)
    print(f"Canonical poses (profile={args.profile}) — expected target signatures:\n")
    for name, (kp, signature) in CANONICAL_POSES.items():
        print(f"# {name}: {signature}")
        _print_analysis(name, kp, tuning)
        print()


def cmd_analyze(args: argparse.Namespace) -> None:
    kp = _load_frame(args.file)
    tuning = tuning_for_source(args.profile)
    _print_analysis(f"{args.file}", kp, tuning)


def cmd_compare(args: argparse.Namespace) -> None:
    """Score a captured frame against a canonical pose, per joint."""

    kp = _load_frame(args.file)
    tuning = tuning_for_source(args.profile)
    ref_kp = _resolve_expect(args.expect)
    got = analyze_frame(kp, tuning)["targets"]
    ref = analyze_frame(ref_kp, tuning)["targets"]
    print(f"Compare {args.file} vs canonical '{args.expect}' (profile={args.profile})")
    print(f"  expected: {CANONICAL_POSES[args.expect][1]}\n")
    total = 0.0
    for name in ACTIVE_JOINT_NAMES:
        delta = got[name] - ref[name]
        total += abs(delta)
        flag = "  <-- off" if abs(delta) > 0.15 else ""
        print(f"    {name:24s} got={got[name]:+.3f}  ref={ref[name]:+.3f}  Δ={delta:+.3f}{flag}")
    print(f"\n  total L1 = {total:.3f} (large per-joint Δ points at the mapping / ROM / neutral)")


def cmd_presets(args: argparse.Namespace) -> None:
    kp = _load_frame(args.file)
    tuning = tuning_for_source(args.profile)
    _presets_on_frame(kp, tuning, args.expect)


def cmd_capture(args: argparse.Namespace) -> None:
    kp = _capture_live(args)
    np.save(args.out, kp)
    print(f"Saved captured frame to {args.out} (shape {kp.shape})")
    _print_analysis("captured", kp, tuning_for_source(args.profile))


def cmd_live(args: argparse.Namespace) -> None:
    kp = _capture_live(args)
    tuning = tuning_for_source(args.profile)
    _print_analysis("live", kp, tuning)


def _presets_on_frame(
    kp: np.ndarray, tuning: RetargeterTuning, expect: str | None
) -> None:
    expected = analyze_frame(_resolve_expect(expect), tuning)["targets"] if expect else None
    print("Frame preset comparison (apply each axis remap to the captured frame):\n")
    target_vectors = []
    for name, transform in GLOVE_FRAME_PRESETS.items():
        moved = kp @ np.asarray(transform, dtype=np.float64).T
        info = analyze_frame(moved, tuning)
        target_vectors.append([info["targets"][j] for j in ACTIVE_JOINT_NAMES])
        line = f"[{name:8s} det={np.linalg.det(transform):+.0f}] {frame_handedness_note(moved)}"
        if expected is not None:
            err = sum(abs(info["targets"][j] - expected[j]) for j in ACTIVE_JOINT_NAMES)
            line += f"  | signature L1 err={err:.3f}"
        print(line)
    spread = float(np.abs(np.array(target_vectors) - np.array(target_vectors[0])).max())
    if spread < 1e-4:
        print(
            "\n=> All presets produce IDENTICAL targets (max Δ "
            f"{spread:.1e}). The postprocess is frame-invariant — the frame is\n"
            "   NOT your problem. Use `compare` to find the off joints, then look at the\n"
            "   node mapping (thumb!), skeleton ROM (--profile glove), and neutral\n"
            "   offset (enable --calibrate-delay). Frame only matters under --no-*-postprocess."
        )


# ─── IO helpers ────────────────────────────────────────────────────────────


def _load_frame(path: str) -> np.ndarray:
    kp = np.load(path)
    if kp.shape != (21, 3):
        raise SystemExit(f"Expected a (21,3) frame in {path}, got {kp.shape}")
    return kp.astype(np.float64)


def _capture_live(args: argparse.Namespace) -> np.ndarray:
    """Subscribe to a glove topic and return the latest ``(21,3)`` frame."""

    import time

    from midas_hand_teleop.manus_glove.glove_subscriber import (
        GLOVE_TOPIC,
        GloveSubscriber,
        parse_glove_array,
        start_data_center_proxy,
    )

    if not args.no_proxy:
        start_data_center_proxy()
    topic = GLOVE_TOPIC.format(side=args.side)
    sub = GloveSubscriber(topic, args.host)
    logger.info("Waiting for a glove frame on %s (%.1fs)...", topic, args.wait)
    deadline = time.monotonic() + args.wait
    latest = None
    while time.monotonic() < deadline:
        arr = sub.poll_latest(timeout_ms=100)
        if arr is not None:
            parsed = parse_glove_array(arr)
            if parsed is not None:
                latest = parsed
    sub.close()
    if latest is None:
        raise SystemExit(
            f"No glove frame received on {topic} within {args.wait}s — is the "
            "bridge or fake_glove_publisher running?"
        )
    return latest.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect/compare hand-tracking sources through the MIDAS retargeter"
    )
    parser.add_argument(
        "--profile",
        default="vision",
        help="Tuning profile (vision/glove) used to compute targets",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_poses = sub.add_parser("poses", help="Print target signatures of canonical poses")
    p_poses.set_defaults(func=cmd_poses)

    p_an = sub.add_parser("analyze", help="Analyze a captured (21,3) .npy frame")
    p_an.add_argument("--file", required=True)
    p_an.set_defaults(func=cmd_analyze)

    p_cmp = sub.add_parser("compare", help="Per-joint diff of a captured frame vs a canonical pose")
    p_cmp.add_argument("--file", required=True)
    p_cmp.add_argument("--expect", required=True, help=f"one of {sorted(CANONICAL_POSES)}")
    p_cmp.set_defaults(func=cmd_compare)

    p_pr = sub.add_parser("presets", help="Show frame presets are invariant on a captured frame")
    p_pr.add_argument("--file", required=True)
    p_pr.add_argument("--expect", default=None, help=f"one of {sorted(CANONICAL_POSES)}")
    p_pr.set_defaults(func=cmd_presets)

    for name, func in (("capture", cmd_capture), ("live", cmd_live)):
        p = sub.add_parser(name, help=f"{name} a live glove frame")
        p.add_argument("--side", choices=["right", "left"], default="right")
        p.add_argument("--host", default="localhost")
        p.add_argument("--wait", type=float, default=5.0, help="Seconds to wait for a frame")
        p.add_argument("--no-proxy", action="store_true")
        if name == "capture":
            p.add_argument("--out", required=True, help="Output .npy path")
        p.set_defaults(func=func)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
