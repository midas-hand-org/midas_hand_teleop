"""manus_teleop.py — Manus glove -> MIDAS hand finger teleop in MuJoCo (fixed base).

Milestone harness: wear a physical Manus glove and watch the MIDAS hand's four
fingers track it in simulation. No wrist motion — the hand base is fixed.

This is the glove-driven sibling of the webcam pipeline
(``midas-hand-teleop --backend mujoco``). The two differ only in their INPUT
source: where ``webcam_demo`` runs OpenCV + MediaPipe to produce 21 hand
landmarks, this driver subscribes to the Manus bridge, which already publishes
MediaPipe-21 keypoints over the data-center ZMQ bus. Everything downstream
is reused.

Single-process pipeline:

    Manus bridge keypoint topic (ZMQ, MediaPipe-21; midas-manus-bridge)
      -> retargeting (one of two modes, see "Mapping") -> 13 joint angles (rad)
      -> EMA smooth the 13-value vector
      -> MujocoBackend (midas_hand_teleop.backends): clip to ctrlrange, step,
         sync the passive viewer

Mapping (``--retarget``):

* ``full`` (default) — the same ``MidasHandRetargeter`` the webcam/hardware
  pipeline uses: keypoints are rotated into the MANO frame, then
  ``retarget_landmarks`` runs the dex_retargeting NLopt vector optimizer, the
  MIDAS geometric postprocess (finger curl / thumb opposition), neutral
  calibration, and PIP-DIP coupling. Tuning transfers to the real robot. Knobs:
  ``--scaling-factor``, ``--no-finger-postprocess`` / ``--no-thumb-postprocess``
  (set both to let the pure optimizer drive the joints), ``--coupling-mode``,
  ``--calibrate-delay``, and the finger/thumb gains.

* ``geometric`` — the lightweight direct map from the first milestone: call the
  postprocess functions on their own (no optimizer, no IK), rotation-invariant.

Both emit exactly the 13 ``ACTIVE_JOINT_NAMES`` (index/middle/ring: mcp_abad,
mcp_pitch, pip; thumb: cmc_roll, cmc_side, mcp, dip) by name, already
range-bounded, so ``MujocoBackend`` maps them straight onto the same-named
actuators — no reorder or sign table.

The passive finger DIP/linkage joints are driven automatically by the MJCF
``equality/connect`` constraints; we only command the 13 active joints.

Prereqs: the Manus bridge publishing keypoints (run it first, on x86 with the
ManusSDK + gloves), or the bundled ``fake_glove_publisher`` for a hardware-free
smoke test; the MIDAS stack installed in ``midas_env``; and the MuJoCo model
discoverable (``MIDAS_HAND_MUJOCO_DIR`` or a sibling ``midas_hand_mujoco`` dir).
The viewer needs a display; use ``--headless`` without one.

Run:
    # terminal 1 — the glove bridge (publishes /data_collection/glove/<side>/keypoint/state)
    midas-manus-bridge --host localhost --rate 120
    # terminal 2 — this driver (needs a display for the viewer)
    midas-manus-teleop --side right --mujoco-viewer --debug-targets
    # headless smoke test (no display): prints solve rate
    midas-manus-teleop --side right --headless --duration 10
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.human import mediapipe_world_to_mano_landmarks
from midas_hand_retargeter.postprocess import (
    finger_joint_targets_from_landmarks,
    thumb_joint_targets_from_landmarks,
)
from midas_hand_retargeter.tuning import RetargeterTuning
from midas_hand_teleop.backends import MujocoBackend
from midas_hand_teleop.manus_glove.glove_subscriber import (
    DATA_OUTPUT_PORT,
    GLOVE_TOPIC,
    GloveSubscriber,
    parse_glove_array,
    start_data_center_proxy,
)

logger = logging.getLogger("manus_teleop")


@dataclass
class _Control:
    """Minimal duck-typed stand-in for ``RetargetingResult``.

    ``MujocoBackend.send`` only reads ``active_joint_positions`` (a
    name -> radians dict), so this is all the geometric map needs to produce.
    """

    active_joint_positions: dict[str, float]


def landmarks_to_joint_targets(
    keypoints: np.ndarray,
    tuning: RetargeterTuning,
    *,
    mano_frame: bool,
    side: str,
) -> dict[str, float]:
    """Geometric 21x3 keypoints -> 13 MIDAS active-joint targets (radians).

    ``mano_frame`` optionally rotates the keypoints into the wrist-centered MANO
    frame first (parity with the webcam path). The geometric functions build
    their own palm basis and are rotation-invariant, so this is OFF by default:
    it changes the output only up to numerical noise while adding the SVD
    sign-flip instability inside ``estimate_frame_from_hand_points``.
    """
    landmarks = keypoints
    if mano_frame:
        hand_type = "Right" if side == "right" else "Left"
        landmarks = mediapipe_world_to_mano_landmarks(keypoints, hand_type=hand_type)
    targets: dict[str, float] = {}
    targets.update(finger_joint_targets_from_landmarks(landmarks, tuning))
    targets.update(thumb_joint_targets_from_landmarks(landmarks, tuning))
    return targets


class _EmaFilter:
    """Per-joint exponential smoother for the 13-value command vector.

    ``alpha`` is the usual low-pass coefficient: 1.0 = no filtering (instant),
    smaller = smoother but laggier. The bare geometric functions do not apply the
    smoothing in ``RetargeterTuning`` (the retargeter wrapper does), so we smooth
    here instead.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._values: dict[str, float] = {}

    def update(self, targets: dict[str, float]) -> dict[str, float]:
        for name, value in targets.items():
            prev = self._values.get(name)
            self._values[name] = (
                float(value)
                if prev is None
                else prev + self.alpha * (float(value) - prev)
            )
        return dict(self._values)


def build_tuning(args: argparse.Namespace) -> RetargeterTuning:
    """Construct a ``RetargeterTuning`` from the CLI gain overrides."""
    return RetargeterTuning(
        finger_curl_gain=args.finger_curl_gain,
        finger_abad_gain=args.finger_abad_gain,
        thumb_cmc_side_gain=args.thumb_cmc_side_gain,
        thumb_cmc_roll_gain=args.thumb_cmc_roll_gain,
        thumb_flexion_gain=args.thumb_flexion_gain,
    )


def build_retargeter(
    args: argparse.Namespace, tuning: RetargeterTuning
) -> MidasHandRetargeter | None:
    """Build the full ``MidasHandRetargeter`` for ``--retarget full``, else None.

    ``None`` selects the lightweight geometric direct-map path. The full
    retargeter is the same one the webcam/hardware pipeline uses (optimizer +
    geometric postprocess + neutral calibration + PIP-DIP coupling), so tuning
    here carries over to the real robot.
    """
    if args.retarget != "full":
        if args.calibrate_delay > 0:
            logger.warning("--calibrate-delay needs --retarget full; ignoring it.")
        return None
    return MidasHandRetargeter.create(
        mujoco_repo=args.mujoco_repo,
        scaling_factor=args.scaling_factor,
        coupling_mode=args.coupling_mode,
        finger_postprocess=args.finger_postprocess,
        thumb_postprocess=args.thumb_postprocess,
        tuning=tuning,
    )


def run(args: argparse.Namespace) -> None:
    glove_topic = GLOVE_TOPIC.format(side=args.side)
    host = args.host
    tuning = build_tuning(args)
    retargeter = build_retargeter(args, tuning)  # None => geometric direct-map
    hand_type = "Right" if args.side == "right" else "Left"
    logger.info(
        "Retargeting mode: %s%s",
        args.retarget,
        f" (scaling={args.scaling_factor}, finger_pp={args.finger_postprocess}, "
        f"thumb_pp={args.thumb_postprocess}, coupling={args.coupling_mode})"
        if retargeter is not None
        else "",
    )

    if args.side != "right":
        logger.warning(
            "MIDAS MJCF is a RIGHT hand; mapping is tuned for the right "
            "hand. Driving from the left glove will mirror splay/thumb — left-hand "
            "mirroring is a follow-up."
        )

    # Relay so the bridge (PUB->5710) reaches our subscriber (SUB<-5711). Returns
    # False if the ports are already bound (real data center or a stale process).
    proxy_started = start_data_center_proxy() if not args.no_proxy else False

    # --- MuJoCo backend (loads the MIDAS MJCF, maps joints->actuators, viewer) ---
    backend = MujocoBackend(
        xml_path=args.xml_path,
        mujoco_repo=args.mujoco_repo,
        render=args.mujoco_viewer,
        steps_per_frame=1,  # overridden below to match real time
    )
    timestep = float(backend.model.opt.timestep)
    period = 1.0 / max(args.control_hz, 1e-6)
    # Step the sim by ~one control period per send so it runs near real time,
    # regardless of control rate (mirrors the webcam path's frame-paced stepping).
    steps = args.steps_per_frame or max(1, round(period / timestep))
    backend.steps_per_frame = steps
    if not args.gravity:
        backend.model.opt.gravity[:] = 0.0
    logger.info(
        "MIDAS MuJoCo loaded (nq=%d, nu=%d), control=%.0f Hz, steps/frame=%d, "
        "gravity=%s",
        backend.model.nq,
        backend.model.nu,
        args.control_hz,
        steps,
        args.gravity,
    )

    # --- glove subscriber (minimal raw ZMQ; see glove_subscriber.GloveSubscriber) ---
    sub = GloveSubscriber(glove_topic, host)
    logger.info("Subscribed to %s on %s:%d", glove_topic, host, DATA_OUTPUT_PORT)

    ema = _EmaFilter(args.filter_alpha)
    last_control = _Control({name: 0.0 for name in ACTIVE_JOINT_NAMES})

    solves = 0
    parse_failures = 0
    calibrated = False
    start = time.monotonic()
    last_report = start
    last_data_time = start
    closed = [False]

    def compute_targets(keypoints: np.ndarray) -> dict[str, float]:
        """Map one (21,3) keypoint frame to the 13 active-joint targets."""
        if retargeter is None:
            return landmarks_to_joint_targets(
                keypoints, tuning, mano_frame=args.mano_frame, side=args.side
            )
        # Full retargeter: feed MANO-frame landmarks (the vector optimizer needs a
        # consistent frame), then read the 13 active joints from the result.
        mano = mediapipe_world_to_mano_landmarks(keypoints, hand_type=hand_type)
        return dict(retargeter.retarget_landmarks(mano).active_joint_positions)

    def maybe_calibrate_neutral() -> None:
        """After --calibrate-delay s of data, capture the held pose as neutral."""
        nonlocal calibrated
        if calibrated or retargeter is None or args.calibrate_delay <= 0:
            return
        if last_data_time - start < args.calibrate_delay:
            return
        calibrated = True
        try:
            retargeter.calibrate_neutral_from_last_frame()
            logger.info(
                "Captured neutral pose after %.1fs — recentering MIDAS zero here.",
                args.calibrate_delay,
            )
        except RuntimeError as e:
            logger.warning("Neutral calibration skipped: %s", e)

    def update_control() -> bool:
        """Poll the glove; refresh ``last_control`` via the active retargeter.

        Returns True if a fresh frame was consumed.
        """
        nonlocal solves, parse_failures, last_data_time, last_control
        arr = sub.poll_latest(timeout_ms=0)
        if arr is None:
            return False
        keypoints = parse_glove_array(arr)
        if keypoints is None:
            parse_failures += 1
            return False
        last_data_time = time.monotonic()
        targets = compute_targets(keypoints)
        maybe_calibrate_neutral()
        last_control = _Control(ema.update(targets))
        solves += 1
        if args.debug_targets:
            logger.info(
                "targets: %s",
                {k: round(v, 3) for k, v in last_control.active_joint_positions.items()},
            )
        return True

    def report_maybe() -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 2.0:
            return
        rate = solves / (now - start) if now > start else 0.0
        stale = now - last_data_time
        hint = ""
        if stale > 1.0:
            hint = f"  [no glove data for {stale:.1f}s]"
            if not args.no_proxy and not proxy_started and solves == 0 and stale > 4.0:
                hint += (
                    " (ports 5710/5711 already bound but nothing is relaying — kill "
                    "any stale publisher/driver: `pkill -f manus_teleop`)"
                )
        logger.info(
            "solves=%d (%.0f Hz) parse_fail=%d%s", solves, rate, parse_failures, hint
        )
        last_report = now

    def release_zmq() -> None:
        # Release the SUB socket AND the proxy's bound ports so the next run can
        # rebind 5710/5711. Idempotent. Done BEFORE the viewer tears down because
        # the viewer can segfault on exit (Wayland) — this guarantees no orphan
        # keeps holding the ports.
        if closed[0]:
            return
        closed[0] = True
        try:
            sub.close()
        except Exception:
            pass
        if proxy_started:
            try:
                import zmq

                zmq.Context.instance().destroy(linger=0)
            except Exception:
                pass

    def viewer_running() -> bool:
        return backend.viewer is None or backend.viewer.is_running()

    next_tick = time.monotonic()
    try:
        if args.headless:
            logger.info("Headless run for %.0fs (no viewer)...", args.duration)
        else:
            logger.info(
                "Viewer open — move your gloved hand. Ctrl-C or close window to stop."
            )
        while True:
            if args.headless and time.monotonic() - start >= args.duration:
                break
            if not args.headless and not viewer_running():
                break
            update_control()
            # Send every tick (latest control, held between glove frames) so the
            # sim keeps stepping and the viewer stays synced at a steady rate.
            backend.send(last_control)
            report_maybe()
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()  # fell behind; resync
        logger.info("Done. total solves=%d, parse_failures=%d", solves, parse_failures)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down cleanly")
    finally:
        release_zmq()
        try:
            backend.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manus glove -> MIDAS hand finger teleop in MuJoCo"
    )
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument(
        "--host",
        default=os.environ.get("MIDAS_DATA_CENTER_HOST", "localhost"),
        help="Data-center host to subscribe to (default: $MIDAS_DATA_CENTER_HOST or "
        "localhost)",
    )
    # --- Retargeting ---
    parser.add_argument(
        "--retarget",
        choices=["full", "geometric"],
        default="full",
        help="full = MidasHandRetargeter (optimizer + postprocess + neutral + "
        "coupling, same as webcam/hardware); geometric = lightweight direct map.",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.15,
        help="(full) dex_retargeting vector scaling — human-to-robot size ratio. "
        "Raise/lower if fingers over/under-reach.",
    )
    parser.add_argument(
        "--no-finger-postprocess",
        dest="finger_postprocess",
        action="store_false",
        help="(full) let the optimizer drive the FINGER joints instead of the "
        "geometric curl/splay heuristic.",
    )
    parser.add_argument(
        "--no-thumb-postprocess",
        dest="thumb_postprocess",
        action="store_false",
        help="(full) let the optimizer drive the THUMB joints instead of the "
        "geometric opposition heuristic.",
    )
    parser.set_defaults(finger_postprocess=True, thumb_postprocess=True)
    parser.add_argument(
        "--coupling-mode",
        choices=["fixed_passive", "pip_dip_lookup"],
        default="fixed_passive",
        help="(full) passive DIP handling. fixed_passive: MJCF equality constraints "
        "drive DIP in sim. pip_dip_lookup: fill DIP from the PIP-DIP lookup table.",
    )
    parser.add_argument(
        "--calibrate-delay",
        type=float,
        default=0.0,
        help="(full) after N s, capture the held pose as the neutral (MIDAS-zero) "
        "reference. 0 = off. Hold a relaxed/open hand during the countdown.",
    )
    parser.add_argument(
        "--mujoco-viewer", action="store_true", help="Launch the MuJoCo passive viewer"
    )
    parser.add_argument(
        "--headless", action="store_true", help="No viewer (needs no display)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Headless run length (s)"
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=60.0,
        help="Control/step loop rate. The sim is stepped ~one period per tick so "
        "it runs near real time.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=0,
        help="MuJoCo steps per control tick (0 = auto from --control-hz for "
        "near-real-time sim).",
    )
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.6,
        help="EMA low-pass on the 13-joint vector: higher = less lag (1.0 = none). "
        "Lower if the fingers jitter.",
    )
    parser.add_argument(
        "--no-gravity",
        dest="gravity",
        action="store_false",
        help="Disable gravity (default: gravity on)",
    )
    parser.set_defaults(gravity=True)
    parser.add_argument(
        "--mano-frame",
        action="store_true",
        help="(geometric mode only) rotate keypoints into the MANO frame first. Off "
        "by default — the geometric map is rotation-invariant. The full retargeter "
        "always uses the MANO frame.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Don't start the built-in data center proxy (use an already-running one)",
    )
    parser.add_argument(
        "--debug-targets",
        action="store_true",
        help="Log the 13 joint targets each solved frame",
    )
    # MuJoCo model location (else MIDAS_HAND_MUJOCO_DIR / sibling repo discovery).
    parser.add_argument("--xml-path", default=None, help="Explicit MJCF path override")
    parser.add_argument(
        "--mujoco-repo", default=None, help="Path to the midas_hand_mujoco repo"
    )
    # Coarse geometric tuning gains (forwarded to RetargeterTuning).
    parser.add_argument("--finger-curl-gain", type=float, default=1.0)
    parser.add_argument("--finger-abad-gain", type=float, default=1.2)
    parser.add_argument("--thumb-cmc-side-gain", type=float, default=1.5)
    parser.add_argument("--thumb-cmc-roll-gain", type=float, default=0.6)
    parser.add_argument("--thumb-flexion-gain", type=float, default=1.2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
