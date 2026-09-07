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
  pipeline uses, in its ``analytic`` mode: the geometric landmark->joint map
  (finger curl / thumb opposition), neutral calibration, and PIP-DIP
  coupling. Tuning transfers to the real robot. Knobs:
  ``--profile`` (glove/vision base tuning), ``--scaling-factor``,
  ``--no-finger-postprocess`` / ``--no-thumb-postprocess`` (set both to let the
  pure optimizer drive the joints), ``--coupling-mode``, ``--calibrate-delay``
  (ON by default for the glove — hold an open hand at startup), and the
  finger/thumb gains (which override the profile).

The default postprocess path is invariant to the input coordinate frame (any
rotation/reflection); if the glove tracks poorly the cause is the keypoint
geometry (node mapping / skeleton ROM), the neutral offset (calibration), or
smoothing — use ``midas-hand-diag`` to pinpoint which. See that module's
docstring.

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
The viewer needs a display; omit ``--mujoco-viewer`` without one.

Run:
    # terminal 1 — the glove bridge (publishes /data_collection/glove/<side>/keypoint/state)
    midas-manus-bridge --host localhost --rate 120
    # terminal 2 — this driver (needs a display for the viewer)
    midas-manus-teleop --side right --mujoco-viewer --debug-targets
    # headless smoke test (no display): prints solve rate
    midas-manus-teleop --side right --duration 10
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.config import (
    ANALYTIC_MODE,
    DEXPILOT_MODE,
    VECTOR_MODE,
)
from midas_hand_retargeter.constants import (
    ACTIVE_JOINT_NAMES,
    HARDWARE_MOTOR_JOINT_NAMES,
)
from midas_hand_retargeter.human import mediapipe_world_to_mano_landmarks
from midas_hand_retargeter.postprocess import as_profile as tuning_profile
from midas_hand_retargeter.tuning import (
    PROFILES,
    RetargeterTuning,
    tuning_for_source,
)

from midas_hand_teleop.backend_cli import (
    add_backend_arguments,
    build_backend,
    check_hardware_preconditions,
    check_side_supported,
)
from midas_hand_teleop.backends import MujocoBackend
from midas_hand_teleop.manus_glove.glove_subscriber import (
    DATA_OUTPUT_PORT,
    GLOVE_TOPIC,
    GloveSubscriber,
    parse_glove_array,
    start_data_center_proxy,
)
from midas_hand_teleop.shutdown import (
    close_quietly,
    exit_without_atexit,
    protected_shutdown,
)

logger = logging.getLogger("manus_teleop")


@dataclass
class _Control:
    """Minimal duck-typed stand-in for ``RetargetingResult``.

    ``MujocoBackend.send`` reads ``active_joint_positions`` (name -> radians),
    but ``HardwareBackend`` reads ``hardware_motor_positions`` — a 13-vector in
    motor order, which is thumb-first and therefore NOT the same ordering. The
    an earlier direct-map path provided only the former, so any hardware send from
    it raised AttributeError.
    """

    active_joint_positions: dict[str, float]

    @property
    def hardware_motor_positions(self) -> np.ndarray:
        """Active targets in HARDWARE_MOTOR_JOINT_NAMES (motor) order."""

        return np.asarray(
            [self.active_joint_positions[name] for name in HARDWARE_MOTOR_JOINT_NAMES],
            dtype=np.float64,
        )


class _EmaFilter:
    """Per-joint exponential smoother for the 13-value command vector.

    ``alpha`` is the usual low-pass coefficient: 1.0 = no filtering (instant),
    smaller = smoother but laggier. The analytic map does not apply the
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
                float(value) if prev is None else prev + self.alpha * (float(value) - prev)
            )
        return dict(self._values)


_GAIN_ARGS = (
    "finger_curl_gain",
    "finger_abad_gain",
    "thumb_cmc_side_gain",
    "thumb_cmc_roll_gain",
    "thumb_flexion_gain",
)


def build_tuning(args: argparse.Namespace):
    """Build the tuning from a preset, or the source profile plus overrides.

    ``--preset`` wins and returns a ``RetargetProfile``, which is the only way
    a tuned DexPilot session reaches this CLI: the legacy ``RetargeterTuning``
    that ``--profile`` produces has no dexpilot section at all, so every solver
    knob would silently fall back to a library default.

    Otherwise the base profile (``--profile``) supplies source-appropriate
    gains, smoothing, and bend normalizers, and any ``--*-gain`` flag the user
    set explicitly overrides the corresponding field.
    """

    if args.preset:
        from midas_hand_retargeter import presets

        profile, neutral = presets.load(args.preset)
        logger.info("Loaded preset %s (%d neutral offsets)", args.preset, len(neutral))
        if args.scaling_factor is not None:
            profile = profile.with_values({"dexpilot.scaling_factor": args.scaling_factor})
        return profile, neutral

    base = tuning_for_source(args.profile)
    overrides = {
        name: getattr(args, name) for name in _GAIN_ARGS if getattr(args, name) is not None
    }
    return (replace(base, **overrides) if overrides else base), {}


def resolve_filter_alpha(args: argparse.Namespace) -> float:
    """Resolve the glove-side EMA alpha, avoiding double smoothing.

    The retargeter already applies per-joint EMA smoothing from the tuning
    profile, so the default here is 1.0 -- no extra stage. An explicit
    ``--filter-alpha`` wins, with a warning that it double-smooths.
    """
    if args.filter_alpha is not None:
        if args.filter_alpha < 1.0:
            logger.warning(
                "--filter-alpha %.2f stacks on the retargeter's internal smoothing "
                "(double smoothing). Full mode already smooths via the profile; use "
                "1.0 here and tune finger/thumb_smoothing_alpha in the profile instead.",
                args.filter_alpha,
            )
        return args.filter_alpha
    return 1.0


def _poll_console_key() -> str | None:
    """Non-blocking read of one keypress line from a terminal.

    The webcam path has had on-demand recalibration ('c' to capture neutral)
    since the beginning, because it owns an OpenCV window that can take key
    events. The glove path had only a blind --calibrate-delay countdown at
    startup, so a drifting neutral meant restarting the session. This gives it
    the same control from the terminal.

    Returns None when there is no TTY (piped or systemd runs) or nothing typed.
    """

    if not sys.stdin or not sys.stdin.isatty():
        return None
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not ready:
        return None
    line = sys.stdin.readline().strip().lower()
    return line[0] if line else None


def build_retargeter(args: argparse.Namespace, tuning: RetargeterTuning) -> MidasHandRetargeter:
    """Build the ``MidasHandRetargeter`` for the selected ``--retarget`` mode.

    The same one the webcam and hardware pipelines use, so tuning here carries
    over to the real robot.
    """
    if args.retarget == "dexpilot":
        logger.info(
            "DexPilot: optimizing 6 inter-fingertip vectors + 4 palm-rooted. "
            "scaling_factor=%.2f is the load-bearing knob here — unlike the "
            "analytic map, this mode is sensitive to your hand size.",
            tuning_profile(tuning).dexpilot.scaling_factor,
        )
        return MidasHandRetargeter.create(
            mode=DEXPILOT_MODE,
            mujoco_repo=args.mujoco_repo,
            coupling_mode=args.coupling_mode,
            tuning=tuning,
        )

    # Turning the analytic layer off has always meant "let the optimizer drive",
    # which is now a named mode. Translate rather than fail: under the analytic
    # default those joints would simply go unwritten.
    mode = ANALYTIC_MODE
    if not (args.finger_postprocess and args.thumb_postprocess):
        mode = VECTOR_MODE
        logger.warning(
            "--no-finger-postprocess/--no-thumb-postprocess selects the vector "
            "optimizer (mode=%s). It needs the [vector] extra, is ~6x slower, "
            "and is not what drives the hand by default.",
            VECTOR_MODE,
        )

    return MidasHandRetargeter.create(
        mode=mode,
        mujoco_repo=args.mujoco_repo,
        scaling_factor=1.15 if args.scaling_factor is None else args.scaling_factor,
        coupling_mode=args.coupling_mode,
        tuning=tuning,
    )


def run(args: argparse.Namespace) -> None:
    # First, before a solver is built or a port is opened: there is nothing
    # useful to do with a left glove and no reason to spend a second finding out.
    check_side_supported(args.side)

    glove_topic = GLOVE_TOPIC.format(side=args.side)
    host = args.host
    tuning, preset_neutral = build_tuning(args)
    retargeter = build_retargeter(args, tuning)
    if preset_neutral:
        # A preset carries the zero pose it was captured with; installing it is
        # the other half of reproducing that session.
        retargeter.set_neutral_offsets(preset_neutral)
        logger.info("Applied the preset's zero pose (%d joints)", len(preset_neutral))
    hand_type = "Right" if args.side == "right" else "Left"
    # Report what the retargeter RESOLVED, not what was asked for. The Cartesian
    # modes override the coupling default, and dexpilot ignores
    # scaling/finger_pp/thumb_pp entirely -- printing those as if they were in
    # force is how an operator comes to believe a dead knob is live.
    source = f"preset={args.preset}" if args.preset else f"profile={args.profile}"
    config = retargeter.config
    if args.retarget == "dexpilot":
        detail = (
            f"scaling={retargeter.profile.dexpilot.scaling_factor:.2f}, "
            f"abduction_limit={retargeter.profile.dexpilot.abduction_limit:.2f}, "
            f"coupling={config.coupling_mode}"
        )
    else:
        detail = (
            f"scaling={config.scaling_factor:.2f}, "
            f"finger_pp={args.finger_postprocess}, "
            f"thumb_pp={args.thumb_postprocess}, "
            f"coupling={config.coupling_mode}"
        )
    logger.info("Retargeting mode: %s (%s) (%s)", config.mode, source, detail)

    # Relay so the bridge (PUB->5710) reaches our subscriber (SUB<-5711). Returns
    # False if the ports are already bound (real data center or a stale process).
    proxy_started = start_data_center_proxy() if not args.no_proxy else False

    # --- backend: print, MuJoCo, or the real hand -----------------------------
    period = 1.0 / max(args.control_hz, 1e-6)
    check_hardware_preconditions(args)
    if args.backend == "hardware":
        backend = build_backend(args, control_hz=args.control_hz)
        logger.info(
            "HARDWARE backend on %s. Torque is %s; current cap %d mA, slew %.3f "
            "rad/tick at %.0f Hz.",
            args.hardware_port,
            "ENABLED" if args.start_armed else "OFF until armed",
            args.hardware_current_limit,
            args.hardware_max_step_rad,
            args.hardware_rate_hz,
        )
    elif args.backend == "print":
        backend = build_backend(args, control_hz=args.control_hz)
    else:
        backend = MujocoBackend(
            xml_path=args.xml_path,
            mujoco_repo=args.mujoco_repo,
            render=args.mujoco_viewer,
            steps_per_frame=1,  # overridden below to match real time
        )
        timestep = float(backend.model.opt.timestep)
        # Step the sim by ~one control period per send so it runs near real
        # time, regardless of control rate.
        backend.steps_per_frame = args.steps_per_frame or max(1, round(period / timestep))
        if not args.gravity:
            backend.model.opt.gravity[:] = 0.0
        logger.info(
            "MIDAS MuJoCo loaded (nq=%d, nu=%d), control=%.0f Hz, steps/frame=%d, gravity=%s",
            backend.model.nq,
            backend.model.nu,
            args.control_hz,
            backend.steps_per_frame,
            args.gravity,
        )

    # --- glove subscriber (minimal raw ZMQ; see glove_subscriber.GloveSubscriber) ---
    sub = GloveSubscriber(glove_topic, host)
    logger.info("Subscribed to %s on %s:%d", glove_topic, host, DATA_OUTPUT_PORT)

    filter_alpha = resolve_filter_alpha(args)
    logger.info(
        "Smoothing: glove EMA alpha=%.2f%s",
        filter_alpha,
        " (off; retargeter smooths internally)" if filter_alpha >= 1.0 else "",
    )
    if args.calibrate_delay > 0:
        logger.info(
            "Neutral calibration ON: hold a relaxed OPEN hand for the first %.1fs "
            "— that pose becomes MIDAS zero.",
            args.calibrate_delay,
        )
    ema = _EmaFilter(filter_alpha)
    last_control = _Control(dict.fromkeys(ACTIVE_JOINT_NAMES, 0.0))

    solves = 0
    parse_failures = 0
    calibrated = False
    start = time.monotonic()
    last_report = start
    last_data_time = start
    closed = [False]
    deadman_tripped = [False]

    def compute_targets(keypoints: np.ndarray) -> dict[str, float]:
        """Map one (21,3) keypoint frame to the 13 active-joint targets."""
        # DexPilot compares 3D vectors (palm->tip and tip->tip) against the
        # robot's own frame, so it is orientation-sensitive in a way nothing
        # else here is. The MANO frame is a different convention, and rotating
        # into it points every target the wrong way — measured: the solver
        # stops curling entirely (index_pip 0.000 instead of -0.23). So feed
        # dexpilot the landmarks as published.
        if retargeter.config.mode == DEXPILOT_MODE:
            return dict(retargeter.retarget_landmarks(keypoints).active_joint_positions)

        # Full retargeter: feed MANO-frame landmarks (the vector optimizer needs a
        # consistent frame), then read the 13 active joints from the result.
        mano = mediapipe_world_to_mano_landmarks(keypoints, hand_type=hand_type)
        return dict(retargeter.retarget_landmarks(mano).active_joint_positions)

    def maybe_calibrate_neutral() -> None:
        """After --calibrate-delay s of data, capture the held pose as neutral."""
        nonlocal calibrated
        if calibrated or args.calibrate_delay <= 0:
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
        logger.info("solves=%d (%.0f Hz) parse_fail=%d%s", solves, rate, parse_failures, hint)
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
        """True while a viewer is open. False when there is no viewer at all.

        This used to return True when ``viewer`` was None, so a run without a
        viewer had no exit condition and spun invisibly forever.
        """

        viewer = getattr(backend, "viewer", None)
        return viewer is not None and viewer.is_running()

    next_tick = time.monotonic()
    try:
        if sys.stdin and sys.stdin.isatty():
            keys = "c = capture neutral, r = clear, q = quit"
            if args.retarget == "dexpilot":
                keys = "c = capture neutral, s = calibrate hand size, r = clear, q = quit"
            logger.info("Type a key then Enter: %s", keys)
        has_viewer = getattr(backend, "viewer", None) is not None
        if has_viewer:
            logger.info("Viewer open — move your gloved hand. Ctrl-C or close window to stop.")
        elif args.duration:
            logger.info("Running for %.0fs (no viewer)...", args.duration)
        else:
            logger.info("Running with no viewer and no --duration. Ctrl-C to stop.")
        while True:
            # --duration applies to every run, viewer or not.
            if args.duration and time.monotonic() - start >= args.duration:
                break
            if has_viewer and not viewer_running():
                break
            key = _poll_console_key()
            if key:
                if key == "c":
                    try:
                        retargeter.calibrate_neutral_from_last_frame()
                        logger.info("Captured neutral pose — this is now MIDAS zero.")
                    except RuntimeError as exc:
                        logger.warning("Neutral calibration skipped: %s", exc)
                elif key == "r":
                    retargeter.clear_neutral_offsets()
                    logger.info("Cleared neutral calibration.")
                elif key == "s" and retargeter.config.mode == DEXPILOT_MODE:
                    try:
                        scaling = retargeter.calibrate_scaling_from_landmarks()
                        logger.info("Calibrated hand scale to %.3f", scaling)
                    except RuntimeError as exc:
                        logger.warning("Scale calibration skipped: %s", exc)
                elif key == "q":
                    logger.info("Quit requested.")
                    break

            update_control()
            # Send every tick (latest control, held between glove frames) so the
            # sim keeps stepping and the viewer stays synced at a steady rate.
            # But a held frame is only safe for so long: without this the loop
            # would keep commanding a pose against a dead publisher forever,
            # which on hardware means leaning on an object indefinitely.
            stale_s = time.monotonic() - last_data_time
            if stale_s > args.stale_timeout > 0:
                if not deadman_tripped[0]:
                    deadman_tripped[0] = True
                    logger.error(
                        "No glove data for %.1fs (> --stale-timeout %.1fs) — "
                        "stopped commanding, and dropping torque if armed. The "
                        "hand goes limp; re-arm to resume.",
                        stale_s,
                        args.stale_timeout,
                    )
                    disarm = getattr(backend, "disarm", None)
                    if callable(disarm):
                        disarm()
            elif solves > 0:
                if deadman_tripped[0]:
                    logger.info("Glove data resumed after %.1fs.", stale_s)
                    deadman_tripped[0] = False
                backend.send(last_control)
            # `solves > 0` is load-bearing, not defensive. last_control is
            # seeded to every joint = 0.0, and last_data_time to the start, so
            # for the first --stale-timeout seconds this branch used to command
            # the all-zeros pose with no glove frame ever received. On hardware
            # that is uncommanded motion at power-on: arm() seeds from the
            # measured pose, then the next tick asks for URDF zero and the slew
            # limiter walks the whole hand there in ~0.3 s -- inside the
            # deadman window, so the deadman fires only after it has finished.
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
        # Guard the whole block: a second Ctrl-C here used to skip backend
        # teardown, leaving the MuJoCo render loop alive so that atexit's
        # glfw.terminate() deadlocked and the process ignored further Ctrl-C.
        used_viewer = getattr(backend, "viewer", None) is not None
        with protected_shutdown(logger):
            # Backend (and therefore the viewer) before the bus, so GLFW is
            # torn down while the interpreter is still healthy.
            clean = close_quietly(logger, "backend", backend.close)
            clean &= close_quietly(logger, "zmq", release_zmq)

        if used_viewer and clean:
            # See shutdown.exit_without_atexit: GLFW's Wayland teardown crashes
            # at interpreter exit even with no code of ours involved.
            exit_without_atexit(logger, 0)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, as its own function so tests can read the defaults.

    Named to match webcam_demo and the tuner, which both already do this.
    """

    parser = argparse.ArgumentParser(
        description="Manus glove -> MIDAS hand finger teleop, in sim or on hardware"
    )
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument(
        "--host",
        default=os.environ.get("MIDAS_DATA_CENTER_HOST", "localhost"),
        help="Data-center host to subscribe to (default: $MIDAS_DATA_CENTER_HOST or localhost)",
    )
    # --- Retargeting ---
    parser.add_argument(
        "--preset",
        default=None,
        help="Load a tuned profile by name from ~/.midas_hand/retarget_presets "
        "(or a path). REQUIRED to reproduce a DexPilot tuning session: without "
        "it every solver knob falls back to a library default, because the "
        "legacy --profile form has no dexpilot section. Also installs the zero "
        "pose the preset was saved with. Pass --retarget dexpilot too; a preset "
        "does not record which mode it was tuned in.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="glove",
        help="Tuning profile: source-appropriate gains/smoothing/bend-normalizers. "
        "'glove' (default) is the starting point for the Manus glove; 'vision' "
        "reproduces the webcam tuning for A/B comparison. Individual --*-gain flags "
        "override the profile.",
    )
    parser.add_argument(
        "--retarget",
        choices=["full", "dexpilot"],
        default="full",
        help="full = the analytic geometric map plus neutral calibration and "
        "PIP-DIP coupling, same as webcam/hardware; dexpilot = optimizer over "
        "inter-fingertip vectors, the only mode that controls where the "
        "fingertips sit relative to each other, and what glove teleop wants.",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=None,
        help="Human-to-robot size ratio. Applies to --retarget full and, with "
        "--preset, to dexpilot. Raise/lower if fingers over/under-reach. "
        "Default: 1.15 for full, or whatever the preset says for dexpilot.",
    )
    parser.add_argument(
        "--no-finger-postprocess",
        dest="finger_postprocess",
        action="store_false",
        help="(full) let the optimizer drive the FINGER joints instead of the "
        "analytic curl/splay heuristic.",
    )
    parser.add_argument(
        "--no-thumb-postprocess",
        dest="thumb_postprocess",
        action="store_false",
        help="(full) let the optimizer drive the THUMB joints instead of the "
        "analytic opposition heuristic.",
    )
    parser.set_defaults(finger_postprocess=True, thumb_postprocess=True)
    parser.add_argument(
        "--coupling-mode",
        choices=["fixed_passive", "pip_dip_lookup"],
        default=None,
        help="Passive DIP handling. fixed_passive: let the MJCF equality "
        "constraints drive DIP in sim. pip_dip_lookup: fill DIP from the "
        "four-bar lookup table. Default: let the mode choose -- the Cartesian "
        "modes need the lookup, because without it the fingertip the solver "
        "aims at is up to 59 mm from where the linkage actually puts it.",
    )
    parser.add_argument(
        "--calibrate-delay",
        type=float,
        default=2.0,
        help="(full) after N s, capture the held pose as the neutral (MIDAS-zero) "
        "reference. 0 = off. Hold a relaxed/open hand during the countdown. Defaults "
        "ON for the glove (vision captures neutral interactively; the glove cannot).",
    )
    parser.add_argument(
        "--mujoco-viewer", action="store_true", help="Launch the MuJoCo passive viewer"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds. Applies with or without a viewer.",
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
        help="MuJoCo steps per control tick (0 = auto from --control-hz for near-real-time sim).",
    )
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=None,
        help="EMA low-pass on the 13-joint vector: higher = less lag (1.0 = none). "
        "Default: full mode -> 1.0 (the retargeter smooths internally; tune the "
        "profile's *_smoothing_alpha). "
        "Setting <1.0 in full mode double-smooths (a warning fires).",
    )
    parser.add_argument(
        "--no-gravity",
        dest="gravity",
        action="store_false",
        help="Disable gravity (default: gravity on)",
    )
    parser.set_defaults(gravity=True)
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
    add_backend_arguments(parser, default="mujoco", include_mujoco=False)
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=0.5,
        help="Stop commanding if no glove frame arrives for this long (s). "
        "0 disables the deadman. On hardware this is what prevents leaning "
        "on an object forever after the publisher dies.",
    )
    parser.add_argument("--xml-path", default=None, help="Explicit MJCF path override")
    parser.add_argument("--mujoco-repo", default=None, help="Path to the midas_hand_mujoco repo")
    # Coarse tuning gains. Default None => inherit from --profile; set to override.
    parser.add_argument("--finger-curl-gain", type=float, default=None)
    parser.add_argument("--finger-abad-gain", type=float, default=None)
    parser.add_argument("--thumb-cmc-side-gain", type=float, default=None)
    parser.add_argument("--thumb-cmc-roll-gain", type=float, default=None)
    parser.add_argument("--thumb-flexion-gain", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
