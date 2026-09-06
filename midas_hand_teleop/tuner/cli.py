"""``midas-hand-tune`` — the browser-based per-finger retargeting tuner.

Typical use, with no hardware at all::

    python -m midas_hand_teleop.manus_glove.fake_glove_publisher &
    midas-hand-tune

With a real glove, driving the simulator::

    midas-manus-bridge &
    midas-hand-tune --side right --mode dexpilot --mujoco-viewer --open

With a real glove driving the REAL HAND. The hand starts disarmed and stays
that way until you press "Arm hardware" in the browser; nothing is energised by
launching this::

    midas-manus-bridge &
    midas-hand-tune --side right --mode dexpilot --backend hardware --open

Then open http://127.0.0.1:8765.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.config import ANALYTIC_MODE, SUPPORTED_RETARGET_MODES
from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.store import ProfileStore
from midas_hand_retargeter.tuning import PROFILES, tuning_for_source

from ..backend_cli import add_backend_arguments, build_backend
from ..shutdown import (
    close_quietly,
    exit_without_atexit,
    protected_shutdown,
    sigterm_as_interrupt,
)
from .loop import LoopConfig, TunerLoop
from .server import serve_in_background
from .state import TunerState

logger = logging.getLogger("midas_hand_teleop.tuner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midas-hand-tune", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--side", default="right", choices=["right", "left"],
                        help="Glove side to subscribe to (default: right).")
    parser.add_argument("--host", default="localhost",
                        help="Data-center host publishing glove keypoints.")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Do not start the built-in XSUB/XPUB relay.")

    # The backend and hardware flags are shared with webcam_demo and
    # manus_teleop rather than redefined, so the safety defaults (disarmed,
    # current cap, slew limit, homing precondition) are the same everywhere.
    add_backend_arguments(parser, default="mujoco", include_mujoco=False)
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="Open the MuJoCo viewer alongside the browser UI.")
    parser.add_argument("--xml-path", default=None, help="Override the MJCF path.")
    parser.add_argument("--mujoco-repo", default=None,
                        help="Path to midas_hand_mujoco, if not auto-discovered.")

    parser.add_argument(
        "--mode", default=ANALYTIC_MODE, choices=list(SUPPORTED_RETARGET_MODES),
        help="Retargeting mode (default: analytic). Use 'dexpilot' to tune the "
             "optimizer that controls fingertip positions relative to each "
             "other — the thing the analytic map structurally cannot do. The "
             "UI shows only the controls the chosen mode actually reads.",
    )
    parser.add_argument("--profile", default="glove", choices=sorted(PROFILES),
                        help="Starting tuning profile (default: glove).")
    parser.add_argument("--preset", default=None,
                        help="Load this preset file at startup instead of a profile.")

    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="HTTP bind address. Defaults to loopback; binding "
                             "publicly exposes live control of a robot hand.")
    parser.add_argument("--open", action="store_true", help="Open a browser tab.")
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="Exit after this many seconds (for smoke tests).")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None,
                        help="Also write logs here. Without this, logs go only "
                             "to stderr and a bare `> file` redirect captures nothing.")
    return parser


def _configure_logging(level: str, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_state(args) -> TunerState:
    # Filled by the preset branch, then queued onto the state below so the
    # control loop installs it on its first frame -- the same path the browser
    # uses, rather than a second one that could drift from it.
    state_neutral: dict[str, float] = {}
    if args.preset:
        from midas_hand_retargeter import presets

        profile, neutral = presets.load(args.preset)
        logger.info("Loaded preset %s (%d neutral offsets)", args.preset, len(neutral))
        state_neutral.update(neutral)
    else:
        profile = RetargetProfile.from_legacy_tuning(tuning_for_source(args.profile))
        profile = RetargetProfile(
            index=profile.index, middle=profile.middle, ring=profile.ring,
            thumb=profile.thumb, name=args.profile, source=args.profile,
        )
    state = TunerState(store=ProfileStore(profile))
    if state_neutral:
        state.request_neutral_offsets(state_neutral)
    return state


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    state = build_state(args)
    state.loop.mode = args.mode
    retargeter = MidasHandRetargeter.create(
        mode=args.mode, mujoco_repo=args.mujoco_repo, tuning=state.profile
    )
    backend = build_backend(args, control_hz=args.control_hz)

    server = serve_in_background(state, host=args.bind, port=args.port)
    url = f"http://{args.bind}:{args.port}"
    logger.info("Tuner UI at %s", url)
    if args.open:
        webbrowser.open(url)

    loop = TunerLoop(
        state, retargeter, backend,
        LoopConfig(side=args.side, host=args.host, control_hz=args.control_hz,
                   duration_s=args.duration, start_proxy=not args.no_proxy),
    )
    def drop_torque():
        """The one thing that must happen on every exit path."""

        disarm = getattr(backend, "disarm", None)
        if callable(disarm):
            disarm()

    try:
        # SIGTERM would otherwise kill the process outright, skipping both the
        # finally below and midas_hand_api's atexit torque-off. `kill` and
        # `pkill -f` are how the runbook says to stop these loops.
        with sigterm_as_interrupt(logger):
            loop.run()
    except KeyboardInterrupt:
        logger.info("Stopping.")
    finally:
        used_viewer = getattr(backend, "viewer", None) is not None
        # A second Ctrl-C must not abort this block. If it did, the MuJoCo
        # viewer would never close and the process would then deadlock in
        # glfw.terminate() at interpreter exit, unkillable by Ctrl-C.
        with protected_shutdown(logger, before_force_exit=drop_torque):
            # Backend first, deliberately: it owns the GUI, and the GLFW
            # render loop has to be stopped before atexit tries to terminate
            # the library underneath it.
            clean = close_quietly(logger, "control loop", loop.close)
            server.shutting_down = True
            clean &= close_quietly(logger, "http server", server.shutdown)
            # shutdown() only stops the accept loop; this releases the socket
            # so an immediate restart does not hit "address already in use".
            clean &= close_quietly(logger, "http socket", server.server_close)

        if used_viewer and clean:
            # GLFW's Wayland teardown crashes at atexit even for a bare MuJoCo
            # viewer, so a successful run would otherwise report status 139.
            exit_without_atexit(logger, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
