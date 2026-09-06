"""``midas-hand-tune`` — the browser-based per-finger retargeting tuner.

Typical use, with no hardware at all::

    python -m midas_hand_teleop.manus_glove.fake_glove_publisher &
    midas-hand-tune

With a real glove::

    midas-manus-bridge &
    midas-hand-tune --side right

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

from ..backends import MujocoBackend, PrintBackend
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

    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "print"],
                        help="Where commands go. Hardware is intentionally not "
                             "offered here yet; tune in sim first.")
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


def build_backend(args):
    if args.backend == "print":
        return PrintBackend()
    return MujocoBackend(
        xml_path=args.xml_path,
        mujoco_repo=args.mujoco_repo,
        render=args.mujoco_viewer,
        steps_per_frame=max(1, round((1.0 / args.control_hz) / 0.002)),
    )


def build_state(args) -> TunerState:
    if args.preset:
        from midas_hand_retargeter import presets

        profile, neutral = presets.load(args.preset)
        logger.info("Loaded preset %s (%d neutral offsets)", args.preset, len(neutral))
    else:
        profile = RetargetProfile.from_legacy_tuning(tuning_for_source(args.profile))
        profile = RetargetProfile(
            index=profile.index, middle=profile.middle, ring=profile.ring,
            thumb=profile.thumb, name=args.profile, source=args.profile,
        )
    return TunerState(store=ProfileStore(profile))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    state = build_state(args)
    state.loop.mode = args.mode
    retargeter = MidasHandRetargeter.create(
        mode=args.mode, mujoco_repo=args.mujoco_repo, tuning=state.profile
    )
    backend = build_backend(args)

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
    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Stopping.")
    finally:
        server.shutting_down = True
        server.shutdown()
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
