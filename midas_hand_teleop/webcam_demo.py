"""Realtime webcam demo entrypoint."""

from __future__ import annotations

import argparse
import time

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.adaptor import SUPPORTED_COUPLING_MODES
from midas_hand_retargeter.tuning import DEFAULT_TUNING, RetargeterTuning

from .backend_cli import add_backend_arguments, build_backend
from .detector import MediaPipeHandDetector
from .pipeline import MidasTeleopPipeline

# print, not hardware: a bare `midas-hand-teleop` must not energise motors.
DEFAULT_BACKEND = "print"
DEFAULT_DEBUG_TARGETS = True
DEFAULT_LOCK_INPUT_HAND = False
DEFAULT_SHOW = True


def _compact_joint_values(values: dict[str, float]) -> dict[str, float]:
    return {
        name: round(value, 3)
        for name, value in values.items()
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, as its own function so tests can read the defaults.

    Named to match manus_teleop and the tuner, which both already do this.
    """

    parser = argparse.ArgumentParser(description="Run MIDAS webcam teleop.")
    parser.add_argument("--camera", default=0, help="OpenCV camera index or video path.")
    parser.add_argument(
        "--hand-type",
        choices=["Right", "Left"],
        default="Right",
        help="Robot hand type to retarget onto.",
    )
    parser.add_argument(
        "--input-hand",
        choices=["auto", "Right", "Left"],
        default="auto",
        help="Physical human hand to accept after MediaPipe handedness correction.",
    )
    parser.add_argument(
        "--selfie",
        action="store_true",
        help="Treat webcam frames as mirrored selfie input for MediaPipe handedness.",
    )
    parser.add_argument(
        "--no-mirror-input",
        action="store_true",
        help="Disable mirroring when the physical hand side differs from the robot hand.",
    )
    parser.add_argument(
        "--no-lock-input-hand",
        dest="lock_input_hand",
        action="store_false",
        help="Allow auto input handedness to switch while running.",
    )
    parser.add_argument(
        "--lock-input-hand",
        dest="lock_input_hand",
        action="store_true",
        help="Lock onto the first accepted physical input hand side.",
    )
    parser.add_argument(
        "--mirror-axis",
        choices=["x", "y", "z", "xy", "xz", "yz", "xyz", "none"],
        default="y",
        help="MANO-frame axis to mirror when mapping left input to a right robot.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW,
        help="Show webcam landmarks.",
    )
    # The shared definitions, not a fourth copy: this CLI had drifted into a
    # hardware group with no --start-armed and no --allow-unhomed, so
    # `midas-hand-teleop --backend hardware` connected, configured, and could
    # then never energise the hand -- build_backend reads start_armed with a
    # getattr default of False and nothing else ever calls arm().
    add_backend_arguments(parser, default=DEFAULT_BACKEND, include_mujoco=False)
    parser.add_argument(
        "--hand-landmarker-model",
        default=None,
        help="Optional MediaPipe Tasks hand_landmarker.task path.",
    )
    parser.add_argument("--mujoco-repo", default=None)
    parser.add_argument("--mujoco-xml", default=None)
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument("--mujoco-steps", type=int, default=30)
    parser.add_argument(
        "--coupling-mode",
        choices=SUPPORTED_COUPLING_MODES,
        default=None,
        help="Passive PIP-DIP coupling model. Default: let the mode choose -- "
        "the Cartesian modes need the lookup or the fingertip the solver aims "
        "at is up to 59 mm from where the four-bar linkage puts it.",
    )
    parser.add_argument("--scaling-factor", type=float, default=1.15)
    parser.add_argument("--finger-curl-gain", type=float, default=DEFAULT_TUNING.finger_curl_gain)
    parser.add_argument("--finger-abad-gain", type=float, default=DEFAULT_TUNING.finger_abad_gain)
    parser.add_argument("--finger-smoothing-alpha", type=float, default=DEFAULT_TUNING.finger_smoothing_alpha)
    parser.add_argument("--thumb-cmc-gain", type=float, default=DEFAULT_TUNING.thumb_cmc_gain)
    parser.add_argument("--thumb-cmc-side-gain", type=float, default=DEFAULT_TUNING.thumb_cmc_side_gain)
    parser.add_argument("--thumb-cmc-roll-gain", type=float, default=DEFAULT_TUNING.thumb_cmc_roll_gain)
    parser.add_argument("--thumb-flexion-gain", type=float, default=DEFAULT_TUNING.thumb_flexion_gain)
    parser.add_argument("--thumb-smoothing-alpha", type=float, default=DEFAULT_TUNING.thumb_smoothing_alpha)
    parser.add_argument(
        "--debug-targets",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG_TARGETS,
        help="Print detected handedness and active joint targets while running.",
    )
    parser.set_defaults(lock_input_hand=DEFAULT_LOCK_INPUT_HAND)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Imported here, not at module scope: opencv and mediapipe are 252 MB and
    # only the webcam path needs them, so they live in the [webcam] extra and
    # the glove path installs without them.
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        raise SystemExit(
            "The webcam path needs opencv and mediapipe, which are not "
            "installed. They are 252 MB and the glove path does not use them, "
            "so they are an extra: pip install -e '.[webcam]'"
        ) from exc

    camera = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise OSError(f"Could not open camera/video: {args.camera}")

    detector = MediaPipeHandDetector(
        hand_type=args.hand_type,
        input_hand_type=args.input_hand,
        selfie=args.selfie,
        mirror_input=not args.no_mirror_input,
        mirror_axis=args.mirror_axis,
        lock_input_hand=args.lock_input_hand,
        model_path=args.hand_landmarker_model,
    )
    retargeter = MidasHandRetargeter.create(
        mujoco_repo=args.mujoco_repo,
        scaling_factor=args.scaling_factor,
        coupling_mode=args.coupling_mode,
        tuning=RetargeterTuning(
            finger_curl_gain=args.finger_curl_gain,
            finger_abad_gain=args.finger_abad_gain,
            finger_smoothing_alpha=args.finger_smoothing_alpha,
            thumb_cmc_gain=args.thumb_cmc_gain,
            thumb_cmc_side_gain=args.thumb_cmc_side_gain,
            thumb_cmc_roll_gain=args.thumb_cmc_roll_gain,
            thumb_flexion_gain=args.thumb_flexion_gain,
            thumb_smoothing_alpha=args.thumb_smoothing_alpha,
        ),
    )
    pipeline = MidasTeleopPipeline(detector=detector, retargeter=retargeter)
    backend = build_backend(args)
    last_debug_print = 0.0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand_frame = detector.detect_rgb(rgb)
            teleop_frame = None
            if hand_frame is not None:
                teleop_frame = pipeline.process_landmark_frame(hand_frame)
                backend.send(teleop_frame.retargeting)
                if args.debug_targets:
                    now = time.monotonic()
                    if now - last_debug_print >= 0.5:
                        last_debug_print = now
                        print(
                            "hand "
                            f"MP={hand_frame.mediapipe_hand_type} "
                            f"input={hand_frame.input_hand_type} "
                            f"robot={hand_frame.robot_hand_type} "
                            f"mirrored={hand_frame.mirrored}: "
                            f"{_compact_joint_values(teleop_frame.active_joint_positions)}"
                        )

            if args.show:
                detector.draw_landmarks(bgr, hand_frame)
                cv2.imshow("midas_hand_teleop", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if teleop_frame is None:
                        print("No hand frame available for neutral calibration.")
                    else:
                        offsets = pipeline.calibrate_neutral_from_last_frame()
                        print(
                            "Captured retargeter neutral offsets: "
                            f"{_compact_joint_values(offsets)}"
                        )
                if key == ord("r"):
                    pipeline.clear_neutral_offsets()
                    print("Cleared retargeter neutral offsets.")
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        pipeline.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
