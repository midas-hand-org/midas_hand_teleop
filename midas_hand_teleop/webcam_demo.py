"""Realtime webcam demo entrypoint."""

from __future__ import annotations

import argparse
import time

import cv2

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.tuning import DEFAULT_TUNING, RetargeterTuning

from .backends import HardwareBackend, MujocoBackend, PrintBackend
from .detector import MediaPipeHandDetector
from .pipeline import MidasTeleopPipeline


def _build_backend(args):
    if args.backend == "print":
        return PrintBackend()
    if args.backend == "mujoco":
        return MujocoBackend(
            xml_path=args.mujoco_xml,
            mujoco_repo=args.mujoco_repo,
            render=args.mujoco_viewer,
            steps_per_frame=args.mujoco_steps,
        )
    if args.backend == "hardware":
        return HardwareBackend(configure=args.configure_hardware)
    raise ValueError(f"Unsupported backend: {args.backend}")


def main() -> None:
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
        action="store_true",
        help="Allow auto input handedness to switch while running.",
    )
    parser.add_argument(
        "--mirror-axis",
        choices=["x", "y", "z", "xy", "xz", "yz", "xyz", "none"],
        default="y",
        help="MANO-frame axis to mirror when mapping left input to a right robot.",
    )
    parser.add_argument("--show", action="store_true", help="Show webcam landmarks.")
    parser.add_argument("--backend", choices=["print", "mujoco", "hardware"], default="print")
    parser.add_argument(
        "--hand-landmarker-model",
        default=None,
        help="Optional MediaPipe Tasks hand_landmarker.task path.",
    )
    parser.add_argument("--mujoco-repo", default=None)
    parser.add_argument("--mujoco-xml", default=None)
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument("--mujoco-steps", type=int, default=30)
    parser.add_argument("--configure-hardware", action="store_true")
    parser.add_argument("--scaling-factor", type=float, default=1.15)
    parser.add_argument("--finger-abad-gain", type=float, default=DEFAULT_TUNING.finger_abad_gain)
    parser.add_argument("--finger-abad-limit", type=float, default=DEFAULT_TUNING.finger_abad_limit)
    parser.add_argument("--finger-abad-deadzone", type=float, default=DEFAULT_TUNING.finger_abad_deadzone)
    parser.add_argument("--finger-abad-alpha", type=float, default=DEFAULT_TUNING.finger_abad_alpha)
    parser.add_argument(
        "--finger-abad-curl-damping",
        type=float,
        default=DEFAULT_TUNING.finger_abad_curl_damping,
    )
    parser.add_argument("--thumb-cmc-roll-open", type=float, default=DEFAULT_TUNING.thumb_cmc_roll_open)
    parser.add_argument("--thumb-cmc-roll-oppose", type=float, default=DEFAULT_TUNING.thumb_cmc_roll_oppose)
    parser.add_argument("--thumb-cmc-side-open", type=float, default=DEFAULT_TUNING.thumb_cmc_side_open)
    parser.add_argument("--thumb-cmc-side-oppose", type=float, default=DEFAULT_TUNING.thumb_cmc_side_oppose)
    parser.add_argument("--thumb-mcp-closed", type=float, default=DEFAULT_TUNING.thumb_mcp_closed)
    parser.add_argument("--thumb-dip-closed", type=float, default=DEFAULT_TUNING.thumb_dip_closed)
    parser.add_argument("--thumb-pinch-gain", type=float, default=DEFAULT_TUNING.thumb_pinch_gain)
    parser.add_argument(
        "--debug-targets",
        action="store_true",
        help="Print detected handedness and active joint targets while running.",
    )
    args = parser.parse_args()

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
        lock_input_hand=not args.no_lock_input_hand,
        model_path=args.hand_landmarker_model,
    )
    retargeter = MidasHandRetargeter.create(
        mujoco_repo=args.mujoco_repo,
        scaling_factor=args.scaling_factor,
        tuning=RetargeterTuning(
            finger_abad_gain=args.finger_abad_gain,
            finger_abad_limit=args.finger_abad_limit,
            finger_abad_deadzone=args.finger_abad_deadzone,
            finger_abad_alpha=args.finger_abad_alpha,
            finger_abad_curl_damping=args.finger_abad_curl_damping,
            thumb_cmc_roll_open=args.thumb_cmc_roll_open,
            thumb_cmc_roll_oppose=args.thumb_cmc_roll_oppose,
            thumb_cmc_side_open=args.thumb_cmc_side_open,
            thumb_cmc_side_oppose=args.thumb_cmc_side_oppose,
            thumb_mcp_closed=args.thumb_mcp_closed,
            thumb_dip_closed=args.thumb_dip_closed,
            thumb_pinch_gain=args.thumb_pinch_gain,
        ),
    )
    pipeline = MidasTeleopPipeline(detector=detector, retargeter=retargeter)
    backend = _build_backend(args)
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
                        compact = {
                            name: round(value, 3)
                            for name, value in teleop_frame.active_joint_positions.items()
                        }
                        print(
                            "hand "
                            f"MP={hand_frame.mediapipe_hand_type} "
                            f"input={hand_frame.input_hand_type} "
                            f"robot={hand_frame.robot_hand_type} "
                            f"mirrored={hand_frame.mirrored}: "
                            f"{compact}"
                        )

            if args.show:
                detector.draw_landmarks(bgr, hand_frame)
                cv2.imshow("midas_hand_teleop", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        pipeline.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
