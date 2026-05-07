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
        return HardwareBackend(
            configure=args.configure_hardware,
            config_path=args.hardware_config,
            port=args.hardware_port,
            baudrate=args.hardware_baudrate,
            current_limit_ma=args.hardware_current_limit,
            command_scale=args.hardware_command_scale,
            max_step_rad=args.hardware_max_step_rad,
            update_rate_hz=args.hardware_rate_hz,
            interpolation_alpha=args.hardware_interpolation_alpha,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def _compact_joint_values(values: dict[str, float]) -> dict[str, float]:
    return {
        name: round(value, 3)
        for name, value in values.items()
    }


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
    parser.add_argument(
        "--hardware-config",
        default=None,
        help="Optional midas_hand_api calibration config. Defaults to ~/.midas_hand/config.yaml.",
    )
    parser.add_argument(
        "--hardware-port",
        default=None,
        help="Optional Dynamixel serial port override, for example /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--hardware-baudrate",
        type=int,
        default=None,
        help="Optional Dynamixel baudrate override.",
    )
    parser.add_argument(
        "--hardware-current-limit",
        type=int,
        default=None,
        help="Optional current limit in mA to apply when --configure-hardware is used.",
    )
    parser.add_argument(
        "--hardware-command-scale",
        type=float,
        default=1.0,
        help="Scale hardware commands around calibrated zero; use 0.3-0.5 for first bring-up.",
    )
    parser.add_argument(
        "--hardware-max-step-rad",
        type=float,
        default=0.1,
        help="Maximum per-hardware-tick motor target change in radians; use 0 to disable slew limiting.",
    )
    parser.add_argument(
        "--hardware-rate-hz",
        type=float,
        default=50.0,
        help="Fixed hardware command update rate in Hz; use 0 to send directly from the vision loop.",
    )
    parser.add_argument(
        "--hardware-interpolation-alpha",
        type=float,
        default=0.35,
        help="Fraction of remaining target distance to move each hardware tick; lower is smoother.",
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
    parser.add_argument("--finger-abad-alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--finger-abad-limit", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--finger-abad-deadzone", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--finger-abad-curl-damping", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-open", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-oppose", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-angle-neutral", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-angle-span", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-angle-deadzone", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-sign", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-roll-signed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--thumb-curl-opposition-gain", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-open", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-angle-neutral", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-angle-gain", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-deadzone", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-sign", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-mcp-closed", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-dip-closed", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-cmc-side-oppose", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thumb-pinch-gain", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--debug-targets",
        action="store_true",
        help="Print detected handedness and active joint targets while running.",
    )
    args = parser.parse_args()
    if args.finger_abad_alpha is not None:
        args.finger_smoothing_alpha = args.finger_abad_alpha
    if args.thumb_cmc_alpha is not None:
        args.thumb_smoothing_alpha = args.thumb_cmc_alpha
    legacy_thumb_gains = []
    if args.thumb_mcp_closed is not None:
        legacy_thumb_gains.append(args.thumb_mcp_closed / -0.88)
    if args.thumb_dip_closed is not None:
        legacy_thumb_gains.append(args.thumb_dip_closed / -0.72)
    if legacy_thumb_gains:
        args.thumb_flexion_gain = max(0.0, sum(legacy_thumb_gains) / len(legacy_thumb_gains))

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
