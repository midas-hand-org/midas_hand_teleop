"""Realtime webcam demo entrypoint.

Pinch calibration workflow:

1. Perform the normal zero calibration by holding your hand open and pressing
   ``c``.
2. Teleop robot thumb and index finger in a stable pinch posture.
3. Press ``r`` to record the robot thumb/index pinch joint values. With the
   hardware backend this reads actual motor feedback; otherwise it records the
   current commanded thumb/index values.
4. Make a good thumb-index pinch gesture in front of the webcam.
5. Press ``p`` to record the human thumb/index pinch feature vector.
6. Press ``s`` to save the calibration.
7. During runtime, when the human hand is close to the calibrated pinch pose,
   only the robot thumb/index joints are blended toward the saved pinch pose.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import cv2

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE, SUPPORTED_COUPLING_MODES
from midas_hand_retargeter.tuning import DEFAULT_TUNING, RetargeterTuning

from .backends import HardwareBackend, MujocoBackend, PrintBackend
from .detector import MediaPipeHandDetector
from .pinch_calibration import (
    PinchCalibration,
    PinchThresholds,
    extract_pinch_features,
    load_pinch_calibration,
    save_pinch_calibration,
)
from .pipeline import MidasTeleopPipeline


DEFAULT_BACKEND = "hardware"
DEFAULT_CONFIGURE_HARDWARE = True
DEFAULT_DEBUG_TARGETS = True
DEFAULT_HARDWARE_PORT = "/dev/ttyUSB0"
DEFAULT_HARDWARE_CURRENT_LIMIT = 350
DEFAULT_HARDWARE_COMMAND_SCALE = 1.0
DEFAULT_HARDWARE_MAX_STEP_RAD = 0.15
DEFAULT_HARDWARE_RATE_HZ = 50.0
DEFAULT_HARDWARE_INTERPOLATION_ALPHA = 0.4
DEFAULT_LOCK_INPUT_HAND = False
DEFAULT_SHOW = True
DEFAULT_PINCH_CALIBRATION_PATH = "config/pinch_calibration.json"


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


def _load_pinch_calibration_if_available(
    path: Path,
    *,
    thresholds: PinchThresholds,
    enabled: bool,
    autoload: bool,
) -> PinchCalibration:
    if autoload and path.exists():
        try:
            calibration = load_pinch_calibration(path)
            calibration.metadata["loaded_from"] = str(path)
            calibration.enabled = enabled
            print(f"Loaded pinch calibration from {path}")
            return calibration
        except Exception as exc:
            print(f"Could not load pinch calibration from {path}: {exc}")
    elif autoload:
        print(f"No pinch calibration found at {path}; use r/p/s to create one.")
    return PinchCalibration(thresholds=thresholds, enabled=enabled)


def _capture_robot_pinch(
    calibration: PinchCalibration,
    backend,
    teleop_frame,
) -> None:
    read_motor_positions = getattr(backend, "read_motor_positions", None)
    if callable(read_motor_positions):
        try:
            vector = calibration.record_robot_from_hardware_positions(
                read_motor_positions()
            )
            print(
                "Captured robot pinch from hardware feedback: "
                f"{vector.round(4).tolist()}"
            )
            return
        except Exception as exc:
            print(
                "Could not read hardware feedback for robot pinch; "
                f"falling back to current command if available ({exc})."
            )

    if teleop_frame is None:
        print("No teleop command available for robot pinch calibration.")
        return
    vector = calibration.record_robot_from_result(teleop_frame.retargeting)
    print(
        "Captured robot pinch from current commanded thumb/index targets: "
        f"{vector.round(4).tolist()}"
    )


def _capture_human_pinch(
    calibration: PinchCalibration,
    teleop_frame,
) -> None:
    if teleop_frame is None:
        print("No hand frame available for human pinch calibration.")
        return
    features = calibration.record_human(teleop_frame.landmarks)
    print(f"Captured human pinch features: {features.round(4).tolist()}")


def _pinch_debug_text(calibration: PinchCalibration) -> str:
    distance = calibration.state.distance
    distance_text = "n/a" if distance is None else f"{distance:.2f}"
    return (
        f"pinch loaded={calibration.is_complete} "
        f"enabled={calibration.enabled} "
        f"dist={distance_text} "
        f"conf={calibration.state.confidence:.2f} "
        f"active={calibration.state.active}"
    )


def _draw_pinch_overlay(bgr, calibration: PinchCalibration) -> None:
    loaded_from = calibration.metadata.get("loaded_from")
    loaded_text = "loaded" if loaded_from else "unsaved"
    if not calibration.is_complete:
        pieces = []
        if not calibration.has_robot:
            pieces.append("robot:r")
        if not calibration.has_human:
            pieces.append("human:p")
        loaded_text = "need " + ",".join(pieces)

    distance = calibration.state.distance
    distance_text = "n/a" if distance is None else f"{distance:.2f}"
    lines = (
        "pinch "
        f"{loaded_text} enabled={calibration.enabled} active={calibration.state.active}",
        f"pinch dist={distance_text} conf={calibration.state.confidence:.2f} "
        "keys: r robot, p human, s save, l load, x toggle, c zero, n clear-zero",
    )
    for index, line in enumerate(lines):
        y = 58 + 24 * index
        cv2.putText(
            bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


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
    parser.add_argument(
        "--backend",
        choices=["print", "mujoco", "hardware"],
        default=DEFAULT_BACKEND,
    )
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
        default=PIP_DIP_LOOKUP_MODE,
        help="Passive PIP-DIP coupling model used by the retargeter.",
    )
    parser.add_argument(
        "--configure-hardware",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIGURE_HARDWARE,
    )
    parser.add_argument(
        "--hardware-config",
        default=None,
        help="Optional midas_hand_api calibration config. Defaults to ~/.midas_hand/config.yaml.",
    )
    parser.add_argument(
        "--hardware-port",
        default=DEFAULT_HARDWARE_PORT,
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
        default=DEFAULT_HARDWARE_CURRENT_LIMIT,
        help="Optional current limit in mA to apply when --configure-hardware is used.",
    )
    parser.add_argument(
        "--hardware-command-scale",
        type=float,
        default=DEFAULT_HARDWARE_COMMAND_SCALE,
        help="Scale hardware commands around calibrated zero; use 0.3-0.5 for first bring-up.",
    )
    parser.add_argument(
        "--hardware-max-step-rad",
        type=float,
        default=DEFAULT_HARDWARE_MAX_STEP_RAD,
        help="Maximum per-hardware-tick motor target change in radians; use 0 to disable slew limiting.",
    )
    parser.add_argument(
        "--hardware-rate-hz",
        type=float,
        default=DEFAULT_HARDWARE_RATE_HZ,
        help="Fixed hardware command update rate in Hz; use 0 to send directly from the vision loop.",
    )
    parser.add_argument(
        "--hardware-interpolation-alpha",
        type=float,
        default=DEFAULT_HARDWARE_INTERPOLATION_ALPHA,
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
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG_TARGETS,
        help="Print detected handedness and active joint targets while running.",
    )
    parser.add_argument(
        "--pinch-calibration-path",
        default=DEFAULT_PINCH_CALIBRATION_PATH,
        help="JSON file for thumb/index pinch calibration.",
    )
    parser.add_argument(
        "--pinch-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable calibrated thumb/index pinch correction.",
    )
    parser.add_argument(
        "--pinch-autoload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load --pinch-calibration-path at startup if it exists.",
    )
    parser.add_argument(
        "--pinch-activate-distance",
        type=float,
        default=PinchThresholds.activate_distance,
        help="Feature-space distance below which pinch correction activates.",
    )
    parser.add_argument(
        "--pinch-release-distance",
        type=float,
        default=PinchThresholds.release_distance,
        help="Feature-space distance above which active pinch correction releases.",
    )
    parser.add_argument(
        "--pinch-full-distance",
        type=float,
        default=PinchThresholds.full_confidence_distance,
        help="Feature-space distance that maps to full pinch confidence.",
    )
    parser.add_argument(
        "--pinch-smoothing-alpha",
        type=float,
        default=PinchThresholds.smoothing_alpha,
        help="Low-pass alpha for pinch confidence/blending.",
    )
    parser.set_defaults(lock_input_hand=DEFAULT_LOCK_INPUT_HAND)
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

    pinch_thresholds = PinchThresholds(
        activate_distance=args.pinch_activate_distance,
        release_distance=args.pinch_release_distance,
        full_confidence_distance=args.pinch_full_distance,
        smoothing_alpha=args.pinch_smoothing_alpha,
    )
    try:
        pinch_thresholds.validate()
    except ValueError as exc:
        parser.error(str(exc))
    pinch_calibration_path = Path(args.pinch_calibration_path).expanduser()
    pinch_calibration = _load_pinch_calibration_if_available(
        pinch_calibration_path,
        thresholds=pinch_thresholds,
        enabled=args.pinch_correction,
        autoload=args.pinch_autoload,
    )

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
            pinch_features = None
            if hand_frame is not None:
                teleop_frame = pipeline.process_landmark_frame(hand_frame)
                pinch_features = extract_pinch_features(teleop_frame.landmarks)
                corrected_retargeting = pinch_calibration.apply(
                    teleop_frame.retargeting,
                    pinch_features,
                )
                teleop_frame = replace(
                    teleop_frame,
                    retargeting=corrected_retargeting,
                )
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
                            f"{_compact_joint_values(teleop_frame.active_joint_positions)} "
                            f"{_pinch_debug_text(pinch_calibration)}"
                        )
            else:
                pinch_calibration.update_confidence(None)

            if args.show:
                detector.draw_landmarks(bgr, hand_frame)
                _draw_pinch_overlay(bgr, pinch_calibration)
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
                if key == ord("n"):
                    pipeline.clear_neutral_offsets()
                    print("Cleared retargeter neutral offsets.")
                if key == ord("r"):
                    _capture_robot_pinch(
                        pinch_calibration,
                        backend,
                        teleop_frame,
                    )
                if key == ord("p"):
                    _capture_human_pinch(pinch_calibration, teleop_frame)
                if key == ord("s"):
                    if not pinch_calibration.is_complete:
                        print(
                            "Pinch calibration is incomplete; press r for robot "
                            "pinch and p for human pinch before saving."
                        )
                    else:
                        save_pinch_calibration(
                            pinch_calibration,
                            pinch_calibration_path,
                        )
                        pinch_calibration.metadata["loaded_from"] = str(
                            pinch_calibration_path
                        )
                        print(f"Saved pinch calibration to {pinch_calibration_path}")
                if key == ord("l"):
                    try:
                        was_enabled = pinch_calibration.enabled
                        pinch_calibration = load_pinch_calibration(
                            pinch_calibration_path
                        )
                        pinch_calibration.enabled = was_enabled
                        pinch_calibration.metadata["loaded_from"] = str(
                            pinch_calibration_path
                        )
                        print(f"Loaded pinch calibration from {pinch_calibration_path}")
                    except Exception as exc:
                        print(
                            f"Could not load pinch calibration from "
                            f"{pinch_calibration_path}: {exc}"
                        )
                if key == ord("x"):
                    pinch_calibration.enabled = not pinch_calibration.enabled
                    state = "enabled" if pinch_calibration.enabled else "disabled"
                    print(f"Pinch correction {state}.")
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        pipeline.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
