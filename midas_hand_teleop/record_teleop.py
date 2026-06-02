"""Record MIDAS teleop joint trajectories while running the webcam pipeline."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE, SUPPORTED_COUPLING_MODES
from midas_hand_retargeter.constants import ACTIVE_JOINT_NAMES
from midas_hand_retargeter.tuning import RetargeterTuning

from .backends import MujocoBackend, PrintBackend
from .detector import MediaPipeHandDetector
from .pipeline import MidasTeleopPipeline
from .webcam_demo import DEFAULT_LOCK_INPUT_HAND, DEFAULT_SHOW, _compact_joint_values


@dataclass
class TrajectoryRecorder:
    output_dir: Path
    joint_names: tuple[str, ...] = ACTIVE_JOINT_NAMES
    timestamps: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    active: bool = False

    def start(self) -> None:
        self.timestamps.clear()
        self.positions.clear()
        self.active = True
        print("Recording started.")

    def stop_and_save(self) -> Path | None:
        self.active = False
        if not self.timestamps:
            print("No frames recorded.")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"teleop_{stamp}.npz"
        np.savez(
            output_path,
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            positions=np.asarray(self.positions, dtype=np.float64),
            joint_names=np.asarray(self.joint_names),
        )
        duration = self.timestamps[-1] - self.timestamps[0]
        print(
            f"Saved {len(self.timestamps)} frames ({duration:.2f}s) to {output_path}"
        )
        return output_path

    def add_frame(self, active_joint_positions: dict[str, float]) -> None:
        if not self.active:
            return
        values = np.asarray(
            [active_joint_positions[name] for name in self.joint_names],
            dtype=np.float64,
        )
        self.timestamps.append(time.monotonic())
        self.positions.append(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record MIDAS webcam teleop joint trajectories."
    )
    parser.add_argument("--camera", default=0, help="OpenCV camera index or device path.")
    parser.add_argument(
        "--backend",
        choices=["print", "mujoco"],
        default="mujoco",
        help="Optional live preview while recording.",
    )
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW)
    parser.add_argument(
        "--debug-targets",
        action="store_true",
        help="Print joint targets to the terminal when a hand is detected.",
    )
    parser.add_argument("--selfie", action="store_true", help="Mirrored webcam handedness for MediaPipe.")
    parser.add_argument(
        "--no-lock-input-hand",
        dest="lock_input_hand",
        action="store_false",
        help="Allow auto input handedness to switch while running.",
    )
    parser.set_defaults(lock_input_hand=DEFAULT_LOCK_INPUT_HAND)
    parser.add_argument("--mujoco-repo", default=None)
    parser.add_argument("--mujoco-xml", default=None)
    parser.add_argument("--mujoco-viewer", action="store_true")
    parser.add_argument("--mujoco-steps", type=int, default=30)
    parser.add_argument(
        "--floating-wrist",
        action="store_true",
        help="Move the whole palm with the webcam wrist (requires desk scene XML).",
    )
    parser.add_argument(
        "--coupling-mode",
        choices=SUPPORTED_COUPLING_MODES,
        default=PIP_DIP_LOOKUP_MODE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "midas_hand_recordings",
        help="Directory for .npz recordings.",
    )
    parser.add_argument("--scaling-factor", type=float, default=1.15)
    parser.add_argument("--finger-curl-gain", type=float, default=1.45)
    parser.add_argument("--finger-abad-gain", type=float, default=1.2)
    parser.add_argument("--finger-smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--thumb-flexion-gain", type=float, default=1.45)
    parser.add_argument("--thumb-smoothing-alpha", type=float, default=0.5)
    args = parser.parse_args()

    camera = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise OSError(f"Could not open camera/video: {args.camera}")

    detector = MediaPipeHandDetector(
        selfie=args.selfie,
        lock_input_hand=args.lock_input_hand,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    retargeter = MidasHandRetargeter.create(
        mujoco_repo=args.mujoco_repo,
        scaling_factor=args.scaling_factor,
        coupling_mode=args.coupling_mode,
        tuning=RetargeterTuning(
            finger_curl_gain=args.finger_curl_gain,
            finger_abad_gain=args.finger_abad_gain,
            finger_smoothing_alpha=args.finger_smoothing_alpha,
            thumb_flexion_gain=args.thumb_flexion_gain,
            thumb_smoothing_alpha=args.thumb_smoothing_alpha,
        ),
    )
    pipeline = MidasTeleopPipeline(detector=detector, retargeter=retargeter)
    recorder = TrajectoryRecorder(output_dir=args.output_dir)

    backend = None
    if args.backend == "print":
        backend = PrintBackend(interval_s=0.5)
    else:
        backend = MujocoBackend(
            xml_path=args.mujoco_xml,
            mujoco_repo=args.mujoco_repo,
            render=args.mujoco_viewer,
            steps_per_frame=args.mujoco_steps,
            floating_wrist=args.floating_wrist,
        )

    print(
        "Controls: s=start recording, e=stop+save, c=neutral calibrate, "
        "r=clear neutral, q=quit"
    )
    print(
        "Tip: the MuJoCo hand moves only when midas_hand_record shows white "
        "skeleton lines on your hand."
    )
    print(
        "Tip: press c with an OPEN hand to calibrate neutral; then fist and "
        "rotate should match better."
    )

    last_debug_print = 0.0
    last_no_hand_print = 0.0

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
                backend.send(teleop_frame.retargeting, hand_frame=hand_frame)
                recorder.add_frame(teleop_frame.active_joint_positions)
                if args.debug_targets:
                    now = time.monotonic()
                    if now - last_debug_print >= 0.5:
                        last_debug_print = now
                        print(
                            "hand "
                            f"input={hand_frame.input_hand_type} "
                            f"mirrored={hand_frame.mirrored}: "
                            f"{_compact_joint_values(teleop_frame.active_joint_positions)}"
                        )
            elif args.debug_targets:
                now = time.monotonic()
                if now - last_no_hand_print >= 1.0:
                    last_no_hand_print = now
                    print("no hand detected — check camera framing and lighting")

            if args.show:
                status = "REC" if recorder.active else "idle"
                cv2.putText(
                    bgr,
                    status,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if recorder.active else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
                if hand_frame is None:
                    cv2.putText(
                        bgr,
                        "NO HAND",
                        (12, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                detector.draw_landmarks(bgr, hand_frame)
                cv2.imshow("midas_hand_record", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    recorder.start()
                if key == ord("e"):
                    recorder.stop_and_save()
                if key == ord("c") and teleop_frame is not None:
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
        if recorder.active:
            recorder.stop_and_save()
        backend.close()
        pipeline.close()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
