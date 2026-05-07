"""Composable MediaPipe -> MIDAS retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from midas_hand_retargeter import MidasHandRetargeter, RetargetingResult

from .detector import HandLandmarkFrame, MediaPipeHandDetector


@dataclass(frozen=True)
class TeleopFrame:
    hand_frame: HandLandmarkFrame
    ref_vectors: np.ndarray
    retargeting: RetargetingResult

    @property
    def landmarks(self) -> np.ndarray:
        return self.hand_frame.landmarks

    @property
    def active_joint_positions(self) -> dict[str, float]:
        return self.retargeting.active_joint_positions

    @property
    def hardware_motor_positions(self) -> np.ndarray:
        return self.retargeting.hardware_motor_positions


class MidasTeleopPipeline:
    """Owns detection and retargeting, but not simulation or hardware."""

    def __init__(
        self,
        *,
        detector: MediaPipeHandDetector | None = None,
        retargeter: MidasHandRetargeter | None = None,
    ):
        self.detector = detector or MediaPipeHandDetector()
        self.retargeter = retargeter or MidasHandRetargeter()

    def process_rgb(self, rgb: np.ndarray) -> TeleopFrame | None:
        hand_frame = self.detector.detect_rgb(rgb)
        if hand_frame is None:
            return None
        return self.process_landmark_frame(hand_frame)

    def process_landmark_frame(self, hand_frame: HandLandmarkFrame) -> TeleopFrame:
        vectors = self.retargeter.landmarks_to_vectors(hand_frame.landmarks)
        retargeting = self.retargeter.retarget_landmarks(hand_frame.landmarks)
        return TeleopFrame(
            hand_frame=hand_frame,
            ref_vectors=vectors,
            retargeting=retargeting,
        )

    def calibrate_neutral_from_last_frame(self) -> dict[str, float]:
        """Use the latest retargeted pose as the neutral command reference."""

        return self.retargeter.calibrate_neutral_from_last_frame()

    def clear_neutral_offsets(self) -> None:
        """Clear retargeter neutral calibration references."""

        self.retargeter.clear_neutral_offsets()

    def close(self) -> None:
        self.detector.close()
