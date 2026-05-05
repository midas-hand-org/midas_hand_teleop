"""MediaPipe hand-landmark wrapper for MIDAS teleop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from midas_hand_retargeter.human import (
    mediapipe_world_to_mano_landmarks,
    mirror_landmarks_for_robot_hand,
)

HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "midas_hand_teleop" / "hand_landmarker.task"


@dataclass(frozen=True)
class HandLandmarkFrame:
    landmarks: np.ndarray
    image_landmarks: object | None
    wrist_rotation: np.ndarray
    mediapipe_hand_type: str = "Unknown"
    input_hand_type: str = "Unknown"
    robot_hand_type: str = "Right"
    mirrored: bool = False


class MediaPipeHandDetector:
    """Detect one hand and return wrist-centered 21x3 landmarks."""

    def __init__(
        self,
        *,
        hand_type: str = "Right",
        input_hand_type: str = "auto",
        selfie: bool = False,
        mirror_input: bool = True,
        mirror_axis: str = "y",
        lock_input_hand: bool = True,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.8,
        model_path: str | Path | None = None,
    ):
        import mediapipe as mp

        self._mp = mp
        self.hand_type = _normalize_hand_type(hand_type, name="hand_type")
        self.input_hand_type = _normalize_input_hand_type(input_hand_type)
        self.selfie = selfie
        self.mirror_input = bool(mirror_input)
        self.mirror_axis = mirror_axis
        self.lock_input_hand = bool(lock_input_hand)
        self._locked_input_hand_type: str | None = None
        if hasattr(mp, "solutions"):
            self._backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._connections = None
        else:
            self._backend = "tasks"
            self._hands = self._create_tasks_landmarker(
                mp,
                model_path=model_path,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._connections = [
                (connection.start, connection.end)
                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            ]
            self._timestamp_ms = 0

    def detect_rgb(self, rgb: np.ndarray) -> HandLandmarkFrame | None:
        if self._backend == "tasks":
            return self._detect_rgb_tasks(rgb)
        return self._detect_rgb_solutions(rgb)

    def _detect_rgb_solutions(self, rgb: np.ndarray) -> HandLandmarkFrame | None:
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_index = -1
        mediapipe_hand_type = "Unknown"
        input_hand_type = "Unknown"
        for index, handedness in enumerate(results.multi_handedness):
            label = _normalize_hand_type(
                handedness.ListFields()[0][1][0].label,
                name="MediaPipe handedness label",
            )
            physical_label = self._physical_hand_type_from_mediapipe_label(label)
            if self._accept_input_hand_type(physical_label):
                hand_index = index
                mediapipe_hand_type = label
                input_hand_type = physical_label
                self._lock_input_hand_type(physical_label)
                break
        if hand_index < 0:
            return None

        world = self._parse_world_landmarks(results.multi_hand_world_landmarks[hand_index])
        landmarks, mirrored = self._retarget_landmarks(world, input_hand_type)
        wrist_rotation = self._estimate_wrist_frame(world - world[0:1])
        return HandLandmarkFrame(
            landmarks=landmarks,
            image_landmarks=results.multi_hand_landmarks[hand_index],
            wrist_rotation=wrist_rotation,
            mediapipe_hand_type=mediapipe_hand_type,
            input_hand_type=input_hand_type,
            robot_hand_type=self.hand_type,
            mirrored=mirrored,
        )

    def _detect_rgb_tasks(self, rgb: np.ndarray) -> HandLandmarkFrame | None:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        self._timestamp_ms += 33
        results = self._hands.detect_for_video(image, self._timestamp_ms)
        if not results.hand_world_landmarks:
            return None

        hand_index = -1
        mediapipe_hand_type = "Unknown"
        input_hand_type = "Unknown"
        for index, handedness in enumerate(results.handedness):
            label = _normalize_hand_type(
                handedness[0].category_name,
                name="MediaPipe handedness label",
            )
            physical_label = self._physical_hand_type_from_mediapipe_label(label)
            if self._accept_input_hand_type(physical_label):
                hand_index = index
                mediapipe_hand_type = label
                input_hand_type = physical_label
                self._lock_input_hand_type(physical_label)
                break
        if hand_index < 0:
            return None

        world = self._parse_tasks_world_landmarks(results.hand_world_landmarks[hand_index])
        landmarks, mirrored = self._retarget_landmarks(world, input_hand_type)
        wrist_rotation = self._estimate_wrist_frame(world - world[0:1])
        return HandLandmarkFrame(
            landmarks=landmarks,
            image_landmarks=results.hand_landmarks[hand_index],
            wrist_rotation=wrist_rotation,
            mediapipe_hand_type=mediapipe_hand_type,
            input_hand_type=input_hand_type,
            robot_hand_type=self.hand_type,
            mirrored=mirrored,
        )

    def draw_landmarks(self, bgr: np.ndarray, frame: HandLandmarkFrame | None) -> np.ndarray:
        if frame is None or frame.image_landmarks is None:
            return bgr
        if self._backend == "tasks":
            return self._draw_status_text(
                self._draw_tasks_landmarks(bgr, frame.image_landmarks),
                frame,
            )
        self._mp.solutions.drawing_utils.draw_landmarks(
            bgr,
            frame.image_landmarks,
            self._mp.solutions.hands.HAND_CONNECTIONS,
            self._mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            self._mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
        return self._draw_status_text(bgr, frame)

    @staticmethod
    def _parse_world_landmarks(world_landmarks) -> np.ndarray:
        keypoints = np.empty((21, 3), dtype=np.float32)
        for index, landmark in enumerate(world_landmarks.landmark):
            keypoints[index] = [landmark.x, landmark.y, landmark.z]
        return keypoints

    @staticmethod
    def _parse_tasks_world_landmarks(world_landmarks) -> np.ndarray:
        keypoints = np.empty((21, 3), dtype=np.float32)
        for index, landmark in enumerate(world_landmarks):
            keypoints[index] = [landmark.x, landmark.y, landmark.z]
        return keypoints

    @staticmethod
    def _estimate_wrist_frame(centered_landmarks: np.ndarray) -> np.ndarray:
        from midas_hand_retargeter.human import estimate_frame_from_hand_points

        return estimate_frame_from_hand_points(centered_landmarks)

    def _physical_hand_type_from_mediapipe_label(self, label: str) -> str:
        # MediaPipe handedness is documented for selfie-style mirrored input.
        # OpenCV webcam frames are normally not mirrored, so keep the legacy
        # dex-retargeting correction unless the caller passes --selfie.
        return label if self.selfie else _opposite_hand_type(label)

    def _accept_input_hand_type(self, hand_type: str) -> bool:
        if self.input_hand_type != "auto":
            return hand_type == self.input_hand_type
        if self.lock_input_hand and self._locked_input_hand_type is not None:
            return hand_type == self._locked_input_hand_type
        return True

    def _lock_input_hand_type(self, hand_type: str) -> None:
        if self.input_hand_type == "auto" and self.lock_input_hand:
            self._locked_input_hand_type = hand_type

    def _retarget_landmarks(
        self,
        world_landmarks: np.ndarray,
        input_hand_type: str,
    ) -> tuple[np.ndarray, bool]:
        landmarks = mediapipe_world_to_mano_landmarks(
            world_landmarks,
            hand_type=input_hand_type,
        )
        mirrored = self.mirror_input and input_hand_type != self.hand_type
        if mirrored:
            landmarks = mirror_landmarks_for_robot_hand(
                landmarks,
                source_hand_type=input_hand_type,
                target_hand_type=self.hand_type,
                axis=self.mirror_axis,
            )
        return landmarks, mirrored

    def _draw_tasks_landmarks(self, bgr: np.ndarray, landmarks) -> np.ndarray:
        import cv2

        height, width = bgr.shape[:2]
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]
        for start, end in self._connections or []:
            cv2.line(bgr, points[start], points[end], (255, 255, 255), 2)
        for point in points:
            cv2.circle(bgr, point, 4, (48, 48, 255), -1)
        return bgr

    def _draw_status_text(self, bgr: np.ndarray, frame: HandLandmarkFrame) -> np.ndarray:
        import cv2

        status = (
            f"MP:{frame.mediapipe_hand_type} "
            f"input:{frame.input_hand_type} "
            f"robot:{frame.robot_hand_type}"
        )
        if frame.mirrored:
            status += f" mirror:{self.mirror_axis}"
        cv2.putText(
            bgr,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return bgr

    @staticmethod
    def _create_tasks_landmarker(
        mp,
        *,
        model_path: str | Path | None,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        model = _ensure_hand_landmarker_model(model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._hands.close()


def _ensure_hand_landmarker_model(model_path: str | Path | None = None) -> Path:
    path = Path(model_path).expanduser() if model_path is not None else DEFAULT_MODEL_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand landmarker model to {path}")
    urlretrieve(HAND_LANDMARKER_TASK_URL, path)
    return path


def _normalize_hand_type(value: str, *, name: str) -> str:
    hand_type = value.title()
    if hand_type not in {"Right", "Left"}:
        raise ValueError(f"{name} must be 'Right' or 'Left', got {value!r}")
    return hand_type


def _normalize_input_hand_type(value: str) -> str:
    if value.lower() == "auto":
        return "auto"
    return _normalize_hand_type(value, name="input_hand_type")


def _opposite_hand_type(hand_type: str) -> str:
    return {"Right": "Left", "Left": "Right"}[hand_type]
