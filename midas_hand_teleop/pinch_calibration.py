"""Thumb/index pinch calibration helpers for webcam teleop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from midas_hand_retargeter import RetargetingResult
from midas_hand_retargeter.constants import (
    ACTIVE_JOINT_NAMES,
    HARDWARE_MOTOR_JOINT_NAMES,
)
from midas_hand_retargeter.human import as_landmarks


PINCH_FEATURE_NAMES = (
    "thumb_index_tip_distance_norm",
    "thumb_to_index_tip_palm_forward_norm",
    "thumb_to_index_tip_palm_lateral_norm",
    "thumb_to_index_tip_palm_normal_norm",
    "thumb_mcp_bend_norm",
    "thumb_ip_bend_norm",
    "index_mcp_bend_norm",
    "index_pip_bend_norm",
    "index_dip_bend_norm",
)

# Per-feature scales for the distance metric. The raw features are already
# normalized by hand scale or pi; these values express "meaningful difference"
# units for comparing a live pose to the calibrated pinch pose.
PINCH_FEATURE_SCALES = np.asarray(
    (0.12, 0.20, 0.20, 0.20, 0.18, 0.18, 0.18, 0.18, 0.18),
    dtype=np.float32,
)

THUMB_INDEX_JOINT_NAMES = tuple(
    name
    for name in HARDWARE_MOTOR_JOINT_NAMES
    if name.startswith("thumb_") or name.startswith("index_")
)
THUMB_INDEX_HARDWARE_INDICES = tuple(
    HARDWARE_MOTOR_JOINT_NAMES.index(name)
    for name in THUMB_INDEX_JOINT_NAMES
)
THUMB_INDEX_ACTIVE_INDICES = tuple(
    ACTIVE_JOINT_NAMES.index(name)
    for name in THUMB_INDEX_JOINT_NAMES
    if name in ACTIVE_JOINT_NAMES
)


@dataclass(frozen=True)
class PinchThresholds:
    """Distance/confidence settings for pinch detection and blending."""

    activate_distance: float = 1.0
    release_distance: float = 1.25
    full_confidence_distance: float = 0.25
    smoothing_alpha: float = 0.25

    def validate(self) -> None:
        if self.activate_distance <= 0.0:
            raise ValueError("activate_distance must be positive")
        if self.release_distance < self.activate_distance:
            raise ValueError("release_distance must be >= activate_distance")
        if self.full_confidence_distance < 0.0:
            raise ValueError("full_confidence_distance must be non-negative")
        if self.full_confidence_distance >= self.activate_distance:
            raise ValueError(
                "full_confidence_distance must be smaller than activate_distance"
            )
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1]")

    def to_json(self) -> dict[str, float]:
        return {
            "activate_distance": float(self.activate_distance),
            "release_distance": float(self.release_distance),
            "full_confidence_distance": float(self.full_confidence_distance),
            "smoothing_alpha": float(self.smoothing_alpha),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "PinchThresholds":
        if not data:
            return cls()
        thresholds = cls(
            activate_distance=float(data.get("activate_distance", cls.activate_distance)),
            release_distance=float(data.get("release_distance", cls.release_distance)),
            full_confidence_distance=float(
                data.get("full_confidence_distance", cls.full_confidence_distance)
            ),
            smoothing_alpha=float(data.get("smoothing_alpha", cls.smoothing_alpha)),
        )
        thresholds.validate()
        return thresholds


@dataclass
class PinchRuntimeState:
    """Smoothed runtime state for hysteretic pinch correction."""

    active: bool = False
    confidence: float = 0.0
    distance: float | None = None


@dataclass
class PinchCalibration:
    """Saved and runtime state for thumb/index pinch correction."""

    human_feature_vector: np.ndarray | None = None
    robot_joint_vector: np.ndarray | None = None
    thresholds: PinchThresholds = field(default_factory=PinchThresholds)
    joint_names: tuple[str, ...] = THUMB_INDEX_JOINT_NAMES
    feature_names: tuple[str, ...] = PINCH_FEATURE_NAMES
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    state: PinchRuntimeState = field(default_factory=PinchRuntimeState)

    @property
    def is_complete(self) -> bool:
        return self.human_feature_vector is not None and self.robot_joint_vector is not None

    @property
    def has_human(self) -> bool:
        return self.human_feature_vector is not None

    @property
    def has_robot(self) -> bool:
        return self.robot_joint_vector is not None

    def record_human(self, landmarks: np.ndarray) -> np.ndarray:
        self.human_feature_vector = extract_pinch_features(landmarks)
        self.metadata["human_recorded_at"] = _utc_now()
        self.state = PinchRuntimeState()
        return self.human_feature_vector.copy()

    def record_robot_from_result(self, result: RetargetingResult) -> np.ndarray:
        self.robot_joint_vector = np.asarray(
            [result.active_joint_positions[name] for name in self.joint_names],
            dtype=np.float32,
        )
        self.metadata["robot_recorded_at"] = _utc_now()
        self.metadata["robot_source"] = "current retargeted command"
        return self.robot_joint_vector.copy()

    def record_robot_from_hardware_positions(
        self,
        motor_positions: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(motor_positions, dtype=np.float32)
        if positions.ndim != 1 or positions.shape[0] < len(HARDWARE_MOTOR_JOINT_NAMES):
            raise ValueError(
                "Expected hardware motor positions in HARDWARE_MOTOR_JOINT_NAMES order"
            )
        self.robot_joint_vector = positions[list(THUMB_INDEX_HARDWARE_INDICES)].copy()
        self.metadata["robot_recorded_at"] = _utc_now()
        self.metadata["robot_source"] = "hardware feedback"
        return self.robot_joint_vector.copy()

    def compute_distance(self, features: np.ndarray) -> float | None:
        if self.human_feature_vector is None:
            return None
        return normalized_feature_distance(features, self.human_feature_vector)

    def confidence_from_distance(self, distance: float | None) -> float:
        if distance is None:
            return 0.0
        return confidence_from_distance(distance, self.thresholds)

    def update_confidence(self, features: np.ndarray | None) -> PinchRuntimeState:
        if not self.enabled or not self.is_complete or features is None:
            self.state.active = False
            self.state.confidence = _blend(
                self.state.confidence,
                0.0,
                self.thresholds.smoothing_alpha,
            )
            self.state.distance = None
            return self.state

        distance = self.compute_distance(features)
        if distance is None:
            raw_confidence = 0.0
            active = False
        else:
            if self.state.active:
                active = distance <= self.thresholds.release_distance
            else:
                active = distance <= self.thresholds.activate_distance
            raw_confidence = self.confidence_from_distance(distance) if active else 0.0

        self.state.active = active
        self.state.confidence = _blend(
            self.state.confidence,
            raw_confidence,
            self.thresholds.smoothing_alpha,
        )
        self.state.distance = distance
        return self.state

    def apply(self, result: RetargetingResult, features: np.ndarray | None) -> RetargetingResult:
        state = self.update_confidence(features)
        alpha = float(np.clip(state.confidence, 0.0, 1.0))
        if (
            alpha <= 1e-6
            or self.robot_joint_vector is None
            or len(self.robot_joint_vector) != len(self.joint_names)
        ):
            return result
        return apply_pinch_correction(
            result,
            joint_names=self.joint_names,
            robot_joint_vector=self.robot_joint_vector,
            alpha=alpha,
        )

    def to_json(self) -> dict[str, Any]:
        now = _utc_now()
        metadata = dict(self.metadata)
        metadata.setdefault("created_at", now)
        metadata["updated_at"] = now
        return {
            "version": 1,
            "metadata": metadata,
            "feature_names": list(self.feature_names),
            "human_pinch_feature_vector": (
                None
                if self.human_feature_vector is None
                else [float(value) for value in self.human_feature_vector]
            ),
            "feature_scales": [float(value) for value in PINCH_FEATURE_SCALES],
            "robot_joint_names": list(self.joint_names),
            "robot_joint_indices": {
                "hardware_motor_indices": list(THUMB_INDEX_HARDWARE_INDICES),
                "active_joint_indices": list(THUMB_INDEX_ACTIVE_INDICES),
            },
            "robot_pinch_joint_vector": (
                None
                if self.robot_joint_vector is None
                else [float(value) for value in self.robot_joint_vector]
            ),
            "thresholds": self.thresholds.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PinchCalibration":
        feature_names = tuple(data.get("feature_names", PINCH_FEATURE_NAMES))
        if feature_names != PINCH_FEATURE_NAMES:
            raise ValueError(
                "Pinch calibration feature_names do not match this code version"
            )
        joint_names = tuple(data.get("robot_joint_names", THUMB_INDEX_JOINT_NAMES))
        if joint_names != THUMB_INDEX_JOINT_NAMES:
            raise ValueError(
                "Pinch calibration robot_joint_names do not match this code version"
            )

        human = data.get("human_pinch_feature_vector")
        robot = data.get("robot_pinch_joint_vector")
        calibration = cls(
            human_feature_vector=(
                None if human is None else np.asarray(human, dtype=np.float32)
            ),
            robot_joint_vector=(
                None if robot is None else np.asarray(robot, dtype=np.float32)
            ),
            thresholds=PinchThresholds.from_json(data.get("thresholds")),
            joint_names=joint_names,
            feature_names=feature_names,
            metadata=dict(data.get("metadata", {})),
        )
        calibration._validate_vectors()
        return calibration

    def _validate_vectors(self) -> None:
        if (
            self.human_feature_vector is not None
            and self.human_feature_vector.shape != (len(PINCH_FEATURE_NAMES),)
        ):
            raise ValueError("human_pinch_feature_vector has the wrong length")
        if (
            self.robot_joint_vector is not None
            and self.robot_joint_vector.shape != (len(THUMB_INDEX_JOINT_NAMES),)
        ):
            raise ValueError("robot_pinch_joint_vector has the wrong length")


def extract_pinch_features(landmarks: np.ndarray) -> np.ndarray:
    """Return normalized thumb/index pinch features from 21x3 landmarks.

    The detector has already transformed landmarks into the robot-hand frame.
    Features avoid image pixels: distances are normalized by palm scale, tip
    displacement is expressed in a palm basis, and joint angles are divided by
    pi so their units are comparable across users.
    """

    points = as_landmarks(landmarks)
    scale = _hand_scale(points)
    palm_forward, palm_lateral, palm_normal = _palm_basis(points)

    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = (points[index] for index in (1, 2, 3, 4))
    index_mcp, index_pip, index_dip, index_tip = (
        points[index] for index in (5, 6, 7, 8)
    )

    thumb_to_index = (index_tip - thumb_tip) / scale
    tip_delta_palm = np.asarray(
        [
            np.dot(thumb_to_index, palm_forward),
            np.dot(thumb_to_index, palm_lateral),
            np.dot(thumb_to_index, palm_normal),
        ],
        dtype=np.float32,
    )

    index_proximal = index_pip - index_mcp
    features = np.asarray(
        [
            np.linalg.norm(thumb_to_index),
            *tip_delta_palm,
            _angle_between(thumb_mcp - thumb_cmc, thumb_ip - thumb_mcp) / np.pi,
            _angle_between(thumb_ip - thumb_mcp, thumb_tip - thumb_ip) / np.pi,
            _angle_between(palm_forward, index_proximal) / np.pi,
            _angle_between(index_proximal, index_dip - index_pip) / np.pi,
            _angle_between(index_dip - index_pip, index_tip - index_dip) / np.pi,
        ],
        dtype=np.float32,
    )
    return features


def normalized_feature_distance(
    current_features: np.ndarray,
    reference_features: np.ndarray,
) -> float:
    current = np.asarray(current_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    if current.shape != reference.shape:
        raise ValueError(
            f"Feature shape mismatch: current={current.shape}, reference={reference.shape}"
        )
    scaled_delta = (current - reference) / PINCH_FEATURE_SCALES
    return float(np.sqrt(np.mean(np.square(scaled_delta))))


def confidence_from_distance(distance: float, thresholds: PinchThresholds) -> float:
    if distance <= thresholds.full_confidence_distance:
        return 1.0
    span = thresholds.activate_distance - thresholds.full_confidence_distance
    confidence = (thresholds.activate_distance - distance) / max(span, 1e-6)
    return float(np.clip(confidence, 0.0, 1.0))


def apply_pinch_correction(
    result: RetargetingResult,
    *,
    joint_names: tuple[str, ...],
    robot_joint_vector: np.ndarray,
    alpha: float,
) -> RetargetingResult:
    """Blend thumb/index active targets toward the calibrated robot pinch pose."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    active = dict(result.active_joint_positions)
    for joint_name, saved_value in zip(joint_names, robot_joint_vector):
        if joint_name not in active:
            continue
        original_value = active[joint_name]
        active[joint_name] = float(
            (1.0 - alpha) * original_value + alpha * float(saved_value)
        )

    hardware = np.asarray(
        [active[name] for name in HARDWARE_MOTOR_JOINT_NAMES],
        dtype=np.float32,
    )
    robot_qpos = np.asarray(result.robot_qpos, dtype=np.float32).copy()
    robot_index_by_name = {
        name: index for index, name in enumerate(result.robot_joint_names)
    }
    for joint_name in joint_names:
        joint_index = robot_index_by_name.get(joint_name)
        if joint_index is not None and joint_name in active:
            robot_qpos[joint_index] = active[joint_name]

    return RetargetingResult(
        robot_qpos=robot_qpos,
        robot_joint_names=result.robot_joint_names,
        ref_vectors=result.ref_vectors,
        active_joint_positions=active,
        hardware_motor_positions=hardware,
        fixed_joint_positions=result.fixed_joint_positions,
    )


def save_pinch_calibration(calibration: PinchCalibration, path: str | Path) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(calibration.to_json(), f, indent=2, sort_keys=True)
        f.write("\n")


def load_pinch_calibration(path: str | Path) -> PinchCalibration:
    with open(Path(path).expanduser()) as f:
        return PinchCalibration.from_json(json.load(f))


def _hand_scale(points: np.ndarray) -> float:
    pairs = ((0, 5), (0, 9), (0, 13), (5, 9), (9, 13))
    distances = [
        float(np.linalg.norm(points[end] - points[start]))
        for start, end in pairs
    ]
    scale = float(np.median([distance for distance in distances if distance > 1e-6]))
    return max(scale, 1e-6)


def _palm_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    ring_mcp = points[13]
    palm_forward = _unit(middle_mcp - wrist)
    palm_lateral = _unit(ring_mcp - index_mcp)
    palm_normal = _unit(np.cross(palm_lateral, palm_forward))
    if np.linalg.norm(palm_normal) < 1e-6:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    palm_lateral = _unit(np.cross(palm_forward, palm_normal))
    return palm_forward, palm_lateral, palm_normal


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _unit(a)
    b_unit = _unit(b)
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.arccos(dot))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def _blend(previous: float, target: float, alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(previous + alpha * (target - previous))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
