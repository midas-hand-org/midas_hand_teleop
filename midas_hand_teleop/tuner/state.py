"""Shared state between the tuning UI and the control loop.

The control loop owns the retargeter and the sim; the HTTP server only reads
and writes this object. Everything crossing that boundary is either an atomic
frozen-object swap (parameters) or a plain snapshot dict (telemetry), so the
web server can never stall or half-update the 60 Hz loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.store import ProfileStore

#: Telemetry frames kept for the UI's rolling traces.
TRACE_LENGTH = 600


@dataclass
class GloveStatus:
    """What the UI needs to distinguish 'bad tuning' from 'bad input'."""

    connected: bool = False
    source: str = "unknown"
    rate_hz: float = 0.0
    age_s: float = float("inf")
    #: Glove -> retarget latency, or None when the publisher sends no timestamp.
    latency_ms: float | None = None
    parse_failures: int = 0

    def to_dict(self) -> dict:
        age = self.age_s
        return {
            "connected": self.connected,
            "source": self.source,
            "rate_hz": round(self.rate_hz, 1),
            "age_s": None if age == float("inf") else round(age, 3),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "parse_failures": self.parse_failures,
            # Never report a latency as good when the frame it came from is old.
            "stale": age > 0.5,
        }


@dataclass
class LoopStatus:
    rate_hz: float = 0.0
    retarget_ms: float = 0.0
    backend: str = "mujoco"
    mode: str = "analytic"
    armed: bool = False
    hardware_available: bool = False

    def to_dict(self) -> dict:
        return {
            "rate_hz": round(self.rate_hz, 1),
            "retarget_ms": round(self.retarget_ms, 3),
            "backend": self.backend,
            "mode": self.mode,
            "armed": self.armed,
            "hardware_available": self.hardware_available,
        }


@dataclass
class TunerState:
    """Everything the browser can see or change."""

    store: ProfileStore = field(default_factory=ProfileStore)
    glove: GloveStatus = field(default_factory=GloveStatus)
    loop: LoopStatus = field(default_factory=LoopStatus)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._telemetry: dict[str, Any] = {}
        self._trace: deque[dict[str, Any]] = deque(maxlen=TRACE_LENGTH)
        self._version = 0
        self._calibration_request: str | None = None
        self._messages: deque[str] = deque(maxlen=32)

    # --- parameters ----------------------------------------------------
    @property
    def profile(self) -> RetargetProfile:
        return self.store.get()

    def apply_updates(self, updates: Mapping[str, Any]) -> RetargetProfile:
        return self.store.apply(updates)

    # --- telemetry (written by the loop, read by the server) -----------
    def publish(
        self,
        *,
        commanded: Mapping[str, float],
        measured: Mapping[str, float],
        intermediates: Mapping[str, Any],
        palm: Mapping[str, Any] | None = None,
    ) -> None:
        frame = {
            "t": time.monotonic(),
            "commanded": {k: round(float(v), 5) for k, v in commanded.items()},
            "measured": {k: round(float(v), 5) for k, v in measured.items()},
            "intermediates": intermediates,
            "palm": palm or {},
            "glove": self.glove.to_dict(),
            "loop": self.loop.to_dict(),
        }
        with self._lock:
            self._telemetry = frame
            self._trace.append(
                {"t": frame["t"], "commanded": frame["commanded"], "measured": frame["measured"]}
            )
            self._version += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame = dict(self._telemetry)
            frame["version"] = self._version
            frame["messages"] = list(self._messages)
            return frame

    def trace(self, joint: str) -> dict[str, list]:
        with self._lock:
            rows = list(self._trace)
        return {
            "t": [row["t"] for row in rows],
            "commanded": [row["commanded"].get(joint) for row in rows],
            "measured": [row["measured"].get(joint) for row in rows],
        }

    def note(self, message: str) -> None:
        with self._lock:
            self._messages.append(message)

    # --- calibration handshake -----------------------------------------
    def request_calibration(self, action: str) -> None:
        """Queue 'capture' or 'clear'; the loop performs it on its next frame.

        Done as a request rather than a direct call so calibration always
        happens on the control thread, holding a real landmark frame.
        """

        if action not in ("capture", "clear"):
            raise ValueError(f"Unknown calibration action {action!r}")
        with self._lock:
            self._calibration_request = action

    def take_calibration_request(self) -> str | None:
        with self._lock:
            request, self._calibration_request = self._calibration_request, None
            return request
