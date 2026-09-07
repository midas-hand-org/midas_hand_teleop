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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.store import ProfileStore


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
    #: The retargeter's live neutral (zero-pose) calibration, republished each
    #: frame by the loop. Saving a preset reads it from here rather than
    #: trusting the browser to echo back what it was last told: the browser is
    #: only ever told on a preset LOAD, so a freshly captured neutral was being
    #: written out as {} under a "saved" message.
    neutral_offsets: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._telemetry: dict[str, Any] = {}
        self._version = 0
        self._calibration_request: str | None = None
        self._pending_neutral: dict[str, float] | None = None
        #: None until a browser has polled once. The watchdog only trips after
        #: a client has been seen, so a deliberate headless --start-armed run
        #: is not disarmed for the crime of having no browser.
        self._last_client_poll: float | None = None
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
            "neutral_offsets": dict(self.neutral_offsets),
        }
        with self._lock:
            self._telemetry = frame
            self._version += 1

    def snapshot(self) -> dict[str, Any]:
        """Also the client heartbeat: the browser polls this continuously, so
        silence here is how the loop learns the tab is gone."""

        with self._lock:
            self._last_client_poll = time.monotonic()
            frame = dict(self._telemetry)
            frame["version"] = self._version
            frame["messages"] = list(self._messages)
            return frame

    def seconds_since_client_poll(self) -> float:
        """Seconds since a browser last polled, or 0.0 if none ever has."""

        with self._lock:
            if self._last_client_poll is None:
                return 0.0
            return time.monotonic() - self._last_client_poll

    def note(self, message: str) -> None:
        with self._lock:
            self._messages.append(message)

    # --- calibration handshake -----------------------------------------
    #: 'capture' zero pose, 'clear' it, or 'scale' the operator's hand size.
    CALIBRATION_ACTIONS = ("capture", "clear", "scale")

    def request_calibration(self, action: str) -> None:
        """Queue a calibration action; the loop performs it on its next frame.

        Done as a request rather than a direct call so calibration always
        happens on the control thread, holding a real landmark frame.
        """

        if action not in self.CALIBRATION_ACTIONS:
            raise ValueError(f"Unknown calibration action {action!r}")
        with self._lock:
            self._calibration_request = action

    def take_calibration_request(self) -> str | None:
        with self._lock:
            request, self._calibration_request = self._calibration_request, None
            return request

    def request_neutral_offsets(self, offsets) -> None:
        """Ask the loop to install a neutral calibration, e.g. from a preset.

        Queued rather than applied, for the same reason the calibration actions
        are: the retargeter belongs to the control thread.
        """

        with self._lock:
            self._pending_neutral = {str(k): float(v) for k, v in dict(offsets).items()}

    def take_neutral_offsets(self):
        with self._lock:
            pending, self._pending_neutral = self._pending_neutral, None
            return pending
