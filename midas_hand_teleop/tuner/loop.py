"""The tuning control loop: glove in, sim out, telemetry to the browser.

Deliberately a separate, simpler loop from ``manus_teleop``: this one exists to
make parameters observable, so it publishes intermediates and measured state
that the teleop driver has no reason to compute.

Threading: this loop owns the retargeter and the backend and runs on the main
thread. The HTTP server runs on daemon threads and only touches ``TunerState``,
so a slow or disconnected browser can never stall the hand.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from midas_hand_retargeter import MidasHandRetargeter
from midas_hand_retargeter.postprocess import analytic_debug

from ..manus_glove.glove_subscriber import (
    GLOVE_TOPIC,
    GloveSubscriber,
    parse_glove_array,
    start_data_center_proxy,
)
from .state import TunerState

logger = logging.getLogger(__name__)

#: Treat the glove as gone after this long with no fresh frame.
GLOVE_TIMEOUT_S = 0.5


@dataclass
class LoopConfig:
    side: str = "right"
    host: str = "localhost"
    control_hz: float = 60.0
    duration_s: float | None = None
    start_proxy: bool = True


class TunerLoop:
    """Drives the retargeter from the glove bus and publishes telemetry."""

    def __init__(self, state: TunerState, retargeter, backend, config: LoopConfig):
        self.state = state
        self.retargeter = retargeter
        self.backend = backend
        self.config = config

        if config.start_proxy:
            # Without a bound XSUB/XPUB relay a publisher and subscriber never
            # meet, and nothing reports an error — messages just vanish.
            start_data_center_proxy()
            time.sleep(0.3)  # PUB/SUB slow joiner

        self.subscriber = GloveSubscriber(
            GLOVE_TOPIC.format(side=config.side), config.host
        )
        self._last_frame_monotonic = 0.0
        self._frame_intervals: list[float] = []
        self._last_landmarks: np.ndarray | None = None

    # --- glove ---------------------------------------------------------
    def _poll_glove(self) -> np.ndarray | None:
        array, metadata = self.subscriber.poll_latest_with_metadata()
        if array is None:
            age = (
                time.monotonic() - self._last_frame_monotonic
                if self._last_frame_monotonic
                else float("inf")
            )
            self.state.glove.age_s = age
            self.state.glove.connected = age < GLOVE_TIMEOUT_S
            return None

        landmarks = parse_glove_array(array)
        if landmarks is None:
            self.state.glove.parse_failures += 1
            return None

        now = time.monotonic()
        if self._last_frame_monotonic:
            interval = now - self._last_frame_monotonic
            self._frame_intervals.append(interval)
            if len(self._frame_intervals) > 60:
                self._frame_intervals.pop(0)
            mean = sum(self._frame_intervals) / len(self._frame_intervals)
            self.state.glove.rate_hz = 1.0 / mean if mean > 0 else 0.0
        self._last_frame_monotonic = now
        self.state.glove.age_s = 0.0
        self.state.glove.connected = True

        published = metadata.get("t_mono_pub")
        # Same-host only: both stamps come from time.monotonic(), which shares
        # an epoch within a machine but not across machines.
        self.state.glove.latency_ms = (
            (now - float(published)) * 1000.0 if isinstance(published, (int, float)) else None
        )
        return landmarks

    # --- calibration ---------------------------------------------------
    def _service_calibration(self) -> None:
        request = self.state.take_calibration_request()
        if request is None:
            return
        if request == "clear":
            self.retargeter.clear_neutral_offsets()
            self.state.note("neutral calibration cleared")
            return
        try:
            captured = self.retargeter.calibrate_neutral_from_last_frame()
            self.state.note(f"captured neutral on {len(captured)} joints")
        except RuntimeError as exc:
            self.state.note(f"calibration failed: {exc}")

    # --- main ----------------------------------------------------------
    def run(self) -> None:
        period = 1.0 / max(self.config.control_hz, 1e-6)
        next_tick = time.monotonic()
        started = next_tick
        ticks = 0
        tick_window = next_tick

        self.state.loop.mode = self.retargeter.config.mode
        self.state.loop.backend = type(self.backend).__name__

        while True:
            if self.config.duration_s and time.monotonic() - started > self.config.duration_s:
                return

            landmarks = self._poll_glove()
            if landmarks is not None:
                self._last_landmarks = landmarks

            if self._last_landmarks is not None:
                # One profile read per frame; a concurrent UI edit lands on the
                # next frame rather than halfway through this one.
                self.retargeter.profile = self.state.store.get()

                solve_start = time.perf_counter()
                result = self.retargeter.retarget_landmarks(self._last_landmarks)
                self.state.loop.retarget_ms = (time.perf_counter() - solve_start) * 1000.0

                self._service_calibration()
                self._send(result)
                self._publish(result, self._last_landmarks)

            ticks += 1
            now = time.monotonic()
            if now - tick_window >= 1.0:
                self.state.loop.rate_hz = ticks / (now - tick_window)
                ticks, tick_window = 0, now

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def _send(self, result) -> None:
        """Forward to the backend, honouring the arm gate for hardware."""

        if self.state.loop.hardware_available and not self.state.loop.armed:
            return
        self.backend.send(result)

    def _publish(self, result, landmarks) -> None:
        try:
            measured = self.backend.measured()
        except Exception:  # a sim/hardware read must never kill the loop
            measured = {}
        debug = analytic_debug(landmarks, self.retargeter.profile)
        self.state.publish(
            commanded=result.active_joint_positions,
            measured=measured,
            intermediates={k: v for k, v in debug.items() if not k.startswith("_")},
            palm=debug.get("_palm", {}),
        )

    def close(self) -> None:
        self.subscriber.close()
        self.backend.close()
