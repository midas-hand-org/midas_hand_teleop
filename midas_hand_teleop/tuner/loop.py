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
    #: Deadman. With no glove frame for this long the hardware is disarmed and
    #: has to be armed again by hand. The loop otherwise keeps re-solving its
    #: last frame forever, which on hardware means leaning on whatever the hand
    #: is holding until someone notices. 0 disables it.
    stale_timeout_s: float = 0.5
    #: Disarm if no browser has polled for this long. The arm switch lives in a
    #: web page, so a closed tab, a slept laptop or lost wifi must not leave a
    #: hand tracking. Only trips once a browser HAS been seen, so a deliberate
    #: headless --start-armed run is left alone. 0 disables it.
    client_timeout_s: float = 3.0


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

        self.subscriber = GloveSubscriber(GLOVE_TOPIC.format(side=config.side), config.host)
        self._last_frame_monotonic = 0.0
        self._frame_intervals: list[float] = []
        self._last_landmarks: np.ndarray | None = None
        # Only a backend that can be armed gates on arming; the sim never does.
        self.state.loop.hardware_available = callable(getattr(backend, "arm", None))
        self._armed_applied = False
        self._deadman_tripped = False

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
        pending = self.state.take_neutral_offsets()
        if pending is not None:
            # A preset's saved zero pose. Applied here rather than at load time
            # because the retargeter belongs to this thread.
            self.retargeter.set_neutral_offsets(pending)
            self.state.note(f"applied neutral calibration on {len(pending)} joints")

        request = self.state.take_calibration_request()
        if request is None:
            return
        if request == "clear":
            self.retargeter.clear_neutral_offsets()
            self.state.note("neutral calibration cleared")
            return
        if request == "scale":
            try:
                scaling = self.retargeter.calibrate_scaling_from_landmarks()
                # Mirror it into the shared store so the slider tracks it.
                self.state.store.apply({"dexpilot.scaling_factor": scaling})
                self.state.note(f"calibrated hand scale to {scaling:.3f}")
            except (RuntimeError, KeyError) as exc:
                self.state.note(f"scale calibration failed: {exc}")
            return
        try:
            captured = self.retargeter.calibrate_neutral_from_last_frame()
            self.state.note(f"captured neutral on {len(captured)} joints")
        except RuntimeError as exc:
            self.state.note(f"calibration failed: {exc}")

    # --- hardware ------------------------------------------------------
    def _service_arming(self) -> None:
        """Apply the browser's arm switch to the backend, on transition only.

        The HTTP thread deliberately does not touch the backend: arming enables
        torque over the same serial bus the hardware command thread already
        owns, and DynamixelClient has no documented thread safety. So the
        browser only sets a flag, and it is acted on here, on the loop thread.
        """

        if not self.state.loop.hardware_available:
            return
        wanted = self.state.loop.armed
        if wanted == self._armed_applied:
            return
        try:
            self.backend.arm() if wanted else self.backend.disarm()
        except Exception as exc:
            # Fail closed: report disarmed rather than leave the UI claiming a
            # torque state the hand is not actually in.
            logger.exception("Arming the hardware failed")
            self.state.loop.armed = False
            self._armed_applied = False
            self.state.note(f"arming failed: {exc}")
            return
        self._armed_applied = wanted
        self.state.note("hardware ARMED" if wanted else "hardware disarmed")

    def _service_client_watchdog(self) -> None:
        """Disarm if no browser is watching.

        The Arm button is in a web page, and closing the tab, sleeping the
        laptop or losing wifi does not reach the loop -- so an armed hand would
        keep tracking with nobody able to stop it short of a terminal. The UI
        polls continuously while open, so silence means nobody is looking.
        """

        timeout = self.config.client_timeout_s
        if timeout <= 0 or not self.state.loop.hardware_available:
            return
        if not (self.state.loop.armed or self._armed_applied):
            return
        age = self.state.seconds_since_client_poll()
        if age <= timeout:
            return
        self.state.loop.armed = False
        self._armed_applied = False
        try:
            self.backend.disarm()
        except Exception:
            logger.exception("Watchdog disarm failed")
        logger.error("No browser has polled for %.1fs — disarmed.", age)
        self.state.note(f"watchdog: no browser for {age:.1f}s, disarmed")

    def _service_deadman(self) -> None:
        """Disarm if the glove has gone quiet, and require a manual re-arm.

        Re-arming automatically on the next frame would let a flapping link
        reconnect straight into whatever pose the operator's hand had drifted
        into while they were not watching.
        """

        timeout = self.config.stale_timeout_s
        if not self.state.loop.hardware_available or timeout <= 0:
            return
        stale = self.state.glove.age_s
        if stale is None or stale <= timeout:
            if self._deadman_tripped and self.state.glove.connected:
                self._deadman_tripped = False
                self.state.note("glove data resumed — arm again to resume tracking")
            return
        if self._deadman_tripped:
            return
        self._deadman_tripped = True
        if self.state.loop.armed or self._armed_applied:
            self.state.loop.armed = False
            self._armed_applied = False
            try:
                self.backend.disarm()
            except Exception:
                logger.exception("Deadman disarm failed")
            logger.error("No glove data for %.1fs — disarmed.", stale)
            self.state.note(f"deadman: no glove data for {stale:.1f}s, disarmed")

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

            # Outside the landmark gate on purpose. A disarm request, and the
            # deadman, must be honoured even when no glove frame has EVER
            # arrived -- otherwise the one situation where you most want to
            # drop torque is the one where the loop never looks.
            self._service_arming()
            self._service_deadman()
            self._service_client_watchdog()

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
        self.state.neutral_offsets = dict(self.retargeter.neutral_joint_offsets)
        self.state.publish(
            commanded=result.active_joint_positions,
            measured=measured,
            intermediates={k: v for k, v in debug.items() if not k.startswith("_")},
            palm=debug.get("_palm", {}),
        )

    def close(self) -> None:
        # Torque off before anything else can fail: close() runs on the way out
        # of a Ctrl-C, and a hand left energised is the one outcome that must
        # not depend on the rest of teardown succeeding.
        disarm = getattr(self.backend, "disarm", None)
        if callable(disarm):
            try:
                disarm()
            except Exception:
                logger.exception("Disarm during shutdown failed")
        self.subscriber.close()
        self.backend.close()
