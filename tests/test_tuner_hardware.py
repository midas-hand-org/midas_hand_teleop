"""The tuner's hardware gate: arming, the deadman, and torque-off on teardown.

Runs with no glove, no sim and no hardware. The backend is a double that only
records what was asked of it, which is enough because the property under test is
*when* the loop arms, disarms, and forwards — not what a servo does with it.
"""

from __future__ import annotations

import numpy as np
import pytest
from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.store import ProfileStore

from midas_hand_teleop.tuner.loop import LoopConfig, TunerLoop
from midas_hand_teleop.tuner.state import TunerState


class FakeHardware:
    """Quacks like HardwareBackend: it can be armed, so the loop gates on it."""

    def __init__(self, arm_error: Exception | None = None):
        self.sent = 0
        self.arm_calls = 0
        self.disarm_calls = 0
        self.closed = False
        self._arm_error = arm_error

    def arm(self):
        self.arm_calls += 1
        if self._arm_error is not None:
            raise self._arm_error

    def disarm(self):
        self.disarm_calls += 1

    def send(self, result):
        self.sent += 1

    def measured(self):
        return {}

    def close(self):
        self.closed = True


class FakeSim:
    """A backend with no arm(), i.e. the simulator. Must never be gated."""

    def __init__(self):
        self.sent = 0

    def send(self, result):
        self.sent += 1

    def measured(self):
        return {}

    def close(self):
        pass


@pytest.fixture()
def loop_factory():
    made = []

    def make(backend, **config):
        state = TunerState(store=ProfileStore(RetargetProfile()))
        loop = TunerLoop(
            state, retargeter=None, backend=backend,
            config=LoopConfig(start_proxy=False, **config),
        )
        made.append(loop)
        return state, loop

    yield make
    for loop in made:
        loop.subscriber.close()


def test_only_an_armable_backend_is_gated(loop_factory):
    state, _ = loop_factory(FakeHardware())
    assert state.loop.hardware_available is True
    sim_state, _ = loop_factory(FakeSim())
    assert sim_state.loop.hardware_available is False


def test_the_sim_is_never_gated(loop_factory):
    """Requiring an arm press to move the simulator would be pure friction."""

    _, loop = loop_factory(FakeSim())
    loop._send(object())
    assert loop.backend.sent == 1


def test_hardware_does_not_move_until_armed(loop_factory):
    state, loop = loop_factory(FakeHardware())
    loop._send(object())
    assert loop.backend.sent == 0, "a launched-but-unarmed hand must not move"

    state.loop.armed = True
    loop._service_arming()
    loop._send(object())
    assert loop.backend.arm_calls == 1
    assert loop.backend.sent == 1


def test_arming_reaches_the_backend_only_on_transition(loop_factory):
    """The browser sets a flag every poll; torque must be toggled once."""

    state, loop = loop_factory(FakeHardware())
    state.loop.armed = True
    for _ in range(5):
        loop._service_arming()
    assert loop.backend.arm_calls == 1

    state.loop.armed = False
    for _ in range(5):
        loop._service_arming()
    assert loop.backend.disarm_calls == 1


def test_a_failed_arm_reports_disarmed(loop_factory):
    """Fail closed: the UI must not claim a torque state the hand is not in."""

    state, loop = loop_factory(FakeHardware(arm_error=RuntimeError("serial down")))
    state.loop.armed = True
    loop._service_arming()
    assert state.loop.armed is False
    loop._send(object())
    assert loop.backend.sent == 0


def test_deadman_disarms_when_the_glove_goes_quiet(loop_factory):
    state, loop = loop_factory(FakeHardware(), stale_timeout_s=0.5)
    state.loop.armed = True
    loop._service_arming()

    state.glove.age_s = 2.0
    loop._service_deadman()
    assert state.loop.armed is False
    assert loop.backend.disarm_calls == 1

    # And it must not keep re-disarming every tick.
    loop._service_deadman()
    assert loop.backend.disarm_calls == 1


def test_deadman_requires_a_manual_rearm(loop_factory):
    """A flapping link must not reconnect straight back into tracking."""

    state, loop = loop_factory(FakeHardware(), stale_timeout_s=0.5)
    state.loop.armed = True
    loop._service_arming()
    state.glove.age_s = 2.0
    loop._service_deadman()

    state.glove.age_s = 0.0
    state.glove.connected = True
    loop._service_deadman()
    loop._service_arming()
    assert state.loop.armed is False, "must stay disarmed until a human re-arms"
    loop._send(object())
    assert loop.backend.sent == 0


def test_deadman_can_be_disabled(loop_factory):
    state, loop = loop_factory(FakeHardware(), stale_timeout_s=0.0)
    state.loop.armed = True
    loop._service_arming()
    state.glove.age_s = 99.0
    loop._service_deadman()
    assert state.loop.armed is True


def test_close_drops_torque_first(loop_factory):
    """close() runs on the way out of a Ctrl-C, so torque-off must not depend
    on the rest of teardown succeeding."""

    state, loop = loop_factory(FakeHardware())
    state.loop.armed = True
    loop._service_arming()
    loop.close()
    assert loop.backend.disarm_calls == 1
    assert loop.backend.closed is True


def test_a_non_finite_target_is_never_commanded():
    """np.clip propagates NaN, so clip_positions cannot catch this."""

    from midas_hand_teleop.backends import HardwareBackend

    backend = HardwareBackend.__new__(HardwareBackend)
    backend.command_scale = 1.0
    backend.hand = type("H", (), {"clip_positions": staticmethod(lambda x: x)})()

    result = type("R", (), {"hardware_motor_positions": np.full(13, np.nan)})()
    with pytest.raises(ValueError, match="non-finite"):
        backend._prepare_target(result)
