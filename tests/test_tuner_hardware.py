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
            state,
            retargeter=None,
            backend=backend,
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


def test_watchdog_disarms_when_the_browser_goes_away(loop_factory):
    """The Arm button lives in a web page. A closed tab, a slept laptop or lost
    wifi never reaches the loop, so an armed hand would keep tracking with
    nobody able to stop it short of a terminal."""

    state, loop = loop_factory(FakeHardware(), client_timeout_s=1.0)
    state.loop.armed = True
    loop._service_arming()

    state.snapshot()  # a browser polls
    loop._service_client_watchdog()
    assert state.loop.armed is True

    state._last_client_poll -= 5.0  # ...and then stops
    loop._service_client_watchdog()
    assert state.loop.armed is False
    assert loop.backend.disarm_calls == 1


def test_watchdog_leaves_a_headless_run_alone(loop_factory):
    """--start-armed with no browser is a deliberate choice, not a lost tab."""

    state, loop = loop_factory(FakeHardware(), client_timeout_s=1.0)
    state.loop.armed = True
    loop._service_arming()
    loop._service_client_watchdog()
    assert state.loop.armed is True, "no browser has EVER polled; nothing was lost"


def _bare_backend(command_scale=1.0):
    from midas_hand_teleop.backends import HardwareBackend

    backend = HardwareBackend.__new__(HardwareBackend)
    backend.command_scale = command_scale
    backend.hand = type("H", (), {"clip_positions": staticmethod(lambda x: x)})()
    return backend


def test_commands_are_clamped_to_the_robots_own_limits():
    """The hand's clip_positions reads limits from ~/.midas_hand/config.yaml,
    and homing writes those as +/-pi -- so on a correctly homed hand it clips
    nothing. These are the URDF's, which is what the mechanism allows."""

    from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES

    from midas_hand_teleop.backends import HardwareBackend

    backend = _bare_backend()
    wild = np.full(13, 5.0)
    result = type("R", (), {"hardware_motor_positions": wild})()
    target = backend._prepare_target(result)

    limits = HardwareBackend._MODEL_LIMITS
    assert np.all(target <= limits[:, 1] + 1e-9)
    assert len(target) == len(HARDWARE_MOTOR_JOINT_NAMES)


def test_command_scale_cannot_push_past_a_mechanical_stop():
    """command_scale is applied before the clamp on purpose: a scale > 1 can
    take a joint the retargeter had bounded correctly straight into a stop."""

    from midas_hand_teleop.backends import HardwareBackend

    at_limit = HardwareBackend._MODEL_LIMITS[:, 0].copy()
    result = type("R", (), {"hardware_motor_positions": at_limit.copy()})()
    target = _bare_backend(command_scale=3.0)._prepare_target(result)
    assert np.allclose(target, at_limit), "scaled past the lower stop"
    assert np.allclose(result.hardware_motor_positions, at_limit), (
        "_prepare_target must not scale the caller's array in place"
    )
