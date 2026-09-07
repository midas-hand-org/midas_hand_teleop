"""HardwareBackend against a fake hand: the paths that move real motors.

Every test here corresponds to a way the hand could be driven somewhere nobody
asked for. None of them needs hardware -- the double records bus traffic.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from midas_hand_retargeter.constants import HARDWARE_MOTOR_JOINT_NAMES

needs_api = pytest.mark.skipif(
    importlib.util.find_spec("midas_hand_api") is None,
    reason="midas_hand_api is not installed",
)

N_MOTORS = len(HARDWARE_MOTOR_JOINT_NAMES)


class FakeHand:
    """Enough MidasHand to exercise the backend's own logic."""

    def __init__(self, *, read_ok=True, position=None):
        self.motor_ids = list(range(N_MOTORS))
        self.last_read_ok = read_ok
        self._position = (
            np.zeros(N_MOTORS) if position is None else np.asarray(position, dtype=float)
        )
        self.commanded = []
        self.torque = None
        self.configured = False
        self.reads = 0

    def configure(self, enable_torque=False):
        self.configured = True

    def read_pos(self):
        self.reads += 1
        return self._position.copy()

    def read_cur(self):
        return np.zeros(N_MOTORS)

    def clip_positions(self, values):
        return np.asarray(values, dtype=float)

    def set_positions(self, values, clip=False):
        self.commanded.append(np.asarray(values, dtype=float).copy())

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        self.torque = False

    def shutdown(self):
        self.torque = False


@pytest.fixture()
def make_backend(monkeypatch):
    """Build a HardwareBackend around a FakeHand."""

    import midas_hand_api

    from midas_hand_teleop.backends import HardwareBackend

    built = []

    def make(hand, **kwargs):
        monkeypatch.setattr(midas_hand_api, "MidasHand", lambda **kw: hand)
        kwargs.setdefault("update_rate_hz", 0.0)  # no thread unless asked
        kwargs.setdefault("configure", False)
        backend = HardwareBackend(**kwargs)
        built.append(backend)
        return backend

    yield make
    for backend in built:
        backend.close()


@needs_api
def test_a_stale_read_refuses_to_start(make_backend):
    """The dangerous failure is silent, not an exception. A dropped sync read
    makes DynamixelReader return its cache, which starts as zeros, and raw 0
    maps to about -3.1 rad on these motors -- roughly 177 degrees from where
    the finger is. Arming onto that drives the whole hand into its stops."""

    hand = FakeHand(read_ok=False, position=np.full(N_MOTORS, -3.09))
    with pytest.raises(RuntimeError, match="Refusing to command or arm"):
        make_backend(hand)
    assert hand.torque is None, "torque must never have been enabled"


@needs_api
def test_a_stale_read_is_retried_before_giving_up(make_backend):
    """One dropped packet on a busy bus should not abort a session."""

    from midas_hand_teleop.backends import HardwareBackend

    hand = FakeHand(read_ok=False, position=np.full(N_MOTORS, 0.1))

    def flaky():
        hand.reads += 1
        hand.last_read_ok = hand.reads >= 3
        return hand._position.copy()

    hand.read_pos = flaky
    backend = make_backend(hand)
    assert hand.reads == 3, "retried until the read was fresh, and no further"
    assert np.allclose(backend._last_command, 0.1)
    assert HardwareBackend._START_READ_ATTEMPTS >= 3


@needs_api
def test_nothing_is_energised_by_construction(make_backend):
    backend = make_backend(FakeHand())
    assert backend.is_armed is False
    assert backend.hand.torque is None


@needs_api
def test_arming_seeds_from_the_measured_pose(make_backend):
    """So arming cannot snap the fingers to a stale target."""

    hand = FakeHand(position=np.linspace(-0.3, -0.1, N_MOTORS))
    backend = make_backend(hand)
    backend.arm()
    assert hand.torque is True
    assert np.allclose(hand.commanded[-1], hand._position)


@needs_api
def test_arming_goes_through_the_command_thread(make_backend):
    """measured() documents that the command thread owns the bus and that
    DynamixelClient is not thread safe. arm() used to drive it anyway, from
    the caller's thread, at exactly the moment the two collide."""

    import threading

    hand = FakeHand()
    seen = []
    real = hand.enable_torque

    def record():
        seen.append(threading.current_thread().name)
        real()

    hand.enable_torque = record
    backend = make_backend(hand, update_rate_hz=50.0)
    caller = threading.current_thread().name
    backend.arm()
    assert seen and seen[0] != caller
    assert "command-loop" in seen[0]


@needs_api
def test_a_failed_arm_propagates_and_leaves_the_hand_disarmed(make_backend):
    hand = FakeHand()

    def boom():
        raise OSError("bus error")

    hand.enable_torque = boom
    backend = make_backend(hand, update_rate_hz=50.0)
    assert backend.is_armed is False
    with pytest.raises(OSError, match="bus error"):
        backend.arm()
    assert backend.is_armed is False


@needs_api
def test_disarm_stops_commanding_even_if_the_bus_call_fails(make_backend):
    hand = FakeHand()
    backend = make_backend(hand)
    backend.arm()
    hand.disable_torque = lambda: (_ for _ in ()).throw(OSError("gone"))
    backend.disarm()
    assert backend.is_armed is False


@needs_api
def test_limits_are_tightened_to_the_measured_hard_stops(make_backend):
    """Homing records where each stop is. Two URDF limits reach past it on
    this hand, and both are commanded by ordinary input."""

    backend = make_backend(FakeHand())
    names = list(HARDWARE_MOTOR_JOINT_NAMES)
    lower, upper = backend._limits[names.index("thumb_mcp_joint")]
    assert lower > -1.57, "thumb MCP must not be commandable past its stop"
    _, side_upper = backend._limits[names.index("thumb_cmc_side_joint")]
    assert side_upper < 0.90
    assert np.all(backend._limits[:, 0] <= backend._limits[:, 1])


@needs_api
def test_constructing_a_backend_never_energises_a_motor(make_backend):
    """The constructor used to default start_armed=True, so `HardwareBackend()`
    in a REPL enabled torque on the spot. Every real caller passes it."""

    import inspect

    from midas_hand_teleop.backends import HardwareBackend

    signature = inspect.signature(HardwareBackend.__init__)
    assert signature.parameters["start_armed"].default is False
    assert make_backend(FakeHand()).hand.torque is None


@needs_api
def test_start_armed_still_works(make_backend):
    hand = FakeHand()
    backend = make_backend(hand, start_armed=True, update_rate_hz=50.0)
    assert backend.is_armed is True
    assert hand.torque is True
