"""Unit tests for the Manus bridge self-healing helpers.

Exercises _HandData timestamps and the pure glove-map invalidation function
without loading the Manus SDK or starting threads.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from midas_hand_teleop.manus_glove.manus_bridge import (
    _SIDE_INVALID,
    _SIDE_LEFT,
    _SIDE_RIGHT,
    MAP_INVALIDATE_SEC,
    _HandData,
    _invalidate_stale_glove_map,
    _read_side_from_sdk,
    _spawn_publish_thread_if_needed,
)


def _make_positions(n: int = 25) -> np.ndarray:
    return np.arange(n * 3, dtype=np.float64).reshape(n, 3)


class TestHandData:
    def test_update_sets_both_wall_and_monotonic_timestamps(self) -> None:
        hd = _HandData("left")
        before_wall = time.time()
        before_mono = time.monotonic()
        hd.update(_make_positions())
        after_wall = time.time()
        after_mono = time.monotonic()

        positions, wall_ts, mono_ts = hd.get()
        assert positions is not None
        assert before_wall <= wall_ts <= after_wall
        assert before_mono <= mono_ts <= after_mono
        assert hd.frame_count == 1

    def test_get_returns_zeros_before_first_update(self) -> None:
        hd = _HandData("right")
        positions, wall_ts, mono_ts = hd.get()
        assert positions is None
        assert wall_ts == 0.0
        assert mono_ts == 0.0

    def test_mark_invalidated_resets_monotonic_only(self) -> None:
        hd = _HandData("left")
        hd.update(_make_positions())
        _, wall_before, _ = hd.get()
        hd.mark_invalidated()
        positions, wall_after, mono_after = hd.get()

        # Wall-clock timestamp and cached positions are preserved so in-flight
        # consumers don't observe a torn state; only the stall clock resets.
        assert positions is not None
        assert wall_after == wall_before
        assert mono_after == 0.0

    def test_get_returns_a_copy(self) -> None:
        hd = _HandData("left")
        original = _make_positions()
        hd.update(original)
        returned, _, _ = hd.get()
        assert returned is not None
        returned[0, 0] = 999.0
        # Mutating the returned array must not corrupt internal state.
        next_returned, _, _ = hd.get()
        assert next_returned is not None
        assert next_returned[0, 0] != 999.0


class TestInvalidateStaleGloveMap:
    def _setup(
        self, left_last_mono: float, right_last_mono: float
    ) -> tuple[dict[str, _HandData], dict[int, str], threading.Lock]:
        left = _HandData("left")
        right = _HandData("right")
        # Inject monotonic timestamps directly to simulate stall states
        # without having to sleep in tests.
        left.last_monotonic = left_last_mono
        right.last_monotonic = right_last_mono
        glove_map = {0xAAAA: "left", 0xBBBB: "right"}
        return (
            {"left": left, "right": right},
            glove_map,
            threading.Lock(),
        )

    def test_fresh_entries_not_invalidated(self) -> None:
        now = 100.0
        hand_data_by_side, glove_map, lock = self._setup(
            left_last_mono=now - 0.5, right_last_mono=now - 0.5
        )

        invalidated = _invalidate_stale_glove_map(
            now_monotonic=now,
            hand_data_by_side=hand_data_by_side,
            glove_id_to_side=glove_map,
            glove_map_lock=lock,
            stale_threshold_sec=MAP_INVALIDATE_SEC,
        )

        assert invalidated == []
        assert glove_map == {0xAAAA: "left", 0xBBBB: "right"}

    def test_stale_side_is_invalidated_and_resets_clock(self) -> None:
        now = 100.0
        hand_data_by_side, glove_map, lock = self._setup(
            left_last_mono=now - (MAP_INVALIDATE_SEC + 5.0),
            right_last_mono=now - 0.5,
        )

        invalidated = _invalidate_stale_glove_map(
            now_monotonic=now,
            hand_data_by_side=hand_data_by_side,
            glove_id_to_side=glove_map,
            glove_map_lock=lock,
            stale_threshold_sec=MAP_INVALIDATE_SEC,
        )

        assert invalidated == [(0xAAAA, "left")]
        assert glove_map == {0xBBBB: "right"}
        # Stall clock reset so the next pass won't re-warn before a new frame arrives.
        assert hand_data_by_side["left"].last_monotonic == 0.0
        # Healthy side untouched.
        assert hand_data_by_side["right"].last_monotonic == now - 0.5

    def test_uninitialized_side_is_skipped(self) -> None:
        now = 100.0
        hand_data_by_side, glove_map, lock = self._setup(left_last_mono=0.0, right_last_mono=0.0)

        invalidated = _invalidate_stale_glove_map(
            now_monotonic=now,
            hand_data_by_side=hand_data_by_side,
            glove_id_to_side=glove_map,
            glove_map_lock=lock,
            stale_threshold_sec=MAP_INVALIDATE_SEC,
        )

        # Sides that never produced a frame (or were just invalidated) must
        # not trigger repeated invalidation of their map entries.
        assert invalidated == []
        assert glove_map == {0xAAAA: "left", 0xBBBB: "right"}

    def test_invalidates_all_gloves_mapped_to_stalled_side(self) -> None:
        # Defensive: if a stray glove ever got mapped to the same side (it
        # normally can't), invalidation should clear every matching entry.
        now = 100.0
        hand_data_by_side, _, lock = self._setup(
            left_last_mono=now - (MAP_INVALIDATE_SEC + 1.0),
            right_last_mono=now - 0.5,
        )
        glove_map = {0xAAAA: "left", 0xCCCC: "left", 0xBBBB: "right"}

        invalidated = _invalidate_stale_glove_map(
            now_monotonic=now,
            hand_data_by_side=hand_data_by_side,
            glove_id_to_side=glove_map,
            glove_map_lock=lock,
            stale_threshold_sec=MAP_INVALIDATE_SEC,
        )

        assert sorted(invalidated) == sorted([(0xAAAA, "left"), (0xCCCC, "left")])
        assert glove_map == {0xBBBB: "right"}


class _FakeThread:
    """Minimal stand-in for threading.Thread that records start() calls
    without actually spawning anything (we don't want test threads
    leaking inside the test process).

    Defaults to alive-after-start; tests that need to simulate a thread
    exiting unexpectedly (to exercise the respawn path) set
    ``._alive = False`` directly.
    """

    def __init__(self) -> None:
        self.started = False
        self._alive = True

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self._alive


class TestSpawnPublishThreadIfNeeded:
    """Late-arriving glove spawn helper.

    Models the cold-start scenario StationOps caught: bridge launches
    before gloves are powered on, so frame_count starts at 0; once a
    glove wakes up and the SDK callback delivers a frame, the next
    helper call must spawn a publish thread for that side and only
    that side.
    """

    def _hd(self, frame_count: int) -> _HandData:
        hd = _HandData("left")
        # Simulate the SDK callback having (or not having) delivered
        # frames without calling .update() repeatedly.
        hd.frame_count = frame_count
        return hd

    def test_no_frames_yet_does_not_spawn(self) -> None:
        spawned: dict[str, threading.Thread] = {}
        hd = self._hd(frame_count=0)
        factory_calls: list[_HandData] = []

        def factory(h: _HandData) -> _FakeThread:
            factory_calls.append(h)
            return _FakeThread()

        result = _spawn_publish_thread_if_needed("left", hd, spawned, factory)

        assert result is False
        assert spawned == {}
        # Crucially: did NOT call factory — no thread object created at all.
        assert factory_calls == []

    def test_first_frame_spawns_and_starts_thread(self) -> None:
        spawned: dict[str, threading.Thread] = {}
        hd = self._hd(frame_count=1)
        fake = _FakeThread()

        result = _spawn_publish_thread_if_needed("left", hd, spawned, thread_factory=lambda _: fake)

        assert result is True
        assert spawned == {"left": fake}
        assert fake.started is True

    def test_idempotent_when_already_spawned(self) -> None:
        existing = _FakeThread()
        existing.started = True
        spawned: dict[str, threading.Thread] = {"left": existing}
        hd = self._hd(frame_count=42)
        factory_calls = 0

        def factory(_: _HandData) -> _FakeThread:
            nonlocal factory_calls
            factory_calls += 1
            return _FakeThread()

        result = _spawn_publish_thread_if_needed("left", hd, spawned, factory)

        assert result is False
        # The previously-spawned thread reference is preserved untouched.
        assert spawned == {"left": existing}
        # Factory must NOT be invoked — that's how we avoid leaking threads
        # on every supervisor tick after spawn.
        assert factory_calls == 0

    def test_late_arrival_spawns_only_that_side(self) -> None:
        """Cold-start: left glove powers on first, right is still off."""
        spawned: dict[str, threading.Thread] = {}
        left_hd = self._hd(frame_count=1)
        right_hd = self._hd(frame_count=0)
        left_thread = _FakeThread()
        right_thread = _FakeThread()

        _spawn_publish_thread_if_needed(
            "left", left_hd, spawned, thread_factory=lambda _: left_thread
        )
        _spawn_publish_thread_if_needed(
            "right", right_hd, spawned, thread_factory=lambda _: right_thread
        )

        assert spawned == {"left": left_thread}
        assert left_thread.started is True
        assert right_thread.started is False  # never even constructed via factory


class _FakeSdkLib:
    """Minimal mock for the Manus SDK used by _read_side_from_sdk.

    Records calls and returns whatever side distribution the test
    configured. Mirrors the real ctypes surface just enough that
    ctypes.byref on the mocked return pointer works.
    """

    def __init__(self, node_sides: list[int] | None, count_rc: int = 0, info_rc: int = 0):
        self._node_sides = node_sides or []
        self._count_rc = count_rc
        self._info_rc = info_rc
        self.calls: list[tuple[str, int]] = []

    def CoreSdk_GetRawSkeletonNodeCount(self, glove_id, p_count):
        self.calls.append(("count", glove_id))
        # p_count is a ctypes byref() wrapper; ._obj is the underlying
        # c_uint32 and .value writes through — mirrors what the real SDK
        # does C-side when it fills the out-param.
        p_count._obj.value = len(self._node_sides)
        return self._count_rc

    def CoreSdk_GetRawSkeletonNodeInfoArray(self, glove_id, arr, count):
        self.calls.append(("info", glove_id))
        for i in range(count):
            arr[i].side = self._node_sides[i]
        return self._info_rc


class TestReadSideFromSdk:
    def test_all_right_returns_right(self) -> None:
        """Unanimous SIDE_RIGHT across 25 nodes → 'right'. Matches arg021 empirical."""
        lib = _FakeSdkLib([_SIDE_RIGHT] * 25)
        assert _read_side_from_sdk(lib, glove_id=0xAC37B5EC) == "right"

    def test_all_left_returns_left(self) -> None:
        lib = _FakeSdkLib([_SIDE_LEFT] * 25)
        assert _read_side_from_sdk(lib, glove_id=0x8FFC0BA4) == "left"

    def test_all_invalid_returns_none_for_fallback(self) -> None:
        """All SIDE_INVALID: return None so caller falls back to geometric.

        Defensive path — the arg021 smoke test observed zero SIDE_INVALID
        across 3367 frames, but future firmware / SDK revisions could
        populate differently.
        """
        lib = _FakeSdkLib([_SIDE_INVALID] * 25)
        assert _read_side_from_sdk(lib, glove_id=0x1234) is None

    def test_majority_wins_when_sides_mixed(self) -> None:
        """Bulk of nodes LEFT + a couple INVALID + one stray RIGHT → 'left'."""
        sides = [_SIDE_LEFT] * 20 + [_SIDE_INVALID] * 4 + [_SIDE_RIGHT]
        lib = _FakeSdkLib(sides)
        assert _read_side_from_sdk(lib, glove_id=0x5678) == "left"

    def test_tie_returns_none_for_fallback(self) -> None:
        """Exact tie on valid sides: return None so caller falls back to
        geometric. A balanced left/right vote is evidence the SDK is
        disagreeing with itself, not evidence of "right."
        """
        lib = _FakeSdkLib([_SIDE_LEFT, _SIDE_RIGHT])
        assert _read_side_from_sdk(lib, glove_id=0xABCD) is None

    def test_majority_right_with_invalid_and_stray_left(self) -> None:
        """Symmetric counterpart to ``test_majority_wins_when_sides_mixed``:
        mostly RIGHT, a few INVALID, one stray LEFT → 'right'. Guards
        against a future refactor that silently hard-codes one side.
        """
        sides = [_SIDE_RIGHT] * 20 + [_SIDE_INVALID] * 4 + [_SIDE_LEFT]
        lib = _FakeSdkLib(sides)
        assert _read_side_from_sdk(lib, glove_id=0xDEAD) == "right"

    def test_mostly_invalid_single_valid_node(self) -> None:
        """Degenerate case: 24 SIDE_INVALID + 1 SIDE_RIGHT → 'right'. One
        valid node is enough — majority is 1 vs 0 among non-invalid.
        """
        lib = _FakeSdkLib([_SIDE_INVALID] * 24 + [_SIDE_RIGHT])
        assert _read_side_from_sdk(lib, glove_id=0xBEEF) == "right"

    def test_oversized_node_count_returns_none(self) -> None:
        """A bogus/uninitialized node_count from the SDK (e.g. 500 nodes)
        must not trigger a large ctypes allocation; sanity-bound makes
        the function return None so we fall back to geometric.
        """
        lib = _FakeSdkLib([_SIDE_LEFT] * 500)
        assert _read_side_from_sdk(lib, glove_id=0xFEED) is None

    def test_node_count_rc_nonzero_returns_none(self) -> None:
        """SDK error on count → None (falls back to geometric)."""
        lib = _FakeSdkLib([_SIDE_LEFT] * 25, count_rc=-1)
        assert _read_side_from_sdk(lib, glove_id=0x1234) is None

    def test_node_count_zero_returns_none(self) -> None:
        """Empty node array → None."""
        lib = _FakeSdkLib([])
        assert _read_side_from_sdk(lib, glove_id=0x1234) is None

    def test_info_rc_nonzero_returns_none(self) -> None:
        """SDK error on info array → None."""
        lib = _FakeSdkLib([_SIDE_LEFT] * 25, info_rc=-1)
        assert _read_side_from_sdk(lib, glove_id=0x1234) is None

    def test_call_order_count_then_info(self) -> None:
        """Must call Count first (to size the array) then InfoArray."""
        lib = _FakeSdkLib([_SIDE_RIGHT] * 25)
        _read_side_from_sdk(lib, glove_id=0xDEAD)
        assert lib.calls == [("count", 0xDEAD), ("info", 0xDEAD)]


# ────────────────────────────────────────────────────────────────────────────
# Struct layout / SDK binding regression guards
# ────────────────────────────────────────────────────────────────────────────


class TestCoordinateSystemVUHStructLayout:
    """VUH struct layout MUST match ManusSDKTypes.h: view, up, handedness,
    unitScale. A regression flipping this back to (handedness, up, view, …)
    is the exact silent-misinterpretation bug this PR resubmits to solve.
    """

    def test_field_order_matches_manus_header(self) -> None:
        import ctypes

        from midas_hand_teleop.manus_glove.manus_bridge import _CoordinateSystemVUH

        assert [f[0] for f in _CoordinateSystemVUH._fields_] == [
            "view",
            "up",
            "handedness",
            "unitScale",
        ]
        # Natural-aligned 4×4-byte struct on x86_64.
        assert ctypes.sizeof(_CoordinateSystemVUH) == 16


class TestNodeInfoStructLayout:
    """NodeInfo layout guards the SDK-native handedness path.

    ``_read_side_from_sdk`` reads ``NodeInfo.side`` to route gloves. A future
    SDK update (or a typo) that shifts ``side``'s offset would let the
    function read whatever garbage happens to live at that offset — if it
    equals ``_SIDE_LEFT`` or ``_SIDE_RIGHT`` the function would confidently
    return a wrong answer, which then gets cached in ``glove_id_to_side``
    and silently mis-routes the glove for the life of the process.
    """

    def test_field_order_matches_manus_header(self) -> None:
        import ctypes

        from midas_hand_teleop.manus_glove.manus_bridge import _NodeInfo

        # Matches ManusSDKTypes.h:1664 — nodeId, parentId, chainType, side,
        # fingerJointType. `side` MUST stay at offset 12 (two uint32s + one
        # int before it); the bug-bot concern this test guards is a future
        # layout change shifting this offset silently.
        assert [f[0] for f in _NodeInfo._fields_] == [
            "nodeId",
            "parentId",
            "chainType",
            "side",
            "fingerJointType",
        ]
        # 5×4-byte fields, natural-aligned on x86_64.
        assert ctypes.sizeof(_NodeInfo) == 20
        # `side` specifically must land at byte offset 12 — the whole
        # SDK-native handedness path trusts this.
        assert _NodeInfo.side.offset == 12


# ────────────────────────────────────────────────────────────────────────────
# DDC host rebuild — closes previous MessageSender to prevent FD leak
# ────────────────────────────────────────────────────────────────────────────


class _RecordingSender:
    """MessageSender stand-in that records close() calls and the host it was
    constructed with — lets us assert exactly-one-close on the old sender
    without touching real ZMQ sockets.
    """

    def __init__(self, host: str = "") -> None:
        self.host = host
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class TestRebuildSenderImpl:
    def test_swaps_sender_ref_and_closes_old(self) -> None:
        from midas_hand_teleop.manus_glove.manus_bridge import _rebuild_sender_impl

        old = _RecordingSender(host="old-host")
        sender_ref = [old]
        sender_lock = threading.Lock()

        _rebuild_sender_impl(
            "new-host",
            sender_ref,
            sender_lock,
            sender_factory=_RecordingSender,
        )

        assert sender_ref[0] is not old
        assert sender_ref[0].host == "new-host"
        assert old.close_count == 1

    def test_close_failure_is_swallowed(self) -> None:
        """Close failures must not propagate — a stuck old socket should not
        break the publish path (the new sender is already swapped in).
        """
        from midas_hand_teleop.manus_glove.manus_bridge import _rebuild_sender_impl

        class _FailingSender(_RecordingSender):
            def close(self) -> None:
                super().close()
                raise OSError("simulated close failure")

        old = _FailingSender(host="old")
        sender_ref = [old]
        sender_lock = threading.Lock()

        _rebuild_sender_impl(
            "new-host",
            sender_ref,
            sender_lock,
            sender_factory=_RecordingSender,
        )

        assert sender_ref[0] is not old
        assert old.close_count == 1  # was attempted even though it raised


# ────────────────────────────────────────────────────────────────────────────
# Dead-thread respawn guard
# ────────────────────────────────────────────────────────────────────────────


class TestSpawnRespawnsDeadThread:
    """If a prior publish thread exited (e.g. unhandled exception), the
    supervisor must respawn it on the next tick — otherwise `spawned[side]`
    lingers with a dead Thread object and the side silently goes offline.
    """

    def _hd(self, frame_count: int) -> _HandData:
        hd = _HandData("left")
        hd.frame_count = frame_count
        return hd

    def test_alive_thread_is_not_respawned(self) -> None:
        hd = self._hd(frame_count=5)

        class _AliveThread(_FakeThread):
            def is_alive(self) -> bool:
                return True

        spawned: dict[str, threading.Thread] = {"left": _AliveThread()}
        factory_calls: list[_HandData] = []

        def factory(h):
            factory_calls.append(h)
            return _FakeThread()

        result = _spawn_publish_thread_if_needed("left", hd, spawned, factory)
        assert result is False
        assert factory_calls == []

    def test_dead_thread_is_respawned(self) -> None:
        hd = self._hd(frame_count=5)

        class _DeadThread(_FakeThread):
            def is_alive(self) -> bool:
                return False

        dead = _DeadThread()
        spawned: dict[str, threading.Thread] = {"left": dead}
        new_thread = _FakeThread()

        result = _spawn_publish_thread_if_needed("left", hd, spawned, lambda _: new_thread)
        assert result is True
        assert spawned["left"] is new_thread
        assert new_thread.started is True


def test_sdk_library_search_order(tmp_path, monkeypatch):
    """The SDK is proprietary and cannot be vendored, so anyone outside this
    machine needs a way to say where theirs is. The README described exactly
    this lookup for a long time before it existed."""

    from midas_hand_teleop.manus_glove.manus_bridge import (
        SDK_LIBRARY_NAME,
        resolve_sdk_library,
    )

    monkeypatch.delenv("MANUS_SDK_LIB", raising=False)
    monkeypatch.delenv("MANUS_SDK_DIR", raising=False)

    explicit = tmp_path / "explicit.so"
    explicit.write_bytes(b"")
    assert resolve_sdk_library(str(explicit)) == str(explicit)

    from_env = tmp_path / "from_env.so"
    from_env.write_bytes(b"")
    monkeypatch.setenv("MANUS_SDK_LIB", str(from_env))
    assert resolve_sdk_library() == str(from_env)
    # An explicit argument still wins over the environment.
    assert resolve_sdk_library(str(explicit)) == str(explicit)

    monkeypatch.delenv("MANUS_SDK_LIB")
    (tmp_path / "lib").mkdir()
    in_dir = tmp_path / "lib" / SDK_LIBRARY_NAME
    in_dir.write_bytes(b"")
    monkeypatch.setenv("MANUS_SDK_DIR", str(tmp_path))
    assert resolve_sdk_library() == str(in_dir)


def test_a_named_sdk_path_that_does_not_exist_is_an_error(monkeypatch):
    """Falling through to a system copy would silently load a different SDK
    than the one asked for."""

    from midas_hand_teleop.manus_glove.manus_bridge import resolve_sdk_library

    monkeypatch.delenv("MANUS_SDK_LIB", raising=False)
    with pytest.raises(FileNotFoundError, match="--sdk-lib"):
        resolve_sdk_library("/definitely/not/here.so")

    monkeypatch.setenv("MANUS_SDK_LIB", "/also/not/here.so")
    with pytest.raises(FileNotFoundError, match="MANUS_SDK_LIB"):
        resolve_sdk_library()
