"""Manus Pro glove → data-center bridge (dual hand, Integrated SDK).

Self-contained in the MIDAS repo: MIDAS owns the glove input end-to-end. The
data-center bus shim (``data_center.py``) and the glove protobuf
(``glove_proto.py``) are vendored alongside; the wire format is the standard
DataCenter framing, so any compatible DataCenter interoperates.

Runs on a separate x86 Linux machine (ManusSDK is x86-64 only). Reads skeleton
data from both gloves and publishes MediaPipe-21 keypoints per side over ZMQ TCP.
Requires the ManusSDK shared library installed at one of the system paths below
(``/usr/local/lib`` or ``/opt/ManusSDK/lib``) — that binary is NOT part of any
repo; only the ctypes bindings live here.

Handedness is resolved once per unknown gloveId, then cached. The SDK-native
path (``_read_side_from_sdk``) is primary; geometric cross-product
(``_detect_handedness``) is the fallback.

Publishes per-side:
  - /data_collection/glove/{side}/keypoint/state  → GloveKeypointStateProto

Usage:
    python -m midas_hand_teleop.manus_glove.manus_bridge --host <consumer-ip>
    # or set MIDAS_DATA_CENTER_HOST; --host wins if both are given
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import signal
import sys
import threading
import time

import numpy as np
import zmq

from midas_hand_teleop.manus_glove.data_center import (
    DATA_OUTPUT_PORT,
    MessageSender,
    _dtype_from_descr,
    send_proto_as_sized_array,
)
from midas_hand_teleop.manus_glove.glove_proto import (
    KEYPOINT_FORMAT_MEDIAPIPE_21,
    GloveKeypointStateProto,
)

logger = logging.getLogger(__name__)

# Haptic output: a sim (or other consumer) publishes a 5-float "powers" array
# (Thumb,Index,Middle,Ring,Pinky; 0..1 amplitude) per side here, and the bridge
# drives the glove's vibration motors via CoreSdk_VibrateFingersForGlove.
HAPTIC_TOPIC = "/data_collection/glove/{side}/haptic/command"
HAPTIC_STALE_SEC = 0.3  # stop vibrating if no command for this long (publisher died)


# ─── Manus SDK ctypes structures ──────────────────────────────────────────


class _ManusVec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _ManusQuaternion(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class _ManusTransform(ctypes.Structure):
    _fields_ = [
        ("position", _ManusVec3),
        ("rotation", _ManusQuaternion),
        ("scale", _ManusVec3),
    ]


class _SkeletonNode(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("transform", _ManusTransform)]


class _SkeletonStreamInfo(ctypes.Structure):
    _fields_ = [("publishTime", ctypes.c_uint64), ("skeletonsCount", ctypes.c_uint32)]


class _RawSkeletonInfo(ctypes.Structure):
    _fields_ = [
        ("gloveId", ctypes.c_uint32),
        ("nodesCount", ctypes.c_uint32),
        ("publishTime", ctypes.c_uint64),
    ]


class _CoordinateSystemVUH(ctypes.Structure):
    # Field order matches ManusSDKTypes.h: view, up, handedness, unitScale.
    _fields_ = [
        ("view", ctypes.c_int),
        ("up", ctypes.c_int),
        ("handedness", ctypes.c_int),
        ("unitScale", ctypes.c_float),
    ]


class _ManusHost(ctypes.Structure):
    _fields_ = [
        ("hostName", ctypes.c_char * 256),
        ("ipAddress", ctypes.c_char * 40),
        ("manusCoreVersion", ctypes.c_char * 16),
    ]


class _NodeInfo(ctypes.Structure):
    # Layout matches ManusSDKTypes.h:1664 (NodeInfo struct). `side` is the
    # per-node Side enum (SIDE_INVALID/LEFT/RIGHT) populated by the glove
    # firmware at pair time — hardware-ground-truth handedness.
    _fields_ = [
        ("nodeId", ctypes.c_uint32),
        ("parentId", ctypes.c_uint32),
        ("chainType", ctypes.c_int),
        ("side", ctypes.c_int),
        ("fingerJointType", ctypes.c_int),
    ]


# Side enum values from ManusSDKTypes.h (Side_Invalid=0, Side_Left=1, Side_Right=2).
_SIDE_INVALID = 0
_SIDE_LEFT = 1
_SIDE_RIGHT = 2


_RAW_SKELETON_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(_SkeletonStreamInfo))

# ─── Manus 25-node → MediaPipe 21-point mapping ──────────────────────────

NODE_TO_MEDIAPIPE = {
    0: 0,  # Wrist
    1: 1,
    2: 2,
    3: 3,
    4: 4,  # Thumb (skip metacarpal)
    6: 5,
    7: 6,
    8: 7,
    9: 8,  # Index
    11: 9,
    12: 10,
    13: 11,
    14: 12,  # Middle
    16: 13,
    17: 14,
    18: 15,
    19: 16,  # Ring
    21: 17,
    22: 18,
    23: 19,
    24: 20,  # Pinky
}


# ─── Bridge tuning ────────────────────────────────────────────────────────

# Supervisor poll period during the initial glove-wait window.
SUPERVISOR_TICK_SEC = 0.1

# Self-healing thresholds.
# Warn + pause publishing this side once the SDK callback has been silent
# for this long. The other side keeps running unaffected.
STALL_WARN_SEC = 2.0
# If a side remains stalled this long, forget its learned gloveId so the
# geometric handedness detector can re-bind when the glove re-appears
# (e.g. after a USB replug the SDK assigns a new gloveId).
MAP_INVALIDATE_SEC = 10.0
# How often the watchdog checks for stalled sides and DataCenter IP change.
WATCHDOG_TICK_SEC = 0.5
# How often the watchdog re-reads MIDAS_DATA_CENTER_HOST. ZMQ PUB auto-reconnects
# TCP transparently, so this only matters for the IP-change case (station move).
ENV_RECHECK_SEC = 30.0
# Short window we explicitly call out as "waiting for gloves" before the bridge
# silently transitions to its steady-state loop. Just for clean log UX during
# the common "gloves already on at startup" path; not a timeout — the bridge
# never exits because gloves are absent.
INITIAL_GLOVE_WAIT_SEC = 10.0
# How often to log "still waiting for X glove" once we're past the initial wait
# and at least one side hasn't published yet.
GLOVE_WAIT_LOG_PERIOD_SEC = 30.0


# ─── Per-hand state buffer ────────────────────────────────────────────────


class _HandData:
    """Thread-safe buffer for one hand's skeleton data.

    Tracks both wall-clock ``timestamp`` (embedded in published protos) and
    ``last_monotonic`` (drift-free stall detection by the watchdog + publish
    loop). EMA smoothing is disabled — the wujihandpy hardware low-pass
    filter (4 Hz cutoff) handles smoothing on the hand side.
    """

    def __init__(self, side: str):
        self.side = side
        self.lock = threading.Lock()
        self.raw_positions: np.ndarray | None = None
        self.timestamp: float = 0.0
        self.last_monotonic: float = 0.0
        self.frame_count: int = 0

    def update(self, positions: np.ndarray) -> None:
        with self.lock:
            self.raw_positions = positions.copy()
            self.timestamp = time.time()
            self.last_monotonic = time.monotonic()
            self.frame_count += 1

    def get(self) -> tuple[np.ndarray | None, float, float]:
        with self.lock:
            if self.raw_positions is not None:
                return self.raw_positions.copy(), self.timestamp, self.last_monotonic
            return None, 0.0, 0.0

    def mark_invalidated(self) -> None:
        """Reset monotonic-age so a restored glove re-learns from scratch."""
        with self.lock:
            self.last_monotonic = 0.0


def _rebuild_sender_impl(
    new_host: str,
    sender_ref: list,
    sender_lock: threading.Lock,
    sender_factory=None,
) -> None:
    """Atomically swap ``sender_ref[0]`` to a new MessageSender for ``new_host``.

    Closes the old sender explicitly outside the lock. Relying on
    ``MessageSender.__del__`` is fragile — CPython GC is not prompt and
    ``zmq.Context.destroy()`` from a finalizer has thread-safety issues.
    Without the explicit close, a long-running bridge through intermittent
    MIDAS_DATA_CENTER_HOST churn leaks a zmq.Context + socket FD per rebuild.

    ``sender_factory`` is injectable for tests (default: real MessageSender).
    """
    factory = sender_factory if sender_factory is not None else MessageSender
    new_sender = factory(host=new_host)
    with sender_lock:
        old_sender = sender_ref[0]
        sender_ref[0] = new_sender
    # Close outside the lock so a publish waiting on sender_lock doesn't
    # stall on the closing socket.
    try:
        old_sender.close()
    except Exception as e:
        logger.warning(f"Failed to close old MessageSender: {e}")
        e.__traceback__ = None


def _spawn_publish_thread_if_needed(
    side: str,
    hand_data: "_HandData",
    spawned: dict[str, threading.Thread],
    thread_factory,
) -> bool:
    """Spawn a publish thread for ``side`` if its first SDK frame has arrived.

    Returns True if a new thread was started. Idempotent: no-ops if a thread is
    alive for this side or if no frame has been delivered yet. If a prior
    thread exited (e.g. exception outside the publish try/except), it is
    respawned so the side doesn't silently go dead.
    Caller owns logging. Exported for unit tests.
    """
    existing = spawned.get(side)
    if existing is not None and existing.is_alive():
        return False
    if hand_data.frame_count == 0:
        return False
    t = thread_factory(hand_data)
    t.start()
    spawned[side] = t
    return True


def _invalidate_stale_glove_map(
    now_monotonic: float,
    hand_data_by_side: dict[str, _HandData],
    glove_id_to_side: dict[int, str],
    glove_map_lock: threading.Lock,
    stale_threshold_sec: float,
) -> list[tuple[int, str]]:
    """Remove gloveId mappings for sides stalled longer than the threshold.

    Returns the list of ``(glove_id, side)`` tuples that were invalidated.
    Pure function — caller owns logging. Exported for unit tests.
    """
    invalidated: list[tuple[int, str]] = []
    for side, hand_data in hand_data_by_side.items():
        with hand_data.lock:
            last_mono = hand_data.last_monotonic
        if last_mono <= 0.0 or now_monotonic - last_mono <= stale_threshold_sec:
            continue
        with glove_map_lock:
            stale_ids = [gid for gid, s in glove_id_to_side.items() if s == side]
            for gid in stale_ids:
                del glove_id_to_side[gid]
                invalidated.append((gid, side))
        if stale_ids:
            hand_data.mark_invalidated()
    return invalidated


# Sanity bound on SDK-reported node count. Real Manus gloves have 25 nodes;
# anything above this is almost certainly a garbage value from an
# uninitialized out-param or a future SDK revision we don't understand.
# Clamping prevents a gigabyte-sized ctypes allocation on a bogus count.
_MAX_PLAUSIBLE_NODE_COUNT = 128


def _read_side_from_sdk(lib, glove_id: int) -> str | None:
    """Read hardware-declared handedness for ``glove_id`` from the Manus SDK.

    Reads each node's ``NodeInfo.side`` (``SIDE_LEFT=1``, ``SIDE_RIGHT=2``,
    ``SIDE_INVALID=0``) which the glove firmware populates at pair time.
    Returns "left" / "right" on an unambiguous majority, else None (caller
    falls back to ``_detect_handedness``). A tie returns None — a balanced
    vote is evidence the SDK is disagreeing with itself, not a clean signal.

    Must be called on the SDK callback thread; the underlying calls read
    per-glove state that the skeleton-stream thread mutates.
    """
    node_count = ctypes.c_uint32(0)
    rc = lib.CoreSdk_GetRawSkeletonNodeCount(glove_id, ctypes.byref(node_count))
    if rc != 0:
        logger.warning(
            "CoreSdk_GetRawSkeletonNodeCount(0x%08X) rc=%d; falling back to geometric",
            glove_id,
            rc,
        )
        return None
    if node_count.value == 0:
        # Normal transient state before the glove fully streams; don't log.
        return None
    if node_count.value > _MAX_PLAUSIBLE_NODE_COUNT:
        logger.warning(
            "Suspicious node_count=%d for glove 0x%08X (sanity max %d); "
            "skipping SDK path, falling back to geometric",
            node_count.value,
            glove_id,
            _MAX_PLAUSIBLE_NODE_COUNT,
        )
        return None
    arr = (_NodeInfo * node_count.value)()
    rc = lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, arr, node_count.value)
    if rc != 0:
        logger.warning(
            "CoreSdk_GetRawSkeletonNodeInfoArray(0x%08X) rc=%d; falling back to geometric",
            glove_id,
            rc,
        )
        return None

    left_count = 0
    right_count = 0
    for i in range(node_count.value):
        s = arr[i].side
        if s == _SIDE_LEFT:
            left_count += 1
        elif s == _SIDE_RIGHT:
            right_count += 1
    # Majority wins; ties return None (see docstring — a tie is evidence
    # of SDK disagreement, not a clean signal).
    if left_count == right_count:
        return None
    return "left" if left_count > right_count else "right"


# Per-glove rate-limiter for the "both detection paths returned None" warning.
# Keyed by glove_id → last-log monotonic timestamp. First occurrence always logs;
# subsequent occurrences for the same glove are throttled to avoid drowning
# station logs at 120 Hz when a glove sits in an ambiguous pose.
_UNRESOLVED_LOG_PERIOD_SEC = 5.0
_unresolved_last_log: dict[int, float] = {}


def _log_unresolved_once(glove_id: int) -> None:
    """Warn (rate-limited) that both handedness paths returned None.

    Without this, an unrouted glove is completely silent — the watchdog
    can't help because no glove_id_to_side entry ever got written. The
    rate-limit keeps a persistently-ambiguous glove from spamming 120
    log lines per second.
    """
    now = time.monotonic()
    last = _unresolved_last_log.get(glove_id, 0.0)
    if now - last < _UNRESOLVED_LOG_PERIOD_SEC:
        return
    _unresolved_last_log[glove_id] = now
    logger.warning(
        "Glove 0x%08X: both SDK and geometric handedness detection returned None; "
        "glove is streaming but cannot be routed. Pose likely ambiguous or "
        "SDK NodeInfo.side all-INVALID for this firmware.",
        glove_id,
    )


def _detect_handedness(positions: np.ndarray) -> str | None:
    """Detect left vs right from skeleton geometry.

    Uses the cross product of (wrist→middle_mcp) × (wrist→index_mcp);
    the Y component sign differs for left vs right hands. Kept as
    fallback for the SDK-native ``_read_side_from_sdk`` path — if the
    SDK returns all ``SIDE_INVALID`` (or errors), geometric inference
    is still a reasonable answer for a glove in a natural pose.
    """
    if positions.shape[0] < 12:
        return None
    wrist = positions[0]
    index_mcp = positions[6]
    middle_mcp = positions[11]
    cross = np.cross(middle_mcp - wrist, index_mcp - wrist)
    if abs(cross[1]) < 1e-6:
        return None
    return "left" if cross[1] > 0 else "right"


def _positions_to_mediapipe(positions: np.ndarray) -> np.ndarray | None:
    """Convert 25-node raw positions to MediaPipe 21-point keypoints."""
    keypoints = np.zeros((21, 3), dtype=np.float64)
    filled = 0
    for node_idx, mp_idx in NODE_TO_MEDIAPIPE.items():
        if node_idx < positions.shape[0]:
            keypoints[mp_idx] = positions[node_idx]
            filled += 1
    if filled < 15:
        return None
    # Negate Y axis to match the wuji-retargeting IK solver's expected
    # coordinate convention. The Manus VUH coordinate system used here
    # outputs Y in the opposite direction from what the retargeter wants.
    keypoints[:, 1] *= -1
    return keypoints


# ─── Bridge main ─────────────────────────────────────────────────────────


def run_bridge(
    swap_sides: bool = False,
    rate_hz: float = 120.0,
    haptics: bool = True,
    host: str | None = None,
) -> None:
    """Run the dual-hand Manus bridge.

    ``host`` is the data-center host to publish to; if None it falls back to
    ``$MIDAS_DATA_CENTER_HOST`` then ``localhost``.
    """
    # ── Resilient publisher ──
    # ZMQ PUB/connect auto-reconnects TCP transparently when the downstream
    # subscriber dies and comes back, so bog-standard DataCenter restarts
    # recover on their own. The one case ZMQ cannot handle is the endpoint
    # *IP* changing (station redeploy on a new host) — the socket keeps
    # retrying the old address forever. The watchdog thread periodically
    # re-reads MIDAS_DATA_CENTER_HOST and rebuilds the socket via
    # _rebuild_sender() when it changes.
    sender_lock = threading.Lock()
    sender_host = host or os.environ.get("MIDAS_DATA_CENTER_HOST", "localhost")
    sender_ref = [MessageSender(host=sender_host)]

    def _publish_proto(topic: str, msg) -> None:
        with sender_lock:
            send_proto_as_sized_array(sender_ref[0], topic, msg)

    def _rebuild_sender(new_host: str) -> None:
        _rebuild_sender_impl(new_host, sender_ref, sender_lock)

    KEYPOINT_TOPICS = {
        "left": "/data_collection/glove/left/keypoint/state",
        "right": "/data_collection/glove/right/keypoint/state",
    }

    left_data = _HandData("left")
    right_data = _HandData("right")
    # Mutated by the C skeleton callback (one writer) and read by the
    # supervisor loop (concurrent reader). Although individual key reads
    # under the GIL are safe, a concurrent dict-resize during iteration
    # (`for k, v in glove_id_to_side.items()`) can raise RuntimeError on
    # some CPython versions. The lock is cheap (held only around dict
    # mutation and iteration) and removes the failure mode entirely.
    glove_id_to_side: dict[int, str] = {}
    glove_map_lock = threading.Lock()
    running = True

    # ── Signal handlers FIRST, before any SDK init ──
    # The SDK init + glove enumeration block can take 30-60s in the worst
    # case (slow gloves waking up). If the operator hits Ctrl-C during
    # that wait, the handlers must already be registered or SIGINT either
    # kills the process abruptly (leaving the SDK / motors in an unknown
    # state) or — worse — interrupts an SDK call mid-flight.
    def _shutdown(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Load SDK ──
    lib_paths = [
        "/usr/local/lib/libManusSDK_Integrated.so",
        "/opt/ManusSDK/lib/libManusSDK_Integrated.so",
    ]
    lib_path = next((p for p in lib_paths if os.path.exists(p)), None)
    if not lib_path:
        logger.error("ManusSDK not found at %s", lib_paths)
        return

    logger.info("Loading ManusSDK from %s", lib_path)
    lib = ctypes.CDLL(lib_path)

    # Set function signatures
    lib.CoreSdk_InitializeIntegrated.restype = ctypes.c_int
    lib.CoreSdk_ShutDown.restype = ctypes.c_int
    lib.CoreSdk_InitializeCoordinateSystemWithVUH.restype = ctypes.c_int
    lib.CoreSdk_InitializeCoordinateSystemWithVUH.argtypes = [_CoordinateSystemVUH, ctypes.c_bool]
    lib.CoreSdk_ConnectToHost.restype = ctypes.c_int
    lib.CoreSdk_ConnectToHost.argtypes = [_ManusHost]
    lib.CoreSdk_RegisterCallbackForRawSkeletonStream.restype = ctypes.c_int
    lib.CoreSdk_RegisterCallbackForRawSkeletonStream.argtypes = [_RAW_SKELETON_CALLBACK_TYPE]
    lib.CoreSdk_GetRawSkeletonInfo.restype = ctypes.c_int
    lib.CoreSdk_GetRawSkeletonInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(_RawSkeletonInfo)]
    lib.CoreSdk_GetRawSkeletonData.restype = ctypes.c_int
    lib.CoreSdk_GetRawSkeletonData.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_SkeletonNode),
        ctypes.c_uint32,
    ]
    # SDK-native handedness via per-node info (different path than the
    # broken dongle-pairing-table API — see _read_side_from_sdk docstring).
    lib.CoreSdk_GetRawSkeletonNodeCount.restype = ctypes.c_int
    lib.CoreSdk_GetRawSkeletonNodeCount.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.CoreSdk_GetRawSkeletonNodeInfoArray.restype = ctypes.c_int
    lib.CoreSdk_GetRawSkeletonNodeInfoArray.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_NodeInfo),
        ctypes.c_uint32,
    ]
    # Haptics: per-finger vibration STRENGTH (5 floats: Thumb,Index,Middle,Ring,Pinky).
    lib.CoreSdk_VibrateFingersForGlove.restype = ctypes.c_int
    lib.CoreSdk_VibrateFingersForGlove.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
    # Initialize
    rc = lib.CoreSdk_InitializeIntegrated()
    if rc != 0:
        logger.error("CoreSdk_InitializeIntegrated failed: %d", rc)
        sys.exit(1)
    time.sleep(1.0)

    # VUH wire values per ManusSDKTypes.h (view=0, up=5, handedness=2).
    vuh = _CoordinateSystemVUH(view=0, up=5, handedness=2, unitScale=1.0)
    rc = lib.CoreSdk_InitializeCoordinateSystemWithVUH(vuh, True)
    if rc != 0:
        # Silent failure here leaves gloves in an undefined coordinate frame,
        # which breaks both the geometric-handedness cross-product sign and
        # the Y-axis negation in _positions_to_mediapipe downstream.
        logger.error("CoreSdk_InitializeCoordinateSystemWithVUH failed: %d", rc)
        sys.exit(1)

    host = _ManusHost()
    ctypes.memset(ctypes.addressof(host), 0, ctypes.sizeof(host))
    rc = lib.CoreSdk_ConnectToHost(host)
    if rc != 0:
        # A failed host-connect leaves the skeleton callback registered
        # against a host that'll never deliver frames — bridge runs with
        # 0 Hz forever, operator can't tell why. Fatal is the right call.
        logger.error("CoreSdk_ConnectToHost failed: %d", rc)
        sys.exit(1)
    time.sleep(2.0)

    # ── Skeleton callback (multiplexed, routes by gloveId) ──
    def on_skeleton(p_info):
        if not p_info:
            return
        try:
            info = p_info.contents
            if info.skeletonsCount == 0:
                return

            raw_info = _RawSkeletonInfo()
            rc = lib.CoreSdk_GetRawSkeletonInfo(0, ctypes.byref(raw_info))
            if rc != 0 or raw_info.nodesCount == 0:
                return

            glove_id = raw_info.gloveId
            nodes = (_SkeletonNode * raw_info.nodesCount)()
            rc = lib.CoreSdk_GetRawSkeletonData(0, nodes, raw_info.nodesCount)
            if rc != 0:
                return

            positions = np.zeros((raw_info.nodesCount, 3), dtype=np.float64)
            for i in range(raw_info.nodesCount):
                p = nodes[i].transform.position
                positions[i] = [p.x, p.y, p.z]

            # Detect handedness on first encounter of this gloveId.
            # The dict is guarded by glove_map_lock — see comment at
            # declaration site. Lock scope kept tight: only around the
            # check / mutation, not around the (slower) detection math.
            with glove_map_lock:
                already_known = glove_id in glove_id_to_side

            if not already_known:
                # SDK-native first: reads hardware-ground-truth handedness
                # from each glove's firmware (see _read_side_from_sdk). Falls
                # through to geometric cross-product inference if the SDK
                # returns all SIDE_INVALID, which empirically never happens
                # on Pro dongles in Integrated mode but we keep the belt
                # and the suspenders.
                # Narrow try/except so a ctypes wiring bug in the SDK path
                # (missing argtypes, symbol rename, segfault-turned-OSError)
                # degrades to geometric instead of killing the whole
                # on_skeleton callback for all gloves.
                try:
                    detected = _read_side_from_sdk(lib, glove_id)
                except (AttributeError, OSError, ctypes.ArgumentError) as e:
                    logger.warning(
                        "SDK handedness path raised for glove 0x%08X: %s; "
                        "falling back to geometric",
                        glove_id,
                        e,
                    )
                    e.__traceback__ = None
                    detected = None
                if detected is not None:
                    detect_source = "sdk"
                else:
                    detected = _detect_handedness(positions)
                    detect_source = "geometric"
                if not detected:
                    # Both paths gave up. Rate-limit so a glove stuck in
                    # an ambiguous pose (palm-up, SDK all-INVALID) doesn't
                    # spam logs at 120 Hz while still surfacing the
                    # debuggable signal.
                    _log_unresolved_once(glove_id)
                if detected:
                    if swap_sides:
                        detected = "right" if detected == "left" else "left"
                    with glove_map_lock:
                        existing = [gid for gid, s in glove_id_to_side.items() if s == detected]
                        if existing:
                            logger.error(
                                "Glove 0x%08X detected as %s (via %s), but glove 0x%08X "
                                "already claims that side. Ignoring — use --swap-sides if wrong.",
                                glove_id,
                                detected.upper(),
                                detect_source,
                                existing[0],
                            )
                        else:
                            glove_id_to_side[glove_id] = detected
                            logger.info(
                                "Glove 0x%08X → %s hand (via %s)",
                                glove_id,
                                detected.upper(),
                                detect_source,
                            )

            with glove_map_lock:
                side = glove_id_to_side.get(glove_id)
            if side is None:
                return
            (left_data if side == "left" else right_data).update(positions)
        except Exception:
            # Log but never let exceptions escape into C callback
            logger.exception("Error in Manus skeleton callback")

    cb = _RAW_SKELETON_CALLBACK_TYPE(on_skeleton)
    rc = lib.CoreSdk_RegisterCallbackForRawSkeletonStream(cb)
    if rc != 0:
        # Fatal — without the callback, on_skeleton never fires and every
        # downstream symptom (0 Hz, stall warnings, map never populated)
        # becomes a red herring.
        logger.error("CoreSdk_RegisterCallbackForRawSkeletonStream failed: %d", rc)
        sys.exit(1)
    _prevent_gc = [cb]  # noqa: F841 — prevent GC of ctypes callback

    # ── Publish loop (one thread per side) ──
    # Stall behavior: if no new SDK frame has arrived for STALL_WARN_SEC,
    # the loop logs one warning and stops publishing (it does NOT publish
    # stale cached frames). The watchdog handles gloveId-map invalidation.
    def publish_loop(hand_data: _HandData) -> None:
        side = hand_data.side
        keypoint_topic = KEYPOINT_TOPICS[side]
        last_frame = 0
        period = 1.0 / rate_hz
        stall_warned = False

        try:
            while running:
                start = time.monotonic()

                positions, _ts, last_mono = hand_data.get()

                if last_mono > 0.0 and (start - last_mono) > STALL_WARN_SEC:
                    if not stall_warned:
                        logger.warning(
                            "%s glove stalled (%.1fs since last SDK frame); "
                            "pausing publish for this side",
                            side.upper(),
                            start - last_mono,
                        )
                        stall_warned = True
                    time.sleep(WATCHDOG_TICK_SEC)
                    continue

                # `last_mono == 0.0` means the watchdog just invalidated this
                # side (or the glove has never streamed). Cached `raw_positions`
                # are stale — sleep at watchdog cadence instead of tight-polling
                # at 0.001s, which would otherwise burn ~500× more CPU on a
                # side that's waiting for a USB replug.
                if last_mono == 0.0:
                    time.sleep(WATCHDOG_TICK_SEC)
                    continue

                if positions is None or hand_data.frame_count == last_frame:
                    time.sleep(0.001)
                    continue

                if stall_warned:
                    logger.info("%s glove recovered; resuming publish", side.upper())
                    stall_warned = False

                last_frame = hand_data.frame_count

                try:
                    keypoints = _positions_to_mediapipe(positions)
                    if keypoints is not None:
                        # The vendored proto carries only keypoints + format (no
                        # timestamp); the MIDAS pipeline doesn't use the timestamp.
                        kp_msg = GloveKeypointStateProto()
                        for i in range(keypoints.shape[0]):
                            pt = kp_msg.keypoints.add()
                            pt.x = float(keypoints[i, 0])
                            pt.y = float(keypoints[i, 1])
                            pt.z = float(keypoints[i, 2])
                        kp_msg.keypoint_format = KEYPOINT_FORMAT_MEDIAPIPE_21
                        _publish_proto(keypoint_topic, kp_msg)
                except Exception as e:
                    logger.exception("Publish error (%s): %s", side, e)
                    e.__traceback__ = None

                elapsed = time.monotonic() - start
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            # Drop ourselves from `spawned` so the supervisor respawns this
            # side if we exited due to an unexpected exception (anything
            # outside the narrow per-iteration try/except above). Without
            # this, a dead Thread object would linger in `spawned` and the
            # side would silently go offline until the process restarts.
            spawned.pop(side, None)

    # ── Self-heal watchdog thread ──
    # Runs once per WATCHDOG_TICK_SEC. Does two jobs:
    #   1. Invalidate the gloveId→side mapping for any side that's been
    #      stalled > MAP_INVALIDATE_SEC so geometric detection re-binds
    #      cleanly when the glove reappears (e.g. after USB replug the
    #      SDK assigns a new gloveId; without invalidation the old ID
    #      lingers in the map forever — upstream bug we are NOT copying).
    #   2. Re-read MIDAS_DATA_CENTER_HOST every ENV_RECHECK_SEC. If the
    #      station moved to a new IP, rebuild the ZMQ PUB socket in
    #      place. ZMQ auto-reconnect handles DataCenter restarts on the
    #      same IP for free; this only covers IP *changes*.
    def watchdog_loop() -> None:
        nonlocal sender_host
        hand_data_by_side = {"left": left_data, "right": right_data}
        last_env_check = time.monotonic()
        while running:
            time.sleep(WATCHDOG_TICK_SEC)
            now = time.monotonic()

            invalidated = _invalidate_stale_glove_map(
                now_monotonic=now,
                hand_data_by_side=hand_data_by_side,
                glove_id_to_side=glove_id_to_side,
                glove_map_lock=glove_map_lock,
                stale_threshold_sec=MAP_INVALIDATE_SEC,
            )
            for gid, side in invalidated:
                logger.warning(
                    "Invalidating stale glove 0x%08X (%s) after %.0fs "
                    "silence; awaiting re-detection",
                    gid,
                    side.upper(),
                    MAP_INVALIDATE_SEC,
                )

            if now - last_env_check >= ENV_RECHECK_SEC:
                last_env_check = now
                new_host = os.environ.get("MIDAS_DATA_CENTER_HOST", sender_host)
                if new_host != sender_host:
                    logger.warning(
                        "MIDAS_DATA_CENTER_HOST changed %s → %s; rebuilding ZMQ PUB socket",
                        sender_host,
                        new_host,
                    )
                    try:
                        _rebuild_sender(new_host)
                        sender_host = new_host
                    except Exception as exc:
                        logger.exception("Failed to rebuild sender: %s", exc)
                        exc.__traceback__ = None

    # ── Haptic output thread ──
    # Subscribes (raw ZMQ SUB) to HAPTIC_TOPIC per side: a 5-float "powers" array
    # (Thumb..Pinky, 0..1 amplitude) from a sim's fingertip contact forces, and
    # drives the connected glove's motors via CoreSdk_VibrateFingersForGlove. The
    # SDK vibrate is momentary, so we re-apply every tick; decay to zero if the
    # publisher goes stale; skip SDK calls while idle. Best-effort — any error is
    # logged and the loop continues (never takes the bridge down).
    def haptic_loop() -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{sender_host}:{DATA_OUTPUT_PORT}")
        for s in ("left", "right"):
            sock.setsockopt_string(zmq.SUBSCRIBE, HAPTIC_TOPIC.format(side=s))
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        cur = {"left": [0.0] * 5, "right": [0.0] * 5}
        last_rx = {"left": 0.0, "right": 0.0}
        sent_zero = {"left": True, "right": True}
        buzzed: set[str] = set()  # one-time info log per side, for observability
        logger.info("Haptic output ready (subscribed to %s)", HAPTIC_TOPIC.format(side="<side>"))
        while running:
            try:
                if poller.poll(timeout=16):
                    while True:
                        try:
                            frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        if len(frames) < 4:
                            continue
                        side = "left" if "/left/" in frames[0].decode() else "right"
                        meta = json.loads(frames[1])
                        dtype = _dtype_from_descr(meta["dtype"])
                        arr = np.frombuffer(frames[3], dtype=dtype).reshape(tuple(meta["shape"]))
                        cur[side] = [max(0.0, min(1.0, float(x))) for x in arr[0]["powers"]]
                        last_rx[side] = time.monotonic()
                now = time.monotonic()
                with glove_map_lock:
                    id_by_side = {sd: gid for gid, sd in glove_id_to_side.items()}
                for side, gid in id_by_side.items():
                    powers = cur[side] if now - last_rx[side] <= HAPTIC_STALE_SEC else [0.0] * 5
                    all_zero = not any(p > 0.0 for p in powers)
                    if all_zero and sent_zero[side]:
                        continue  # idle: don't spam the SDK with zeros
                    lib.CoreSdk_VibrateFingersForGlove(gid, (ctypes.c_float * 5)(*powers))
                    if not all_zero and side not in buzzed:
                        logger.info(
                            "Haptic: vibrating %s glove (powers=%s)",
                            side,
                            [round(p, 2) for p in powers],
                        )
                        buzzed.add(side)
                    sent_zero[side] = all_zero
            except Exception as e:
                logger.debug("haptic loop error: %s", e)
                e.__traceback__ = None
                time.sleep(0.05)
        sock.close(linger=0)

    # ── Reactive thread supervisor ──
    # Spawns each side's publish thread the moment its first frame arrives.
    # The bridge never exits because gloves are absent: if no gloves are
    # connected at startup (operator powered on the station first), it
    # waits patiently and the main run loop below keeps polling
    # _maybe_spawn so a late-arriving glove gets a publish thread within
    # ~100 ms. The ctypes callback we registered above stays live for the
    # life of the process, so the SDK delivers frames to us as soon as a
    # glove powers on, with no SDK re-init required.
    spawned: dict[str, threading.Thread] = {}

    def _make_publish_thread(hand_data: _HandData) -> threading.Thread:
        return threading.Thread(target=publish_loop, args=(hand_data,), daemon=True)

    def _maybe_spawn(side: str, hand_data: _HandData) -> None:
        if _spawn_publish_thread_if_needed(side, hand_data, spawned, _make_publish_thread):
            logger.info("Publishing %s glove data", side.upper())

    logger.info(
        "Waiting for gloves (initial %.0fs window; bridge stays alive indefinitely "
        "if none connect — late arrivals are picked up by the watchdog)...",
        INITIAL_GLOVE_WAIT_SEC,
    )

    initial_wait_start = time.monotonic()
    while running and time.monotonic() - initial_wait_start < INITIAL_GLOVE_WAIT_SEC:
        _maybe_spawn("left", left_data)
        _maybe_spawn("right", right_data)
        if len(spawned) == 2:
            break
        time.sleep(SUPERVISOR_TICK_SEC)

    if spawned:
        with glove_map_lock:
            mapping_snapshot = {f"0x{k:08X}": v for k, v in glove_id_to_side.items()}
        logger.info("Bridge running (%d glove(s); mapping=%s)", len(spawned), mapping_snapshot)
    else:
        logger.warning(
            "No gloves connected after initial %.0fs wait; bridge will keep running "
            "and publish as soon as gloves appear.",
            INITIAL_GLOVE_WAIT_SEC,
        )

    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True, name="manus-watchdog")
    watchdog_thread.start()

    if haptics:
        threading.Thread(target=haptic_loop, daemon=True, name="manus-haptic").start()

    # ── Run loop with per-side callback-rate diagnostics + late-spawn ──
    # Per-second log discriminates between "SDK / radio is dropping frames"
    # (rates asymmetric or both below the dongle datarate) vs "publish loop
    # is throttled" (callbacks high but published topic rate low). Also
    # picks up gloves that connect AFTER the initial wait window — the
    # watchdog handles the spawn (via _maybe_spawn) within one tick.
    LOG_PERIOD = 1.0
    last_log = time.monotonic()
    last_left_count = left_data.frame_count
    last_right_count = right_data.frame_count
    last_wait_log = time.monotonic()

    try:
        while running:
            time.sleep(0.1)
            now = time.monotonic()

            # Late-arriving gloves: spawn the moment the first frame lands.
            # Cheap (already-spawned check is O(1)), runs at 10 Hz here.
            _maybe_spawn("left", left_data)
            _maybe_spawn("right", right_data)

            if now - last_log >= LOG_PERIOD:
                cur_left = left_data.frame_count
                cur_right = right_data.frame_count
                dt = now - last_log
                logger.info(
                    "callback rate: L=%.1f Hz R=%.1f Hz",
                    (cur_left - last_left_count) / dt,
                    (cur_right - last_right_count) / dt,
                )
                last_left_count = cur_left
                last_right_count = cur_right
                last_log = now

            # Periodic "still waiting" log so operators know the bridge is
            # alive and intentionally idle, not crashed.
            missing = [s for s in ("left", "right") if s not in spawned]
            if missing and now - last_wait_log >= GLOVE_WAIT_LOG_PERIOD_SEC:
                logger.warning(
                    "Still waiting for %s glove(s) — power them on to start publishing.",
                    "/".join(s.upper() for s in missing),
                )
                last_wait_log = now
    except KeyboardInterrupt:
        running = False

    lib.CoreSdk_ShutDown()
    logger.info("Bridge stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manus Pro dual-glove → data-center bridge")
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="Manual override: swap the SDK-reported handedness mapping",
    )
    parser.add_argument("--rate", type=float, default=120.0, help="Max publish rate per hand (Hz)")
    parser.add_argument(
        "--host",
        default=None,
        help="Data-center host to publish to (default: $MIDAS_DATA_CENTER_HOST or localhost)",
    )
    parser.add_argument(
        "--no-haptics",
        action="store_true",
        help="Disable haptic output (vibration driven by sim fingertip contact forces)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_bridge(
        swap_sides=args.swap_sides,
        rate_hz=args.rate,
        haptics=not args.no_haptics,
        host=args.host,
    )


if __name__ == "__main__":
    main()
