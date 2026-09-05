"""ZMQ ingest for Manus-glove MediaPipe-21 keypoints.

The Manus bridge (``midas_hand_teleop.manus_glove.manus_bridge``) publishes one
``GloveKeypointStateProto`` per side on the data-center bus:

    /data_collection/glove/{side}/keypoint/state

This module is the consumer half. It is deliberately kept dependency-light: a raw
``zmq.SUB`` socket plus the 4-frame sized-array wire format that
``MessageSender.send_array`` produces, namely

    [topic, system_metadata_json, user_metadata_json, array_buffer]

with ``system_metadata = {"dtype": array.dtype.descr, "shape": array.shape}``.

Self-contained: the data-center bus shim lives in ``data_center.py`` (ports,
proxy, dtype helper) and the glove protobuf is vendored in ``glove_proto.py``
(rebuilt at runtime via the protobuf descriptor API). The ports, proxy, and dtype
helper are re-exported here for callers that import them from this module.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import zmq

from midas_hand_teleop.manus_glove.data_center import (
    DATA_OUTPUT_PORT,
    DATA_SOURCE_PORT,
    _dtype_from_descr,
    start_data_center_proxy,
)
from midas_hand_teleop.manus_glove.glove_proto import GloveKeypointStateProto

logger = logging.getLogger("manus_glove.subscriber")

GLOVE_TOPIC = "/data_collection/glove/{side}/keypoint/state"

__all__ = [
    "DATA_OUTPUT_PORT",
    "DATA_SOURCE_PORT",
    "GLOVE_TOPIC",
    "GloveKeypointStateProto",
    "GloveSubscriber",
    "parse_glove_array",
    "start_data_center_proxy",
]


class GloveSubscriber:
    """Minimal raw-ZMQ subscriber for one sized-array glove topic.

    A SUB socket on the data center's output port reading the 4-frame wire format
    ``MessageSender.send_array`` produces — no heavyweight subscriber runtime.
    """

    def __init__(self, topic: str, host: str, port: int = DATA_OUTPUT_PORT) -> None:
        self._topic = topic
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    def poll_latest(self, timeout_ms: int = 0) -> np.ndarray | None:
        """Drain the buffer; return the most recent message's structured array, or None."""
        array, _ = self.poll_latest_with_metadata(timeout_ms)
        return array

    def poll_latest_with_metadata(
        self, timeout_ms: int = 0
    ) -> tuple[np.ndarray | None, dict]:
        """Like :meth:`poll_latest`, but also return the user-metadata frame.

        Frame 2 of the wire format carries the publisher's monotonic stamp, so
        this is what makes an end-to-end latency measurement possible. Returns
        ``(None, {})`` when nothing is waiting.
        """

        if not dict(self._poller.poll(timeout_ms)):
            return None, {}
        latest = None
        while True:
            try:
                latest = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None or len(latest) < 4:
            return None, {}
        sysmeta = json.loads(latest[1])
        try:
            usermeta = json.loads(latest[2]) or {}
        except (ValueError, TypeError):
            usermeta = {}
        dtype = _dtype_from_descr(sysmeta["dtype"])
        shape = tuple(sysmeta["shape"])
        return np.frombuffer(latest[3], dtype=dtype).reshape(shape), usermeta

    def close(self) -> None:
        self._sock.close(linger=0)


def parse_glove_array(arr: np.ndarray) -> np.ndarray | None:
    """Decode a sized-array glove message to (21, 3) keypoints, or None if malformed."""
    try:
        size = int(arr[0]["size"])
        payload = arr[0]["payload"][:size].tobytes()
        proto = GloveKeypointStateProto()
        proto.ParseFromString(payload)
    except Exception as e:
        logger.warning("Failed to parse glove message: %s", e)
        return None
    if len(proto.keypoints) != 21:
        logger.warning("Unexpected keypoint count %d (expected 21)", len(proto.keypoints))
        return None
    kp = np.empty((21, 3), dtype=np.float64)
    for i, pt in enumerate(proto.keypoints):
        kp[i] = (pt.x, pt.y, pt.z)
    return kp
