"""Minimal data-center bus shim — self-contained, no external repo deps.

Provides the small ZMQ pub/sub layer the Manus bridge and the teleop driver use:
a PUB-side ``MessageSender``, the ``send_proto_as_sized_array`` helper, the
XSUB/XPUB relay, and the shared port constants. The wire format is the standard
DataCenter sized-array framing, so this interoperates with any compatible
DataCenter:

    frames = [topic, {"dtype": dtype.descr, "shape": shape}, user_metadata, buf]

The DataCenter is a simple XSUB(5710)->XPUB(5711) ZMQ proxy: publishers PUB to
5710, subscribers SUB from 5711. ``start_data_center_proxy`` runs that relay
in-process for a standalone 2-process setup (no full DataCenter needed).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import zmq

logger = logging.getLogger("manus_glove.data_center")

# DataCenter pub/sub relay ports (standard well-known values).
DATA_SOURCE_PORT = 5710  # publishers PUB here (XSUB frontend)
DATA_OUTPUT_PORT = 5711  # subscribers SUB here (XPUB backend)


def _dtype_from_descr(descr: list) -> np.dtype:
    """Rebuild a numpy (structured) dtype from a JSON-roundtripped ``dtype.descr``.

    JSON turns the descr's inner tuples into lists and any field shape into a
    list; numpy needs tuples for both.
    """
    fields = []
    for field in descr:
        field = list(field)
        if len(field) == 3 and isinstance(field[2], list):
            field[2] = tuple(field[2])
        fields.append(tuple(field))
    return np.dtype(fields)


class MessageSender:
    """Minimal PUB-side sender: connects to the DataCenter source port (5710).

    Each instance owns a private ZMQ context so ``close()`` can tear it down
    without touching the shared global context (the bridge rebuilds the sender on
    a DataCenter IP change and closes the old one).
    """

    def __init__(self, host: str = "localhost") -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.connect(f"tcp://{host}:{DATA_SOURCE_PORT}")
        self._lock = threading.Lock()

    def send_array(
        self,
        topic: str,
        user_metadata: dict,
        array: np.ndarray,
        flags: int = 0,
        copy: bool = True,
    ) -> None:
        """Send a structured array as the 4-frame DataEvent wire format."""
        buf = np.ascontiguousarray(array)
        system_metadata = {"dtype": array.dtype.descr, "shape": array.shape}
        with self._lock:
            self._sock.send_string(topic, flags | zmq.SNDMORE)
            self._sock.send_json(system_metadata, flags | zmq.SNDMORE)
            self._sock.send_json(user_metadata, flags | zmq.SNDMORE)
            self._sock.send(memoryview(buf), flags, copy=copy)

    def close(self) -> None:
        if not self._sock.closed:
            self._sock.close(linger=0)
        self._ctx.destroy(linger=0)


def send_proto_as_sized_array(
    message_sender: MessageSender,
    topic: str,
    proto_message,
    user_metadata: dict | None = None,
) -> None:
    """Publish a protobuf message in the ``(size, payload)`` sized-array format.

    Mirrors ``robot_controller.motion_controller_utils.send_proto_as_sized_array``
    — the format motion-control / glove topics use so subscribers can decode with
    a fixed ``[("size","u4"), ("payload","u1",(N,))]`` dtype.

    ``user_metadata`` is frame 2 of the wire format and is free-form; when it is
    omitted a monotonic publish stamp is added so consumers can measure
    end-to-end latency. This deliberately does NOT touch the protobuf: the
    upstream schema is not vendored here, so inventing field numbers there could
    break compatibility, whereas every existing consumer already ignores frame 2.
    """
    proto_bytes = proto_message.SerializeToString()
    size = len(proto_bytes)
    dtype = np.dtype([("size", "u4"), ("payload", "u1", (size,))])
    array = np.empty(1, dtype=dtype)
    array["size"][0] = size
    array["payload"][0] = np.frombuffer(proto_bytes, dtype="u1")
    metadata = dict(user_metadata or {})
    # Monotonic, not wall clock: an NTP step or a suspend must not make a fresh
    # frame look ancient (or a dead one look fresh).
    metadata.setdefault("t_mono_pub", time.monotonic())
    message_sender.send_array(topic=topic, user_metadata=metadata, array=array)


def start_data_center_proxy() -> bool:
    """Start a minimal XSUB(5710)->XPUB(5711) proxy — the DataCenter's relay.

    This is exactly what a DataCenter does for message forwarding (XSUB bind 5710
    / XPUB bind 5711 / ``zmq.proxy``) without a full DataCenter runtime. Lets a
    publisher (PUB->5710) and a subscriber (SUB<-5711) talk in a standalone setup.

    Returns True if started, False if the ports are already bound (a real data
    center — or a stale process — is already up; we just use it).
    """
    ctx = zmq.Context.instance()
    try:
        frontend = ctx.socket(zmq.XSUB)
        frontend.bind(f"tcp://*:{DATA_SOURCE_PORT}")
        backend = ctx.socket(zmq.XPUB)
        backend.bind(f"tcp://*:{DATA_OUTPUT_PORT}")
    except zmq.ZMQError as e:
        logger.info(
            "Not starting built-in proxy (ports %d/%d busy: %s) — assuming a data center is up",
            DATA_SOURCE_PORT,
            DATA_OUTPUT_PORT,
            e,
        )
        return False

    def _run() -> None:
        # zmq.proxy blocks until the context is destroyed at shutdown, which
        # raises ContextTerminated (or a related ZMQError as sockets close).
        # Swallow all — this daemon thread is just exiting.
        try:
            zmq.proxy(frontend, backend)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="dc-proxy").start()
    logger.info(
        "Started built-in data center proxy (XSUB %d -> XPUB %d)",
        DATA_SOURCE_PORT,
        DATA_OUTPUT_PORT,
    )
    return True
