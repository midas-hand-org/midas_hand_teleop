"""fake_glove_publisher.py — hardware-free Manus-glove keypoint publisher.

The real Manus bridge needs an x86 host, the ManusSDK ``.so``, and physical
gloves. This stand-in publishes the SAME wire format on the SAME topic
(``/data_collection/glove/{side}/keypoint/state``) so the whole subscribe ->
geometric-map -> MuJoCo pipeline is testable end-to-end with nothing plugged in.

It emits a synthetic right hand whose four fingers + thumb open and close in a
slow sine cycle. Each finger is built as a kinematic chain (MCP -> PIP -> DIP ->
tip), so curling produces real bend ANGLES between phalanx segments — exactly
what ``finger_joint_targets_from_landmarks`` / ``thumb_joint_targets_from_landmarks``
measure — rather than just translating points.

It publishes through the same vendored ``data_center.MessageSender`` +
``send_proto_as_sized_array`` the real bridge uses, so the wire format is
identical:

    frames = [topic, {"dtype": dtype.descr, "shape": shape}, {}, array_buffer]
    array  = np.empty(1, dtype=[("size","u4"), ("payload","u1",(N,))])

Run:
    # publishes to localhost (the driver starts the 5710/5711 proxy)
    python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right --rate 60
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time

import numpy as np

from midas_hand_teleop.manus_glove.data_center import MessageSender, send_proto_as_sized_array
from midas_hand_teleop.manus_glove.glove_proto import GloveKeypointStateProto
from midas_hand_teleop.manus_glove.glove_subscriber import GLOVE_TOPIC

logger = logging.getLogger("fake_glove_publisher")

# MediaPipe-21 finger landmark groups (MCP, PIP, DIP, tip) and thumb (CMC, MCP,
# IP, tip). Lateral x offset of each finger base, in a wrist-origin frame where
# fingers point +y, the palm normal is +z, and +x is toward the pinky.
_FINGERS = {
    "index": (5, 6, 7, 8, -0.030),
    "middle": (9, 10, 11, 12, -0.010),
    "ring": (13, 14, 15, 16, 0.010),
    "pinky": (17, 18, 19, 20, 0.030),
}
_SEG_LEN = 0.035  # phalanx length (m)
_MCP_Y = 0.045  # finger-base distance from the wrist along +y


def _finger_chain(base_xyz: np.ndarray, curl: float) -> np.ndarray:
    """Four points (MCP, PIP, DIP, tip) for one finger bending by ``curl`` (0..1).

    Bend is about the lateral (+x) axis in the y-z plane; the flex accumulates
    down the chain so adjacent segments form a real angle (PIP/DIP bend).
    """
    mcp_flex = 0.5 * curl
    pip_flex = 1.2 * curl
    dip_flex = 0.8 * curl
    pts = [base_xyz.astype(np.float64)]
    angle = mcp_flex
    for extra in (pip_flex, dip_flex, 0.0):
        direction = np.array([0.0, math.cos(angle), -math.sin(angle)])
        pts.append(pts[-1] + _SEG_LEN * direction)
        angle += extra
    return np.array(pts)  # (4, 3): MCP, PIP, DIP, tip


def synthetic_hand(curl: float) -> np.ndarray:
    """Build a (21, 3) MediaPipe right-hand pose curled by ``curl`` in [0, 1]."""
    kp = np.zeros((21, 3), dtype=np.float64)
    for mcp_i, pip_i, dip_i, tip_i, x in _FINGERS.values():
        base = np.array([x, _MCP_Y, 0.0])
        chain = _finger_chain(base, curl)
        kp[mcp_i], kp[pip_i], kp[dip_i], kp[tip_i] = chain
    # Thumb: CMC, MCP, IP, tip (indices 1..4); wrist stays at origin (index 0).
    cmc = np.array([-0.035, 0.010, 0.005])
    base_dir = np.array([-0.5, 0.8, 0.0])
    base_dir /= np.linalg.norm(base_dir)
    mcp_flex, ip_flex, dip_flex = 0.7 * curl, 0.9 * curl, 0.6 * curl
    pts = [cmc]
    angle = mcp_flex
    for extra in (ip_flex, dip_flex):
        bend = np.array([base_dir[0], base_dir[1] * math.cos(angle), -math.sin(angle)])
        pts.append(pts[-1] + _SEG_LEN * bend)
        angle += extra
    kp[1], kp[2], kp[3], kp[4] = cmc, pts[1], pts[2], pts[2] + _SEG_LEN * np.array(
        [base_dir[0], base_dir[1] * math.cos(angle), -math.sin(angle)]
    )
    return kp


def run(side: str, rate: float, duration: float | None, host: str) -> None:
    topic = GLOVE_TOPIC.format(side=side)
    sender = MessageSender(host=host)
    logger.info("Publishing synthetic glove on %s -> %s (rate %.0f Hz)", topic, host, rate)
    # PUB/SUB slow-joiner: give the proxy/subscriber a moment to connect.
    time.sleep(0.3)

    period = 1.0 / max(rate, 1e-6)
    start = time.monotonic()
    next_tick = start
    sent = 0
    try:
        while True:
            now = time.monotonic()
            if duration is not None and now - start >= duration:
                break
            # Slow open/close cycle (~0.2 Hz): curl in [0, 1].
            curl = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.2 * (now - start)))
            kp = synthetic_hand(curl)

            proto = GloveKeypointStateProto()
            for x, y, z in kp:
                pt = proto.keypoints.add()
                pt.x, pt.y, pt.z = float(x), float(y), float(z)
            send_proto_as_sized_array(sender, topic, proto)
            sent += 1
            if sent % int(max(rate, 1)) == 0:
                logger.info("sent=%d curl=%.2f", sent, curl)

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping publisher (sent=%d)", sent)
    finally:
        sender.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Manus-glove keypoint publisher")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--rate", type=float, default=60.0, help="Publish rate (Hz)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run until Ctrl-C)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MIDAS_DATA_CENTER_HOST", "localhost"),
        help="Data-center host to publish to (default: $MIDAS_DATA_CENTER_HOST or localhost)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run(args.side, args.rate, args.duration, args.host)


if __name__ == "__main__":
    main()
