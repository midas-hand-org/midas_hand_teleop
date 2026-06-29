"""Self-contained glove-keypoint protobuf — vendored, no external repo deps.

The Manus bridge publishes a ``GloveKeypointStateProto``. To keep the MIDAS repos
fully self-contained, this module rebuilds a *minimal* equivalent at runtime via
the protobuf descriptor API — so it needs no ``.proto`` compiler and no generated
file, only the ``protobuf`` runtime (a normal PyPI dependency).

It declares the two fields the bridge emits and the driver reads: ``keypoints``
(field 2, repeated ``Point3{double x=1, y=2, z=3}``) and ``keypoint_format``
(field 3, enum-on-the-wire, kept as int32). The ``timestamp`` (field 1) is
omitted — nothing in this pipeline uses it. Because protobuf is tag-based, this
schema is wire-compatible with the full upstream message in BOTH directions:

* parsing the bridge's output, the timestamp is ignored as an unknown field;
* this side's output parses cleanly under the full upstream proto too.

If the keypoint wire layout ever changes on the bridge side, mirror it here.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_F = descriptor_pb2.FieldDescriptorProto


def _build_message_class():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "midas_manus_glove/glove.proto"
    file_proto.package = "midas_manus_glove"
    file_proto.syntax = "proto3"

    # foxglove.Point3 layout: double x=1, y=2, z=3.
    point3 = file_proto.message_type.add()
    point3.name = "Point3"
    for name, number in (("x", 1), ("y", 2), ("z", 3)):
        field = point3.field.add()
        field.name = name
        field.number = number
        field.label = _F.LABEL_OPTIONAL
        field.type = _F.TYPE_DOUBLE

    glove = file_proto.message_type.add()
    glove.name = "GloveKeypointStateProto"
    keypoints = glove.field.add()
    keypoints.name = "keypoints"
    keypoints.number = 2  # must match the bridge's field number
    keypoints.label = _F.LABEL_REPEATED
    keypoints.type = _F.TYPE_MESSAGE
    keypoints.type_name = ".midas_manus_glove.Point3"
    # keypoint_format is an enum upstream; an int32 here is wire-identical
    # (both are varints at field 3) and avoids vendoring the enum type.
    keypoint_format = glove.field.add()
    keypoint_format.name = "keypoint_format"
    keypoint_format.number = 3
    keypoint_format.label = _F.LABEL_OPTIONAL
    keypoint_format.type = _F.TYPE_INT32

    # Private pool so this never collides with another GloveKeypointStateProto if
    # both happen to be importable in the same process.
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("midas_manus_glove.GloveKeypointStateProto")
    )


# KeypointFormat enum value (MEDIAPIPE_21) — the only one we emit.
KEYPOINT_FORMAT_MEDIAPIPE_21 = 1

# Built once at import; parse/serialize failures would surface here immediately.
GloveKeypointStateProto = _build_message_class()
