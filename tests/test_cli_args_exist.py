"""Every ``args.<name>`` a CLI reads must be defined by its own parser.

Written after a cleanup commit deleted ``--hand-landmarker-model`` from
webcam_demo's argument list -- it happened to sit inside a block being replaced
wholesale -- and left the ``args.hand_landmarker_model`` reader behind.
``midas-hand-teleop`` then raised AttributeError on every run, and nothing
noticed, because no test constructs that pipeline and the read is 20 lines past
the last point a smoke test could reach without a camera.

This is a static check on purpose: it needs no camera, no glove and no robot.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pathlib

import pytest

MODULES = {
    "midas-hand-teleop": "midas_hand_teleop.webcam_demo",
    "midas-manus-teleop": "midas_hand_teleop.manus_glove.manus_teleop",
    "midas-hand-tune": "midas_hand_teleop.tuner.cli",
}

#: Set on the namespace at runtime rather than by add_argument.
ALLOWED_EXTRA = {
    "midas-hand-teleop": {
        "finger_smoothing_alpha",
        "thumb_smoothing_alpha",
        "thumb_flexion_gain",
        "lock_input_hand",
    },
}


def parser_dests(module_name: str) -> set[str]:
    parser = importlib.import_module(module_name).build_parser()
    dests = set()
    for action in parser._actions:
        if action.dest and action.dest != argparse.SUPPRESS:
            dests.add(action.dest)
    # set_defaults(...) additions are real too.
    dests |= set(vars(parser.parse_args([])))
    return dests


def args_attributes_read(module_name: str) -> set[str]:
    path = pathlib.Path(importlib.import_module(module_name).__file__)
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            found.add(node.attr)
    return found


@pytest.mark.parametrize("name,module", sorted(MODULES.items()))
def test_every_args_attribute_is_a_real_flag(name, module):
    missing = args_attributes_read(module) - parser_dests(module) - ALLOWED_EXTRA.get(name, set())
    assert not missing, (
        f"{name} reads args attributes its parser never defines: {sorted(missing)}. "
        "Either the flag was deleted and its reader left behind, or the reader "
        "is a typo -- both raise AttributeError at runtime."
    )


@pytest.mark.parametrize("name,module", sorted(MODULES.items()))
def test_a_bare_parse_produces_every_attribute_the_module_reads(name, module):
    """The same property from the other direction: parse with no arguments and
    confirm each read resolves, which is what actually happens on `--help`-less
    invocation."""

    args = importlib.import_module(module).build_parser().parse_args([])
    for attribute in sorted(args_attributes_read(module) - ALLOWED_EXTRA.get(name, set())):
        assert hasattr(args, attribute), f"{name}: args.{attribute} does not exist"
