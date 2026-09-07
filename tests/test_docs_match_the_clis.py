"""Every flag shown in a doc must exist on the CLI it is shown with.

Written because an audit found the README documenting a Manus SDK lookup
(`--sdk-lib`, `$MANUS_SDK_LIB`, `$MANUS_SDK_DIR`) that did not exist in the
code at all -- a hard blocker for anyone outside this machine, sitting in the
install instructions. A doc that names a flag nobody implemented is worse than
no doc, and nothing was checking.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: CLIs with an importable parser factory.
PARSER_MODULES = {
    "midas-hand-tune": "midas_hand_teleop.tuner.cli",
    "midas-manus-teleop": "midas_hand_teleop.manus_glove.manus_teleop",
    "midas-hand-teleop": "midas_hand_teleop.webcam_demo",
}

#: The bridge builds its parser inside main(), so its flags are read from source.
BRIDGE_SOURCE = (
    REPO / "midas_hand_teleop" / "midas_hand_teleop" / "manus_glove" / "manus_bridge.py"
)

DOCS = (
    REPO / "midas_hand_teleop" / "README.md",
    REPO / "midas_hand_teleop" / "RUNBOOK.md",
    REPO / "midas_hand_retargeter" / "README.md",
)

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")


def cli_flags() -> dict[str, set[str]]:
    flags = {
        name: {
            option
            for action in importlib.import_module(module).build_parser()._actions
            for option in action.option_strings
        }
        for name, module in PARSER_MODULES.items()
    }
    flags["midas-manus-bridge"] = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', BRIDGE_SOURCE.read_text()))
    return flags


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_documented_flags_exist(doc):
    flags = cli_flags()
    problems = []
    for lineno, line in enumerate(doc.read_text().splitlines(), 1):
        named = [name for name in flags if name in line]
        if len(named) != 1:
            # Zero commands, or two on one line -- attributing a flag to the
            # right one is guesswork, so skip rather than report a false alarm.
            continue
        name = named[0]
        for flag in FLAG.findall(line):
            if flag not in flags[name]:
                problems.append(f"{doc.name}:{lineno} shows `{name} {flag}`, which does not exist")
    assert not problems, "\n".join(problems)


def test_every_entry_point_has_an_importable_parser():
    """So this check keeps working, and so defaults stay testable."""

    for module in PARSER_MODULES.values():
        assert callable(importlib.import_module(module).build_parser)
