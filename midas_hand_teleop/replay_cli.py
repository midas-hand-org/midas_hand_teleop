"""Launch the MuJoCo trajectory replay script from the sibling mujoco repo."""

from __future__ import annotations

import runpy
import sys

from midas_hand_retargeter.paths import find_mujoco_repo


def main() -> None:
    script = find_mujoco_repo() / "assets" / "replay_trajectory.py"
    if not script.exists():
        raise FileNotFoundError(f"Replay script not found: {script}")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
