"""DexPilot is orientation-sensitive; the analytic map is not.

This distinction is easy to get wrong because every other path here is frame
agnostic. The glove driver used to rotate landmarks into the MANO frame before
retargeting, which is harmless for the analytic map and destroys DexPilot.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from midas_hand_retargeter import MidasHandRetargeter, RetargetProfile
from midas_hand_retargeter.config import DEXPILOT_MODE
from midas_hand_retargeter.human import mediapipe_world_to_mano_landmarks

needs_optimizer = pytest.mark.skipif(
    importlib.util.find_spec("dex_retargeting") is None,
    reason="requires the [vector] extra",
)


@needs_optimizer
def test_dexpilot_is_sensitive_to_the_input_frame():
    """Pins WHY the glove driver must not MANO-transform for this mode."""

    from midas_hand_teleop.manus_glove.fake_glove_publisher import synthetic_hand

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.profile = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.55}
    )
    landmarks = synthetic_hand(0.5)

    retargeter.reset()
    for _ in range(25):
        raw = retargeter.retarget_landmarks(landmarks).active_vector()

    retargeter.reset()
    rotated = mediapipe_world_to_mano_landmarks(landmarks, hand_type="Right")
    for _ in range(25):
        mano = retargeter.retarget_landmarks(rotated).active_vector()

    assert np.abs(raw - mano).max() > 0.1, (
        "if this ever becomes frame-invariant, the pairwise vectors have stopped "
        "doing anything and the mode is pointless"
    )


@needs_optimizer
def test_glove_driver_does_not_mano_transform_for_dexpilot():
    import inspect

    from midas_hand_teleop.manus_glove import manus_teleop

    source = inspect.getsource(manus_teleop.run)
    assert "DEXPILOT_MODE" in source
    index = source.index("DEXPILOT_MODE")
    # The dexpilot branch must return before the MANO conversion below it.
    assert "mediapipe_world_to_mano_landmarks" in source[index:]
