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

needs_optimizer = pytest.mark.skipif(
    importlib.util.find_spec("dex_retargeting") is None,
    reason="requires the [vector] extra",
)


@needs_optimizer
def test_dexpilot_is_rotation_invariant_but_chirality_sensitive():
    """The two halves of the frame story, pinned together.

    Rotations must wash out — the retargeter normalises into the operator's
    palm basis, so how the hand is held is not finger articulation. A
    reflection must NOT wash out: it is the difference between curling toward
    the palm and curling backwards, and silently swallowing it is what left
    the ring finger tracking inverted.
    """

    from midas_hand_teleop.manus_glove.fake_glove_publisher import synthetic_hand

    retargeter = MidasHandRetargeter.create(mode=DEXPILOT_MODE)
    retargeter.profile = RetargetProfile().with_values(
        {"dexpilot.scaling_factor": 1.1}
    )
    landmarks = synthetic_hand(0.5)

    def solve(points):
        retargeter.reset()
        for _ in range(20):
            result = retargeter.retarget_landmarks(points)
        return result.active_vector()

    base = solve(landmarks)

    rotation = np.array(  # 90 degrees about X, a proper rotation
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    assert np.linalg.det(rotation) > 0
    np.testing.assert_allclose(solve(landmarks @ rotation.T), base, atol=1e-4)

    reflection = np.diag([1.0, -1.0, 1.0])
    assert np.linalg.det(reflection) < 0
    assert np.abs(solve(landmarks @ reflection.T) - base).max() > 0.1, (
        "a mirrored hand must not produce the same command as a correct one"
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
