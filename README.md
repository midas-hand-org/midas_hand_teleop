# MIDAS Hand Teleop

MediaPipe webcam wrappers for producing MIDAS retargeting inputs. This repo
does not own the optimizer, MuJoCo model, or hardware API:

- retargeting: `midas_hand_retargeter`
- simulation: `midas_hand_mujoco`
- hardware: `midas_hand_api`

Install the sibling packages during development:

```bash
pip install -e ../midas_hand_api
pip install -e ../midas_hand_retargeter
pip install -e .
```

Print retargeted active joint targets from a webcam:

```bash
midas-hand-teleop --show --backend print
```

The MIDAS MuJoCo model is currently a right hand. By default the teleop demo
accepts either physical hand and mirrors a left-hand input into the right-hand
robot convention. The webcam overlay shows both the raw MediaPipe handedness
label and the corrected input hand:

```bash
midas-hand-teleop --show --backend print --debug-targets
```

When `--show` is enabled, hold the human hand in the pose that should command
MIDAS zero and press `c` to capture retargeter neutral calibration. Press `r`
to clear it. This calibration is applied before print, MuJoCo, or hardware
output while preserving the robot joint limits on each side of zero.

For an unmirrored OpenCV webcam feed, MediaPipe's raw handedness label is often
opposite the physical hand. If you mirror the camera image before detection,
pass `--selfie`.

With newer `mediapipe` wheels, the package uses the MediaPipe Tasks API and
caches `hand_landmarker.task` at `~/.cache/midas_hand_teleop/` on first run.
You can also provide the model explicitly:

```bash
midas-hand-teleop --hand-landmarker-model /path/to/hand_landmarker.task
```

Send commands to the MIDAS MuJoCo model:

```bash
midas-hand-teleop --backend mujoco --mujoco-viewer --show --debug-targets
```

Common live tuning overrides:

```bash
midas-hand-teleop --backend mujoco --mujoco-viewer --show --debug-targets \
  --finger-curl-gain 1.0 \
  --finger-abad-gain 0.8 \
  --finger-smoothing-alpha 0.16 \
  --thumb-cmc-gain 1.0 \
  --thumb-cmc-side-gain 1.0 \
  --thumb-cmc-roll-gain 1.0 \
  --thumb-flexion-gain 1.0 \
  --thumb-smoothing-alpha 0.25
```

For persistent defaults, edit `midas_hand_retargeter/tuning.py`.
Use `--input-hand Left` or `--input-hand Right` if the MediaPipe handedness
label flips while running. With the default `--input-hand auto`, teleop locks
onto the first corrected physical hand to avoid convention switching jitter.

Send commands to hardware after calibration. The hardware backend runs its own
fixed-rate command loop, so vision frames update the target while motors receive
interpolated commands at `--hardware-rate-hz`. Start with a low command scale
and slow per-tick step, then increase after checking that the signs and limits
are correct:

```bash
midas-hand-teleop --backend hardware --show --debug-targets \
  --configure-hardware \
  --hardware-command-scale 0.3 \
  --hardware-max-step-rad 0.03 \
  --hardware-rate-hz 50 \
  --hardware-interpolation-alpha 0.25
```

Useful hardware overrides:

```bash
midas-hand-teleop --backend hardware --show --debug-targets \
  --configure-hardware \
  --hardware-config ~/.midas_hand/config.yaml \
  --hardware-port /dev/ttyUSB0 \
  --hardware-current-limit 350 \
  --hardware-command-scale 0.3 \
  --hardware-max-step-rad 0.03 \
  --hardware-rate-hz 50 \
  --hardware-interpolation-alpha 0.25
```
