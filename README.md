# MIDAS Hand Teleop

Teleoperate the MIDAS robot hand — in MuJoCo and on real hardware — from a
Manus Haptic Pro glove or a webcam, with a browser UI for tuning each finger's
retargeting live.

This repo owns the input sources, the control loops and the command sinks. It
does not own:

| | |
|---|---|
| retargeting | [`midas_hand_retargeter`](https://github.com/midas-hand-org/midas_hand_retargeter) |
| robot model | [`midas_hand_mujoco`](https://github.com/midas-hand-org/midas_hand_mujoco) |
| hardware API | [`midas_hand_api`](https://github.com/midas-hand-org/midas_hand_api) |

## Install

```bash
pip install -e ../midas_hand_retargeter
pip install -e ".[manus,mujoco]"        # add `hardware` for the real hand
```

The Manus SDK is proprietary and is **not** vendored here. The bridge binds it
through `ctypes` and looks for `libManusSDK_Integrated.so` via `--sdk-lib`,
`$MANUS_SDK_LIB`, `$MANUS_SDK_DIR`, the loader path, then `/usr/local/lib`.
Everything except the live-glove path runs without it.

## Quickstart, no hardware at all

The synthetic publisher speaks the same wire format as the real bridge, so the
whole pipeline runs with no glove, no camera and no robot:

```bash
python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right &
midas-manus-teleop --backend mujoco --mujoco-viewer
```

## Tuning UI

```bash
python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right &   # or: midas-manus-bridge &
midas-hand-tune --open
```

Then edit any finger's curl scale, output range, splay or smoothing and watch
that finger change in the sim on the next frame. The page shows the glove rate
and measured end-to-end latency, the analytic intermediates (curl, splay, thumb
angles) behind each joint target, and commanded-vs-measured position per joint.
Presets save to `~/.midas_hand/retarget_presets/` and carry the neutral
calibration with them.

Output-range sliders are bounded by the robot's real joint limits, and show
what percentage of each joint's travel the profile actually commands — the
built-in defaults reach only 75% of MCP pitch and 84% of PIP.

## Glove teleop

```bash
midas-manus-bridge &                       # glove -> ZMQ
midas-manus-teleop --backend mujoco --mujoco-viewer
```

Right hand only. `--side left` is refused rather than warned about: the
analytic map is reflection-invariant, so mirroring the input produces
byte-identical joint targets and does **not** give you a left hand.

## Webcam teleop

```bash
midas-hand-teleop --backend mujoco --mujoco-viewer
```

Keys in the camera window: `c` capture neutral calibration, `r` clear it,
`q` quit.

## Driving the real hand

> The default backend is `print` in every entry point. Nothing energises a
> motor unless you ask for `--backend hardware`.

Before the first run, **home the hand** (see `midas_hand_api`) so
`~/.midas_hand/config.yaml` exists. Without it the motor zero has no defined
relationship to the URDF zero and the API's joint-limit clamp does nothing, so
`--backend hardware` is refused; `--allow-unhomed` overrides that if you know
why you want it.

```bash
midas-manus-teleop --backend hardware \
    --hardware-current-limit 200 \
    --hardware-command-scale 0.4 \
    --start-armed
```

Bring-up order: `--backend print` first to check the targets look sane, then
`--backend mujoco`, then hardware with a low current limit and a reduced
command scale. Without `--start-armed` the backend connects and configures but
leaves torque off until armed, and the first commanded pose is always the
measured pose, so arming cannot jump.

A deadman stops commanding and disarms if no glove frame arrives for
`--stale-timeout` seconds (0.5 by default). Without it the loop would hold a
commanded pose against a dead publisher indefinitely.

## Entry points

| command | what it does |
|---|---|
| `midas-manus-bridge` | Manus SDK → ZMQ keypoints |
| `midas-manus-teleop` | glove → retarget → print / MuJoCo / hardware |
| `midas-hand-teleop` | webcam → retarget → print / MuJoCo / hardware |
| `midas-hand-tune` | browser tuning UI |
| `midas-hand-diag` | text diagnostics: palm basis, canonical poses, frame presets |

## Architecture

```
Manus glove ──ctypes──▶ manus_bridge ──ZMQ 5710/5711──▶ subscriber
                                                            │
                                                    (21,3) landmarks
                                                            ▼
                                          midas_hand_retargeter (analytic)
                                                            │
                                                   13 joint targets
                                                            ▼
                                        PrintBackend │ MujocoBackend │ HardwareBackend
```

The bridge publishes at 120 Hz; control loops run at 60 Hz; the hardware
backend commands on its own 50 Hz thread with slew limiting, so an irregular
input rate never becomes an irregular command rate.

## Tests

```bash
pytest
```

Everything runs with no glove, no SDK and no hardware.

## License

MIT. See `LICENSE`.
