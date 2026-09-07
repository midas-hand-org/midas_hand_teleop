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
through `ctypes` and looks for `libManusSDK_Integrated.so` in this order:
`--sdk-lib`, `$MANUS_SDK_LIB`, `$MANUS_SDK_DIR/lib`, `/usr/local/lib`, then
`/opt/ManusSDK/lib`. A `--sdk-lib` or `$MANUS_SDK_LIB` that points at nothing
is an error rather than a fall-through to a different copy. Everything except
the live-glove path runs without the SDK.

## Quickstart, no hardware at all

The synthetic publisher speaks the same wire format as the real bridge, so the
whole pipeline runs with no glove, no camera and no robot:

```bash
python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right &
midas-manus-teleop --backend mujoco --mujoco-viewer
```

## Retargeting modes

Two things can drive the joints, and which one you want depends on the task:

| `--mode` | what it does | use it when |
|---|---|---|
| `analytic` | reads angles off the hand and maps them per joint | you want predictable, per-finger control and no solver |
| `dexpilot` | optimises **fingertip positions relative to each other** | you care about pinches and where the fingertips are with respect to one another |

`analytic` is blind to absolute hand geometry — scaling a hand 0.6×–3× changes
its output by ~3e-6 rad — so it structurally cannot place fingertips relative
to each other. `dexpilot` can, at the cost of being sensitive to hand size
(`scaling_factor`) and to input chirality.

Two further modes, `vector` and `refine`, exist for comparison and are not
what you want day to day.

## Tuning UI

```bash
# a real glove: midas-manus-bridge &
python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right &
midas-hand-tune --mode dexpilot --open
```

Edit any finger's curl scale, output range, splay or smoothing and watch that
finger change on the next frame. The page shows the glove rate and measured
end-to-end latency, the intermediates behind each joint target, and
commanded-vs-measured position per joint. It renders only the controls the
running mode actually reads, so a slider is never shown dead.

Three separate calibrations, easy to confuse:

| | what it measures | how |
|---|---|---|
| **hand size** | your reach and finger spacing vs the robot's | hold a flat open hand, press **Calibrate hand size**; sets `scaling_factor` and `spread_scale` |
| **zero pose** | which of your poses means "robot at zero" | hold the rest pose, press **Capture neutral** |
| **homing** | where each motor's encoder zero is | `python -m midas_hand_api --home`, once per hand |

Presets save to `~/.midas_hand/retarget_presets/` and carry the zero pose with
them, so `--preset <name>` reproduces a session. They do **not** record which
mode they were tuned in — pass `--mode` too.

Output-range sliders are bounded by the robot's real joint limits, and show
what percentage of each joint's travel the profile actually commands.

## Glove teleop

```bash
midas-manus-bridge &                       # glove -> ZMQ
midas-manus-teleop --backend mujoco --mujoco-viewer
```

Keys in the terminal while it runs:

| key | |
|---|---|
| `c` | capture the current pose as the zero pose |
| `r` | clear that calibration |
| `s` | calibrate hand size from a held open pose (dexpilot only) |
| `q` | quit |

To run the profile you tuned, pass both the preset and the mode — a preset does
not record which mode it was tuned in:

```bash
midas-manus-teleop --retarget dexpilot --preset my-hand --backend mujoco --mujoco-viewer
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

> Nothing energises a motor unless you ask for `--backend hardware`, **and**
> then arm it. `midas-hand-tune` and `midas-hand-teleop` default to
> `--backend print`; `midas-manus-teleop` defaults to `mujoco`.

Before the first run, **home the hand** (see `midas_hand_api`) so
`~/.midas_hand/config.yaml` exists. Without it the motor zero has no defined
relationship to the URDF zero, so `--backend hardware` is refused;
`--allow-unhomed` overrides that if you know why you want it.

The recommended path is the tuner, because it is the only entry point that
loads a preset — so it is the only one that runs the settings you tuned:

```bash
midas-manus-bridge &
midas-hand-tune --mode dexpilot --preset my-hand --backend hardware --open \
    --hardware-current-limit 150 \
    --hardware-command-scale 0.3 \
    --hardware-max-step-rad 0.05
```

It starts **disarmed**. Press *Arm hardware* in the browser when the status bar
shows a healthy glove. The first commanded pose is the measured pose by
construction, so arming cannot jump.

Raise `--hardware-command-scale` toward 1.0, then the current limit, then the
slew limit — one at a time.

The headless equivalent, without the UI or the preset:

```bash
midas-manus-teleop --backend hardware --retarget dexpilot \
    --hardware-current-limit 150 --hardware-command-scale 0.3 --start-armed
```

Note `--retarget dexpilot`: without it this runs the analytic map.

### Stopping

Five things drop torque, and all of them are tested:

| | |
|---|---|
| *Disarm* in the browser | immediate |
| closing the browser tab | ~3 s, client watchdog |
| no glove frame for 0.5 s | deadman; re-arming is manual and deliberate |
| Ctrl-C, twice if impatient | torque is dropped even on the force-quit path |
| `kill` / SIGTERM | routed through the same cleanup as Ctrl-C |

Commands are clamped to the robot model's joint limits inside the backend.
This matters because homing writes ±π into `config.yaml`, which makes the hand
API's own `clip_positions` a no-op on a correctly homed hand.

## Entry points

| command | what it does |
|---|---|
| `midas-manus-bridge` | Manus SDK → ZMQ keypoints |
| `midas-manus-teleop` | glove → retarget → print / MuJoCo / hardware |
| `midas-hand-teleop` | webcam → retarget → print / MuJoCo / hardware |
| `midas-hand-tune` | glove → retarget → print / MuJoCo / hardware, with a browser tuning UI |
| `midas-hand-diag` | text diagnostics: palm basis, canonical poses, frame presets |

## Architecture

```
Manus glove ──ctypes──▶ manus_bridge ──ZMQ 5710/5711──▶ subscriber
                                                            │
                                                    (21,3) landmarks
                                                            ▼
                                    midas_hand_retargeter (analytic / dexpilot)
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
