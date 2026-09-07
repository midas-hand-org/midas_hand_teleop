# MIDAS glove teleop — verification runbook

Work through Part A now (glove + sim, everything is on your desk). Part B waits
for the hand. Part C is the future tactile-haptics phase.

Steps marked **`RECORD:`** produce a value a later step needs — write it in the
blank so nothing has to be re-derived. Steps marked **`TODO(hardware-day)`**
are placeholders that only hardware can resolve.

Environment for every command:

```bash
cd /home/dyna/midas
source midas_env/bin/activate        # or use midas_env/bin/... directly
```

---

## Part A — verifiable now (glove + sim)

### A0. Both suites green  ☐

```bash
(cd midas_hand_retargeter && pytest -q)
(cd midas_hand_teleop     && pytest -q)
```

Expected: **152 passed** and **128 passed**. Was 29 + 49 before this work.

### A1. The default install is clean  ☐

This is the release blocker that used to make `pip install midas-hand-retargeter`
unable to retarget a single frame.

```bash
python -m venv /tmp/cleanenv
/tmp/cleanenv/bin/pip install ./midas_hand_retargeter
/tmp/cleanenv/bin/pip list
```

Expected: exactly `midas-hand-retargeter`, `numpy`, `pip`. No torch, no
dex_retargeting, no pinocchio.

```bash
/tmp/cleanenv/bin/midas-retargeter-smoke && echo "exit $?"
```

Expected: 13 joint values printed, exit 0 — with no URDF and no sibling repo.

### A2. Lint clean  ☐

```bash
(cd midas_hand_retargeter && ruff check .)
(cd midas_hand_teleop     && ruff check .)
```

### A3. Pipeline works with no glove at all  ☐

Prove the synthetic path first, so a later failure with the real glove is
unambiguously about the glove.

```bash
python -m midas_hand_teleop.manus_glove.fake_glove_publisher --side right &
midas-manus-teleop --backend mujoco --duration 5
```

Expected: `solves=... (60 Hz) parse_fail=0`, then `Done.`

### A4. Real glove → sim  ☐  ← **the first thing that needs you present**

Plug in the Manus dongle, put the glove on, then:

```bash
pkill -f fake_glove_publisher
midas-manus-bridge &
midas-manus-teleop --backend mujoco --mujoco-viewer
```

Watch for, in order:

- bridge logs `callback rate: L=... R=... Hz` — the SDK is delivering
- teleop logs `solves=... (60 Hz) parse_fail=0`
- the MuJoCo hand follows yours

**`RECORD:` glove callback rate = ________ Hz** → sanity for A6.

If the bridge exits immediately, the SDK was not found; the error now names
every path it tried (this used to exit 0 and look like success).

**`TODO(hardware-day)` dual-glove check.** `manus_bridge` reads skeleton index
0 unconditionally, so if the SDK ever delivers more than one skeleton per
callback the second glove is silently dropped. With both gloves on, confirm
`glove_id_to_side` resolves two ids. Not fixed yet — it needs two gloves to
verify, and guessing would be worse than leaving it documented.

### A5. Tuning UI, per-finger  ☐  ← **the main deliverable**

```bash
midas-hand-tune --open              # add --backend mujoco --mujoco-viewer for the 3D view
```

Then, in order:

1. **Status bar sane** — glove Hz green, latency shown in ms, loop 60 Hz,
   mode `analytic`.
   **`RECORD:` idle latency = ________ ms** → the baseline for A6.
2. **Per-finger independence.** Index tab → drag `curl_gain` up. Only the index
   finger changes in the sim; middle and ring hold. This is the thing that was
   impossible before — every gain used to be global.
3. **Reclaim the unused travel.** Index tab → `mcp_pitch_range`, drag *closed*
   from −1.35 to −1.8. The `% of ROM` readout goes from 75% to 100% and the
   finger visibly closes further. Same for `pip_range`: −1.22 → −1.45 (84% →
   100%).
4. **Intermediates.** Curl your index finger and watch `curl` and `splay_rad`
   in the right panel. If `curl` saturates at 1.0 before your hand is actually
   closed, raise `curl_max_bend` — not `curl_gain`.
5. **Calibration.** Hold a relaxed open hand → *Capture neutral*. The sim hand
   should sit at zero in that pose.
6. **Preset round trip.** Name it, Save, then Reset, then Load. Values and the
   neutral calibration both come back.
   **`RECORD:` preset name = ______________** → used in B5.
7. **Rejected edits are safe.** Nothing you can do in the UI should be able to
   command past a joint limit — the sliders are bounded by the URDF.

**`TODO(hardware-day)`** the *Arm hardware* button stays disabled here; it needs
a hardware backend, which is B4.

### A5b. DexPilot mode — fingertip geometry  ☐  ← **run this if fingers look wrong relative to each other**

The analytic map reads joint *angles* only. It is provably blind to absolute
geometry (scaling a hand 0.6x-3x moves its output by 3e-6 rad) and gives three
equally-curled fingers bit-identical commands, so it structurally cannot place
fingertips relative to one another. DexPilot optimises six pairwise
inter-fingertip vectors plus four palm-rooted ones, which is that missing
capability.

```bash
midas-manus-bridge &
midas-hand-tune --mode dexpilot --mujoco-viewer --open
```

**Calibrate `scaling_factor` FIRST — nothing else matters until it is right.**
It is your hand size relative to the robot's, and it is the one parameter the
analytic map never had. A 0.7x-1.5x change moves joints by ~1.5 rad.

You do not have to estimate it any more. Hold your hand flat and open and press
**Calibrate hand size**: it fits the scale from the median of your index,
middle and ring reach against the robot's, and reports what it set.

Note it sets **two** parameters. The second is `spread_scale` — finger SPACING
rather than reach, fitted from your fingertip span at rest. That is deliberate,
but be aware one button moves two sliders, and `spread_scale` costs real
accuracy below ~0.85 (mean inter-fingertip error 5.5 mm at 1.0, 6.4 mm at 0.86,
13.5 mm at 0.65).

**`RECORD:` calibrated scaling_factor = ________  spread_scale = ________**

If you want to sanity-check it against a ruler, the MIDAS hand reaches ~218 mm
from `palm_base` to fingertip with the hand open, so `218 / your_reach_mm` is
roughly what the button should produce.

Then, on the slider:

- **too low** → fingers over-curl and rail into their limits, and the pose
  stops responding to your hand at all. This is the most common failure.
- **too high** → fingers never close far enough.
- Open your hand fully: the sim should be open, not slightly curled.
  Close it: it should close without saturating early.

Only then touch the rest, in this order:

| knob | what it fixes |
|---|---|
| `abduction_limit` | fingers leaning toward the thumb when you simply curl. Default 0.25 rad; 0 locks them parallel. Do not go below ~0.15 — the clipping becomes its own artifact. |
| `project_dist` | the gap at which a fingertip pair snaps together. **Raise it if pinches leave too big a gap.** It is measured against your RAW landmark gap, and landmarks sit inside your fingers — a pinch you feel as contact still reads ~10 mm, so anything below that disables the snap and a pinch lands ~26 mm open. |
| `escape_dist` | **the stickiness knob.** Lower it if pinched fingertips cling to each other. Keep it a few mm above `project_dist`; that band is hysteresis. At 0.03/0.05 a real trace was held snapped for 33% of its frames. 0.020/0.024 gives a 1.1 mm pinch and 0% stuck. |
| `eta1` | how close a snapped thumb-finger pinch gets. |
| `norm_delta` | smoother but laggier. |
| `thumb_vector_scale` | (advanced) a straighter thumb, at 1-3 mm of fingertip accuracy. |

Then press **Capture neutral** with your hand in its rest pose, and save a
preset. The preset carries the zero pose with it, so `--preset <name>`
reproduces the session — but it does **not** record the mode, so always pass
`--mode dexpilot` alongside it.

Then A/B it honestly against `--mode analytic` on the same motion. Analytic is
6x cheaper and rock-solid for curl; DexPilot is the one that gets relative
fingertip placement right. Which you want depends on the task.

**`TODO(hardware-day)` fingertip offsets.** DexPilot aims at the tip frames in
`urdf.TIP_LINKS`, which are CAD estimates (`thumb_tip` at `0 0.042 -0.010`,
fingers at `0 0.036 -0.009`). In this mode they are load-bearing — a wrong
offset means the solver optimises toward the wrong point. Measure them on the
real hand and correct them.

The thumb one was outright **wrong** until this session: the sign was inverted,
putting `thumb_tip` 28 mm *closer* to the palm than the thumb DIP (cos −0.84
against the distal direction, now +0.96). Every thumb measurement taken before
that fix is void, including the one that concluded the thumb was too short —
it is proportionally long. Worth re-measuring the magnitudes on the real hand
even so.

### A6. Latency is honest  ☐

With the tuner open, unplug the glove dongle. Within ~0.5 s the glove chip must
go red/stale and the latency chip must stop claiming a good number. Replug and
confirm it recovers. A tuner that reports a stale latency as healthy is worse
than one that reports none.

### A7. Webcam path still works  ☐

```bash
midas-hand-teleop --backend mujoco --mujoco-viewer
```

`c` captures neutral, `r` clears, `q` quits. Note the default backend is now
`print`, not `hardware` — a bare invocation no longer energises motors.

### A8. Nothing downstream broke  ☐

`midas-piper-control` imports `PIP_DIP_LOOKUP_MODE` from the retargeter in six
places and drives the real hand.

```bash
python -c "from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE; print('ok')"
python -c "import sys; from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE; assert 'torch' not in sys.modules; print('torch-free ok')"
```

---

## Part B — hardware day (the real hand)

**Status: homing is done.** `~/.midas_hand/config.yaml` holds 13 motors with
per-motor home offsets, all `joint_signs` `+1.0`. So B2 below is complete and
`--backend hardware` will start. Go straight to B1 → B4.

Do these strictly in order. Each is a stop-gate: if one fails, stop.

### B0. The 60-second version  ☐

If you only read one block, read this one. It is the whole path, at bring-up
settings, with nothing energised until you click a button in the browser:

```bash
# glove -> ZMQ
midas-manus-bridge &

# sim first: confirm the retargeting is the one you tuned
midas-hand-tune --mode dexpilot --preset zmz --mujoco-viewer --open

# then the real hand, low and slow. Starts DISARMED.
midas-hand-tune --mode dexpilot --preset zmz --backend hardware --open \
    --hardware-current-limit 150 \
    --hardware-command-scale 0.3 \
    --hardware-max-step-rad 0.05
```

Then press **Arm hardware** in the browser. Press it again, close the tab, kill
the bridge, or Ctrl-C to stop — all four drop torque.

Why the tuner and not `midas-manus-teleop`: it is the only entry point that
loads a preset, so it is the only one that runs the settings you actually
tuned. `midas-manus-teleop --backend hardware` runs the **analytic** map at
library defaults unless you also pass `--retarget dexpilot`, and even then it
cannot read `zmz`.

### B1. Bus and motors  ☐

```bash
ls -l /dev/serial/by-id/                # the adapter should appear here
python -c "
from midas_hand_api import MidasHand
h = MidasHand()
print('port:', h.port)
print('ping:', h.ping())
print('models:', h.verify_models())
h.close()
"
```

Expected: 13 motors answer, all model 1710. The port is discovered, not
assumed — `--hardware-port` exists but you should not need it.

**`RECORD:` any motor that did not answer = ______________**

### B2. Home the hand  ☐ — **already done**

`~/.midas_hand/config.yaml` exists. Redo it only if the hand has been
disassembled or a motor replaced:

```bash
python -m midas_hand_api --home        # or --home-thumb / --home-fingers
```

Note what homing writes for joint limits: **±π on every joint**, which makes
`MidasHand.clip_positions` a no-op. Commands are therefore bounded by the
retargeter's own model limits, applied in `HardwareBackend._prepare_target`
before anything reaches a motor. That is the only thing standing between a
`--hardware-command-scale` typo and a mechanical stop.

### B3. The PIP four-bar sign  ☐ — **settled, no hardware needed**

This was carried as a stop-gate on the belief that the two packages held
opposite conventions. They do not. The comparison had been made against the
wrong function.

`midas_hand_api` exposes the same lookup table twice:

| function | takes | flips sign |
|---|---|---|
| `pip_to_dip_position` | lookup-space PIP (**positive**) | no |
| `passive_dip_from_pip_motor` | motor-space PIP (**negative in flexion**) | yes |

A hardware angle is motor-space, so the second is the only correct
comparison — and against it `LookupPassiveCoupling` matches to **1e-6** across
the whole range:

| PIP (motor) | −0.20 | −0.50 | −0.90 | −1.20 | −1.45 |
|---|---|---|---|---|---|
| both packages → DIP | −0.4758 | −0.9787 | −1.4509 | −1.7173 | −1.9054 |

Feeding a motor-space (negative) angle to `pip_to_dip_position` falls below the
table's domain and clamps to `0.0`, which is what looked like a sign
disagreement and was really a unit error.

It could not have reached a motor in any case: the finger DIPs are passive
four-bar links with no servo, so they are absent from
`HARDWARE_MOTOR_JOINT_NAMES`. Pinned by
`midas_hand_retargeter/tests/test_coupling_agrees_with_hardware.py`.

### B4. Glove → real hand, low and slow  ☐

```bash
midas-manus-bridge &

# 1. nothing energised: just look at the numbers
midas-hand-tune --mode dexpilot --preset zmz --backend print --duration 20

# 2. sim tracks you, with the tuned profile
midas-hand-tune --mode dexpilot --preset zmz --mujoco-viewer --open

# 3. the hand, disarmed: connects, configures, torque OFF
midas-hand-tune --mode dexpilot --preset zmz --backend hardware --open \
    --hardware-current-limit 150 --hardware-command-scale 0.3 \
    --hardware-max-step-rad 0.05
```

Step 3 starts disarmed. Before pressing **Arm hardware**, check the status bar:
glove connected, a sane rate, `backend HardwareBackend`. Then arm, and check in
this order:

- **no jump on arming.** The first commanded pose is the measured pose by
  construction; if a finger snaps, disarm and report it.
- **direction.** Curl one finger. If the robot extends, stop — that is a sign
  convention, not a tuning problem.
- **cmd vs meas rows line up finger-for-finger.** A mismatch means the
  thumb-first motor order and index-first joint order got crossed — the exact
  bug the name-keyed `measured()` API exists to prevent.
- **disarm actually drops torque** — the fingers should go limp, not merely
  stop updating.

Then raise the limits one at a time, never together:

| flag | bring-up | default | what it means |
|---|---|---|---|
| `--hardware-command-scale` | 0.3 | 1.0 | fraction of the commanded angle |
| `--hardware-max-step-rad` | 0.05 | 0.15 | × 50 Hz = 2.5 rad/s vs 7.5 rad/s |
| `--hardware-current-limit` | 150 | 350 | goal current cap, mA |

**`RECORD:` highest safe command scale = ________**
**`RECORD:` working current limit = ________ mA**
**`RECORD:` working max step = ________ rad**

### B5. Every way of stopping  ☐

All five must drop torque. Test each one deliberately, with the hand armed and
holding a light object:

| action | expected |
|---|---|
| **Disarm** in the browser | limp within one control tick |
| close the browser tab | limp within ~3 s (client watchdog) |
| `pkill -f midas-manus-bridge` | limp within 0.5 s (deadman), and a note in the UI |
| Ctrl-C in the terminal | limp, clean exit |
| `kill <pid>` (SIGTERM) | limp, clean exit |

After the deadman trips, re-arming is **manual** and deliberate: restarting the
bridge does not resume tracking on its own.

**`RECORD:` any of the five that did not drop torque = ______________**

### B6. Retune against the real hand  ☐

The tuner is the same one you used in sim, now driving the hand, so retune in
place and save:

- adjust per-finger ranges and, in dexpilot mode, `scaling_factor`,
  `abduction_limit` and `project_dist` while watching the hand
- press **Capture neutral** with your hand in a flat open pose to set the zero
- save under a new name — the preset now carries that zero pose with it

**`RECORD:` hardware preset name = ______________**

---

## Part C — future: Paxini fingertip haptics

Not started. The glove side is **already complete** — `HAPTIC_TOPIC`,
`haptic_loop` and the `CoreSdk_VibrateFingersForGlove` binding all exist and
re-tick every 16 ms with a 300 ms decay. There is simply no publisher:
`grep -rn '/haptic' /home/dyna/midas` returns exactly one hit, the topic
definition.

Order of work, and why:

1. **Sensor gaps that actually block a loop.** `read_latest()` hands out its
   shared internal dict without a copy; the frame parser / LRC / resync path
   has zero tests and decides whether a corrupt frame becomes a full-power
   buzz; there is no host-side baseline fallback if the board's `_RECALIBRATE`
   silently fails; `reconnect()` cannot reopen its own `exclusive=True` port.
2. **Use the resultant force already on the wire.** Each finger block carries 6
   bytes of int16 Fx/Fy/Fz that the driver parses and discards. That is a
   cheaper and better signal than reducing 283 taxels. Settle its scale and
   endianness with a known-force fit — the block *width* is already pinned by
   `connect()` and needs no experiment.
3. **Feel.** Schmitt hysteresis plus a 60–100 ms minimum-on time, with the
   onset boost latched per contact *episode*. Without it a finger resting at
   the deadband machine-guns the buzz at the command rate.
   Set `publish_rate_hz` to 200–250, **not** 83: promotion is already
   sequence-gated, so matching the board's rate makes the clocks beat and adds
   jitter.
4. **Sim-side source first**, so this is developable with no sensors: extract
   contacts *inside* `MujocoBackend.send()` right after `mj_step`. A separate
   process cannot work (nothing publishes the hand's pose) and `MjData` is not
   thread-safe.
5. **Bridge fixes:** zero-vibration on exit, a monotonic rate limiter on the
   vibrate call (raising the poll timeout does not bound it — poll returns
   early on message arrival), and a real reconnect for the pinned `sender_host`.

The hand has **no pinky**, so element 4 of the 5-float power vector is always 0.

---

## Known limitations, deliberately not fixed

| | why |
|---|---|
| Left hand unsupported | Mirroring does nothing — the analytic map is reflection-invariant (verified max \|Δ\| = 0.0). Real support needs sign-aware splay and a signed thumb opposition, or a left model. |
| Thumb cannot tell palmar from dorsal deviation | `abs()` on the opposition angle. Affects analytic mode only; DexPilot has no such term. |
| Analytic mode cannot place fingertips relative to each other | Structural: it reads angles, not positions. Use `--mode dexpilot` (A5b). |
| DexPilot fingertip offsets are CAD estimates | Load-bearing in that mode; measure on hardware. See A5b. |
| Dual glove untested | See A4. |
| ~~PIP sign disagreement~~ | Resolved: the two packages agree to 1e-6. See B3. |
