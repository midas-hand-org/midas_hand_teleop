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

Expected: **110 passed** and **68 passed**. Was 29 + 49 before this work.

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
midas-hand-tune --mode dexpilot --open
```

**Calibrate `scaling_factor` FIRST — nothing else matters until it is right.**
It is your hand size relative to the robot's, and it is the one parameter the
analytic map never had. A 0.7x-1.5x change moves joints by ~1.5 rad.

Estimate it before touching the slider:

```bash
python - <<'EOF'
import numpy as np
from midas_hand_retargeter import MidasHandRetargeter
r = MidasHandRetargeter.create(mode="dexpilot")
rb = r.dex_retargeting.optimizer.robot
rb.compute_forward_kinematics(np.zeros(19))
inv = np.linalg.inv(rb.get_link_pose(rb.get_link_index("palm_base")))
for n, link in (("index","index_tip"),("middle","middle_tip"),("ring","ring_tip")):
    p = (inv @ rb.get_link_pose(rb.get_link_index(link)))[:3,3]
    print(f"robot {n}: {np.linalg.norm(p)*1000:.0f} mm")
EOF
```

The MIDAS hand reaches ~218 mm from `palm_base` to fingertip with the hand
open. Measure your own wrist-to-fingertip with a ruler, then start at
`robot / yours` — about **1.2 for a 180 mm hand, 1.55 for a 140 mm hand**. The
inherited default of 1.15 is almost certainly too low.

**`RECORD:` my wrist-to-fingertip = ________ mm → starting scaling = ________**

Then, on the slider:

- **too low** → fingers over-curl and rail into their limits, and the pose
  stops responding to your hand at all. This is the most common failure.
- **too high** → fingers never close far enough.
- Open your hand fully: the sim should be open, not slightly curled.
  Close it: it should close without saturating early.

**`RECORD:` final scaling_factor = ________**

Only then touch the rest: `project_dist` (the gap at which a fingertip pair
snaps together — raise it if pinches hover, lower it if fingers stick to each
other), `eta1` (how close a snapped thumb-finger pinch gets), and `norm_delta`
(smoother but laggier).

Then A/B it honestly against `--mode analytic` on the same motion. Analytic is
6x cheaper and rock-solid for curl; DexPilot is the one that gets relative
fingertip placement right. Which you want depends on the task.

**`TODO(hardware-day)` fingertip offsets.** DexPilot aims at the tip frames in
`urdf.TIP_LINKS`, which are CAD estimates (`thumb_tip` at `0 -0.042 -0.010`,
fingers at `0 0.036 -0.009`). In this mode they are load-bearing — a wrong
offset means the solver optimises toward the wrong point. Measure them on the
real hand and correct them.

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

Do these strictly in order. Each is a stop-gate: if one fails, stop.

### B1. Bus and motors  ☐

```bash
cd midas_hand_api
./setup_dynamixel_latency.sh          # needs sudo; sets the FTDI latency timer to 1
python -c "
from midas_hand_api import MidasHand
h = MidasHand()
print('ping:', h.ping())
print('models:', h.verify_models())
h.close()
"
```

Expected: 13 motors answer, all model 1710.
**`RECORD:` any motor that did not answer = ______________**

### B2. Home the hand  ☐  ← **blocks everything below**

`~/.midas_hand/config.yaml` does not exist on this machine, so the hand has
never been homed here. Until it does, motor zero has no defined relationship to
URDF zero and the API's joint-limit clamp is a no-op — which is why
`--backend hardware` refuses to start.

```bash
python -m midas_hand_api --home        # or --home-thumb / --home-fingers
ls -l ~/.midas_hand/config.yaml
```

**`RECORD:` homing offsets written = ______________**

### B3. Settle the PIP four-bar sign  ☐  ← **`TODO(hardware-day)`**

The two packages disagree, both self-consistently:

| | convention |
|---|---|
| `midas_hand_retargeter.coupling.LookupPassiveCoupling` | `pip_to_lookup_sign = -1.0` |
| `midas_hand_api.kinematics.pip_to_dip_position` | no sign flip |

The retargeter's convention matches a settled MJCF loop to 3.8e-3 rad, but the
hardware path uses the other one. Measure, do not guess:

```bash
python -c "
from midas_hand_api import MidasHand
import numpy as np, time
h = MidasHand(); h.configure()
q = np.zeros(13); q[4] = -0.90          # index PIP, motor index 4
h.set_positions_blocking(q, timeout_s=3)
time.sleep(0.5)
print('measured 16-DOF joint vector:', np.round(h.read_joint_pos(), 4))
h.shutdown()
"
```

Compare the measured **index DIP** against the two predictions:

- retargeter convention → **−1.4509 rad**
- api convention → run `pip_to_dip_position(-0.90)` and record it

**`RECORD:` measured index DIP at PIP −0.90 = ________ rad**
**`RECORD:` which convention matches = ______________**

Then make it one shared implementation — do not leave two, and do not add a
third. Also note the lookup table's domain is `[0, 1.3963]` while the URDF PIP
range is `[-1.45, 0]`, so the last 0.054 rad is clamped.

### B4. Glove → real hand, low and slow  ☐

```bash
# 1. targets look sane, nothing energised
midas-manus-teleop --backend print --duration 10

# 2. sim tracks you
midas-manus-teleop --backend mujoco --mujoco-viewer --duration 30

# 3. hardware, disarmed: connects, configures, torque OFF
midas-manus-teleop --backend hardware \
    --hardware-current-limit 150 --hardware-command-scale 0.3 --duration 20

# 4. hardware, armed
midas-manus-teleop --backend hardware \
    --hardware-current-limit 150 --hardware-command-scale 0.3 --start-armed
```

Check, in order:

- **no jump on arming.** The first commanded pose is the measured pose by
  construction; if a finger snaps, stop and report it.
- **disarm actually drops torque** — the fingers should go limp, not just stop
  updating.
- **command scale.** Raise 0.3 → 0.5 → 1.0 only once each step looks right.
  **`RECORD:` highest safe command scale = ________**
- **current limit.** Raise from 150 mA only as needed.
  **`RECORD:` working current limit = ________ mA**

### B5. Deadman  ☐

Mid-session, kill the bridge:

```bash
pkill -f midas-manus-bridge
```

Expected within 0.5 s: `No glove data for 0.5s (> --stale-timeout 0.5s) —
holding position and disarming if armed.` The hand must stop commanding, not
keep leaning on whatever it was touching.

### B6. Tuner against hardware  ☐

```bash
midas-hand-tune --open        # then use the Arm button
```

- **cmd vs meas rows line up finger-for-finger.** A mismatch here means the
  thumb-first motor order and index-first joint order got crossed — the exact
  bug the name-keyed `measured()` API exists to prevent.
- Retune the per-finger ranges against the real hand and save a hardware preset.
  **`RECORD:` hardware preset name = ______________**

**`TODO(hardware-day)`** the tuner currently offers only `print` and `mujoco`
backends; wiring its `--backend hardware` is a small follow-up once B4 passes.

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
| PIP sign disagreement | See B3. |
