# Changelog

## 0.2.0 (unreleased)

**Milestone: glove → real hand works end to end.** A Manus Haptic Pro glove
drives all 13 motors of the physical MIDAS hand through DexPilot retargeting,
run from the browser tuner, with the profile that was tuned in simulation.

### Added

- **`midas-hand-tune` drives hardware.** `--backend hardware` plus the *Arm
  hardware* button in the browser. It is the only entry point that loads a
  tuned preset, so it is the one that runs the settings you actually tuned.
- **`--preset` on `midas-manus-teleop`** too, for the headless path. It also
  installs the zero pose the preset was saved with.
- **A shared arm gate and three independent ways to drop torque**: the browser
  switch, a 0.5 s glove deadman (re-arming is manual, so a flapping link cannot
  reconnect into tracking), and a 3 s client watchdog for a closed tab or a
  slept laptop. SIGTERM and the second-Ctrl-C force-quit both drop torque too.
- **`abduction_limit`** on the DexPilot profile, and the browser controls for
  every solver knob the running mode actually reads.
- A `webcam` extra, so the glove path no longer pulls 252 MB of opencv and
  mediapipe it never imports.

### Fixed

- **A dropped startup sync read could have driven every motor ~177°.** It
  returns cached data — zeros — which map to about −3.09 rad on twelve of
  thirteen motors, and that was written unslewed before torque was enabled.
  `last_read_ok` is now checked and retried, and refuses to arm otherwise.
- **Nothing bounded a command to the mechanism.** Homing writes ±π joint
  limits, which makes the hand API's own clamp a no-op, and two URDF limits
  reach past the real hard stops. Commands are now clamped to the URDF limits
  intersected with the stops the homing table implies.
- **`midas-manus-teleop` commanded the all-zeros pose before any glove frame**,
  for the first `--stale-timeout` seconds — uncommanded motion at power-on.
- `arm()`/`disarm()` drove the serial bus from the caller's thread while the
  50 Hz command thread was reading it; both now go through that thread.
- `HardwareBackend(start_armed=...)` defaulted to `True`, so constructing one
  in a REPL enabled torque.
- A non-finite target reached `set_positions` unchecked (`np.clip` propagates
  NaN); the frame is now dropped and the previous target held.
- The glove skeleton was published mirrored, which the analytic map cannot see
  — it is reflection-invariant — but which asked DexPilot to bend the fingers
  backwards.
- The Manus SDK lookup the README documented did not exist; `--sdk-lib`,
  `$MANUS_SDK_LIB` and `$MANUS_SDK_DIR` now work, and a missing SDK exits
  nonzero instead of looking like a clean start.
- `midas-hand-teleop --backend hardware` could never energise the hand: its
  duplicated argument group had no `--start-armed`.
- The tuner's zero-pose calibration never reached the preset it was saved into.
- The Cartesian modes were not getting the four-bar coupling, because the CLI
  default overrode the mode-dependent one.

### Removed

- Four CLI flags the code itself labelled deprecated, six duplicated
  `DEFAULT_HARDWARE_*` constants, and `mediapipe`/`opencv-python` as required
  dependencies.
