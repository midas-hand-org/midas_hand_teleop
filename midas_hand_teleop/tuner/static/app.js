"use strict";
/* MIDAS retarget tuner.
 *
 * The form is generated from /api/schema rather than hand-written, so adding a
 * parameter in Python cannot leave this file silently out of date.
 */

const state = {
  schema: null,
  params: {},       // dotted path -> value
  defaults: {},
  section: null,      // chosen from the schema; modes expose different sections
  telemetry: null,
};

const $ = (id) => document.getElementById(id);
const FINGER_JOINTS = {
  thumb: ["thumb_cmc_roll_joint", "thumb_cmc_side_joint", "thumb_mcp_joint", "thumb_dip_joint"],
  index: ["index_mcp_abad_joint", "index_mcp_pitch_joint", "index_pip_joint"],
  middle: ["middle_mcp_abad_joint", "middle_mcp_pitch_joint", "middle_pip_joint"],
  ring: ["ring_mcp_abad_joint", "ring_mcp_pitch_joint", "ring_pip_joint"],
};
let jointLimits = {};

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function showError(message) {
  const banner = $("banner");
  banner.textContent = message;
  banner.hidden = !message;
}

/* ---------- form ---------- */

function controlValue(path) {
  return state.params[path];
}

function isModified(path) {
  return JSON.stringify(state.params[path]) !== JSON.stringify(state.defaults[path]);
}

async function setParam(path, value) {
  try {
    const payload = await api("/api/profile", { updates: { [path]: value } });
    state.params = payload.parameters;
    showError("");
    renderControls();
  } catch (err) {
    // A rejected edit leaves the live profile untouched; re-render to snap back.
    showError(`Rejected: ${err.message}`);
    renderControls();
  }
}

function makeScalar(control) {
  const wrap = document.createElement("div");
  wrap.className = "control" + (isModified(control.path) ? " modified" : "");
  const value = Number(controlValue(control.path));
  wrap.innerHTML = `
    <div class="head"><span class="name">${control.label}</span>
      <span class="value">${value.toFixed(3)}</span></div>
    <input type="range" min="${control.min}" max="${control.max}" step="${control.step}" value="${value}">
    ${control.help ? `<div class="help">${control.help}</div>` : ""}`;
  const slider = wrap.querySelector("input");
  const readout = wrap.querySelector(".value");
  slider.addEventListener("input", () => { readout.textContent = Number(slider.value).toFixed(3); });
  slider.addEventListener("change", () => setParam(control.path, Number(slider.value)));
  return wrap;
}

function makeBool(control) {
  const wrap = document.createElement("div");
  wrap.className = "control" + (isModified(control.path) ? " modified" : "");
  const on = Boolean(controlValue(control.path));
  wrap.innerHTML = `
    <div class="head"><span class="name">${control.label}</span>
      <span class="value"><input type="checkbox" ${on ? "checked" : ""}></span></div>
    ${control.help ? `<div class="help">${control.help}</div>` : ""}`;
  wrap.querySelector("input").addEventListener("change", (event) =>
    setParam(control.path, event.target.checked));
  return wrap;
}

/* The output-range control. The track is the joint's REAL limit, so the part
 * of it the commanded span does not cover is visibly unused travel — this is
 * how the historical 25% of mcp_pitch and 16% of pip become obvious. */
function makeRange(control) {
  const wrap = document.createElement("div");
  wrap.className = "control" + (isModified(control.path) ? " modified" : "");
  const [open, closed] = controlValue(control.path).map(Number);
  const lo = control.min, hi = control.max;
  const span = hi - lo || 1;
  const pct = (v) => ((v - lo) / span) * 100;
  const a = Math.min(open, closed), b = Math.max(open, closed);
  const usedPct = (Math.abs(closed - open) / span) * 100;

  wrap.innerHTML = `
    <div class="head"><span class="name">${control.label}</span>
      <span class="value">${control.joint}</span></div>
    <div class="rangewrap">
      <div class="track" data-joint="${control.joint}">
        <div class="span" style="left:${pct(a)}%;width:${Math.max(pct(b) - pct(a), 0.5)}%"></div>
        <i class="live" hidden></i>
      </div>
      <div class="inputs">
        <label>open <input type="number" class="open" step="${control.step}" min="${lo}" max="${hi}" value="${open}"></label>
        <label>closed <input type="number" class="closed" step="${control.step}" min="${lo}" max="${hi}" value="${closed}"></label>
        <span class="rom${usedPct > 99 ? " full" : ""}">${usedPct.toFixed(0)}% of ROM</span>
      </div>
      ${control.help ? `<div class="help">${control.help}</div>` : ""}
    </div>`;

  const commit = () => {
    const openValue = Number(wrap.querySelector(".open").value);
    const closedValue = Number(wrap.querySelector(".closed").value);
    setParam(control.path, [openValue, closedValue]);
  };
  wrap.querySelectorAll("input").forEach((input) =>
    input.addEventListener("change", commit));
  return wrap;
}

function renderControls() {
  const host = $("controls");
  const advanced = $("show-advanced").checked;
  const section = state.schema.sections.find((s) => s.name === state.section);
  host.textContent = "";
  if (!section) {
    // e.g. mode=vector, which exposes nothing tunable. Say so rather than
    // rendering an empty panel that looks broken.
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent =
      state.schema.note || "This mode exposes no tunable parameters.";
    host.appendChild(note);
    return;
  }
  for (const control of section.controls) {
    if (control.advanced && !advanced) continue;
    if (control.kind === "bool") host.appendChild(makeBool(control));
    else if (control.kind === "range") host.appendChild(makeRange(control));
    else host.appendChild(makeScalar(control));
  }
}

function renderTabs() {
  const host = $("tabs");
  host.textContent = "";
  for (const section of state.schema.sections) {
    const button = document.createElement("button");
    button.textContent = section.label;
    button.className = section.name === state.section ? "active" : "";
    button.addEventListener("click", () => {
      state.section = section.name;
      renderTabs();
      renderControls();
      renderLive();
    });
    host.appendChild(button);
  }
}

/* ---------- live readout ---------- */

function chip(id, text, level) {
  const element = $(id);
  element.textContent = text;
  element.className = "chip" + (level ? ` ${level}` : "");
}

function renderStatus() {
  const frame = state.telemetry;
  if (!frame || !frame.glove) return;
  const g = frame.glove, l = frame.loop || {};

  chip("chip-glove", `glove ${g.rate_hz ?? 0} Hz`,
       !g.connected ? "bad" : g.stale ? "warn" : "ok");
  // Never show a latency number as trustworthy when its frame is stale.
  chip("chip-latency",
       g.latency_ms == null ? "latency n/a" : `latency ${g.latency_ms} ms`,
       g.latency_ms == null ? "" : g.stale ? "warn" : g.latency_ms > 60 ? "warn" : "ok");
  chip("chip-loop", `loop ${l.rate_hz ?? 0} Hz · ${l.retarget_ms ?? 0} ms`);
  chip("chip-mode", `mode ${l.mode ?? "?"}`);
  const neutral = Object.keys(frame.neutral_offsets ?? {}).length;
  chip("chip-neutral", neutral ? `zero pose · ${neutral} joints` : "zero pose not set",
       neutral ? "ok" : "");
  chip("chip-backend", `backend ${l.backend ?? "?"}${l.armed ? " · ARMED" : ""}`,
       l.armed ? "bad" : "");

  const armButton = $("arm-button");
  armButton.disabled = !l.hardware_available;
  armButton.textContent = l.armed ? "Disarm" : "Arm hardware";
  armButton.classList.toggle("armed", Boolean(l.armed));

  // These are always computed from the landmarks by the analytic map, which in
  // the optimizer modes is a READ of your hand rather than what drives the
  // joints. Saying so, because a readout that looks like a cause but is not is
  // the same trap as a slider that does nothing.
  $("intermediates-hint").textContent = l.mode === "analytic"
    ? "What the analytic map computed before it hit the joint ranges."
    : `A read of your hand from the analytic map. In ${l.mode} these do NOT `
      + "drive the joints, so they explain your pose, not the robot's.";

  $("provenance").textContent =
    l.mode === "analytic"
      ? "All 13 joints come from the analytic map, so every slider here is live."
      : `mode=${l.mode}: the optimizer also writes joints, so a slider may not be the whole story.`;
}

function renderLive() {
  const frame = state.telemetry;
  if (!frame || !frame.commanded) return;
  // Solver modes (dexpilot) have no per-finger tab, so show all 13 joints.
  const joints = FINGER_JOINTS[state.section] || Object.keys(frame.commanded).sort();
  const host = $("joints");
  host.textContent = "";

  for (const joint of joints) {
    const cmd = frame.commanded[joint];
    const meas = frame.measured ? frame.measured[joint] : undefined;
    const limit = jointLimits[joint] || { lower: -1, upper: 1 };
    const span = limit.upper - limit.lower || 1;
    const pct = (v) => Math.max(0, Math.min(100, ((v - limit.lower) / span) * 100));

    const row = document.createElement("div");
    row.className = "jointrow";
    row.innerHTML = `
      <div class="lbl"><span class="n">${joint.replace(/_joint$/, "")}</span>
        <span class="v">${cmd == null ? "—" : cmd.toFixed(3)}${
          meas == null ? "" : ` <span style="color:var(--meas)">/ ${meas.toFixed(3)}</span>`}</span></div>
      <div class="bar">
        <i class="zero" style="left:${pct(0)}%"></i>
        ${cmd == null ? "" : `<i class="cmd" style="left:${pct(cmd)}%"></i>`}
        ${meas == null ? "" : `<i class="meas" style="left:${pct(meas)}%"></i>`}
      </div>`;
    host.appendChild(row);
  }

  // Live command marker inside the matching output-range track.
  document.querySelectorAll(".track[data-joint]").forEach((track) => {
    const joint = track.dataset.joint;
    const value = frame.commanded[joint];
    const marker = track.querySelector(".live");
    const limit = jointLimits[joint];
    if (value == null || !limit) { marker.hidden = true; return; }
    const span = limit.upper - limit.lower || 1;
    marker.hidden = false;
    marker.style.left = `${Math.max(0, Math.min(100, ((value - limit.lower) / span) * 100))}%`;
  });

  const all = frame.intermediates || {};
  const inter = FINGER_JOINTS[state.section]
    ? Object.entries(all[state.section] || {})
    : Object.entries(all).flatMap(([digit, values]) =>
        Object.entries(values || {}).map(([k, v]) => [`${digit}.${k}`, v]));
  $("intermediates").innerHTML = inter
    .map(([k, v]) => `<div class="kv"><span>${k}</span><span>${
      typeof v === "number" ? v.toFixed(4) : v}</span></div>`)
    .join("") || `<div class="hint">no intermediates available</div>`;

  $("log").innerHTML = (frame.messages || []).slice(-6).reverse()
    .map((m) => `<li>${m}</li>`).join("");
}

/* ---------- wiring ---------- */

function connectStream() {
  const source = new EventSource("/api/stream");
  source.onmessage = (event) => {
    state.telemetry = JSON.parse(event.data);
    renderStatus();
    renderLive();
  };
  source.onerror = () => {
    // A page reload or a brief drop should not look like a failure; the browser
    // reconnects on its own, and the loop keeps running regardless.
    chip("chip-loop", "loop reconnecting…", "warn");
  };
}

async function refreshPresets() {
  const { presets } = await api("/api/presets");
  $("preset-list").innerHTML =
    `<option value="">—</option>` + presets.map((p) => `<option>${p}</option>`).join("");
}

async function main() {
  state.schema = await api("/api/schema");
  state.defaults = state.schema.defaults;
  jointLimits = Object.fromEntries(state.schema.joints.map((j) => [j.name, j]));
  const profile = await api("/api/profile");
  state.params = profile.parameters;
  // The default tab used to be hardcoded to "index", which does not exist in
  // dexpilot mode — the schema is mode-specific, so take the first section it
  // actually offers.
  state.section = state.schema.sections.length
    ? state.schema.sections[0].name
    : null;

  renderTabs();
  renderControls();
  connectStream();
  await refreshPresets();

  $("show-advanced").addEventListener("change", renderControls);
  for (const [id, path] of [["undo", "/api/profile/undo"], ["redo", "/api/profile/redo"], ["reset", "/api/profile/reset"]]) {
    $(id).addEventListener("click", async () => {
      state.params = (await api(path, {})).parameters;
      renderControls();
    });
  }
  // Hand-size calibration only exists for the solver mode that uses it.
  const scaleMode = state.schema.mode === "dexpilot";
  $("cal-scale-row").hidden = !scaleMode;
  $("cal-scale-hint").hidden = !scaleMode;
  $("cal-scale").addEventListener("click", async () => {
    await api("/api/calibrate", { action: "scale" });
    const profile = await api("/api/profile");
    state.params = profile.parameters;
    renderControls();
  });
  $("cal-capture").addEventListener("click", () => api("/api/calibrate", { action: "capture" }));
  $("cal-clear").addEventListener("click", () => api("/api/calibrate", { action: "clear" }));
  $("preset-save").addEventListener("click", async () => {
    const name = $("preset-name").value.trim();
    if (!name) return showError("Enter a preset name first.");
    try {
      // The server takes the neutral calibration from the live loop state, not
      // from here: the page is only ever told about one on a preset load, so
      // echoing it back wrote {} over a freshly captured zero pose.
      await api("/api/presets/save", { name });
      showError("");
      await refreshPresets();
    } catch (err) { showError(err.message); }
  });
  $("preset-load").addEventListener("click", async () => {
    const name = $("preset-list").value;
    if (!name) return;
    try {
      const payload = await api("/api/presets/load", { name });
      state.params = payload.parameters;
      showError("");
      renderControls();
    } catch (err) { showError(err.message); }
  });
  $("arm-button").addEventListener("click", async () => {
    const armed = !(state.telemetry?.loop?.armed);
    if (armed && !confirm("Arm the real hand? It will start tracking your glove.")) return;
    try { await api("/api/arm", { armed }); showError(""); }
    catch (err) { showError(err.message); }
  });
}

main().catch((err) => showError(`Failed to start: ${err.message}`));
