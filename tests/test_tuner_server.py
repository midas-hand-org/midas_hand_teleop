"""The tuner's HTTP surface, exercised against a real server on a real socket.

These run with no glove, no sim and no hardware: the server only ever talks to
TunerState, which is the property that makes the tuner CI-able.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from midas_hand_retargeter.params import RetargetProfile
from midas_hand_retargeter.store import ProfileStore

from midas_hand_teleop.tuner.schema import build_schema
from midas_hand_teleop.tuner.server import make_server
from midas_hand_teleop.tuner.state import TunerState


@pytest.fixture()
def server(tmp_path):
    import threading

    state = TunerState(store=ProfileStore(RetargetProfile()))
    httpd = make_server(state, port=0, preset_dir=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield state, f"http://{host}:{port}"
    finally:
        httpd.shutting_down = True
        httpd.shutdown()
        httpd.server_close()


def get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as response:
        return json.loads(response.read())


def post(base, path, payload):
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_schema_describes_four_digits_and_no_pinky():
    schema = build_schema()
    names = [section["name"] for section in schema["sections"]]
    assert names == ["thumb", "index", "middle", "ring"]
    assert "pinky" not in json.dumps(schema)


def test_range_controls_are_bounded_by_real_joint_limits():
    schema = build_schema()
    controls = {
        control["path"]: control
        for section in schema["sections"]
        for control in section["controls"]
    }
    pitch = controls["index.mcp_pitch_range"]
    assert (pitch["min"], pitch["max"]) == (-1.8, 0.0)
    assert controls["index.pip_range"]["max"] == 0.0
    assert controls["index.pip_range"]["min"] == -1.45


def test_static_assets_are_served_by_allowlist(server):
    _, base = server
    with urllib.request.urlopen(f"{base}/", timeout=5) as response:
        assert b"MIDAS Retarget Tuner" in response.read()
    for name in ("app.js", "style.css"):
        with urllib.request.urlopen(f"{base}/static/{name}", timeout=5) as response:
            assert response.status == 200

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{base}/static/passwd", timeout=5)
    assert excinfo.value.code == 404


def test_parameter_edit_applies_and_is_readable(server):
    state, base = server
    post(base, "/api/profile", {"updates": {"index.curl_gain": 1.75}})
    assert state.profile.index.curl_gain == 1.75
    assert get(base, "/api/profile")["parameters"]["index.curl_gain"] == 1.75
    # ...and only that finger moved.
    assert state.profile.middle.curl_gain == 1.0


def test_rejected_edit_is_a_400_and_leaves_the_profile_untouched(server):
    state, base = server
    post(base, "/api/profile", {"updates": {"index.curl_gain": 1.5}})

    for bad in ({"pinky.curl_gain": 1.0}, {"index.curl_scale": 1.0}, {"nodot": 1.0}):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(base, "/api/profile", {"updates": bad})
        assert excinfo.value.code == 400
        body = json.loads(excinfo.value.read())
        assert not body["error"].startswith("'"), "KeyError repr leaked to the UI"

    assert state.profile.index.curl_gain == 1.5


def test_undo_redo_reset(server):
    state, base = server
    post(base, "/api/profile", {"updates": {"ring.splay_gain": 3.0}})
    assert post(base, "/api/profile/undo", {})["parameters"]["ring.splay_gain"] == 1.2
    assert post(base, "/api/profile/redo", {})["parameters"]["ring.splay_gain"] == 3.0
    assert post(base, "/api/profile/reset", {})["parameters"]["ring.splay_gain"] == 1.2


def test_presets_round_trip_over_http(server, tmp_path):
    state, base = server
    post(base, "/api/profile", {"updates": {"thumb.cmc_roll_gain": 0.9}})
    post(base, "/api/presets/save", {"name": "unit", "neutral_offsets": {}})
    assert (tmp_path / "unit.json").exists()
    assert "unit" in get(base, "/api/presets")["presets"]

    post(base, "/api/profile/reset", {})
    payload = post(base, "/api/presets/load", {"name": "unit"})
    assert payload["parameters"]["thumb.cmc_roll_gain"] == 0.9


def test_preset_names_cannot_escape_the_preset_directory(server):
    _, base = server
    for name in ("../evil", "a/b", ".hidden"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(base, "/api/presets/save", {"name": name})
        assert excinfo.value.code == 400


def test_arming_is_refused_without_hardware(server):
    state, base = server
    assert state.loop.hardware_available is False
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(base, "/api/arm", {"armed": True})
    assert excinfo.value.code == 400
    assert state.loop.armed is False


def test_calibration_is_queued_for_the_control_thread(server):
    state, base = server
    post(base, "/api/calibrate", {"action": "capture"})
    assert state.take_calibration_request() == "capture"
    assert state.take_calibration_request() is None

    with pytest.raises(urllib.error.HTTPError):
        post(base, "/api/calibrate", {"action": "explode"})


def test_telemetry_reports_staleness_rather_than_lying(server):
    state, base = server
    state.glove.connected = True
    state.glove.latency_ms = 12.0
    state.glove.age_s = 3.0  # publisher died three seconds ago
    state.publish(commanded={}, measured={}, intermediates={})

    glove = get(base, "/api/telemetry")["glove"]
    assert glove["stale"] is True
    assert glove["latency_ms"] == 12.0, "the number is kept, but flagged stale"


def test_schema_is_mode_aware():
    """The UI must not render controls the running mode ignores.

    In dexpilot mode the per-finger analytic parameters do nothing at all, and
    vice versa. Showing them anyway is how ~20 dead CLI flags accumulated in
    webcam_demo.
    """

    analytic = build_schema(mode="analytic")
    dexpilot = build_schema(mode="dexpilot")
    vector = build_schema(mode="vector")

    assert [s["name"] for s in analytic["sections"]] == [
        "thumb",
        "index",
        "middle",
        "ring",
    ]
    assert [s["name"] for s in dexpilot["sections"]] == ["dexpilot"]
    assert vector["sections"] == []
    assert vector["note"], "a mode with no tunables must say so, not look broken"


def test_dexpilot_schema_leads_with_scaling_factor():
    """It is the load-bearing knob, so it must not be hidden behind 'advanced'."""

    controls = {
        c["name"]: c for s in build_schema(mode="dexpilot")["sections"] for c in s["controls"]
    }
    assert controls["scaling_factor"]["advanced"] is False
    assert "hand size" in controls["scaling_factor"]["help"].lower()
    # 9 solver knobs plus the thumb reach, finger spread and abduction bound.
    assert len(controls) == 12
    assert controls["thumb_vector_scale"]["advanced"] is True
    # Finger spread is a calibrated primary knob, not an advanced one.
    assert controls["spread_scale"]["advanced"] is False
    # The abduction bound is the fix for fingers leaning sideways on a curl,
    # so it must be reachable without expanding 'advanced'.
    assert controls["abduction_limit"]["advanced"] is False
    assert controls["abduction_limit"]["min"] == 0.0
    # Pinch snapping must be switchable off; its floor used to be 5 mm.
    assert controls["project_dist"]["min"] == 0.0
    # The two pinch knobs must point at the right symptoms. The old text sent
    # an operator with sticky fingertips to project_dist, and lowering that
    # disables the snap instead of releasing it -- so a pinch then lands ~26 mm
    # open, which is exactly what happened.
    assert "too big a gap" in controls["project_dist"]["help"]
    assert "stickiness knob" in controls["escape_dist"]["help"]


@pytest.mark.parametrize("mode", ["analytic", "dexpilot", "refine", "vector"])
def test_every_mode_yields_a_renderable_schema(mode):
    """The page picks its first tab from the schema, so that contract matters.

    The UI used to default its selected tab to "index", which does not exist
    in dexpilot mode — the page died with "Cannot read properties of undefined
    (reading 'controls')". Either there is a first section to select, or there
    is a note explaining why there is nothing to tune.
    """

    schema = build_schema(mode=mode)
    assert schema["mode"] == mode

    if schema["sections"]:
        first = schema["sections"][0]
        assert first["name"] and first["controls"], "a section must be renderable"
        for section in schema["sections"]:
            for control in section["controls"]:
                assert control["path"].startswith(f"{section['name']}.")
                assert control["kind"] in {"scalar", "range", "bool"}
    else:
        assert schema["note"], "an empty schema must explain itself, not look broken"


def test_defaults_cover_every_control_the_ui_can_render():
    """The page reads defaults[path] to mark a control as modified."""

    for mode in ("analytic", "dexpilot"):
        schema = build_schema(mode=mode)
        for section in schema["sections"]:
            for control in section["controls"]:
                assert control["path"] in schema["defaults"], control["path"]


def test_saving_a_preset_keeps_the_solver_knobs(server):
    """Regression: the save path rebuilt the profile field by field and omitted
    `dexpilot=`, so every DexPilot solver knob an operator tuned was reset to
    defaults on save — under a "saved preset" confirmation."""

    state, base = server
    post(
        base,
        "/api/profile",
        {
            "updates": {
                "dexpilot.scaling_factor": 1.33,
                "dexpilot.thumb_vector_scale": 1.15,
                "index.curl_gain": 1.7,
            }
        },
    )
    post(base, "/api/presets/save", {"name": "solver"})
    post(base, "/api/profile/reset", {})

    loaded = post(base, "/api/presets/load", {"name": "solver"})["parameters"]
    assert loaded["dexpilot.scaling_factor"] == 1.33
    assert loaded["dexpilot.thumb_vector_scale"] == 1.15
    assert loaded["index.curl_gain"] == 1.7


def test_loading_a_preset_cannot_escape_the_preset_directory(server):
    """Load validates its name exactly as save does. It joins the name onto
    preset_dir, so this check is the only thing keeping a request inside it."""

    _, base = server
    for name in ("../escape", "..", ".hidden", "a/b", ""):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post(base, "/api/presets/load", {"name": name})
        assert excinfo.value.code == 400, name


def test_saving_a_preset_captures_the_live_zero_pose(server, tmp_path):
    """The browser is only ever TOLD about a neutral on a preset load, so
    echoing its copy back wrote {} over a freshly captured zero pose."""

    state, base = server
    state.neutral_offsets = {"index_pip_joint": -0.12, "thumb_mcp_joint": -0.34}
    post(base, "/api/presets/save", {"name": "zero"})

    saved = json.loads((tmp_path / "zero.json").read_text())
    assert saved["neutral_offsets"] == {"index_pip_joint": -0.12, "thumb_mcp_joint": -0.34}


def test_loading_a_preset_queues_its_zero_pose_for_the_loop(server):
    """A preset is a complete artifact, so its calibration must be installed,
    not merely displayed. The loop applies it; the server may not touch the
    retargeter."""

    state, base = server
    state.neutral_offsets = {"index_pip_joint": -0.12}
    post(base, "/api/presets/save", {"name": "zero"})
    state.neutral_offsets = {}

    post(base, "/api/presets/load", {"name": "zero"})
    assert state.take_neutral_offsets() == {"index_pip_joint": -0.12}
    assert state.take_neutral_offsets() is None, "consumed exactly once"


def test_the_profile_payload_carries_the_history_state(server):
    """The page has Undo and Redo buttons and the payload has always carried
    can_undo/can_redo, but nothing read them -- so both buttons stayed enabled
    with an empty history, offering an action that does nothing."""

    state, base = server
    fresh = get(base, "/api/profile")
    assert fresh["can_undo"] is False
    assert fresh["can_redo"] is False

    edited = post(base, "/api/profile", {"updates": {"index.curl_gain": 1.4}})
    assert edited["can_undo"] is True

    undone = post(base, "/api/profile/undo", {})
    assert undone["can_redo"] is True

    # And nothing else: keys with no reader are not published.
    assert set(fresh) == {"parameters", "can_undo", "can_redo"}
