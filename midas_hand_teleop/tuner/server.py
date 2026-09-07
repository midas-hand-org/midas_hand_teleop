"""Stdlib HTTP + SSE server for the tuning UI.

No web framework, no npm, no CDN: the page is vanilla HTML/CSS/JS served from
package data. That keeps the dependency set honest for an open-source release
(nothing to license-audit) and lets the tuner run in an air-gapped robot cell.

Threading model: ThreadingHTTPServer, so a long-lived SSE stream cannot block
parameter POSTs. Handlers only touch ``TunerState``; they never call into the
control loop directly, so a slow browser can never stall the hand.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from midas_hand_retargeter import presets

from .schema import build_schema

logger = logging.getLogger(__name__)

#: Only these files are ever served, by exact name. Not a directory walk.
STATIC_FILES = {"index.html", "app.js", "style.css"}

#: Telemetry frames per second pushed to the browser. Well under the control
#: rate on purpose: the UI needs to look live, not to receive every frame.
STREAM_HZ = 20.0


def _static_path(name: str) -> Path | None:
    if name not in STATIC_FILES:
        return None
    resource = resources.files("midas_hand_teleop.tuner").joinpath("static", name)
    path = Path(str(resource))
    return path if path.is_file() else None


class TunerRequestHandler(BaseHTTPRequestHandler):
    server_version = "MidasTuner/1.0"

    # Injected by make_server.
    state = None
    preset_dir = presets.DEFAULT_PRESET_DIR

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # --- helpers -------------------------------------------------------
    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _profile_payload(self) -> dict:
        profile = self.state.profile
        return {
            "parameters": profile.to_flat_dict(),
            # Read by the page to enable/disable its Undo and Redo buttons.
            "can_undo": self.state.store.can_undo,
            "can_redo": self.state.store.can_redo,
        }

    # --- routing -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        route = urlparse(self.path)
        path = route.path
        try:
            if path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/api/schema":
                # Mode-specific: the browser must not render controls the
                # running mode ignores.
                return self._send_json(build_schema(mode=self.state.loop.mode))
            if path == "/api/profile":
                return self._send_json(self._profile_payload())
            if path == "/api/telemetry":
                return self._send_json(self.state.snapshot())
            if path == "/api/presets":
                return self._send_json(
                    {"presets": [p.stem for p in presets.list_presets(self.preset_dir)]}
                )
            if path == "/api/stream":
                return self._stream()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("GET %s failed", path)
            return self._send_json({"error": str(exc)}, status=500)
        self._send_json({"error": f"no such endpoint: {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/profile":
                self.state.apply_updates(body.get("updates") or {})
                return self._send_json(self._profile_payload())
            if path == "/api/profile/undo":
                self.state.store.undo()
                return self._send_json(self._profile_payload())
            if path == "/api/profile/redo":
                self.state.store.redo()
                return self._send_json(self._profile_payload())
            if path == "/api/profile/reset":
                self.state.store.reset()
                return self._send_json(self._profile_payload())
            if path == "/api/presets/save":
                return self._save_preset(body)
            if path == "/api/presets/load":
                return self._load_preset(body)
            if path == "/api/calibrate":
                self.state.request_calibration(body.get("action", "capture"))
                return self._send_json({"ok": True})
            if path == "/api/arm":
                return self._set_armed(bool(body.get("armed")))
        except (KeyError, ValueError, TypeError) as exc:
            # A rejected parameter edit is a client error, and the live profile
            # is untouched — that is the contract the UI relies on.
            # KeyError.__str__ wraps its message in repr quotes; unwrap it so
            # the browser shows the sentence, not "'...'".
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return self._send_json({"error": str(message)}, status=400)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("POST %s failed", path)
            return self._send_json({"error": str(exc)}, status=500)
        self._send_json({"error": f"no such endpoint: {path}"}, status=404)

    # --- handlers ------------------------------------------------------
    def _serve_static(self, name: str) -> None:
        path = _static_path(name)
        if path is None:
            return self._send_json({"error": "not found"}, status=404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _preset_name(body: dict) -> str:
        """Validate a preset name from the browser.

        Presets are addressed by bare name and joined onto ``preset_dir``, so
        this is the only thing keeping a request from walking out of that
        directory. Deliberately NOT ``presets.resolve``: that accepts paths on
        purpose, for the command line, where the caller is the operator.
        """

        name = str(body.get("name") or "").strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Preset name must be a simple filename")
        return name

    def _save_preset(self, body: dict) -> None:
        name = self._preset_name(body)
        profile = self.state.profile
        path = presets.save(
            Path(self.preset_dir) / f"{name}.json",
            # dataclasses.replace, not a field-by-field rebuild: the previous
            # version omitted dexpilot=, so every solver knob an operator tuned
            # was silently reset to defaults on save, under a "saved" message.
            replace(profile, name=name),
            # From the live state, not the body: see TunerState.neutral_offsets.
            neutral_offsets=dict(self.state.neutral_offsets),
        )
        self.state.note(f"saved preset {name}")
        self._send_json({"ok": True, "path": str(path)})

    def _load_preset(self, body: dict) -> None:
        name = self._preset_name(body)
        profile, neutral = presets.load(Path(self.preset_dir) / f"{name}.json")
        self.state.store.set(profile)
        # A preset is meant to be a complete, reproducible artifact, so its
        # zero-pose calibration has to be installed too, not just displayed.
        self.state.request_neutral_offsets(neutral)
        self.state.note(f"loaded preset {name}")
        self._send_json({**self._profile_payload(), "neutral_offsets": neutral})

    def _set_armed(self, armed: bool) -> None:
        if armed and not self.state.loop.hardware_available:
            raise ValueError("No hardware backend is attached; nothing to arm.")
        self.state.loop.armed = armed
        self.state.note("ARMED" if armed else "disarmed")
        self._send_json({"armed": self.state.loop.armed})

    def _stream(self) -> None:
        """Server-sent telemetry. Ends quietly when the browser goes away."""

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        period = 1.0 / STREAM_HZ
        try:
            while not getattr(self.server, "shutting_down", False):
                payload = json.dumps(self.state.snapshot())
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError):
            pass  # normal: the tab was closed or reloaded


def make_server(state, *, host: str = "127.0.0.1", port: int = 8765, preset_dir=None):
    """Build (but do not start) the tuner HTTP server."""

    handler = type(
        "BoundTunerRequestHandler",
        (TunerRequestHandler,),
        {"state": state, "preset_dir": preset_dir or presets.DEFAULT_PRESET_DIR},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.shutting_down = False
    return server


def serve_in_background(state, *, host: str = "127.0.0.1", port: int = 8765, preset_dir=None):
    """Start the server on a daemon thread and return it."""

    server = make_server(state, host=host, port=port, preset_dir=preset_dir)
    thread = threading.Thread(
        target=server.serve_forever, name="midas-tuner-http", daemon=True
    )
    thread.start()
    return server
