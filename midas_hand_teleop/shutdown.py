"""Cleanup that a second Ctrl-C cannot half-abort.

Why this exists
---------------
A single Ctrl-C raises KeyboardInterrupt in the control loop, which is caught
and starts cleanup. A *second* Ctrl-C during that cleanup raises again, this
time inside the ``finally`` block, which aborts the remaining steps.

That is not merely untidy. If the MuJoCo viewer never gets closed, the process
then reaches interpreter shutdown with the GLFW render loop still running, and
``mujoco.viewer`` registers ``glfw.terminate`` with :mod:`atexit`. On Wayland
``glfw.terminate()`` deadlocks against the live render loop, and by then the
main thread is inside a C call during finalization, so Python signal handlers
no longer run — further Ctrl-C does nothing and the process can only be killed
with SIGKILL/SIGABRT.

Observed thread dump of exactly that state::

    Current thread (main): glfw/__init__.py:832 in terminate
    Thread: mujoco/viewer.py:525 in _launch_internal   # render loop still live

So cleanup runs with SIGINT neutralised, each step is isolated, and a second
Ctrl-C force-exits with ``os._exit`` — which deliberately skips the atexit
handler that would otherwise hang.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import threading

#: Conventional exit status for "terminated by Ctrl-C".
SIGINT_EXIT_CODE = 130


@contextlib.contextmanager
def protected_shutdown(
    logger: logging.Logger,
    *,
    exit_code: int = SIGINT_EXIT_CODE,
    before_force_exit=None,
):
    """Run a cleanup block that Ctrl-C cannot interrupt halfway.

    The first Ctrl-C inside the block is absorbed with a hint; a second one
    force-exits immediately rather than letting a stuck GUI teardown hold the
    terminal hostage.

    ``before_force_exit`` runs on that force path, before ``os._exit``. Pass
    the thing that MUST happen even when the operator is hammering Ctrl-C --
    dropping motor torque. ``os._exit`` skips atexit, which is where
    ``midas_hand_api`` registers its own torque-off, so without this an
    impatient operator can leave a hand energised.

    Only effective on the main thread — :func:`signal.signal` raises anywhere
    else — so it degrades to a no-op rather than failing.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    presses = 0

    def on_interrupt(signum, frame):  # noqa: ARG001 - signal handler signature
        nonlocal presses
        presses += 1
        if presses == 1:
            logger.warning("Shutting down — press Ctrl-C again to force quit.")
        else:
            logger.warning("Forcing exit; skipping remaining cleanup.")
            if before_force_exit is not None:
                # Suppress everything: this is the last chance to drop torque,
                # and a failure here must not stop the exit.
                with contextlib.suppress(BaseException):
                    before_force_exit()
            # os._exit, not sys.exit: this must not run atexit handlers, since
            # glfw.terminate is registered there and is what hangs.
            os._exit(exit_code)

    try:
        previous = signal.signal(signal.SIGINT, on_interrupt)
    except (ValueError, OSError):  # pragma: no cover - non-main thread
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGINT, previous)


def close_quietly(logger: logging.Logger, label: str, close) -> bool:
    """Run one cleanup step; never let it prevent the next one.

    Catches ``BaseException`` on purpose. ``KeyboardInterrupt`` and
    ``SystemExit`` are not ``Exception`` subclasses, and those are precisely
    the ones that used to skip the rest of the teardown.

    Returns whether the step succeeded, so a caller can decide it is safe to
    skip the interpreter's own shutdown.
    """

    try:
        close()
        return True
    except BaseException as exc:  # noqa: BLE001 - deliberate, see docstring
        logger.warning("Ignoring error while closing %s: %r", label, exc)
        return False


def exit_without_atexit(logger: logging.Logger, code: int = 0) -> None:
    """Exit immediately, skipping :mod:`atexit`.

    Call this ONLY once our own cleanup has fully succeeded.

    MuJoCo's viewer registers ``glfw.terminate`` with atexit, and on Wayland
    that teardown crashes or deadlocks — verified with a 20-line reproducer
    using MuJoCo and GLFW alone, no code from this project involved. The result
    is exit status 139 (or a hang) long after all real work is done, which
    scripts and CI read as a failure.

    The reason this is safe here and not in general: the atexit handler that
    genuinely matters is ``midas_hand_api``'s, which disables motor torque, and
    ``HardwareBackend.close()`` has already done that explicitly. If any
    cleanup step failed we fall through to a normal exit instead, so that
    safety net still runs.
    """

    logger.debug("Exiting without atexit to avoid the GLFW/Wayland teardown.")
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)


@contextlib.contextmanager
def sigterm_as_interrupt(logger: logging.Logger):
    """Make SIGTERM take the same path as Ctrl-C.

    Python's default SIGTERM action terminates the process outright: no
    ``finally``, no ``atexit``, so motor torque stays on. That matters because
    ``kill`` is the documented way to stop these loops, and ``pkill -f`` is in
    the runbook. Raising KeyboardInterrupt instead routes it through the
    cleanup that already exists.

    Main thread only, like every ``signal.signal`` caller; a no-op elsewhere.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_terminate(signum, frame):  # noqa: ARG001 - signal handler signature
        logger.warning("SIGTERM received — shutting down.")
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, on_terminate)
    except (ValueError, OSError):  # pragma: no cover - non-main thread
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGTERM, previous)
