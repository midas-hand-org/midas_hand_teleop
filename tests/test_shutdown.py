"""Cleanup must survive a second Ctrl-C.

The failure this guards against: a second Ctrl-C during teardown skipped the
remaining steps, so the MuJoCo viewer was never closed, and the process then
deadlocked in glfw.terminate() at interpreter exit — where Python signal
handlers no longer run, making it immune to further Ctrl-C.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from midas_hand_teleop.shutdown import close_quietly, protected_shutdown


def test_close_quietly_reports_success():
    assert close_quietly(logging.getLogger("t"), "ok", lambda: None) is True


@pytest.mark.parametrize(
    "boom",
    [
        pytest.param(KeyboardInterrupt, id="KeyboardInterrupt"),
        pytest.param(SystemExit, id="SystemExit"),
        pytest.param(RuntimeError, id="RuntimeError"),
    ],
)
def test_close_quietly_swallows_even_base_exceptions(boom):
    """KeyboardInterrupt and SystemExit are not Exception subclasses.

    Catching only Exception is exactly why teardown used to be abandoned
    half-done.
    """

    def raiser():
        raise boom("nope")

    assert close_quietly(logging.getLogger("t"), "boom", raiser) is False


def test_later_steps_still_run_after_an_interrupted_one():
    log = logging.getLogger("t")
    done = []

    def interrupted():
        raise KeyboardInterrupt

    close_quietly(log, "first", interrupted)
    close_quietly(log, "second", lambda: done.append("second"))
    close_quietly(log, "third", lambda: done.append("third"))
    assert done == ["second", "third"], "one bad step must not skip the rest"


def test_protected_shutdown_absorbs_an_interrupt_and_restores_the_handler():
    original = signal.getsignal(signal.SIGINT)
    log = logging.getLogger("t")
    completed = []

    with protected_shutdown(log):
        os.kill(os.getpid(), signal.SIGINT)  # would normally raise here
        completed.append("survived")

    assert completed == ["survived"]
    assert signal.getsignal(signal.SIGINT) is original


def test_second_interrupt_forces_exit():
    """A genuinely stuck teardown must not hold the terminal hostage."""

    script = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        sys.path.insert(0, %r)
        from midas_hand_teleop.shutdown import protected_shutdown
        logging.basicConfig(level=logging.WARNING)
        with protected_shutdown(logging.getLogger("t")):
            os.kill(os.getpid(), signal.SIGINT)   # absorbed, with a hint
            os.kill(os.getpid(), signal.SIGINT)   # forces os._exit
            time.sleep(5)
            print("NOT REACHED")
        """
    ) % os.getcwd()

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 130, result.stderr
    assert "NOT REACHED" not in result.stdout
    assert "force quit" in result.stderr
