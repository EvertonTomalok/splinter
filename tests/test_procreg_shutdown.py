"""procreg.shutdown: no new children after it, every descendant signalled."""

from __future__ import annotations

import signal

import pytest

from splinter import procreg


def test_shutdown_kills_descendants_and_refuses_new_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[int, int]] = []
    alive = [[101, 102], [102], [102]]
    monkeypatch.setattr(procreg, "_descendants", lambda _root: alive.pop(0) if alive else [102])
    monkeypatch.setattr(procreg.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(procreg.time, "sleep", lambda _s: None)
    try:
        procreg.shutdown(grace=0.01)
        assert (101, signal.SIGTERM) in sent and (102, signal.SIGTERM) in sent
        assert (102, signal.SIGKILL) in sent
        with pytest.raises(procreg.ShuttingDownError):
            procreg.run_subprocess(["true"], timeout=1)
    finally:
        procreg._shutting_down.clear()
        procreg.clear_stop()
