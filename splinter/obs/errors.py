"""Dedicated ``errors.jsonl`` channel for observability failures (US-003).

Observability must never crash a run — but a failure must be *visible*, not
silently swallowed. Every obs write/read/decode failure is recorded as one JSON
line in ``{session.dir}/errors.jsonl`` instead of a bare ``except: pass``.

The reporter is itself best-effort: if even the error write fails it degrades to
a stderr log and never raises back into the caller, preserving the invariant
that telemetry problems can't abort the run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splinter.memory.session import Session

_log = logging.getLogger(__name__)


def report_obs_error(
    session: Session,
    source: str,
    op: str,
    exc: BaseException,
    *,
    detail: str = "",
) -> None:
    """Record one observability failure to ``errors.jsonl``; never raise.

    Args:
        session: session whose dir holds the ``errors.jsonl`` stream.
        source: dotted origin, e.g. ``"agentic.append_jsonl"``.
        op: operation that failed, e.g. ``"write"`` / ``"read"`` / ``"decode"``.
        exc: the caught exception (its type + message are recorded).
        detail: optional extra context (file path, skipped-count, …).
    """
    record: dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "op": op,
        "error": f"{type(exc).__name__}: {exc}",
    }
    if detail:
        record["detail"] = detail
    try:
        session._ensure_dir()
        with open(session.dir / "errors.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as report_exc:  # noqa: BLE001 — last-resort, must not raise
        _log.warning(
            "obs error report failed (%s/%s): %s; original: %s",
            source,
            op,
            report_exc,
            record["error"],
        )
