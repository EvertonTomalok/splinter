"""Incremental run entries from an append-only session event log."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from splinter.memory.session import Session
from splinter.obs.events import decode_run_entries, events_path, migrate_legacy

if TYPE_CHECKING:
    from splinter.obs.trace import RunEntry

log = logging.getLogger(__name__)


class RunEntryReader:
    def __init__(self) -> None:
        self.entries: list[RunEntry] = []
        self.revision = 0
        self._path: Path | None = None
        self._signature: tuple[int, ...] = ()
        self._offset = 0
        self._tail = b""

    def read(self, session: Session) -> list[RunEntry]:
        path = events_path(session)
        if self._path != path:
            self._reset()
            self._path = path
            migrate_legacy(session)
        try:
            stat = path.stat()
        except FileNotFoundError:
            if self._signature:
                self._reset()
            return self.entries
        signature = self._file_signature(stat)
        if signature == self._signature:
            return self.entries
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            signature = self._file_signature(stat)
            previous = self._signature
            if previous and (
                signature[:2] != previous[:2]
                or stat.st_size < previous[2]
                or (stat.st_size == previous[2] and signature != previous)
            ):
                self._reset()
            if self._tail:
                stream.seek(self._offset - len(self._tail))
                if stream.read(len(self._tail)) != self._tail:
                    self._reset()
            stream.seek(self._offset)
            added = decode_run_entries(self._records(stream, stat.st_size))
        if added:
            self.entries.extend(added)
            self.revision += 1
        self._signature = signature
        return self.entries

    def _records(self, stream: BinaryIO, size: int) -> Iterator[dict[str, Any]]:
        malformed = 0
        offset = self._offset
        tail = self._tail
        while offset < size:
            line = stream.readline(size - offset)
            if not line.endswith(b"\n"):
                break
            offset += len(line)
            tail = (tail + line[-256:])[-256:]
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("event must be an object")
            except ValueError:
                malformed += 1
                continue
            if record.get("type") == "run":
                yield record
        if malformed:
            log.warning("events: skipped %d malformed line(s) in %s", malformed, self._path)
        self._offset = offset
        self._tail = tail

    def _reset(self) -> None:
        self.entries = []
        self.revision += 1
        self._signature = ()
        self._offset = 0
        self._tail = b""

    @staticmethod
    def _file_signature(stat: os.stat_result) -> tuple[int, ...]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
