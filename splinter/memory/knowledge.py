from __future__ import annotations

import re
from pathlib import Path

from splinter.memory.session import Session


class KnowledgeStore:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.dir = session.knowledge_dir()

    def write_note(self, topic: str, md: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", topic)
        p = self.dir / f"{safe}.md"
        p.write_text(md)
        return p

    def list_notes(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.md"))

    def read_note(self, topic: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", topic)
        p = self.dir / f"{safe}.md"
        if p.exists():
            return p.read_text()
        return ""

    def query(self, q: str) -> list[str]:
        q_lower = q.lower()
        matches: list[str] = []
        for note_path in sorted(self.dir.glob("*.md")):
            text = note_path.read_text()
            headers = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
            searchable = f"{note_path.stem} {' '.join(headers)}".lower()
            if q_lower in searchable:
                matches.append(note_path.stem)
                continue
            for word in q_lower.split():
                if len(word) > 2 and word in searchable:
                    matches.append(note_path.stem)
                    break
        return matches
