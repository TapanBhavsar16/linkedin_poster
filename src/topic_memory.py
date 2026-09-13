"""Persistent memory of topics posted recently."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_FILE_LOCK = Lock()


def normalize_topic(topic: str) -> str:
    """Create a stable comparison key for a topic."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", topic.casefold())).strip()


class TopicMemory:
    """Store successfully posted topics and retain only the last seven days."""

    def __init__(self, path: str | Path | None = None, retention_days: int = 7) -> None:
        configured_path = path or os.getenv("POSTED_TOPICS_MEMORY_FILE", "data/posted_topics.json")
        self.path = Path(configured_path)
        self.retention = timedelta(days=retention_days)

    def recent(self, now: datetime | None = None) -> list[dict[str, str | None]]:
        """Return recent entries, pruning expired or malformed entries on read."""
        current = now or datetime.now(timezone.utc)
        entries = self._read()
        cutoff = current - self.retention
        recent: list[dict[str, str | None]] = []
        for entry in entries:
            try:
                posted_at = datetime.fromisoformat(str(entry["posted_at"]).replace("Z", "+00:00"))
                if posted_at.tzinfo is None:
                    posted_at = posted_at.replace(tzinfo=timezone.utc)
                topic = str(entry["topic"]).strip()
                if topic and posted_at >= cutoff:
                    recent.append({
                        "topic": topic,
                        "posted_at": posted_at.astimezone(timezone.utc).isoformat(),
                        "post_id": entry.get("post_id"),
                    })
            except (KeyError, TypeError, ValueError):
                continue
        self._write(recent)
        return recent

    def contains(self, topic: str, now: datetime | None = None) -> bool:
        """Return whether a normalized topic was posted during the retention window."""
        key = normalize_topic(topic)
        return bool(key) and any(normalize_topic(str(item["topic"])) == key for item in self.recent(now))

    def record(self, topic: str, post_id: str | None = None, now: datetime | None = None) -> None:
        """Record a topic after LinkedIn confirms the post was created."""
        topic = topic.strip()
        if not topic:
            return
        current = now or datetime.now(timezone.utc)
        entries = self.recent(current)
        if not any(normalize_topic(str(item["topic"])) == normalize_topic(topic) for item in entries):
            entries.append({
                "topic": topic,
                "posted_at": current.astimezone(timezone.utc).isoformat(),
                "post_id": post_id,
            })
            self._write(entries)

    def _read(self) -> list[dict[str, Any]]:
        with _FILE_LOCK:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return []
        return value if isinstance(value, list) else []

    def _write(self, entries: list[dict[str, Any]]) -> None:
        with _FILE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(entries, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                os.replace(temporary, self.path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
