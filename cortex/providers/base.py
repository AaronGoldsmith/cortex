"""Provider base — shared types and protocol for ingestion providers."""

from __future__ import annotations

import sqlite3
from typing import Iterator, NamedTuple, Optional, Protocol


class IngestEntry(NamedTuple):
    """Normalized entry ready for insert_entry()."""

    content: str
    entry_type: str  # "raw", "observation", etc.
    source_model: str  # "claude", "goose/chatgpt_codex", etc.
    source_project: Optional[str]
    session_id: Optional[str]
    confidence: float
    turn_index: Optional[int]


class IngestProvider(Protocol):
    """Protocol that all ingestion providers must satisfy."""

    name: str

    def detect(self) -> bool:
        """Return True if this provider's data source exists on this machine."""
        ...

    def iter_entries(self, conn: sqlite3.Connection) -> Iterator[IngestEntry]:
        """Yield normalized entries. Provider manages its own cursor/state."""
        ...

    @property
    def metadata(self) -> dict:
        """Return provider info: data paths, format, etc."""
        ...
