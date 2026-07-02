"""JSONL audit trail logger per pipeline run."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from core.config import AUDIT_DIR


class AuditLogger:
    """Append-only JSONL audit trail keyed by run_id."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_file = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, event: str, agent: str, details: dict = None, **kwargs) -> None:
        """Append one event record to the JSONL file.

        Args:
            event: Short event name, e.g. 'started', 'completed', 'failed'.
            agent: Name of the agent or component emitting this event.
            details: Optional dict of extra context fields.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": event,
        }
        if details:
            entry.update(details)
        entry.update(kwargs)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_logs(self) -> list:
        """Return all records for this run as a list of dicts."""
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
