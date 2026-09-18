import uuid
from datetime import datetime, timezone


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProvenanceLog:
    """Ordered, timestamped record of every action the agent takes, so the
    final report can show exactly how each conclusion was produced."""

    def __init__(self):
        self.entries = []

    def log(self, stage: str, action: str, details: dict = None):
        self.entries.append({
            "timestamp": now_iso(),
            "stage": stage,
            "action": action,
            "details": details or {},
        })

    def as_list(self) -> list:
        return list(self.entries)