"""Ledger — one row per batch, the audit trail for a generation mission.

Tracks each batch from creation (make-batches) through generation (run / manual paste) to
ingest (valid/invalid counts). Backed by a plain CSV so it's git-diffable and inspectable.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

FIELDS = ["batch_id", "task", "style", "backend", "n_seeds", "status",
          "created", "received", "ingested", "valid", "invalid"]


class Ledger:
    """CSV-backed batch ledger, keyed by batch_id."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(newline="") as f:
                for row in csv.DictReader(f):
                    self.rows[row["batch_id"]] = row

    def upsert(self, batch_id: str, **fields) -> None:
        row = self.rows.get(batch_id, {k: "" for k in FIELDS})
        row["batch_id"] = batch_id
        row.update({k: str(v) for k, v in fields.items()})
        self.rows[batch_id] = row

    def get(self, batch_id: str) -> dict | None:
        return self.rows.get(batch_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for bid in sorted(self.rows):
                w.writerow({k: self.rows[bid].get(k, "") for k in FIELDS})

    @staticmethod
    def today() -> str:
        return date.today().isoformat()
