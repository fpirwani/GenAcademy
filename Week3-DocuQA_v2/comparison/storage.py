"""Ordered version sessions, atomic persistence and revision protection."""

import hashlib
import json
import re
from datetime import datetime, timezone
from time import time_ns

from storage import DocumentGone


class ComparisonStore:
    def __init__(self, store):
        self.store = store
        self.root = store.root / "comparisons"
        self.root.mkdir(exist_ok=True)

    def path(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise ValueError("Invalid comparison ID")
        path = self.root / (identifier + ".json")
        if path.is_symlink():
            raise ValueError("Unsafe comparison path")
        return path

    def check(self, session):
        for doc in session["documents"]:
            current = self.store.read(doc["id"], "metadata")
            if current["generation"] != doc["generation"]:
                raise DocumentGone("A comparison version was replaced. Select the versions again.")

    def open(self, earlier, later):
        if earlier["id"] == later["id"]:
            raise ValueError("Select two different document versions.")
        docs = [dict(earlier), dict(later)]
        identifier = hashlib.sha256(
            json.dumps([(d["id"], d["generation"]) for d in docs]).encode()
        ).hexdigest()
        with self.store.lock:
            session = {
                "id": identifier,
                "documents": docs,
                "revision": time_ns(),
                "chat": [],
                "report": None,
                "created": datetime.now(timezone.utc).isoformat(),
            }
            self.check(session)
            if self.path(identifier).exists():
                return self.read(identifier)
            self.store._save(self.path(identifier), session)
            return session

    def read(self, identifier):
        with self.store.lock:
            path = self.path(identifier)
            if not path.exists():
                raise DocumentGone("This comparison is no longer available.")
            session = json.loads(path.read_text(encoding="utf-8"))
            self.check(session)
            return session

    def list(self):
        result = []
        with self.store.lock:
            for path in self.root.glob("*.json"):
                try:
                    result.append(self.read(path.stem))
                except DocumentGone:
                    path.unlink(missing_ok=True)
                except (ValueError, OSError):
                    continue
        return sorted(result, key=lambda s: s["created"], reverse=True)

    def save_result(self, snapshot, result, question=None):
        with self.store.lock:
            session = self.read(snapshot["id"])
            if session["revision"] != snapshot["revision"]:
                raise ValueError("Comparison history was cleared while processing. Please retry.")
            if snapshot.get("fresh"):
                session["chat"] = []
                session["report"] = None
            if question is None:
                session["report"] = result
                session["chat"] = []
                session["created"] = datetime.now(timezone.utc).isoformat()
            else:
                session["chat"].extend(
                    [{"role": "user", "content": question}, {"role": "assistant", "result": result}]
                )
            self.store._save(self.path(session["id"]), session)
            if question is None:
                # Save the replacement first. A failed run/write must preserve the old report.
                pair = {(d["id"], d["generation"]) for d in session["documents"]}
                for other in self.list():
                    other_pair = {(d["id"], d["generation"]) for d in other["documents"]}
                    if other["id"] != session["id"] and other_pair == pair:
                        self.path(other["id"]).unlink(missing_ok=True)

    def clear(self, identifier):
        with self.store.lock:
            session = self.read(identifier)
            session["chat"] = []
            session["revision"] += 1
            self.store._save(self.path(identifier), session)

    def delete(self, identifier):
        """Remove only this saved report and conversation, preserving source files."""
        with self.store.lock:
            self.path(identifier).unlink(missing_ok=True)
