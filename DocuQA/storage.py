"""Atomic local JSON storage. No persisted model or graph memory outside a document."""

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from filelock import FileLock


class DocumentGone(RuntimeError):
    """The selected generation was deleted or replaced."""


class Store:
    names = {"metadata", "blocks", "chunks", "summary", "chat"}

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.docs = self.root / "documents"
        self.docs.mkdir(parents=True, exist_ok=True)
        # One app-wide lock: no per-document lock files remain after deletion.
        self.lock = FileLock(str(self.root / ".store.lock"), timeout=10)

    def folder(self, doc_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", doc_id):
            raise ValueError("Invalid document ID")
        path = self.docs / doc_id
        if path.resolve().parent != self.docs or path.is_symlink():
            raise ValueError("Unsafe document path")
        return path

    @staticmethod
    def _save(path: Path, value):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _metadata(self, doc_id: str):
        folder = self.folder(doc_id)
        if (folder / ".deleting").exists() or not (folder / "metadata.json").exists():
            raise DocumentGone("This document was deleted. Select another file.")
        return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    def add(self, filename: str, payload: bytes):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise ValueError("Choose a PDF or DOCX file.")
        doc_id = hashlib.sha256(payload).hexdigest()
        with self.lock:
            folder = self.folder(doc_id)
            if (folder / ".deleting").exists():
                raise DocumentGone("Finish deleting this file before uploading it again.")
            if (folder / "metadata.json").exists():
                return self._metadata(doc_id), False
            folder.mkdir(exist_ok=True)
            (folder / ("original" + suffix)).write_bytes(payload)
            metadata = {
                "id": doc_id,
                "generation": uuid4().hex,
                "filename": Path(filename.replace("\\", "/")).name,
                "suffix": suffix,
                "size": len(payload),
                "status": "uploaded",
                "created": datetime.now(timezone.utc).isoformat(),
            }
            self._save(folder / "chat.json", [])
            self._save(folder / "metadata.json", metadata)
            return metadata, True

    def list(self):
        result = []
        with self.lock:
            for folder in self.docs.iterdir():
                if not folder.is_dir() or folder.is_symlink():
                    continue
                if (folder / ".deleting").exists():
                    result.append(
                        {
                            "id": folder.name,
                            "filename": "Pending deletion",
                            "status": "deleting",
                            "created": "",
                        }
                    )
                    continue
                try:
                    result.append(self._metadata(folder.name))
                except (DocumentGone, ValueError, OSError):
                    continue
        return sorted(result, key=lambda item: item["created"], reverse=True)

    def read(self, doc_id: str, name: str, default=None):
        if name not in self.names:
            raise ValueError("Invalid record name")
        with self.lock:
            self._metadata(doc_id)
            path = self.folder(doc_id) / f"{name}.json"
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def write(self, doc_id: str, generation: str, name: str, value):
        if name not in self.names or name == "metadata":
            raise ValueError("Invalid record name")
        with self.lock:
            if self._metadata(doc_id)["generation"] != generation:
                raise DocumentGone("This document was replaced. Please reopen it.")
            self._save(self.folder(doc_id) / f"{name}.json", value)

    def update(self, doc_id: str, generation: str, **values):
        with self.lock:
            metadata = self._metadata(doc_id)
            if metadata["generation"] != generation:
                raise DocumentGone("This document was replaced.")
            if not set(values) <= {"status", "error", "block_count"}:
                raise ValueError("Invalid metadata update")
            metadata.update(values)
            self._save(self.folder(doc_id) / "metadata.json", metadata)

    def clear_chat(self, doc_id: str, generation: str):
        """Clear one conversation and invalidate answers already in flight."""
        with self.lock:
            metadata = self._metadata(doc_id)
            if metadata["generation"] != generation:
                raise DocumentGone("This document was replaced. Please reopen it.")
            metadata["chat_revision"] = metadata.get("chat_revision", 0) + 1
            self._save(self.folder(doc_id) / "metadata.json", metadata)
            self._save(self.folder(doc_id) / "chat.json", [])

    def append_chat(self, doc_id: str, generation: str, messages: list, *, revision=None):
        with self.lock:
            metadata = self._metadata(doc_id)
            if metadata["generation"] != generation:
                raise DocumentGone("This document was replaced. Please reopen it.")
            if revision is not None and metadata.get("chat_revision", 0) != revision:
                return False
            history = self.read(doc_id, "chat", [])
            self.write(doc_id, generation, "chat", history + messages)
            return True

    def original(self, doc_id: str) -> Path:
        with self.lock:
            metadata = self._metadata(doc_id)
            return self.folder(doc_id) / ("original" + metadata["suffix"])

    def delete(self, doc_id: str):
        with self.lock:
            folder = self.folder(doc_id)
            if not folder.exists():
                return
            (folder / ".deleting").touch()
            # Leave tombstone until every other entry has gone, allowing retry.
            for path in folder.iterdir():
                if path.name == ".deleting":
                    continue
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            (folder / ".deleting").unlink()
            folder.rmdir()
