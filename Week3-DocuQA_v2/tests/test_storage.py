import pytest

from storage import DocumentGone, Store


def test_duplicate_reopens_and_restart_preserves_chat(tmp_path):
    store = Store(tmp_path)
    doc, created = store.add("notes.pdf", b"original bytes")
    assert created
    store.write(doc["id"], doc["generation"], "chat", [{"role": "user", "content": "Hello"}])
    again, created = Store(tmp_path).add("renamed.pdf", b"original bytes")
    assert not created
    assert again["id"] == doc["id"]
    assert Store(tmp_path).read(doc["id"], "chat")[0]["content"] == "Hello"
    assert store.original(doc["id"]).read_bytes() == b"original bytes"


def test_delete_removes_all_data_and_rejects_late_writes(tmp_path):
    store = Store(tmp_path)
    doc, _ = store.add("notes.pdf", b"one")
    other, _ = store.add("other.pdf", b"two")
    store.write(doc["id"], doc["generation"], "chunks", [{"text": "private"}])
    store.delete(doc["id"])
    assert not (tmp_path / "documents" / doc["id"]).exists()
    assert [d["id"] for d in store.list()] == [other["id"]]
    with pytest.raises(DocumentGone):
        store.write(doc["id"], doc["generation"], "chat", [])
    new, _ = store.add("notes.pdf", b"one")
    assert new["generation"] != doc["generation"]
    with pytest.raises(DocumentGone):
        store.write(doc["id"], doc["generation"], "summary", {"text": "stale"})


def test_paths_cannot_escape_storage(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.delete("../outside")
    doc, _ = store.add("../../notes.pdf", b"one")
    assert store.original(doc["id"]).parent == tmp_path / "documents" / doc["id"]
    with pytest.raises(ValueError):
        store.write(doc["id"], doc["generation"], "../outside", {})


def test_failed_delete_stays_unavailable_until_retry(tmp_path, monkeypatch):
    from pathlib import Path

    store = Store(tmp_path)
    doc, _ = store.add("notes.pdf", b"one")
    original_unlink = Path.unlink

    def blocked_unlink(path, *args, **kwargs):
        if path.name == "original.pdf":
            raise PermissionError("File is in use")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", blocked_unlink)
        with pytest.raises(PermissionError):
            store.delete(doc["id"])
    assert store.list()[0]["status"] == "deleting"
    with pytest.raises(DocumentGone):
        store.read(doc["id"], "chat")
    with pytest.raises(DocumentGone):
        store.write(doc["id"], doc["generation"], "chat", [])
    store.delete(doc["id"])
    assert store.list() == []
