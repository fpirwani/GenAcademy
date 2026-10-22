from pathlib import Path

import pytest
from langchain_core.tools import ToolException
from streamlit.testing.v1 import AppTest
from test_app import setup_app
from test_tools import upload
from test_workflows import FakeModel

from storage import DocumentGone, Store
from tools import build_tools


def test_clear_tool_requires_confirmation_and_bound_current_document(tmp_path):
    store = Store(tmp_path)
    doc = upload(store)
    store.write(doc["id"], doc["generation"], "chat", [{"role": "user", "content": "Saved"}])
    with pytest.raises(ToolException):
        build_tools(store, document=doc)["clear_qa_history"].invoke({})
    tool = build_tools(store, document=doc, clear_qa_confirmed=True)["clear_qa_history"]
    assert tool.args == {}
    assert tool.invoke({}) == {"status": "cleared", "document_id": doc["id"]}
    assert store.read(doc["id"], "chat") == []
    assert store.read(doc["id"], "chunks")
    payload = store.original(doc["id"]).read_bytes()
    store.delete(doc["id"])
    new, _ = store.add("planets.pdf", payload)
    store.write(new["id"], new["generation"], "chat", ["New history"])
    with pytest.raises(DocumentGone):
        tool.invoke({})
    assert store.read(new["id"], "chat") == ["New history"]
    with pytest.raises(ToolException):
        build_tools(store, clear_qa_confirmed=True)["clear_qa_history"].invoke({})


def test_clear_history_ui_preserves_document_and_other_chat(tmp_path, monkeypatch):
    store, doc = setup_app(tmp_path, monkeypatch)
    history = [{"role": "user", "content": "Old question"}]
    store.write(doc["id"], doc["generation"], "chat", history)
    other, _ = store.add("other.pdf", b"other content")
    store.write(other["id"], other["generation"], "chat", history)
    before = {
        p.name: p.read_bytes()
        for p in store.folder(doc["id"]).iterdir()
        if p.is_file() and p.name not in {"chat.json", "metadata.json"}
    }
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.radio[0].set_value(doc["id"]).run()
    quote_key = f"location_{doc['id']}_{doc['generation']}_quote"
    app.session_state[quote_key] = "An old citation"

    def click(label):
        next(b for b in app.button if b.label == label).click().run()

    click("Clear Q&A history")
    assert store.read(doc["id"], "chat") == history
    click("Keep history")
    assert store.read(doc["id"], "chat") == history
    click("Clear Q&A history")
    click("Clear permanently")
    assert not app.exception
    assert not app.chat_message
    assert store.read(doc["id"], "chat") == []
    assert store.read(other["id"], "chat") == history
    assert quote_key not in app.session_state
    for name, data in before.items():
        assert (store.folder(doc["id"]) / name).read_bytes() == data
    assert next(b for b in app.button if b.label == "Clear Q&A history").disabled


def test_clear_during_generation_does_not_restore_history(tmp_path):
    store = Store(tmp_path)
    doc = upload(store)

    class ClearDuringAnswer(FakeModel):
        def answer(self, question, chunks):
            store.clear_chat(doc["id"], doc["generation"])
            return super().answer(question, chunks)

    result = build_tools(store, document=doc, model_factory=ClearDuringAnswer)[
        "qa_document"
    ].invoke({"question": "Mercury"})
    assert result["status"] == "error"
    assert "cleared" in result["answer"]
    assert store.read(doc["id"], "chat") == []
    fresh = build_tools(store, document=doc, model_factory=FakeModel)["qa_document"]
    assert fresh.invoke({"question": "Mercury"})["status"] == "answered"
    assert len(store.read(doc["id"], "chat")) == 2


def test_stale_clear_cannot_touch_reuploaded_document(tmp_path):
    store = Store(tmp_path)
    doc, _ = store.add("file.pdf", b"content")
    store.delete(doc["id"])
    new, _ = store.add("file.pdf", b"content")
    store.write(new["id"], new["generation"], "chat", ["new history"])
    with pytest.raises(DocumentGone):
        store.clear_chat(doc["id"], doc["generation"])
    assert store.read(new["id"], "chat") == ["new history"]
