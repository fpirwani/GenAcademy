import pytest
from langchain_core.tools import BaseTool, ToolException
from test_documents import pdf_bytes
from test_workflows import FakeModel

from storage import DocumentGone, Store
from tools import build_tools


def upload(store):
    return build_tools(store, upload=("planets.pdf", pdf_bytes()))["upload_document"].invoke({})[
        "document"
    ]


def test_upload_tool_indexes_and_reopens_without_losing_chat(tmp_path):
    store = Store(tmp_path)
    payload = pdf_bytes()
    tools = build_tools(store, upload=("planets.pdf", payload))
    assert all(isinstance(item, BaseTool) for item in tools.values())
    first = tools["upload_document"].invoke({})
    doc = first["document"]
    assert first["created"]
    assert len(store.read(doc["id"], "chunks")) == 2
    store.write(doc["id"], doc["generation"], "chat", [{"role": "user", "content": "Saved"}])
    again = build_tools(store, upload=("renamed.pdf", payload))["upload_document"].invoke({})
    assert not again["created"]
    assert again["document"]["id"] == doc["id"]
    assert store.read(doc["id"], "chat")[0]["content"] == "Saved"
    chunks = store.read(doc["id"], "chunks")
    for chunk in chunks:
        chunk.pop("chunk_version", None)
    store.write(doc["id"], doc["generation"], "chunks", chunks)
    upgraded = tools["upload_document"].invoke({})
    assert upgraded["document"]["generation"] == doc["generation"]
    assert all(c["chunk_version"] == "structure-v2" for c in store.read(doc["id"], "chunks"))
    assert store.read(doc["id"], "chat")[0]["content"] == "Saved"
    (store.folder(doc["id"]) / "chunks.json").unlink()
    tools["upload_document"].invoke({})
    assert len(store.read(doc["id"], "chunks")) == 2


def test_summary_and_qa_tools_persist_results_for_selected_file(tmp_path):
    store = Store(tmp_path)
    doc = upload(store)
    tools = build_tools(store, document=doc, model_factory=FakeModel)
    summary = tools["summarize_document"].invoke({})
    assert summary["status"] == "ready"
    assert store.read(doc["id"], "summary")["text"] == summary["text"]
    answer = tools["qa_document"].invoke({"question": "Mercury"})
    assert answer["status"] == "answered"
    assert answer["sources"][0]["location"] == 1
    history = store.read(doc["id"], "chat")
    assert len(history) == 2
    assert history[1]["sources"] == answer["sources"]


def test_model_failure_does_not_save_a_factual_answer(tmp_path):
    class Offline(FakeModel):
        def answer(self, *args):
            raise ConnectionError("offline")

        def summarize(self, *args):
            raise ConnectionError("offline")

    store = Store(tmp_path)
    doc = upload(store)
    tools = build_tools(store, document=doc, model_factory=Offline)
    assert tools["qa_document"].invoke({"question": "Mercury"})["status"] == "error"
    assert store.read(doc["id"], "chat") == []
    with pytest.raises(ConnectionError):
        tools["summarize_document"].invoke({})
    assert store.read(doc["id"], "metadata")["status"] == "indexed"


def test_delete_tool_requires_app_confirmation_and_rejects_stale_context(tmp_path):
    store = Store(tmp_path)
    doc = upload(store)
    with pytest.raises(ToolException):
        build_tools(store, document=doc)["delete_document"].invoke({})
    assert store.read(doc["id"], "metadata")
    tools = build_tools(store, document=doc, delete_confirmed=True)
    assert tools["delete_document"].invoke({})["status"] == "deleted"
    assert store.list() == []


def test_upload_retry_cannot_resurrect_a_deleted_document(tmp_path):
    store = Store(tmp_path)
    payload = pdf_bytes()
    doc = build_tools(store, upload=("planets.pdf", payload))["upload_document"].invoke({})[
        "document"
    ]
    retry = build_tools(store, document=doc, upload=("planets.pdf", payload))["upload_document"]
    store.delete(doc["id"])
    with pytest.raises(DocumentGone):
        retry.invoke({})
    assert store.list() == []


def test_old_tools_cannot_access_or_delete_reuploaded_generation(tmp_path):
    store = Store(tmp_path)
    payload = pdf_bytes()
    doc = build_tools(store, upload=("planets.pdf", payload))["upload_document"].invoke({})[
        "document"
    ]
    tools = build_tools(store, document=doc, delete_confirmed=True, model_factory=FakeModel)
    store.delete(doc["id"])
    new = build_tools(store, upload=("planets.pdf", payload))["upload_document"].invoke({})[
        "document"
    ]
    assert new["generation"] != doc["generation"]
    for name, args in [
        ("delete_document", {}),
        ("summarize_document", {}),
        ("qa_document", {"question": "Mercury"}),
    ]:
        with pytest.raises(DocumentGone):
            tools[name].invoke(args)
    assert store.read(new["id"], "metadata")["generation"] == new["generation"]


def test_document_and_upload_context_are_not_model_arguments(tmp_path):
    tools = build_tools(Store(tmp_path))
    assert set(tools["qa_document"].args) == {"question"}
    assert tools["upload_document"].args == {}
    assert tools["delete_document"].args == {}
    with pytest.raises(ToolException):
        tools["upload_document"].invoke({})


def test_duplicate_repair_deleted_during_extraction_stays_deleted(tmp_path):
    store = Store(tmp_path)
    payload = pdf_bytes()
    doc, _ = store.add("planets.pdf", payload)
    deleted = False

    def deleted_elsewhere(_):
        nonlocal deleted
        if not deleted:
            store.delete(doc["id"])
            deleted = True

    repair = build_tools(store, upload=("planets.pdf", payload), progress=deleted_elsewhere)
    with pytest.raises(DocumentGone):
        repair["upload_document"].invoke({})
    assert store.list() == []


def test_keyword_fallback_warning_is_saved_with_answer(tmp_path):
    class OfflineEmbeddings:
        @property
        def fingerprint(self):
            raise ConnectionError("offline")

    store = Store(tmp_path)
    doc = upload(store)
    result = build_tools(
        store, document=doc, model_factory=FakeModel, embedding_factory=OfflineEmbeddings
    )["qa_document"].invoke({"question": "Mercury"})
    assert result["status"] == "answered"
    assert "keyword" in result["warning"]
    assert store.read(doc["id"], "chat")[-1]["warning"] == result["warning"]
