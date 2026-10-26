from pathlib import Path

from streamlit.testing.v1 import AppTest

import config
import models


def test_workspace_empty_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models, "model_status", lambda: (False, "Start Ollama"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert [tab.label for tab in app.tabs] == ["Summary", "Compare"]
    app.session_state["workspace_tabs"] = "Compare"
    app.run()
    assert not app.exception
    assert any("two" in item.value.lower() for item in app.info)
    app.session_state["workspace_tabs"] = "Summary"
    app.run()
    assert not app.exception
    assert any("Upload a PDF" in item.value for item in app.info)


def test_pair_selection_and_report_without_question(pair, monkeypatch):
    from test_comparison_workflow import ScriptedModel

    import comparison.ui as ui

    store, docs = pair
    monkeypatch.setattr(config, "DATA_DIR", store.root)
    monkeypatch.setattr(models, "model_status", lambda: (True, "Ready"))
    original = ui.run_comparison
    monkeypatch.setattr(
        ui,
        "run_comparison",
        lambda *a, **kw: original(*a, **kw, model=ScriptedModel(), semantic=False),
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.session_state["workspace_tabs"] = "Compare"
    app.run()
    assert not app.exception
    assert app.selectbox(key="earlier_version").label == "Document One"
    assert app.selectbox(key="later_version").label == "Document Two"
    assert not app.slider
    assert not any(e.label == "Analysis settings" for e in app.expander)
    next(b for b in app.button if b.label == "Compare documents").click().run(timeout=20)
    assert not app.exception
    assert any("thirty" in t.value for t in app.text)
    assert not app.download_button
    assert not any(e.label == "Text diff" for e in app.expander)
    assert len(list((store.root / "comparisons").glob("*.json"))) == 1
    source = next(b for b in app.button if b.label.startswith("View Document One"))
    source.click().run()
    assert not app.exception
    assert app.radio(key="comparison_view").value == "Comparison report"
    assert any("Block 1" in c.value for c in app.caption)
    assert any("thirty" in t.value for t in app.text) or any("fifteen" in t.value for t in app.text)
    assert app.radio(key="comparison_view").options == ["Comparison report", "View sources"]
    assert not app.chat_input
    next(b for b in app.button if b.label == "Compare documents").click().run(timeout=20)
    assert len(list((store.root / "comparisons").glob("*.json"))) == 1
    app.radio(key="comparison_view").set_value("View sources").run()
    assert not app.exception
    assert len(app.download_button) == 2


def test_missing_rows_are_hidden_only_when_both_are_missing():
    from comparison.ui import visible_rows

    rows = [
        {"one": {"status": a}, "two": {"status": b}}
        for a, b in [
            ("not_found", "not_found"),
            ("found", "not_found"),
            ("found", "found"),
            ("unverified", "not_found"),
        ]
    ]
    assert visible_rows(rows) == rows[1:3]


def test_compare_library_labels_and_filter(pair, monkeypatch):
    store, docs = pair
    store.update(docs[0]["id"], docs[0]["generation"], document_type="Mortgage")
    store.update(docs[1]["id"], docs[1]["generation"], document_type=" mortgage ")
    assert store.read(docs[1]["id"], "metadata")["document_type"] == "Mortgage"
    monkeypatch.setattr(config, "DATA_DIR", store.root)
    monkeypatch.setattr(models, "model_status", lambda: (False, "Offline"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert any(b.label == "Mortgage" for b in app.button)
    assert not any(s.label == "Document type" for s in app.selectbox)
    app.session_state["workspace_tabs"] = "Compare"
    app.run()
    assert not app.exception
    assert not any(r.label == "Saved documents" for r in app.radio)
    assert not any(b.label == "Delete file" for b in app.button)
    assert not any(s.label == "Document type" for s in app.selectbox)
    app.selectbox(key="document_type_filter").set_value("Mortgage").run()
    assert len(app.selectbox(key="earlier_version").options) == 2
    store.update(docs[1]["id"], docs[1]["generation"], document_type="Policy")
    app.run()
    assert not app.exception
    assert any("At least two documents of this type" in i.value for i in app.info)


def test_label_edit_and_save(pair, monkeypatch):
    store, docs = pair
    monkeypatch.setattr(config, "DATA_DIR", store.root)
    monkeypatch.setattr(models, "model_status", lambda: (False, "Offline"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    selected = app.session_state["file_selection"]
    app.button(key="label_edit_" + selected).click().run()
    editor = next(t for t in app.selectbox if t.label == "Label")
    editor.options.append("Mortgage")  # AppTest does not simulate accept_new_options typing.
    editor.set_value("Mortgage")
    next(b for b in app.button if b.label == "Save").click().run()
    assert not app.exception
    assert store.read(selected, "metadata")["document_type"] == "Mortgage"
    app.button(key="label_edit_" + selected).click().run()
    editor = next(t for t in app.selectbox if t.label == "Label")
    editor.options.append("Contract")
    editor.set_value("Contract")
    next(b for b in app.button if b.label == "Save").click().run()
    assert store.read(selected, "metadata")["document_type"] == "Contract"


def test_new_session_hides_saved_report_until_opened(pair, monkeypatch):
    from comparison.storage import ComparisonStore

    store, docs = pair
    sessions = ComparisonStore(store)
    session = sessions.open(*docs)
    result = {
        "rows": [
            {
                "topic": "Saved topic",
                "one": {"status": "found", "value": "First", "sources": []},
                "two": {"status": "found", "value": "Second", "sources": []},
            }
        ]
    }
    sessions.save_result(session, result)
    monkeypatch.setattr(config, "DATA_DIR", store.root)
    monkeypatch.setattr(models, "model_status", lambda: (False, "Offline"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.session_state["workspace_tabs"] = "Compare"
    app.run()
    assert not any(t.value == "Saved topic" for t in app.text)
    app.selectbox(key="saved_comparison").set_value(session["id"]).run()
    next(b for b in app.button if b.label == "Open").click().run()
    assert not app.exception
    assert any(t.value == "Saved topic" for t in app.text)
    next(b for b in app.button if b.label == "Delete").click().run()
    next(b for b in app.button if b.label == "Delete comparison permanently").click().run()
    assert not app.exception
    assert not any(t.value == "Saved topic" for t in app.text)
    assert not any(s.label == "Saved comparisons" for s in app.selectbox)
    assert len(store.list()) == 2


def test_delete_saved_comparison_preserves_documents(pair):
    import pytest

    from comparison.storage import ComparisonStore
    from storage import DocumentGone

    store, docs = pair
    sessions = ComparisonStore(store)
    original = sessions.open(*docs)
    sessions.save_result(original, {"answer": "saved"})
    sessions.delete(original["id"])
    assert sessions.list() == []
    assert len(store.list()) == 2
    with pytest.raises(DocumentGone):
        sessions.save_result(original, {"answer": "stale"})
    sessions.open(*docs)
    with pytest.raises(ValueError):
        sessions.save_result(original, {"answer": "stale"})


def test_expanded_pdf_source_shows_only_cited_excerpt():
    app = AppTest.from_string("""
from comparison.ui import source_details
source_details({"version": "earlier", "filename": "sample.pdf", "label": "Page 2",
                "quote": "Annual rate: 5.750%", "text": "Unrelated provisions and a long page of text."}, "citation")
""").run()
    app.button(key="citation").click().run()
    assert not app.exception
    assert [t.value for t in app.text] == ["Annual rate: 5.750%"]
    assert any("sample.pdf - Page 2" == c.value for c in app.caption)


def test_questions_require_report_and_incompatible_reason_is_expanded(pair, monkeypatch):
    import comparison.ui as ui

    store, docs = pair
    monkeypatch.setattr(config, "DATA_DIR", store.root)
    monkeypatch.setattr(models, "model_status", lambda: (True, "Ready"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.session_state["workspace_tabs"] = "Compare"
    app.run()
    assert not app.chat_input
    monkeypatch.setattr(
        ui,
        "run_comparison",
        lambda *a, **kw: {"status": "incompatible", "answer": "The subject and purpose differ."},
    )
    app.radio(key="comparison_view").set_value("Comparison report").run()
    next(b for b in app.button if b.label == "Compare documents").click().run()
    assert not app.exception
    assert any(e.value == "The subject and purpose differ." for e in app.error)
    status = next(s for s in app.status if s.label == "Unable to compare")
    assert status.proto.expanded

