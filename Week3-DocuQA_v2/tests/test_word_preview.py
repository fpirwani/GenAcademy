from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader
from test_documents import pdf_bytes

import word_preview
from storage import DocumentGone, Store


def test_unique_quote_navigation():
    assert (
        word_preview.page_for_quote(["Introduction", "The amount is $400."], "The amount is $400.")
        == 2
    )
    assert word_preview.page_for_quote(["Repeated", "Repeated"], "Repeated") is None
    assert word_preview.page_for_quote(["Other text"], "Missing") is None


def test_cache_reused_and_deleted_with_document(tmp_path, monkeypatch):
    store = Store(tmp_path)
    doc, _ = store.add("test.docx", b"example")
    calls = []
    monkeypatch.setattr(word_preview, "convert", lambda *args: calls.append(args) or pdf_bytes())
    first = word_preview.preview_pdf(store, doc)
    assert word_preview.preview_pdf(store, doc) == first
    assert len(calls) == 1
    store.delete(doc["id"])
    assert not store.folder(doc["id"]).exists()


def test_deleted_during_conversion_cannot_restore_preview(tmp_path, monkeypatch):
    store = Store(tmp_path)
    doc, _ = store.add("test.docx", b"example")

    def convert(*args):
        store.delete(doc["id"])
        return pdf_bytes()

    monkeypatch.setattr(word_preview, "convert", convert)
    with pytest.raises(DocumentGone):
        word_preview.preview_pdf(store, doc)
    assert not store.folder(doc["id"]).exists()


def test_real_python_conversion_without_office():
    payload = (Path(__file__).parents[1] / "samples/policy-v1.docx").read_bytes()
    result = word_preview.convert(payload)
    pages = PdfReader(BytesIO(result)).pages
    assert pages
    assert "30 days" in " ".join(page.extract_text() for page in pages)
