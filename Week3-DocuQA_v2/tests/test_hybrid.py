import pytest

from retrieval import HybridSearch
from storage import DocumentGone, Store


class Embeddings:
    fingerprint = "test-embeddings-v1"

    def __init__(self):
        self.documents_called = 0

    def embed_documents(self, texts):
        self.documents_called += 1
        return [[1.0, 0.0] if "terminate" in t else [0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def saved(store, payload=b"first"):
    doc, _ = store.add("contract.pdf", payload)
    chunks = [
        {
            "id": "c1",
            "text": "Either party may terminate with thirty days notice.",
            "location": 1,
            "label": "Page 1",
        },
        {"id": "c2", "text": "Invoices arrive monthly.", "location": 2, "label": "Page 2"},
    ]
    store.write(doc["id"], doc["generation"], "chunks", chunks)
    return doc, chunks


def test_semantic_match_without_shared_words_persists_and_reuses_vectors(tmp_path):
    store = Store(tmp_path)
    doc, chunks = saved(store)
    embeddings = Embeddings()
    search = HybridSearch(store, doc, embeddings)
    assert search.search(chunks, "Exit agreement early", k=1)[0]["id"] == "c1"
    assert embeddings.documents_called == 1
    assert (
        HybridSearch(store, doc, embeddings).search(chunks, "Exit agreement", k=1)[0]["location"]
        == 1
    )
    assert embeddings.documents_called == 1
    assert (store.folder(doc["id"]) / "chroma").is_dir()
    store.delete(doc["id"])
    assert not store.folder(doc["id"]).exists()


def test_search_is_document_scoped_and_old_generation_cannot_recreate_index(tmp_path):
    store = Store(tmp_path)
    doc, chunks = saved(store)
    other, _ = saved(store, b"other")
    other_chunks = [
        {"id": "c1", "text": "Private account number 987654.", "location": 7, "label": "Page 7"}
    ]
    store.write(other["id"], other["generation"], "chunks", other_chunks)
    HybridSearch(store, other, Embeddings()).ensure(other_chunks)
    search = HybridSearch(store, doc, Embeddings())
    assert all(c in chunks for c in search.search(chunks, "Private account number"))
    store.delete(doc["id"])
    saved(store)
    with pytest.raises(DocumentGone):
        search.search(chunks, "agreement")
    assert not (store.folder(doc["id"]) / "chroma").exists()


def test_delete_during_embedding_does_not_resurrect_document(tmp_path):
    store = Store(tmp_path)
    doc, chunks = saved(store)

    class Deleted(Embeddings):
        def embed_documents(self, texts):
            store.delete(doc["id"])
            return super().embed_documents(texts)

    with pytest.raises(DocumentGone):
        HybridSearch(store, doc, Deleted()).ensure(chunks)
    assert not store.folder(doc["id"]).exists()


def test_fallback_is_explicit_and_embedding_changes_rebuild(tmp_path):
    store = Store(tmp_path)
    doc, chunks = saved(store)
    first = Embeddings()
    HybridSearch(store, doc, first).ensure(chunks)
    second = Embeddings()
    second.fingerprint = "test-embeddings-v2"
    search = HybridSearch(store, doc, second)
    search.ensure(chunks)
    assert second.documents_called == 1

    class Offline(Embeddings):
        def embed_query(self, text):
            raise ConnectionError("offline")

    search = HybridSearch(store, doc, Offline())
    assert search.search(chunks, "Invoices")[0]["id"] == "c2"
    assert "keyword" in search.warning.lower()


def test_fusion_rewards_agreement_without_duplicate_chunks():
    from retrieval import reciprocal_rank_fusion

    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=3) == ["b", "a", "c"]


def test_interrupted_index_is_rebuilt_and_changed_chunks_replace_old_vectors(tmp_path):
    store = Store(tmp_path)
    doc, chunks = saved(store)
    embeddings = Embeddings()
    search = HybridSearch(store, doc, embeddings)
    search.ensure(chunks)
    with store.lock, search._client() as client:
        client.get_collection("passages", embedding_function=None).delete(ids=["c1"])
    search.ensure(chunks)
    assert embeddings.documents_called == 2
    replacement = [dict(chunks[0], id="new")]
    store.write(doc["id"], doc["generation"], "chunks", replacement)
    assert [c["id"] for c in search.search(replacement, "exit")] == ["new"]
    with store.lock, search._client() as client:
        assert client.get_collection("passages", embedding_function=None).count() == 1
