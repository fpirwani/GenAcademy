"""Keep routine tool/UI tests independent of any running Ollama service."""

import pytest

from storage import Store


@pytest.fixture
def pair(tmp_path):
    store = Store(tmp_path)
    docs = []
    for i, text in enumerate(
        [
            "Cancellation requires thirty days written notice.",
            "Cancellation requires fifteen days written notice.",
        ]
    ):
        doc, _ = store.add(f"v{i}.docx", str(i).encode())
        block = {"text": text, "location": 1, "label": "Block 1"}
        store.write(doc["id"], doc["generation"], "blocks", [block])
        store.write(doc["id"], doc["generation"], "chunks", [{**block, "id": "c1"}])
        docs.append(doc)
    return store, docs


class TestEmbeddings:
    fingerprint = "offline-test-v1"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture(autouse=True)
def local_test_embeddings(monkeypatch):
    import models
    import tools

    monkeypatch.setattr(tools, "LocalEmbeddings", TestEmbeddings)
    monkeypatch.setattr(models, "embedding_status", lambda: (True, "Semantic search ready"))
