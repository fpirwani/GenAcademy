"""Keep routine tool/UI tests independent of any running Ollama service."""

import pytest


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
