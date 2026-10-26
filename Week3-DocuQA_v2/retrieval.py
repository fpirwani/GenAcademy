"""Document-scoped BM25 and local Chroma search with reciprocal rank fusion."""

import hashlib
import json
import re
from contextlib import contextmanager

from rank_bm25 import BM25Plus

from storage import DocumentGone

STOPWORDS = set(
    "a an the is are was were be been being to of for in on at by with and or "
    "what which who how when where why do does did can could would should "
    "please tell me about this that it its document file".split()
)


def tokens(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"\w+", text.casefold())
        if word not in STOPWORDS and len(word) > 1
    ]


def retrieve(chunks: list[dict], question: str, k: int = 5) -> list[dict]:
    query = tokens(question)
    candidates = [(chunk, tokens(chunk["text"])) for chunk in chunks]
    candidates = [(chunk, words) for chunk, words in candidates if words]
    if not query or not candidates:
        return []
    # BM25Plus avoids zero/negative IDF surprises in one- or two-passage documents.
    index = BM25Plus([words for _, words in candidates])
    scores = index.get_scores(query)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for (chunk, words), _ in ranked if set(query).intersection(words)][:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 5) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]


class HybridSearch:
    """Short-lived Chroma clients keep Windows files deletable between operations.

    Embedding calls run outside the store lock. Every filesystem operation then
    rechecks the selected generation under the same lock used by deletion.
    """

    def __init__(self, store, document, embeddings):
        self.store = store
        self.document = dict(document)
        self.embeddings = embeddings
        self.warning = ""

    def _current(self):
        doc = self.store.read(self.document["id"], "metadata")
        if doc["generation"] != self.document["generation"]:
            raise DocumentGone("This document was replaced. Please reopen it.")
        return self.store.folder(doc["id"]) / "chroma"

    @contextmanager
    def _client(self):
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(self._current()), settings=Settings(anonymized_telemetry=False)
        )
        try:
            yield client
        finally:
            client.close()

    def _signature(self, chunks):
        return hashlib.sha256(
            json.dumps([self.embeddings.fingerprint, chunks], sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _ready(client, signature, count):
        from chromadb.errors import NotFoundError

        try:
            collection = client.get_collection("passages", embedding_function=None)
        except NotFoundError:
            return False
        return (
            collection.metadata == {"signature": signature, "complete": True}
            and collection.count() == count
        )

    def ensure(self, chunks):
        """Create or repair the index; existing files migrate without reuploading."""
        signature = self._signature(chunks)
        with self.store.lock:
            if self._current().exists():
                with self._client() as client:
                    if self._ready(client, signature, len(chunks)):
                        return
        vectors = []
        for start in range(0, len(chunks), 32):
            vectors.extend(
                self.embeddings.embed_documents(
                    [chunk["text"] for chunk in chunks[start : start + 32]]
                )
            )
        if len(vectors) != len(chunks):
            raise ValueError("The embedding model returned an incomplete batch.")
        with self.store.lock:
            self._current()
            if self.store.read(self.document["id"], "chunks", []) != chunks:
                raise DocumentGone("Document search content changed. Please reopen it.")
            with self._client() as client:
                if self._ready(client, signature, len(chunks)):
                    return
                from chromadb.errors import NotFoundError

                try:
                    client.delete_collection("passages")
                except NotFoundError:
                    pass
                collection = client.create_collection(
                    "passages",
                    embedding_function=None,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata={"signature": signature, "complete": False},
                )
                for start in range(0, len(chunks), 128):
                    batch = chunks[start : start + 128]
                    collection.add(
                        ids=[c["id"] for c in batch],
                        embeddings=vectors[start : start + 128],
                        metadatas=[
                            {
                                "document_id": self.document["id"],
                                "generation": self.document["generation"],
                                "location": c["location"],
                            }
                            for c in batch
                        ],
                    )
                collection.modify(metadata={"signature": signature, "complete": True})

    def search(self, chunks, question, k=5):
        self.warning = ""
        if not chunks or not question.strip() or k <= 0:
            return []
        lexical = retrieve(chunks, question, k=max(10, k))
        try:
            self.ensure(chunks)
            vector = self.embeddings.embed_query(question)
            with self.store.lock:
                with self._client() as client:
                    result = client.get_collection("passages", embedding_function=None).query(
                        query_embeddings=[vector],
                        n_results=min(max(10, k), len(chunks)),
                        where={
                            "$and": [
                                {"document_id": self.document["id"]},
                                {"generation": self.document["generation"]},
                            ]
                        },
                        include=["distances"],
                    )
            ids = reciprocal_rank_fusion([[c["id"] for c in lexical], result["ids"][0]], k=k)
            by_id = {c["id"]: c for c in chunks}
            return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]
        except DocumentGone:
            raise
        except Exception:
            self.warning = (
                "Semantic search is unavailable; this answer used keyword search only. "
                "Check Ollama and download nomic-embed-text, then retry."
            )
            return lexical[:k]
