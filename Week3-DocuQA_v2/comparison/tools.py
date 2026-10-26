"""Read-only LangChain tools with trusted, generation-bound document scopes."""

from langchain_core.tools import tool

from comparison.alignment import sections
from models import LocalEmbeddings
from retrieval import HybridSearch, retrieve
from storage import DocumentGone


def research_tools(store, document, version, *, semantic=True, on_warning=lambda message: None):
    selected = dict(document)

    def current(name):
        with store.lock:
            doc = store.read(selected["id"], "metadata")
            if doc["generation"] != selected["generation"]:
                raise DocumentGone("This document was replaced.")
            return store.read(doc["id"], name, [])

    def source(item):
        identifier = item.get("id", item.get("section_id"))
        return {
            **item,
            "ref": f"{version}:{identifier}",
            "version": version,
            "document_id": selected["id"],
            "generation": selected["generation"],
            "filename": selected["filename"],
        }

    @tool
    def search_passages(query: str) -> list[dict]:
        """Search only the assigned document version for relevant cited passages."""
        if not query.strip() or len(query) > 2000:
            raise ValueError("Search query must be 1–2000 characters.")
        chunks = current("chunks")
        if semantic:
            search = HybridSearch(store, selected, LocalEmbeddings())
            found = search.search(chunks, query, k=4)
            if search.warning:
                on_warning(search.warning)
        else:
            found = retrieve(chunks, query, k=4)
        current("metadata")
        return [source(item) for item in found]

    @tool
    def list_sections() -> list[dict]:
        """List up to 80 section IDs, locations and short text previews in this version."""
        return [
            {"section_id": s["section_id"], "label": s["label"], "preview": s["text"][:100]}
            for s in sections(current("blocks"))[:80]
        ]

    @tool
    def read_section(section_id: str) -> list[dict]:
        """Read a section by the ID returned by list_sections, within this version."""
        found = [source(s) for s in sections(current("blocks")) if s["section_id"] == section_id]
        if not found:
            raise ValueError("Unknown section ID")
        return found

    @tool
    def read_adjacent_chunks(chunk_id: str) -> list[dict]:
        """Read a known chunk and its immediate neighbors to inspect qualifications."""
        chunks = current("chunks")
        index = next((i for i, c in enumerate(chunks) if c["id"] == chunk_id), None)
        if index is None:
            raise ValueError("Unknown chunk ID")
        return [source(c) for c in chunks[max(0, index - 1) : index + 2]]

    return {t.name: t for t in [search_passages, list_sections, read_section, read_adjacent_chunks]}
