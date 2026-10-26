"""Application actions exposed as LangChain tools, with trusted app-bound context.

The UI invokes these directly. They do not give the model autonomous file access
or permission to delete. Each factory call binds one document generation.
"""

import hashlib
from collections.abc import Callable

from langchain_core.tools import BaseTool, ToolException, tool

from documents import extract, make_chunks
from models import LocalEmbeddings, LocalModel
from retrieval import HybridSearch
from storage import DocumentGone, Store
from workflows import MODEL_ERROR, answer_question, summarize


def build_tools(
    store: Store,
    *,
    document: dict | None = None,
    upload: tuple[str, bytes] | None = None,
    delete_confirmed: bool = False,
    clear_qa_confirmed: bool = False,
    model_factory: Callable = LocalModel,
    embedding_factory: Callable | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, BaseTool]:
    """Bind trusted UI inputs; only the Q&A question is a public tool argument."""
    selected = dict(document) if document else None
    embedding_factory = embedding_factory or LocalEmbeddings

    def prepare_search(doc):
        try:
            progress("Preparing local semantic search...")
            HybridSearch(store, doc, embedding_factory()).ensure(
                store.read(doc["id"], "chunks", [])
            )
            return ""
        except DocumentGone:
            raise
        except Exception:
            return (
                "File saved. Semantic search will be retried when you ask a question. "
                "Check Ollama and download nomic-embed-text."
            )

    def current():
        if selected is None:
            raise ToolException("Select a document first.")
        metadata = store.read(selected["id"], "metadata")
        if metadata["generation"] != selected.get("generation"):
            raise DocumentGone("This document was replaced. Please reopen it.")
        return metadata

    @tool
    def upload_document() -> dict:
        """Validate and save the file provided by the app, or reopen its saved workspace.

        Extracts preview blocks and searchable passages. Repairs an incomplete
        upload without replacing existing chat. File bytes are supplied by the UI.
        """
        if upload is None:
            raise ToolException("Choose a PDF or DOCX file to upload first.")
        filename, payload = upload
        doc_id = hashlib.sha256(payload).hexdigest()
        with store.lock:
            if selected is not None and current()["id"] != doc_id:
                raise ToolException("Retry must use the selected document original.")
            existing = next((item for item in store.list() if item["id"] == doc_id), None)
            if existing:
                if existing["status"] == "deleting":
                    raise DocumentGone(
                        "Finish the pending deletion before uploading this file again."
                    )
                complete = store.read(doc_id, "blocks", []) and store.read(doc_id, "chunks", [])
                complete = complete and all(
                    c.get("chunk_version") == "structure-v2" for c in complete
                )
            else:
                complete = False
        if complete:
            warning = prepare_search(existing)
            return {"document": existing, "created": False, "warning": warning}
        progress("Reading your document…")
        blocks = extract(filename, payload)
        with store.lock:
            # Both explicit retries and duplicate-upload repairs retain their generation.
            if existing is not None:
                doc = store.read(doc_id, "metadata")
                if doc["generation"] != existing["generation"]:
                    raise DocumentGone("This document was replaced. Please reopen it.")
                created = False
            else:
                doc, created = store.add(filename, payload)
        progress("Preparing document search…")
        store.write(doc_id, doc["generation"], "blocks", blocks)
        store.write(doc_id, doc["generation"], "chunks", make_chunks(blocks))
        store.update(doc_id, doc["generation"], status="indexed", block_count=len(blocks), error="")
        warning = prepare_search(doc)
        return {"document": store.read(doc_id, "metadata"), "created": created, "warning": warning}

    @tool
    def summarize_document() -> dict:
        """Generate and save a high-level summary of the selected document's full text."""
        with store.lock:
            doc = current()
            blocks = store.read(doc["id"], "blocks", [])
        if not blocks:
            raise ToolException("Finish processing the document before generating a summary.")
        try:
            store.update(doc["id"], doc["generation"], status="summarizing", error="")
            text = summarize(blocks, model_factory(), progress)
            store.write(doc["id"], doc["generation"], "summary", {"text": text})
            store.update(doc["id"], doc["generation"], status="ready", error="")
            return {"status": "ready", "text": text}
        except Exception:
            try:
                store.update(
                    doc["id"],
                    doc["generation"],
                    status="indexed",
                    error="Summary interrupted. Your file is safe; you can retry.",
                )
            except DocumentGone:
                pass
            raise

    @tool
    def qa_document(question: str) -> dict:
        """Answer a question using only the selected document and save its conversation.

        Returns the answer, status, and validated navigable sources. Missing
        evidence ends gracefully; model errors are not saved as factual answers.
        """
        with store.lock:
            doc = current()
            chunks = store.read(doc["id"], "chunks", [])
            history = store.read(doc["id"], "chat", [])
        if not chunks:
            raise ToolException("Finish processing the document before asking questions.")
        try:
            search = HybridSearch(store, doc, embedding_factory())
            result = answer_question(question, chunks, history, model_factory(), search.search)
            if search.warning:
                result["warning"] = search.warning
        except Exception:
            return {"status": "error", "answer": MODEL_ERROR, "sources": []}
        if result["status"] != "error":
            saved = store.append_chat(
                doc["id"],
                doc["generation"],
                [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "status": result["status"],
                        "warning": result.get("warning", ""),
                    },
                ],
                revision=doc.get("chat_revision", 0),
            )
            if not saved:
                return {
                    "status": "error",
                    "answer": "This conversation was cleared while the answer was generating. Please ask again.",
                    "sources": [],
                }
        return result

    @tool
    def clear_qa_history() -> dict:
        """Clear the selected document's Q&A after confirmation in the app.

        Removes saved questions, answers and their source references while
        retaining the original, summary and search index. Confirmation and
        document identity are app-bound, never model-controlled arguments.
        """
        if not clear_qa_confirmed:
            raise ToolException("Confirm in the app before clearing Q&A history.")
        with store.lock:
            doc = current()
            store.clear_chat(doc["id"], doc["generation"])
        return {"status": "cleared", "document_id": doc["id"]}

    @tool
    def delete_document() -> dict:
        """Delete the selected file and all its data after the app confirms deletion.

        Also retries pending cleanup. Confirmation is app-bound, not a model
        argument. A stale tool cannot delete a newly reuploaded document.
        """
        if not delete_confirmed:
            raise ToolException("Confirm deletion in the app before deleting this document.")
        if selected is None:
            raise ToolException("Select a document first.")
        with store.lock:
            if selected.get("status") == "deleting":
                folder = store.folder(selected["id"])
                if folder.exists() and not (folder / ".deleting").exists():
                    raise DocumentGone("This document was replaced. Please reopen it.")
            else:
                current()
            store.delete(selected["id"])
        return {"status": "deleted", "document_id": selected["id"]}

    return {
        item.name: item
        for item in (
            upload_document,
            summarize_document,
            qa_document,
            clear_qa_history,
            delete_document,
        )
    }
