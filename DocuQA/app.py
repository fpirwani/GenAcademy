"""Run with: uv run streamlit run app.py"""

import streamlit as st
from langchain_core.tools import ToolException

from config import DATA_DIR, EMBEDDING_MODEL, MODEL
from documents import render_page
from models import OLLAMA_MISSING, OLLAMA_STOPPED, embedding_status, model_status
from storage import DocumentGone, Store
from tools import build_tools


def go_to_source(key: str, source: dict):
    st.session_state[key] = source["location"]
    st.session_state[key + "_quote"] = source["quote"]


def clear_quote(key: str):
    st.session_state.pop(key + "_quote", None)


def forget_document(doc_id: str):
    for key in list(st.session_state):
        if doc_id in str(key) or key in {"file_selection", "confirm_delete", "pending_summary"}:
            del st.session_state[key]


def generate_summary(store: Store, doc: dict):
    try:
        with st.status("Writing your summary…", expanded=True) as status:
            progress = st.empty()
            build_tools(store, document=doc, progress=progress.write)["summarize_document"].invoke(
                {}
            )
            status.update(label="Summary ready", state="complete", expanded=False)
        st.rerun()
    except DocumentGone as exc:
        st.warning(str(exc))
    except Exception:
        st.error("The summary could not be completed. Check Ollama and retry. Your file is saved.")


def upload_document(store: Store, uploaded, ready: bool):
    try:
        with st.status("Reading your document…", expanded=True) as status:
            result = build_tools(
                store, upload=(uploaded.name, uploaded.getvalue()), progress=st.write
            )["upload_document"].invoke({})
            status.update(label="Document saved", state="complete", expanded=False)
        doc_id = result["document"]["id"]
        st.session_state.file_selection = doc_id
        if not result["created"]:
            st.toast("Opened your saved document and conversation.")
        if result.get("warning"):
            st.toast(result["warning"])
        if ready and not store.read(doc_id, "summary", {}):
            st.session_state.pending_summary = doc_id
        st.rerun()
    except (ValueError, OSError, DocumentGone, ToolException) as exc:
        st.error(str(exc))


def file_panel(store: Store, ready: bool):
    st.subheader("Your files")
    st.caption("Saved on this computer until you delete them.")
    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx"],
        help="Text PDFs (up to 200 pages) or Word DOCX. Maximum 25 MB.",
    )
    if st.button("Open document", type="primary", disabled=uploaded is None, width="stretch"):
        upload_document(store, uploaded, ready)
    st.divider()
    documents = store.list()
    active = [doc for doc in documents if doc["status"] != "deleting"]
    for doc in documents:
        if doc["status"] == "deleting":
            st.warning("A deletion needs to finish.")
            if st.button("Retry cleanup", key="cleanup_" + doc["id"]):
                try:
                    build_tools(store, document=doc, delete_confirmed=True)[
                        "delete_document"
                    ].invoke({})
                    forget_document(doc["id"])
                    st.rerun()
                except OSError:
                    st.error("Close any programs using the file and retry.")
    if not active:
        st.caption("Your documents will appear here.")
        return None
    by_id = {doc["id"]: doc for doc in active}
    if st.session_state.get("file_selection") not in by_id:
        st.session_state.file_selection = active[0]["id"]
    selected = st.radio(
        "Saved documents",
        list(by_id),
        key="file_selection",
        format_func=lambda doc_id: by_id[doc_id]["filename"],
        label_visibility="collapsed",
    )
    doc = by_id[selected]
    st.caption(f"{len(active)} saved file{'s' if len(active) != 1 else ''}")
    st.divider()
    if st.button("Delete file", width="stretch"):
        st.session_state.confirm_delete = (selected, doc["generation"])
    if st.session_state.get("confirm_delete") == (selected, doc["generation"]):
        st.warning("Delete this file, its summary, search data, and entire conversation?")
        st.text(doc["filename"])
        if st.button("Delete permanently", type="primary", width="stretch"):
            try:
                build_tools(store, document=doc, delete_confirmed=True)["delete_document"].invoke(
                    {}
                )
                forget_document(selected)
                st.rerun()
            except OSError:
                st.error("Deletion is pending. Close programs using the file, then retry cleanup.")
        if st.button("Cancel", width="stretch"):
            st.session_state.pop("confirm_delete", None)
            st.rerun()
    return doc


def document_panel(store: Store, doc: dict, ready: bool):
    st.subheader("Document")
    st.text(doc["filename"])
    payload = store.original(doc["id"]).read_bytes()
    download_action, summary_action = st.columns(2)
    download_action.download_button(
        "Download original",
        payload,
        file_name=doc["filename"],
        mime="application/pdf"
        if doc["suffix"] == ".pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        width="stretch",
    )
    blocks = store.read(doc["id"], "blocks", [])
    if not blocks or not store.read(doc["id"], "chunks", []):
        st.warning("File processing was interrupted. Retry to prepare its preview and search.")
        if st.button("Retry file processing"):
            try:
                build_tools(store, document=doc, upload=(doc["filename"], payload))[
                    "upload_document"
                ].invoke({})
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        return
    summary = store.read(doc["id"], "summary", {})
    automatic = not summary and st.session_state.get("pending_summary") == doc["id"] and ready
    if (
        summary_action.button(
            "Regenerate summary" if summary else "Generate summary",
            disabled=not ready,
            width="stretch",
        )
        or automatic
    ):
        st.session_state.pop("pending_summary", None)
        generate_summary(store, doc)
    st.divider()
    st.markdown("**Preview**")
    key = f"location_{doc['id']}_{doc['generation']}"
    unit = "Page" if doc["suffix"] == ".pdf" else "Block"
    location = st.number_input(
        unit,
        min_value=1,
        max_value=len(blocks),
        step=1,
        key=key,
        on_change=clear_quote,
        args=(key,),
    )
    st.caption(f"{unit} {location} of {len(blocks)} · Click an answer source to jump here.")
    quote = st.session_state.get(key + "_quote")
    if quote:
        with st.container(border=True):
            st.caption("Cited passage")
            st.text(quote)
    with st.container(height=540, border=False):
        if doc["suffix"] == ".pdf":
            try:
                st.image(render_page(payload, location), width="stretch")
            except Exception:
                st.warning("Page image unavailable. Showing extracted text instead.")
                st.text(blocks[location - 1]["text"])
        else:
            st.caption("Word text preview · Layout may differ from the original file.")
            for block in blocks[max(0, location - 2) : min(len(blocks), location + 1)]:
                with st.container(border=block["location"] == location):
                    st.caption(block["label"])
                    st.text(block["text"])

    with st.expander("Summary", expanded=True):
        if summary:
            st.text(summary["text"])
        else:
            st.caption("Use Generate summary above for a short overview of this document.")
            if doc.get("error"):
                st.warning(doc["error"])
            if not ready:
                st.caption("Start the local model to generate a summary.")


def chat_panel(store: Store, doc: dict, ready: bool):
    st.subheader("Questions & answers")
    st.caption("Answers use only the selected document.")
    st.text(doc["filename"])
    history = store.read(doc["id"], "chat", [])
    chunks = store.read(doc["id"], "chunks", [])
    key = f"location_{doc['id']}_{doc['generation']}"
    confirmation_key = f"clear_chat_{doc['id']}_{doc['generation']}"
    if st.button("Clear Q&A history", disabled=not history, width="stretch"):
        st.session_state[confirmation_key] = True
    if st.session_state.get(confirmation_key):
        st.warning("Permanently clear this document's questions, answers and source references?")
        st.caption("Your file, summary and search index will be kept.")
        confirm, cancel = st.columns(2)
        if confirm.button("Clear permanently", type="primary", width="stretch"):
            build_tools(store, document=doc, clear_qa_confirmed=True)["clear_qa_history"].invoke({})
            st.session_state.pop(confirmation_key, None)
            clear_quote(key)
            st.toast("Q&A history cleared.")
            st.rerun()
        if cancel.button("Keep history", width="stretch"):
            st.session_state.pop(confirmation_key, None)
            st.rerun()
    with st.container(height=610, border=False):
        if not history:
            st.info(
                "Ask about a topic, name, or term in your document. Sources will open beside the answer."
            )
        for index, message in enumerate(history):
            with st.chat_message(message["role"]):
                # Plain text prevents document/model links from loading remote resources.
                if message["role"] == "assistant":
                    st.markdown("**Answer**")
                st.text(message["content"])
                if message.get("warning"):
                    st.warning(message["warning"])
                if message.get("sources"):
                    with st.container(border=True):
                        _, sources_column, _ = st.columns([0.04, 0.92, 0.04], gap="small")
                        with sources_column:
                            st.caption("Sources · click to view")
                            for number, source in enumerate(message["sources"]):
                                st.text(f"“{source['quote']}”")
                                st.button(
                                    source["label"],
                                    key=f"source_{doc['id']}_{index}_{number}",
                                    on_click=go_to_source,
                                    args=(key, source),
                                    help=source["quote"],
                                )
    question = st.chat_input(
        "Ask about this document…",
        key="question_" + doc["id"],
        disabled=not ready or not chunks,
        max_chars=2000,
    )
    if question:
        with st.spinner("Finding supporting passages…"):
            result = build_tools(store, document=doc)["qa_document"].invoke({"question": question})
        if result["status"] == "error":
            st.error(result["answer"])
        else:
            st.rerun()
    if not ready:
        st.caption("Q&A becomes available when the local model is ready.")


def main():
    st.set_page_config(page_title="DocuQA", page_icon="📄", layout="wide")
    st.title("DocuQA")
    st.caption("Your documents. Clear summaries. Answers with sources. All local.")
    ready, status = model_status()
    with st.expander("● Local model ready" if ready else "○ Local model setup", expanded=not ready):
        st.write(status)
        if not ready:
            if status == OLLAMA_MISSING:
                st.warning(
                    "DocuQA needs Ollama for summaries and Q&A. Would you like to install it?"
                )
                st.link_button("Install Ollama", "https://ollama.com/download", type="primary")
                st.caption(
                    "Opens the official download page. Complete the installation, start Ollama, then click Check again."
                )
                st.caption(
                    "Already installed in a custom location? Start Ollama, then check again."
                )
            elif status == OLLAMA_STOPPED:
                st.info("Open Ollama from your applications, or run this in a separate terminal:")
                st.code("ollama serve", language="shell")
            else:
                st.write("Start Ollama if needed. Download the model once with:")
                st.code(f"ollama pull {MODEL}", language="shell")
            st.caption(
                "Ollama is separate from this uv project. Model download needs internet; document processing is local."
            )
            if st.button("Check again"):
                st.rerun()
    if ready:
        semantic_ready, _ = embedding_status()
        if not semantic_ready:
            st.warning(
                "Semantic search is not ready. Questions currently use keyword search only. "
                "Download the search model once to enable meaning-based search:"
            )
            st.code(f"ollama pull {EMBEDDING_MODEL}", language="shell")
            if st.button("Check search model again"):
                st.rerun()
        else:
            st.caption("Search combines keywords and meaning within the selected document.")
    store = Store(DATA_DIR)
    left, middle, right = st.columns([0.20, 0.45, 0.35], gap="medium")
    try:
        with left:
            with st.container(border=True):
                doc = file_panel(store, ready)
        with middle:
            with st.container(border=True):
                if doc:
                    document_panel(store, doc, ready)
                else:
                    st.subheader("Document & summary")
                    st.info("Upload a PDF or Word document to get started.")
                    st.caption("Your preview and summary will appear here.")
        with right:
            with st.container(border=True):
                if doc:
                    chat_panel(store, doc, ready)
                else:
                    st.subheader("Questions & answers")
                    st.caption("Select a document to start a conversation.")
    except DocumentGone:
        st.warning("This document was deleted or replaced in another session. Select another file.")
        if st.button("Refresh files"):
            st.rerun()
    except ToolException as exc:
        st.warning(str(exc))
    except OSError:
        st.error(
            "DocuQA could not access a local file. Check folder permissions and available disk space."
        )


if __name__ == "__main__":
    main()
