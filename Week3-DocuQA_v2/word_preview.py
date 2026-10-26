"""Local DOCX-to-PDF display cache; source blocks remain stable for citations."""

import re
from io import BytesIO

from pypdf import PdfReader

from storage import DocumentGone


def convert(payload):
    from word2pdf import convert as render_docx

    result = render_docx(payload)
    if not PdfReader(BytesIO(result)).pages:
        raise ValueError("Conversion produced no pages")
    return result


def preview_pdf(store, document):
    def check():
        if store.read(document["id"], "metadata")["generation"] != document["generation"]:
            raise DocumentGone("This document was replaced.")

    with store.lock:
        check()
        path = store.folder(document["id"]) / "preview.pdf"
        if path.exists():
            return path.read_bytes()
        payload = store.original(document["id"]).read_bytes()
    result = convert(payload)
    with store.lock:
        check()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(result)
        temporary.replace(path)
    return result


def page_for_quote(pages, quote):
    def normalize(text):
        return re.sub(r"\W+", "", text).casefold()

    needle = normalize(quote)
    matches = [i + 1 for i, text in enumerate(pages) if needle and needle in normalize(text)]
    return matches[0] if len(matches) == 1 else None


def show_word_preview(store, doc, *, key, quote=None):
    """Return False to let the caller display its existing block fallback."""
    import streamlit as st

    from documents import render_page

    failure_key = "word2pdf_preview_failure_" + doc["id"] + doc["generation"]
    if failure_key not in st.session_state:
        try:
            with st.spinner("Preparing Word page preview…"):
                payload = preview_pdf(store, doc)
        except DocumentGone:
            raise
        except Exception:
            st.session_state[failure_key] = (
                "Word page preview could not be converted. Check the document and "
                "then retry. Showing extracted text below."
            )
    if failure_key in st.session_state:
        st.info(st.session_state[failure_key])
        if st.button("Retry Word page preview", key=key + "_retry"):
            del st.session_state[failure_key]
            st.rerun()
        return False
    try:
        pages = [p.extract_text() or "" for p in PdfReader(BytesIO(payload)).pages]
        page_key = key + "_rendered_page"
        if quote and st.session_state.get(key + "_last_quote") != quote:
            match = page_for_quote(pages, quote)
            if match:
                st.session_state[page_key] = match
            st.session_state[key + "_last_quote"] = quote
        if quote:
            st.caption("Cited passage")
            st.text(quote)
            if page_for_quote(pages, quote) is None:
                st.caption(
                    "An exact unique page match was not found. Use the quote to locate the passage."
                )
        page = st.number_input("Page", min_value=1, max_value=len(pages), key=page_key)
        st.caption(
            f"Page {page} of {len(pages)} · Locally rendered Word preview; layout may differ from Microsoft Word."
        )
        st.image(render_page(payload, page), width="stretch")
        return True
    except Exception:
        st.warning("Could not render the converted document. Showing extracted text below.")
        return False
