"""Shared PDF text-block display, retaining original page locations."""

import streamlit as st

from documents import make_chunks


def text_blocks(block):
    chunks = make_chunks([block])
    if not chunks:
        st.info("No readable text on this page.")
    for index, chunk in enumerate(chunks, 1):
        with st.container(border=True):
            st.caption(f"{block['label']} - Text block {index}")
            st.text(chunk["text"])
