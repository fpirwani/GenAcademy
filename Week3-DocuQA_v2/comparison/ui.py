"""Streamlit comparison workspace; agent actions never own UI state or persistence."""

import streamlit as st

from comparison.display import clean_text
from comparison.presentation import concise_difference
from comparison.storage import ComparisonStore
from comparison.workflow import run_comparison
from documents import render_page

LABELS = {"earlier": "Document One", "later": "Document Two"}


def visible_warnings(warnings):
    """Hide retired notices, including those stored in older comparison reports."""
    retired = (
        "An unsupported profile observation was omitted.",
        "Some generated claims failed exact citation validation and were omitted.",
        "Comparison is limited to five fields per topic.",
        "Some terms could not be verified. They are marked in the table.",
        "means not located, not proven absent.",
    )
    return [
        warning.split(" Reviewer feedback:", 1)[0]
        for warning in warnings
        if not any(text in warning for text in retired)
    ]


def visible_rows(rows):
    verified = {"found", "not_stated", "not_applicable"}
    return [
        r
        for r in rows
        if all(r.get(side, {}).get("status") in verified | {"not_found"} for side in ("one", "two"))
        and any(r.get(side, {}).get("status") in verified for side in ("one", "two"))
    ]


def source_details(source, key):
    state_key = key + "_expanded"
    if st.button(f"View {LABELS[source['version']]}: {source['label']}", key=key):
        st.session_state[state_key] = not st.session_state.get(state_key, False)
    if st.session_state.get(state_key):
        with st.container(border=True):
            st.caption(f"{source['filename']} - {source['label']}")
            st.text(clean_text(source["quote"]))


def source_preview(store, doc, version, session_id):
    st.subheader(doc["filename"])
    st.caption(f"({LABELS[version]})")
    blocks = store.read(doc["id"], "blocks", [])
    if not blocks:
        st.info("Process this document in the document workspace first.")
        return
    from word_preview import show_word_preview

    if doc["suffix"] == ".docx" and show_word_preview(
        store,
        doc,
        key=f"word_compare_{session_id}_{version}",
    ):
        return
    location = 1
    key = f"compare_page_{session_id}_{version}_{location}"
    page = st.number_input(
        "Page" if doc["suffix"] == ".pdf" else "Block",
        min_value=1,
        max_value=len(blocks),
        value=min(location, len(blocks)),
        key=key,
    )
    payload = store.original(doc["id"]).read_bytes()
    if doc["suffix"] == ".pdf":
        try:
            st.image(render_page(payload, page), width="stretch")
        except Exception:
            st.text(blocks[page - 1]["text"])
    else:
        st.caption("Extracted Word text · original layout may differ")
        st.text(blocks[page - 1]["text"])


def comparison_panel(store, ready):
    st.subheader("Compare documents")
    documents = [d for d in store.list() if d["status"] != "deleting"]
    if len(documents) < 2:
        st.info("Upload at least two different documents using Your files to compare them.")
        return
    sessions = ComparisonStore(store)
    saved = [s for s in sessions.list() if s.get("report")]
    if st.session_state.get("saved_comparison") not in {s["id"] for s in saved}:
        st.session_state["saved_comparison"] = ""
    if saved:
        by_session = {s["id"]: s for s in saved}
        saved_column, actions = st.columns([4, 2], vertical_alignment="bottom")
        selection = saved_column.selectbox(
            "Saved comparisons",
            [""] + list(by_session),
            format_func=lambda key: (
                "Choose a saved comparison"
                if not key
                else " → ".join(d["filename"] for d in by_session[key]["documents"])
            ),
            key="saved_comparison",
        )
        with actions.container(horizontal=True, gap="small"):
            open_clicked = st.button("Open", disabled=not selection)
            delete_clicked = st.button("Delete", disabled=not selection)
        if open_clicked:
            docs = by_session[selection]["documents"]
            st.session_state["earlier_version"] = docs[0]["id"]
            st.session_state["later_version"] = docs[1]["id"]
            st.session_state["document_type_filter"] = None
            st.session_state["comparison_active_" + selection] = True
            st.session_state["comparison_view"] = "Comparison report"
            st.rerun()
        if delete_clicked:
            st.session_state["delete_saved_compare"] = selection
        if selection and st.session_state.get("delete_saved_compare") == selection:
            st.warning("Delete this saved comparison? Source documents will be kept.")
            yes, no = st.columns(2)
            if yes.button("Delete comparison permanently"):
                sessions.delete(selection)
                for key in list(st.session_state):
                    if selection in str(key) or key == "delete_saved_compare":
                        st.session_state.pop(key, None)
                st.rerun()
            if no.button("Keep comparison"):
                st.session_state.pop("delete_saved_compare", None)
                st.rerun()
    types = sorted({d.get("document_type", "") for d in documents} - {""})
    options = [None, ""] + types
    if st.session_state.get("document_type_filter") not in options:
        st.session_state["document_type_filter"] = None
    filter_column, a, b = st.columns([1, 2, 2])
    selected_type = filter_column.selectbox(
        "Filter by document type",
        options,
        format_func=lambda v: "All types" if v is None else (v or "Unlabelled"),
        key="document_type_filter",
    )
    if selected_type is not None:
        documents = [d for d in documents if d.get("document_type", "") == selected_type]
    if len(documents) < 2:
        st.info(
            "At least two documents of this type are needed. Label files under Your files on the Summary tab or choose another type."
        )
        return
    by_id = {d["id"]: d for d in documents}
    keys = list(by_id)
    if st.session_state.get("earlier_version") not in by_id:
        st.session_state["earlier_version"] = keys[-1]
    if st.session_state.get("later_version") not in by_id:
        st.session_state["later_version"] = keys[0]
    earlier_id = a.selectbox(
        "Document One",
        keys,
        key="earlier_version",
        format_func=lambda key: by_id[key]["filename"],
    )
    later_id = b.selectbox(
        "Document Two", keys, key="later_version", format_func=lambda key: by_id[key]["filename"]
    )
    st.caption(
        "If comparing two versions of the same document, choose the original document as Document One and the new document as Document Two."
    )
    if earlier_id == later_id:
        st.info("Select two different documents.")
        return
    earlier, later = by_id[earlier_id], by_id[later_id]
    if any(not store.read(d["id"], "chunks", []) for d in [earlier, later]):
        st.warning("Finish processing both documents in the document workspace before comparing.")
        return
    session = sessions.open(earlier, later)
    session_id = session["id"]
    active_key = "comparison_active_" + session_id
    if not st.session_state.get(active_key):
        session = {**session, "report": None, "chat": [], "fresh": True}
    if st.session_state.get("comparison_view") not in {"Comparison report", "View sources"}:
        st.session_state["comparison_view"] = "Comparison report"
    mode = st.radio(
        "Comparison view",
        ["Comparison report", "View sources"],
        horizontal=True,
        key="comparison_view",
        label_visibility="collapsed",
    )

    def execute():
        with st.status("Comparing your local documents...", expanded=True) as status:
            message = st.empty()
            partial = st.empty()

            def show_rows(rows):
                partial.dataframe(
                    [
                        {
                            "Term or section": r["topic"],
                            f"{earlier['filename']}": r["one"].get("value", "Pending"),
                            f"{later['filename']}": r["two"].get("value", "Pending"),
                        }
                        for r in visible_rows(rows)
                    ],
                    hide_index=True,
                    width="stretch",
                )

            result = run_comparison(
                store,
                earlier,
                later,
                progress=message.write,
                on_rows=show_rows,
            )
            if result["status"] in {"error", "incompatible", "verification_required"}:
                status.update(
                    label="Unable to compare"
                    if result["status"] == "incompatible"
                    else "Verification required"
                    if result["status"] == "verification_required"
                    else "Comparison interrupted",
                    state="error",
                    expanded=True,
                )
                st.error(result["answer"])
                return
            sessions.save_result(session, result)
            st.session_state[active_key] = True
            status.update(label="Comparison saved", state="complete", expanded=False)
        st.rerun()

    if mode == "Comparison report":
        if st.button("Compare documents", type="primary", disabled=not ready):
            execute()
        report = session["report"]
        if report and report.get("rows"):
            for warning in visible_warnings(report.get("warnings", [])):
                st.caption(warning)
            st.subheader("Side-by-side comparison")
            header = st.columns([0.24, 0.38, 0.38])
            header[0].markdown("**Term or section**")
            for col, doc, label in zip(header[1:], [earlier, later], LABELS.values()):
                with col.container(gap=None):
                    st.text(doc["filename"])
                    st.caption(f"({label})")
            rows = visible_rows(report["rows"])
            if not rows:
                st.info("No verified comparison rows are available.")
            groups = report.get("presentation")
            if groups is None:
                groups = [{"topic": r["topic"], "details": [r], "assessments": {}} for r in rows]
            if not groups:
                st.info("No substantive differences identified in the verified findings.")
            for group_index, group in enumerate(groups):
                st.divider()
                st.text(clean_text(group["topic"]))
                for index, row in enumerate(visible_rows(group["details"])):
                    columns = st.columns([0.24, 0.38, 0.38])
                    columns[0].text(row["topic"])
                    reason = concise_difference(
                        row, group.get("difference_summaries", {}).get(row["topic"], "")
                    )
                    if reason:
                        columns[0].caption(clean_text("Difference: " + reason).replace("$", r"\$"))
                    for column, side in zip(columns[1:], ["one", "two"]):
                        with column:
                            cell = row[side]
                            st.text(clean_text(cell["value"]))
                            for n, source in enumerate(cell.get("sources", [])):
                                source_details(
                                    source, f"table_{session_id}_{group_index}_{index}_{side}_{n}"
                                )
        elif report:
            st.info(
                "This saved report uses the previous format. Click Compare documents to generate a topic-based comparison."
            )
        else:
            st.info("Choose Compare documents to create a side-by-side comparison.")
    else:
        for col, version, doc in zip(st.columns(2), ["earlier", "later"], [earlier, later]):
            col.download_button(
                "Download original",
                store.original(doc["id"]).read_bytes(),
                file_name=doc["filename"],
                key=f"download_{session_id}_{version}",
            )
        for col, version, doc in zip(st.columns(2), ["earlier", "later"], [earlier, later]):
            with col:
                source_preview(store, doc, version, session_id)
    if not ready:
        st.caption("Start the local model to generate comparisons. Saved results remain available.")
