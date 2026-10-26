# DocuQA_v2 architecture

DocuQA_v2 is a local Streamlit application for text PDF and DOCX documents.
Ollama supplies inference and embeddings. Documents and results remain in the local
store. OCR and external model services are not part of this application.

## Workspaces

Summary supports upload, labels, previews, document summaries and single-document
questions. Compare selects two documents, checks compatibility, creates a report,
and displays source evidence. Compare has only Comparison report and View sources. Saved comparisons retain snapshots;
opening a saved result does not rerun inference.

## Evidence-first comparison

`comparison/workflow.py` validates selections and document generations, checks
compatibility, then builds the comparison report. Compatibility is a model judgment,
not a guarantee that the documents can be meaningfully compared in every respect.

Compare builds reports and displays source previews. Summary uses the root
`workflows.py` for its single-document features.

## Automatic reports

`comparison/inventory.py` caches field labels discovered from source chunks and
explicit headings. Structured documents use a section inventory; mixed or unstructured
content uses model discovery. It does not generate descriptive profiles or learn domain rules.
`comparison/holistic.py` extracts individual fields in batches of five and validates
citations, numbers and meanings separately for each document. Accepted fields are
cached by model/version and matching context; uncertain fields are rechecked.
`comparison/presentation.py` groups and ranks reviewed rows only after extraction.
The UI displays at most five groups and five fields per group; raw rows are retained.
Display limits apply after verified facts have been extracted.

`profiles.py` contains only source-derived topic discovery used by the inventory.
No shared learned profile or model-directed research loop is used.

## Sources and storage

`documents.py` extracts native text and retains headings, block/page locations and
structure-preserving chunks. Long blocks use overlapping splits. This does not
reconstruct scanned or image-only content. `retrieval.py` provides keyword and
semantic retrieval; generation-bound tools prevent cross-document access.
`storage.py` and `comparison/storage.py` handle atomic writes, generations, locks,
originals, indexes, previews, histories and saved comparisons.

Single-document features continue to use `tools.py`, `models.py` and `workflows.py`.
Comparison orchestration is a bounded Python pipeline; document workflows use
LangGraph. Rendering uses plain text for monetary values and muted difference notes.

## Validation and limits

Exact citations and number matching are necessary but cannot establish semantic
correctness. Model review can still miss errors or omit relevant facts. Long-document
retrieval may miss provisions. Comparison quality still depends on extraction and model review.
Run automated tests and live sample comparisons when changing prompts or models.
Inspect values, qualifications, attribution and cited passages together.

Saved reports are replaced for the same document pair regardless of selection order.
The new report is written before the prior opposite-order record is removed, preserving
source documents and avoiding loss of a saved report if generation fails.
