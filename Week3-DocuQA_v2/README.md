# DocuQA_v2

A local Streamlit app for text-based PDF and DOCX summaries, questions and
side-by-side comparison. Ollama supplies local inference and embeddings.
No OCR, external model service or document-type training is required.

## Run

From this directory:

```powershell
uv sync
ollama pull qwen3:4b
ollama pull nomic-embed-text
uv run streamlit run app.py --server.port 8502
```

Start Ollama before generating answers. `start.bat` also launches the app.
If another project's virtual environment is active, deactivate it; `uv run`
uses this project's environment. Model downloads require internet access;
document processing uses the local Ollama endpoint.

## Summary workspace

Upload a text PDF or DOCX, select it in Your files, and view its preview,
summary or source-backed answers. Add or edit labels under filenames; existing
labels can be reused. Labels filter document selection but are not evidence
of document compatibility. Original files can be downloaded and deleted.

DOCX previews use the installed `word2pdf` library. Layout can differ from
Microsoft Word; extracted blocks remain available when conversion fails.
PDFs support page and text-block views. Image-only PDFs are unsupported.

## Compare workspace

Choose a type filter and two different documents. Revisions and similar documents
from different sources are supported. Compatibility is checked before processing.

- **Comparison report:** discovers source fields, independently extracts and validates
  values, then displays up to five ranked groups with five fields per group.
  Short muted notes explain differences with source-backed names, key values and
  conditions; source controls expand supporting text.
- **View sources:** previews both originals side by side with page controls, without a PDF view selector.
- **Saved comparisons:** Open or Delete saved reports. Comparing the same pair again
  replaces its previous saved report, including when document order is reversed.
  The previous report is kept until the replacement succeeds.

New sessions start with a fresh workspace; use Open to load saved results.

## Evidence pipeline

Source structure -> scoped retrieval -> independent extraction -> deterministic
quotation/number checks -> semantic review -> concise presentation.

Structured documents use their heading inventory; unstructured documents use model
discovery. Comparison uses a fixed workflow without descriptive profile generation or
model-directed research loops. No shared learned document-type profile is used.

Reports cache verified fields; display grouping happens after extraction. All source
references are bound to document identity and generation.

## Storage and configuration

Data defaults to `data/`; override with `DOCUQA_V2_DATA_DIR`. Original documents,
chunks, indexes, previews, summaries, histories and comparison snapshots persist
locally. `DOCUQA_MODEL` overrides the inference model; defaults are in `config.py`.
Re-upload older originals to upgrade chunking while preserving document identity,
labels and chat history. Regenerate comparisons afterward.

## Validation

```powershell
uv run pytest -q
uv run ruff check .
```

Live checks should use temporary stores and the configured local model.
Do not treat accepted facts or a complete status as proof of exhaustive coverage.
Model review and summaries can still fail; consult sources for important decisions.

See [Architecture](docs/ARCHITECTURE.md) and [Development](docs/DEVELOPMENT.md).
