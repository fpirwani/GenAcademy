# DocuQA

A simple local document assistant for an educational project. Upload a text PDF
or Word `.docx`, read a summary, and ask questions grounded in that document.

The screen has three panels: **Your files | Document and summary | Q&A**.
Source buttons jump to the PDF page or Word text block in the middle panel.
Download original returns the exact uploaded file.

## Run on Windows

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Ollama](https://ollama.com/download/windows). Python is managed by uv.
Ollama is a separate installation; this project is not a standalone executable.

1. Start Ollama and download the answer model (approximately 2.5 GB) and
   the embedding model (approximately 274 MB) once:

   ```powershell
   ollama pull qwen3:4b
   ollama pull nomic-embed-text
   ```

2. In this project's directory, install the locked Python dependencies:

   ```powershell
   uv sync --locked
   ```

3. Run:

   ```powershell
   uv run streamlit run app.py
   ```

4. Open **http://127.0.0.1:8501**. Stop with Ctrl+C in the terminal.
   On Windows, `start.bat` is also provided as a launcher.

The app can open, upload, preview, download, delete files, and show saved history
without Ollama. Summary generation and new answers need the running local model.
When Ollama cannot be found, the UI asks whether you want to install it and offers
an **Install Ollama** link to the official download page. It does not silently
install software. Finish installation and click **Check again**. An installed but
stopped Ollama gets startup instructions instead; a running Ollama with no model
gets the model-download command. Detection checks PATH and standard installation
folders; custom installations can be recognized by starting their local service.
First-time package/model downloads need internet; document processing does not.
The UI checks the embedding model separately. If it is missing or semantic
search fails, questions use BM25 keyword search and show an explicit warning.
Uploading and summarizing still work. After downloading the model, click
**Check search model again**; existing files are indexed on their next question.
For Ollama's explicit local-only mode, set `OLLAMA_NO_CLOUD=1` in its environment
and restart Ollama. DocuQA itself connects only to `127.0.0.1:11434`.

## How it works

- **Streamlit** renders the UI and directly calls the Python application modules.
- **LangChain** splits text and connects to Ollama through `ChatOllama` and `OllamaEmbeddings`.
- **LangGraph** coordinates summary and question-answer workflows.
- **LangChain tools** expose upload, summary, Q&A, clearing Q&A history and deletion. The interface
  invokes them directly; there is no autonomous agent choosing file actions.
- **Hybrid retrieval** combines BM25Plus keyword matches with local
  `nomic-embed-text` embeddings in **Chroma**. Reciprocal rank fusion combines
  up to ten candidates from each method and returns up to five unique passages.
  Chroma runs embedded in Python; no database server, FastAPI service or Docker
  is needed. Its internal SQLite files require no separate SQLite installation.
- **Qwen3 4B** runs through Ollama with thinking disabled.

Upload: validate → extract pages/blocks → save → split → local embeddings → hierarchical summary.
Question: resolve follow-up → BM25 and semantic search within selected file → fuse → answer with
evidence or end gracefully. Source IDs and verbatim evidence quotes are checked
against the retrieved passages before being displayed.

Semantic search helps match synonyms and paraphrases, while BM25 retains exact
names, numbers and identifiers. There is no calibrated relevance threshold or
reranker; the answer model must still decline unsupported questions.
An exact quote check validates source existence,
not the truth of every generated claim; read the sources when accuracy matters.
Summary is generated from all extracted text in bounded batches, not just the
top search hits. It can take several minutes on long documents.

## Supported files and limits

- Text-based PDF: up to 200 pages.
- Word `.docx`: body paragraphs and tables, with a readable text preview.
- Maximum upload: 25 MiB; maximum extracted text: 600,000 characters.
- Password-protected PDFs, scanned-only files, old `.doc` files and OCR are not supported.
- Word headers/footers, floating text boxes, embedded images and exact layout
  are not extracted. DOCX has no reliable page numbers; citations use text blocks.
- PDF source clicks change the rendered page and display the quoted passage;
  they do not highlight coordinates on the PDF page itself.
- Intended for one user on a local desktop. Small screens may stack columns.

## Saved data and deletion

```text
DocuQA/
  app.py             # Three-panel interface
  config.py          # Local settings and limits
  documents.py       # Extraction, chunking, PDF rendering
  storage.py         # Files, atomic JSON, locking, deletion
  retrieval.py       # Document-scoped BM25, Chroma and rank fusion
  models.py          # Ollama boundary and structured answers
  tools.py           # Five @tool actions with app-bound document context
  workflows.py       # LangGraph workflows
  tests/             # Storage, extraction, graph, and UI tests
  data/
    documents/<sha256>/
      original.pdf   # Or original.docx
      metadata.json
      blocks.json
      chunks.json
      summary.json
      chat.json
      chroma/        # Local vectors and index metadata for this document only
```

Identical bytes reopen the existing file and chat, even under another filename.
Changed content becomes a separate document. User files are never sent to a
hosted model. Generated/document text is displayed as plain text so embedded
links/images do not silently fetch external content. Optional tracing and
Streamlit usage statistics, Chroma telemetry and OpenTelemetry export are disabled.

Delete removes the document folder, including original, extracted text,
summary, search passages, Chroma index, chat and citation references. No persistent BM25
index or LangGraph checkpoints exist elsewhere. A tombstone blocks writes while
cleanup runs; failed cleanup can be retried. A generation token prevents a late
answer from attaching to a deleted/reuploaded document. Ollama's shared model
files are not deleted. This is ordinary file deletion, not secure disk erasure;
external backups and OS copies are outside the app's control.

Each document has its own Chroma directory. Clients close after each operation
so Windows file handles do not prevent deletion. Generation checks under the
storage lock prevent an in-flight embedding request from resurrecting a deleted
file. Index signatures include the embedding model digest, retrieval prefixes
and chunks, so model updates, changed text and incomplete indexes trigger a
rebuild without touching saved summaries or chats. Original files need not be
uploaded again. An exported evaluation/report remains a separate snapshot.

The updated Word report describes the current four-document corpus and hybrid
implementation. Run `uv run python -X utf8 evaluation/run_hybrid_evaluation.py`
to evaluate the questions in `evaluation/cases.json`; results are written to
`evaluation/current_results.json` without changing saved chats. Missing sources
or outdated gold anchors are reported as skipped. Review the manifest when
changing test documents and manually reassess answers after every run;
`current_assessment.json` is tied to the reported run timestamp. Older result
files are historical snapshots, excluded from the current report.

The current 15-case run achieved hybrid anchor recall at 5 of 0.917 versus
0.667 for BM25, and strict answer correctness of 10/15. Table interpretation,
follow-up rewriting and exact-quote refusals remain limitations. One question
was correctly declined in prose but returned the wrong status. These scores
are a small, selected evaluation, not a general accuracy guarantee.

Optional environment variables (set before launching):

- `DOCUQA_MODEL`: another installed local Ollama model; default `qwen3:4b`.
  The model must support structured JSON output. Cloud models are rejected.
- `DOCUQA_DATA_DIR`: absolute directory for saved user data; default `data/` here.

## Invoke the tools from Python

Use **Clear Q&A history** above the conversation to remove the selected file's
questions, answers and source references after confirmation. The file, summary
and search index remain available. This also works without Ollama. Answers that
were still generating when history was cleared cannot restore the old conversation.

Run from the project directory. Upload bytes, selected document metadata, model
factory and progress callback are bound by the app, outside the tool input
schema. This keeps large files out of model prompts and prevents a tool caller
from choosing another document or filesystem path.

```python
from pathlib import Path
from config import DATA_DIR
from storage import Store
from tools import build_tools

store = Store(DATA_DIR)
file = Path("example.pdf")
upload_tools = build_tools(store, upload=(file.name, file.read_bytes()))
result = upload_tools["upload_document"].invoke({})
document = result["document"]

document_tools = build_tools(store, document=document)
summary = document_tools["summarize_document"].invoke({})
answer = document_tools["qa_document"].invoke({"question": "What does it say about costs?"})
# Summary and Q&A results are saved by the tools, including validated sources.

# Only after the user confirms deletion in the application:
delete_tools = build_tools(store, document=document, delete_confirmed=True)
delete_tools["delete_document"].invoke({})
```

To clear only the conversation after the user confirms in the interface:

```python
clear_tools = build_tools(store, document=document, clear_qa_confirmed=True)
clear_tools["clear_qa_history"].invoke({})
```

Summary and Q&A tools call the existing LangGraph workflows. Upload and delete
perform deterministic local operations. Tool exceptions surface invalid/missing
context; model failures return a recoverable error for Q&A, while summary failure
leaves a retryable saved document. Generation checks reject stale tool instances.
Deletion confirmation is a trusted application decision, never an LLM argument.

## Verification

```powershell
uv run pytest -q -p no:cacheprovider
uv run ruff check .
```

Tests use real extraction, file storage, BM25 and LangGraph. Model responses are
controlled test doubles; they do not measure real model answer quality.
Streamlit AppTest exercises source navigation and deletion. To test actual
inference, start Ollama, upload a short document, and check a supported question,
an unsupported question, a follow-up, and the source buttons.

## Share with instructors

Share the source, `pyproject.toml`, `uv.lock`, `.python-version`, `.streamlit/`,
tests, docs, and third-party notices. Exclude `.venv/`, `data/`, and caches.
The instructor runs `uv sync --locked` and installs/downloads Ollama as above.
CPU inference is possible but can be slower; suitable RAM and GPU drivers still
depend on their machine. See `THIRD_PARTY_NOTICES.md` for licensing pointers.
