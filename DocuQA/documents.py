"""Extract stable citation locations and render one PDF page at a time."""

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pypdfium2 as pdfium
from docx import Document
from docx.table import Table
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import MAX_BYTES, MAX_CHARS, MAX_PAGES


def extract(filename: str, payload: bytes) -> list[dict]:
    if len(payload) > MAX_BYTES:
        raise ValueError("Choose a file smaller than 25 MB.")
    suffix = Path(filename).suffix.lower()
    blocks = []
    try:
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(payload))
            if reader.is_encrypted:
                raise ValueError("This PDF is encrypted. Upload a copy without a password.")
            if len(reader.pages) > MAX_PAGES:
                raise ValueError("Version one supports PDFs of up to 200 pages.")
            for index, page in enumerate(reader.pages, 1):
                blocks.append(
                    {"location": index, "label": f"Page {index}", "text": page.extract_text() or ""}
                )
        elif suffix == ".docx":
            with ZipFile(BytesIO(payload)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("The expanded Word document is too large (100 MB limit).")
            document = Document(BytesIO(payload))
            heading = ""
            for item in document.iter_inner_content():
                if isinstance(item, Table):
                    text = "\n".join(
                        " | ".join(cell.text for cell in row.cells) for row in item.rows
                    )
                else:
                    text = item.text
                    if item.style and item.style.name.startswith("Heading"):
                        heading = text
                if text.strip():
                    location = len(blocks) + 1
                    blocks.append(
                        {
                            "location": location,
                            "label": f"Block {location}" + (f" · {heading}" if heading else ""),
                            "text": text,
                        }
                    )
        else:
            raise ValueError("Upload a text-based PDF or a .docx Word document.")
    except ValueError:
        raise
    except (Exception, BadZipFile) as exc:
        raise ValueError(
            "This file could not be read. Check that it is a valid PDF or DOCX."
        ) from exc
    if not any(block["text"].strip() for block in blocks):
        raise ValueError(
            "No readable text found. Scanned documents need OCR and are not supported."
        )
    if sum(len(block["text"]) for block in blocks) > MAX_CHARS:
        raise ValueError("This document exceeds the 600,000-character text limit.")
    return blocks


def make_chunks(blocks: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    chunks = []
    for block in blocks:
        for text in splitter.split_text(block["text"]):
            chunks.append(
                {
                    "id": f"c{len(chunks) + 1}",
                    "text": text,
                    "location": block["location"],
                    "label": block["label"],
                }
            )
    return chunks


def render_page(payload: bytes, page_number: int) -> bytes:
    with pdfium.PdfDocument(payload) as pdf:
        page = pdf[page_number - 1]
        try:
            scale = min(1.6, 1600 / max(page.get_size()))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil()
                output = BytesIO()
                image.save(output, format="PNG")
                image.close()
                return output.getvalue()
            finally:
                bitmap.close()
        finally:
            page.close()
