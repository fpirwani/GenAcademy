from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from documents import extract, make_chunks


def test_native_text_preserves_page_locations():
    data = BytesIO()
    pdf = canvas.Canvas(data)
    pdf.showPage()
    pdf.drawString(50, 700, "This page contains selectable native text.")
    pdf.save()
    blocks = extract("text.pdf", data.getvalue())
    assert blocks[0]["text"] == ""
    assert make_chunks(blocks)[0]["location"] == 2


def test_image_only_pdf_is_rejected():
    data = BytesIO()
    pdf = canvas.Canvas(data)
    pdf.showPage()
    pdf.save()
    with pytest.raises(ValueError, match="text-based PDF or DOCX"):
        extract("scan.pdf", data.getvalue())
