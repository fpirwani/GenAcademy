from documents import make_chunks


def test_short_table_stays_with_labels_and_qualifications():
    text = "TERMS\n" + "\n".join(
        f"Item {i} | {i} units, subject to the stated conditions." for i in range(40)
    )
    chunks = make_chunks([{"text": text, "location": 1, "label": "Page 1"}])
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_heading_is_retained_in_paragraph_context():
    chunks = make_chunks(
        [
            {
                "text": "Payment is estimated and excludes insurance.",
                "heading": "Repayment conditions",
                "location": 2,
                "label": "Block 2",
            }
        ]
    )
    assert "Repayment conditions" in chunks[0]["text"]
