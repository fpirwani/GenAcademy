from comparison.display import clean_text


def test_display_repairs_known_encoding_without_changing_currency():
    assert (
        clean_text("Checking\u00e2\u20ac\u00a6 $950 and ($1,600)") == "Checking… $950 and ($1,600)"
    )
    assert clean_text("Café – $950") == "Café – $950"
