"""Validate numeric values against cited source quotations."""

import re
from decimal import Decimal, InvalidOperation


def numeric_support(claim):
    def tokens(text):
        values = set()
        for token in re.findall(r"\d+(?:[.,]\d+)*", text):
            number = re.sub(r",(?=\d{3}(?:[,\.]|$))", "", token)
            try:
                values.add(Decimal(number))
            except InvalidOperation:
                values.add(token)  # Ambiguous locale/date notation requires an exact match.
        return values

    numbers = tokens(claim["text"])
    source_numbers = tokens(" ".join(s["quote"] for s in claim["sources"]))
    return numbers <= source_numbers
