"""Document-independent field identity and semantic label reconciliation."""

import re

from pydantic import BaseModel, Field


class ConceptMatch(BaseModel):
    source: str
    canonical: str


class ConceptMatches(BaseModel):
    matches: list[ConceptMatch] = Field(max_length=24)


class EquivalenceDecision(BaseModel):
    source: str
    equivalent: bool


class ConceptReview(BaseModel):
    decisions: list[EquivalenceDecision] = Field(max_length=24)


def field_name(text):
    # Explicit field/value delimiters never belong in the field identity.
    return re.split(r":\s+|\s+\|\s+", " ".join(text.replace("_", " ").split()), maxsplit=1)[
        0
    ].strip(" .:")[:120]


def explicit_fields(chunks):
    fields = []
    for chunk in chunks:
        for line in chunk["text"].splitlines():
            if re.search(r"[:|]\s*[_ .-]{3,}", line):
                continue
            heading = re.match(r"^\d+[.)]\s+([A-Za-z][^.!?]{2,80})$", line.strip())
            table = re.match(r"^([A-Za-z][A-Za-z /()&_-]{2,65})\s+\|\s+\S", line.strip())
            if table and table[1].strip().isupper():
                continue  # All-caps title banners are not label/value table rows.
            if heading or table:
                fields.append(field_name((heading or table)[1]))
                continue
            match = re.match(r"^([A-Za-z][A-Za-z /()&_-]{2,65}):\s*\S", line.strip())
            if match:
                fields.append(field_name(match[1]))
            elif (
                line.strip().isupper()
                and re.fullmatch(r"[A-Z][A-Z &/()-]{3,64}", line.strip())
                and len(line.split()) <= 6
            ):
                fields.append(field_name(line))
    return list(dict.fromkeys(fields))


def reconcile(model, topics, progress):
    """Preserve field identity; model guesses must never erase a discovered field.

    Only formatting-equivalent labels share an identity. Semantic correspondence is
    resolved by extraction against each document and checked with its evidence.
    """
    progress("Aligning field labels without merging different meanings...")
    concepts, normalized = {}, {}
    for topic in topics:
        key = " ".join(re.findall(r"\w+", topic.casefold()))
        canonical = normalized.setdefault(key, topic)
        concepts.setdefault(canonical, [])
        if topic not in concepts[canonical]:
            concepts[canonical].append(topic)
    return concepts


def literal_value(topic, aliases, passages):
    """Verify simple label/value facts directly; ambiguous/multiline facts use the model."""
    names = {field_name(topic).casefold()}
    matches = []
    for passage in passages:
        for line in passage["text"].splitlines():
            match = re.match(r"^([^:]{3,80}):\s*(\S.*)$", line.strip())
            if (
                match
                and field_name(match[1]).casefold() in names
                and not re.fullmatch(r"[_ .-]+(?:Date:.*)?", match[2].strip(), re.IGNORECASE)
                and len(line.strip()) >= 15
                and len(match[2].split()) <= 12
            ):
                matches.append((match[2].strip(), {**passage, "quote": line.strip()}))
    if not matches or len({v for v, _ in matches}) != 1:
        return None
    value, source = matches[0]
    return {"status": "found", "value": value, "sources": [source]}
