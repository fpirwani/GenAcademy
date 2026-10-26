"""Cache source-derived field labels, without model-generated document profiles."""

import hashlib
import json
import re

from comparison.concepts import explicit_fields, field_name
from comparison.profiles import discover_topics
from config import MODEL


def inventory(store, doc, model, progress):
    chunks = store.read(doc["id"], "chunks", [])
    fingerprint = hashlib.sha256(
        json.dumps(
            ["inventory-v2", MODEL, type(model).__name__, doc["generation"], chunks], sort_keys=True
        ).encode()
    ).hexdigest()
    saved = store.read(doc["id"], "document_profile", {})
    if saved.get("fingerprint") == fingerprint:
        return saved
    topics = explicit_fields(chunks)
    has_sections = any(
        re.fullmatch(r"[A-Z][A-Z &/()-]{3,64}", line.strip())
        or re.fullmatch(r"\d+[.)]\s+[A-Za-z][^.!?:]{2,80}", line.strip())
        for chunk in chunks
        for line in chunk["text"].splitlines()
    )
    for index, chunk in enumerate([] if has_sections and len(topics) >= 3 else chunks):
        progress(f"Discovering fields in {doc['filename']}: {index + 1}/{len(chunks)}")
        # Unstructured documents need model discovery; structured documents already
        # supply a section inventory. Details remain inside their source sections.
        topics.extend(discover_topics(model, [{**chunk, "ref": chunk["id"]}]).topics)
    names = {}
    for topic in topics:
        label = field_name(topic)
        if label:
            names.setdefault(label.casefold(), label)
    result = {"fingerprint": fingerprint, "topics": list(names.values()), "warnings": []}
    store.write(doc["id"], doc["generation"], "document_profile", result)
    return result
