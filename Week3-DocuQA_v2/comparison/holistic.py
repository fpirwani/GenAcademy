"""Full-document topic discovery and cached, batched side-by-side extraction."""

import hashlib
import json

from comparison.concepts import literal_value, reconcile
from comparison.extraction import REVIEW_RULES
from comparison.extraction import extract_fields as extract_document_fields
from comparison.holistic_schemas import FieldReview
from comparison.independent import numeric_support
from comparison.inventory import inventory
from comparison.presentation import organize_report
from comparison.tools import research_tools
from comparison.validation import validate_claims
from config import MODEL
from retrieval import retrieve
from storage import DocumentGone

CACHE_VERSION = "evidence-v1"
LABELS = {
    "not_found": "No corresponding section found",
    "not_stated": "Not stated",
    "not_applicable": "Explicitly not applicable",
    "unverified": "Could not verify this term",
}


def unique(items):
    seen, result = set(), []
    for item in items:
        value = " ".join(item.split()).strip(" .:")[:120]
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def build_report(
    store,
    one,
    two,
    *,
    model,
    document_type,
    progress=lambda message: None,
    on_rows=lambda rows: None,
):
    trace, warnings = [], []
    identity = f"{CACHE_VERSION}:{MODEL}:{type(model).__name__}"

    def current(doc):
        if store.read(doc["id"], "metadata")["generation"] != doc["generation"]:
            raise DocumentGone("A selected document was replaced.")

    def load(doc):
        with store.lock:
            current(doc)
            saved = store.read(doc["id"], "comparison_profile", {})
            return (
                saved if saved.get("identity") == identity else {"identity": identity, "fields": {}}
            )

    def save(doc, cache):
        store.write(doc["id"], doc["generation"], "comparison_profile", cache)

    profiles = [load(d) for d in (one, two)]
    topics = []
    members = {}
    for doc in (one, two):
        profile = inventory(store, doc, model, progress)
        topics.extend(profile["topics"])
        for topic, fields in profile.get("topic_groups", {}).items():
            members.setdefault(topic, []).extend(fields)
        warnings.extend(profile["warnings"])
    topics = unique(topics)
    if not topics:
        raise ValueError("No meaningful comparison topics could be identified.")

    concept_key = hashlib.sha256(json.dumps([topics, members], sort_keys=True).encode()).hexdigest()
    saved_alignment = profiles[0].get("alignment", {})
    if saved_alignment.get("key") == concept_key:
        concepts = saved_alignment["concepts"]
    else:
        aligned = reconcile(model, topics, progress)
        concepts = aligned
        profiles[0]["alignment"] = {"key": concept_key, "concepts": concepts}
        save(one, profiles[0])
    topics = list(concepts)
    # Do not reuse a value whose matching context has changed with a new partner.
    context = hashlib.sha256(json.dumps(concepts, sort_keys=True).encode()).hexdigest()
    for doc, cache in zip((one, two), profiles):
        if cache.get("field_context") != context:
            cache["fields"] = {}
            cache["field_context"] = context
            save(doc, cache)

    output = {"one": {}, "two": {}}
    for doc, cache, side, version in zip(
        (one, two), profiles, ("one", "two"), ("earlier", "later")
    ):
        tools = research_tools(store, doc, version, semantic=False)
        chunks = store.read(doc["id"], "chunks", [])
        short = sum(len(c["text"]) for c in chunks) <= 18000
        all_passages = [
            {
                **c,
                "ref": version + ":" + c["id"],
                "version": version,
                "document_id": doc["id"],
                "generation": doc["generation"],
                "filename": doc["filename"],
            }
            for c in chunks
        ]
        for topic in topics:
            if topic.casefold() not in cache["fields"]:
                fact = (
                    literal_value(topic, concepts[topic], all_passages)
                    if len(concepts[topic]) == 1
                    else None
                )
                if fact:
                    cache["fields"][topic.casefold()] = fact
        save(doc, cache)
        pending = [t for t in topics if t.casefold() not in cache["fields"]]
        for start in range(0, len(pending), 5):
            group = pending[start : start + 5]
            progress(
                f"Extracting {doc['filename']}: fields {start + 1}–{start + len(group)} of {len(pending)}…"
            )
            passages = {}
            if short:
                # Include the whole short document: no repeated embedding/model swaps.
                passages = {
                    f"{version}:{c['id']}": {
                        **c,
                        "ref": f"{version}:{c['id']}",
                        "version": version,
                        "document_id": doc["id"],
                        "generation": doc["generation"],
                        "filename": doc["filename"],
                    }
                    for c in chunks
                }
            else:
                for topic in group:
                    found = tools["search_passages"].invoke(
                        {"query": " ".join([topic] + concepts[topic])}
                    )
                    trace.append(
                        {
                            "agent": f"Document {side} researcher",
                            "tool": "search_passages",
                            "arguments": {"query": topic},
                            "results": len(found),
                        }
                    )
                    for passage in found[:3]:
                        passages[passage["ref"]] = passage
            evidence = list(passages.values())

            def extract_fields(requested, retry=False, selected_evidence=None):
                return extract_document_fields(
                    model,
                    requested,
                    selected_evidence if selected_evidence is not None else evidence,
                    aliases={t: concepts[t] for t in requested},
                    correction="Previous output was missing or had invalid citations. Check field status and supply exact source quotes."
                    if retry
                    else "",
                )

            fields = extract_fields(group)
            # A negative extraction is not evidence of absence. Recheck against focused text.
            for field in list(fields.fields):
                if field.status != "not_found" or field.topic not in group:
                    continue
                focused = retrieve(evidence, " ".join([field.topic] + concepts[field.topic]), k=3)
                if focused:
                    retry = extract_fields([field.topic], retry=True, selected_evidence=focused)
                    replacements = [f for f in retry.fields if f.topic == field.topic]
                    if len(replacements) == 1:
                        fields.fields = [
                            f for f in fields.fields if f.topic != field.topic
                        ] + replacements
            failed = []
            for topic in group:
                matches = [f for f in fields.fields if f.topic.casefold() == topic.casefold()]
                if len(matches) != 1 or (
                    matches[0].status != "not_found"
                    and not validate_claims(
                        [
                            {
                                "text": matches[0].value or LABELS.get(matches[0].status, ""),
                                "evidence": [e.model_dump() for e in matches[0].evidence],
                            }
                        ],
                        evidence,
                    )
                ):
                    failed.append(topic)
            if failed:
                progress(
                    f"Rechecking source citations for {len(failed)} terms in {doc['filename']}..."
                )
                repaired = extract_fields(failed, retry=True)
                fields.fields = [
                    f for f in fields.fields if f.topic not in failed
                ] + repaired.fields

            candidates, results = [], {}
            for topic in group:
                matching = [f for f in fields.fields if f.topic.casefold() == topic.casefold()]
                if len(matching) != 1:
                    results[topic] = {
                        "status": "unverified",
                        "value": LABELS["unverified"],
                        "sources": [],
                    }
                    continue
                field = matching[0]
                if field.status == "not_found":
                    results[topic] = {
                        "status": "not_found",
                        "value": LABELS["not_found"],
                        "sources": [],
                    }
                    continue
                valid = validate_claims(
                    [
                        {
                            "text": field.value or LABELS.get(field.status, ""),
                            "evidence": [e.model_dump() for e in field.evidence],
                        }
                    ],
                    evidence,
                )
                if valid and numeric_support(valid[0]):
                    candidates.append(
                        {
                            "topic": topic,
                            "members": concepts[topic],
                            "status": field.status,
                            "value": field.value,
                            "sources": valid[0]["sources"],
                        }
                    )
                else:
                    results[topic] = {
                        "status": "unverified",
                        "value": LABELS["unverified"],
                        "sources": [],
                    }
            if candidates:
                progress(f"Checking {len(candidates)} terms together in {doc['filename']}…")
                verdict = model.structured(
                    FieldReview,
                    REVIEW_RULES,
                    {"fields": candidates, "passages": evidence},
                )
                for candidate in candidates:
                    votes = [d for d in verdict.decisions if d.topic == candidate["topic"]]
                    topic = candidate.pop("topic")
                    if len(votes) == 1 and votes[0].supported:
                        if candidate["status"] != "found":
                            candidate["value"] = LABELS[candidate["status"]]
                        results[topic] = candidate
                    else:
                        results[topic] = {
                            "status": "unverified",
                            "value": LABELS["unverified"],
                            "sources": [],
                        }
            for topic, value in results.items():
                output[side][topic] = value
                # Cache only reviewed facts. Missing/uncertain fields are rechecked next time.
                if value["status"] not in ("unverified", "not_found"):
                    cache["fields"][topic.casefold()] = value
            save(doc, cache)
            other = "two" if side == "one" else "one"
            if output[other]:
                on_rows(
                    [
                        {
                            "topic": t,
                            "one": output["one"].get(t, {}),
                            "two": output["two"].get(t, {}),
                        }
                        for t in topics
                        if t in output["two"]
                    ]
                )
        for topic in topics:
            if topic not in output[side]:
                value = json.loads(json.dumps(cache["fields"][topic.casefold()]))
                for source in value.get("sources", []):
                    source["version"] = version
                    source["ref"] = version + ":" + source["ref"].split(":", 1)[1]
                output[side][topic] = value
    for doc in (one, two):
        current(doc)
    rows = [{"topic": t, "one": output["one"][t], "two": output["two"][t]} for t in topics]
    if any(row[s]["status"] == "unverified" for row in rows for s in ("one", "two")):
        warnings.append("Some terms could not be verified. They are marked in the table.")
    warnings.append(
        "“No corresponding section found” means not located, not proven absent. "
        "Topic discovery and extraction can miss provisions; inspect important sources."
    )
    progress("Grouping substantive differences and assessing importance...")
    presentation = organize_report(
        model, rows, document_names={"one": one["filename"], "two": two["filename"]}
    )
    for group in presentation:
        group["details"] = group["details"][:5]
    return {
        "presentation": presentation,
        "format": CACHE_VERSION,
        "concepts": concepts,
        "status": "partial"
        if any(
            row[s]["status"] in ("unverified", "not_found") for row in rows for s in ("one", "two")
        )
        else "complete",
        "rows": rows,
        "claims": [],
        "changes": [],
        "warnings": warnings,
        "trace": trace,
        "coverage": {"topics": len(topics)},
        "answer": f"Compared {len(topics)} topics side by side.",
    }
