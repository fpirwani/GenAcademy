from comparison.concepts import ConceptMatches
from comparison.holistic import build_report
from comparison.holistic_schemas import FieldReview, Fields, Topics
from comparison.presentation import ReportOrganization


class ExtractionModel:
    def __init__(self):
        self.calls = []

    def structured(self, schema, instruction, payload):
        self.calls.append(schema)
        if schema is ReportOrganization:
            return schema(groups=[])
        if schema is ConceptMatches:
            return ConceptMatches(
                matches=[{"source": t, "canonical": t} for t in payload["source_labels"]]
            )
        if schema is Topics:
            return Topics(topics=["Cancellation", "Portability"])
        if schema is Fields:
            passage = payload["passages"][0]
            return Fields(
                fields=[
                    {
                        "topic": t,
                        "status": "found" if t == "Cancellation" else "not_found",
                        "value": passage["text"] if t == "Cancellation" else "",
                        "evidence": [{"ref": passage["ref"], "quote": passage["text"]}]
                        if t == "Cancellation"
                        else [],
                    }
                    for t in payload["topics"]
                ]
            )
        if schema is FieldReview:
            return FieldReview(
                decisions=[{"topic": c["topic"], "supported": True} for c in payload["fields"]]
            )
        raise AssertionError(schema)


def test_side_by_side_fields_missing_sections_and_cache(pair):
    store, docs = pair
    model = ExtractionModel()
    result = build_report(store, *docs, model=model, document_type="Policy")
    row = next(r for r in result["rows"] if r["topic"] == "Cancellation")
    assert "thirty" in row["one"]["value"]
    assert "fifteen" in row["two"]["value"]
    assert row["one"]["sources"][0]["document_id"] == docs[0]["id"]
    missing = next(r for r in result["rows"] if r["topic"] == "Portability")
    assert missing["two"]["value"] == "No corresponding section found"
    calls = len(model.calls)
    build_report(store, *docs, model=model, document_type="Policy")
    # Only missing fields are rechecked; discovery and reviewed facts are reused.
    assert model.calls[calls:] == [Fields, Fields, ReportOrganization]
    reversed_result = build_report(store, *reversed(docs), model=model, document_type="Policy")
    assert reversed_result["rows"][0]["one"]["sources"][0]["version"] == "earlier"
    assert reversed_result["rows"][0]["one"]["sources"][0]["document_id"] == docs[1]["id"]


def test_invalid_quote_is_not_displayed_as_a_value(pair):
    class Invalid(ExtractionModel):
        def structured(self, schema, instruction, payload):
            output = super().structured(schema, instruction, payload)
            if schema is Fields:
                for item in output.fields:
                    if item.evidence:
                        item.evidence[0].quote = "An invented rate of one percent."
            return output

    store, docs = pair
    result = build_report(store, *docs, model=Invalid(), document_type="Policy")
    assert result["rows"][0]["one"]["status"] == "unverified"
    assert result["rows"][0]["one"]["value"] == "Could not verify this term"


def test_rejected_review_is_not_cached(pair):
    class Reject(ExtractionModel):
        def structured(self, schema, instruction, payload):
            result = super().structured(schema, instruction, payload)
            if schema is FieldReview:
                for decision in result.decisions:
                    decision.supported = False
            return result

    store, docs = pair
    model = Reject()
    result = build_report(store, *docs, model=model, document_type="Policy")
    assert result["rows"][0]["one"]["status"] == "unverified"
    assert not result["rows"][0]["one"]["sources"]
    for doc in docs:
        assert not store.read(doc["id"], "comparison_profile")["fields"]


def test_extraction_receives_unambiguous_refs_and_retries_missing_evidence(pair):
    class Recover(ExtractionModel):
        def __init__(self):
            super().__init__()
            self.attempts = {}

        def structured(self, schema, instruction, payload):
            if schema is Fields:
                assert all("id" not in p for p in payload["passages"])
                ref = payload["passages"][0]["ref"]
                self.attempts[ref] = self.attempts.get(ref, 0) + 1
                result = super().structured(schema, instruction, payload)
                if self.attempts[ref] == 1:
                    for field in result.fields:
                        field.evidence = []
                return result
            return super().structured(schema, instruction, payload)

    store, docs = pair
    result = build_report(store, *docs, model=Recover(), document_type="Policy")
    assert result["rows"][0]["one"]["status"] == "found"
    assert result["rows"][0]["two"]["status"] == "found"


def test_extracted_value_and_evidence_are_required():
    from comparison.holistic_schemas import FieldValue

    required = FieldValue.model_json_schema()["required"]
    assert "value" in required
    assert "evidence" in required


def test_explicit_labels_do_not_hide_unheaded_content(pair):
    class MixedContent(ExtractionModel):
        def structured(self, schema, instruction, payload):
            if schema is Topics:
                self.calls.append(schema)
                assert "standard_topics" not in payload
                return Topics(topics=["Storage conditions"])
            return super().structured(schema, instruction, payload)

    store, docs = pair
    for doc in docs:
        chunks = store.read(doc["id"], "chunks")
        chunks[0]["text"] = (
            "Identifier: specimen alpha\nOrigin: northern site\nColour: deep purple\n"
            "Keep the specimen in a dry, dark room."
        )
        store.write(doc["id"], doc["generation"], "chunks", chunks)
    model = MixedContent()
    result = build_report(store, *docs, model=model, document_type="Unfamiliar document")
    assert model.calls.count(Topics) == 2
    assert {r["topic"] for r in result["rows"]} == {
        "Identifier",
        "Origin",
        "Colour",
        "Storage conditions",
    }
    model.calls.clear()
    build_report(store, *docs, model=model, document_type="Unfamiliar document")
    assert Topics not in model.calls


def test_premature_not_found_is_retried_with_matching_evidence(pair):
    class Missed(ExtractionModel):
        def structured(self, schema, instruction, payload):
            result = super().structured(schema, instruction, payload)
            if schema is Fields and not payload.get("correction"):
                for field in result.fields:
                    field.status, field.value, field.evidence = "not_found", "", []
            return result

    store, docs = pair
    result = build_report(store, *docs, model=Missed(), document_type="Unknown")
    row = next(r for r in result["rows"] if r["topic"] == "Cancellation")
    assert row["one"]["status"] == row["two"]["status"] == "found"
