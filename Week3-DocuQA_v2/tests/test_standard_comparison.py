def test_unrelated_pair_rejected_before_profiling(pair):
    from comparison.schemas import Compatibility
    from comparison.workflow import run_comparison

    class Model:
        def structured(self, schema, instruction, payload):
            assert schema is Compatibility
            assert "disclaimer" in instruction.lower()
            return schema(
                comparable=False,
                document_one_type="Insurance",
                document_two_type="Mortgage",
                reason="Risk coverage versus secured borrowing.",
            )

    store, docs = pair
    result = run_comparison(store, *docs, model=Model())
    assert result["status"] == "incompatible"
    assert store.read(docs[0]["id"], "document_profile", None) is None


def test_grouped_extraction_repairs_omitted_member_values():
    from comparison.extraction import extract_fields
    from comparison.holistic_schemas import Fields

    class Model:
        def structured(self, schema, instruction, payload):
            assert schema is Fields
            topics = payload["topics"]
            if topics == ["Terms"]:
                return Fields(
                    fields=[dict(topic="Terms", status="found", value="Amount: 100", evidence=[])]
                )
            return Fields(
                fields=[
                    dict(topic=t, status="found", value=v, evidence=[])
                    for t, v in zip(topics, ["100", "5 years"])
                ]
            )

    result = extract_fields(Model(), ["Terms"], [], aliases={"Terms": ["Amount", "Duration"]})
    assert len(result.fields) == 1
    assert result.fields[0].value == "Amount: 100\nDuration: 5 years"


def test_group_with_more_than_six_members_combines_batches():
    from comparison.extraction import extract_fields
    from comparison.holistic_schemas import Fields

    class Model:
        def structured(self, schema, instruction, payload):
            return Fields(
                fields=[
                    dict(topic=t, status="found", value="Answer", evidence=[])
                    for t in payload["topics"]
                ]
            )

    members = [f"Field {i}" for i in range(12)]
    result = extract_fields(Model(), ["Group"], [], aliases={"Group": members})
    assert len(result.fields) == 1
    assert all(m + ":" in result.fields[0].value for m in members)


def test_validation_error_does_not_blame_ollama(pair):
    from comparison.holistic_schemas import Fields
    from comparison.workflow import run_comparison

    class Broken:
        def structured(self, *args):
            return Fields(fields=[{}])

    store, docs = pair
    result = run_comparison(store, *docs, model=Broken())
    assert result["error_type"] == "ValidationError"
    assert "structured results failed validation" in result["answer"]
