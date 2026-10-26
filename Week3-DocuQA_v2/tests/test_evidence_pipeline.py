from test_holistic import ExtractionModel

from comparison.holistic import build_report
from comparison.holistic_schemas import Fields, Topics


def test_report_extracts_before_applying_display_limit(pair):
    class Model(ExtractionModel):
        def structured(self, schema, instruction, payload):
            if schema is Topics:
                return Topics(topics=[f"Field {i}" for i in range(7)])
            if schema is Fields:
                p = payload["passages"][0]
                return Fields(
                    fields=[
                        {
                            "topic": t,
                            "status": "found",
                            "value": p["text"],
                            "evidence": [{"ref": p["ref"], "quote": p["text"]}],
                        }
                        for t in payload["topics"]
                    ]
                )
            return super().structured(schema, instruction, payload)

    store, docs = pair
    result = build_report(store, *docs, model=Model(), document_type="Unknown")
    assert len(result["rows"]) == 7
    assert all(r["one"]["status"] == r["two"]["status"] == "found" for r in result["rows"])
    assert all(len(g["details"]) <= 5 for g in result["presentation"])


def test_report_rejects_changed_numbers_even_when_reviewer_accepts(pair):
    class Model(ExtractionModel):
        def structured(self, schema, instruction, payload):
            result = super().structured(schema, instruction, payload)
            if schema is Fields:
                for field in result.fields:
                    if field.status == "found":
                        field.value = "999 days"
            return result

    store, docs = pair
    result = build_report(store, *docs, model=Model(), document_type="Unknown")
    assert result["rows"][0]["one"]["status"] == "unverified"


def test_numeric_validation_accepts_formatting_but_not_changed_values():
    from comparison.independent import numeric_support

    sources = [{"quote": "The amount is 1,000.00 and the rate is 6.250%."}]
    assert numeric_support({"text": "1000 at 6.25%", "sources": sources})
    assert not numeric_support({"text": "1000 at 6.75%", "sources": sources})
