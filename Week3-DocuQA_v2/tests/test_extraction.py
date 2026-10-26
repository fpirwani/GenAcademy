from comparison.extraction import extract_fields
from comparison.holistic_schemas import Fields


def test_deferred_value_is_not_stated():
    class Model:
        def structured(self, *args):
            return Fields(
                fields=[
                    {
                        "topic": "Processing duration",
                        "status": "found",
                        "value": "The final agreement would specify the processing duration.",
                        "evidence": [
                            {
                                "ref": "c1",
                                "quote": "The final agreement would specify the processing duration.",
                            }
                        ],
                    }
                ]
            )

    result = extract_fields(Model(), ["Processing duration"], [])
    assert result.fields[0].status == "not_stated"


def test_dropped_qualification_is_repaired_or_omitted():
    class Model:
        calls = 0

        def structured(self, schema, instruction, payload):
            self.calls += 1
            value = "Estimated $20; excludes delivery" if self.calls > 1 else "$20"
            return Fields(
                fields=[
                    {
                        "topic": "Price",
                        "status": "found",
                        "value": value,
                        "evidence": [{"ref": "c1", "quote": "Estimated $20; excludes delivery"}],
                    }
                ]
            )

    model = Model()
    result = extract_fields(model, ["Price"], [])
    assert model.calls == 2
    assert result.fields[0].value == "Estimated $20; excludes delivery"


def test_unrepaired_qualification_uses_complete_source_wording():
    class Model:
        def structured(self, *args):
            return Fields(
                fields=[
                    {
                        "topic": "Price",
                        "status": "found",
                        "value": "$20",
                        "evidence": [{"ref": "c1", "quote": "Estimated $20; excludes delivery"}],
                    }
                ]
            )

    assert (
        extract_fields(Model(), ["Price"], []).fields[0].value == "Estimated $20; excludes delivery"
    )
