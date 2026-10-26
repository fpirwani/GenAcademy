from test_holistic import ExtractionModel

from comparison.schemas import Compatibility
from comparison.workflow import run_comparison


class ScriptedModel(ExtractionModel):
    def structured(self, schema, instruction, payload):
        if schema is Compatibility:
            return Compatibility(
                comparable=True,
                document_one_type="Policy",
                document_two_type="Policy",
                reason="Both describe cancellation terms.",
            )
        return super().structured(schema, instruction, payload)


def test_automatic_report_contains_topic_rows(pair):
    store, docs = pair
    result = run_comparison(store, *docs, model=ScriptedModel(), semantic=False)
    assert result["rows"][0]["topic"] == "Cancellation"
    assert "thirty" in result["rows"][0]["one"]["value"]
    assert "fifteen" in result["rows"][0]["two"]["value"]


def test_failed_model_returns_error_without_factual_answer(pair):
    class Broken:
        def structured(self, *args):
            raise RuntimeError("offline")

    store, docs = pair
    result = run_comparison(store, *docs, model=Broken(), semantic=False)
    assert result["status"] == "error"
    assert result["claims"] == []


def test_unrelated_documents_stop_before_research_in_both_modes(pair):
    class Unrelated:
        def structured(self, schema, instruction, payload):
            if schema is not Compatibility:
                return ExtractionModel().structured(schema, instruction, payload)
            return Compatibility(
                comparable=False,
                document_one_type="Employment contract",
                document_two_type="Recipe",
                reason="The subject and purpose differ.",
            )

    store, docs = pair
    for _ in range(1):
        result = run_comparison(store, *docs, model=Unrelated(), semantic=False)
        assert result["status"] == "incompatible"
        assert "Cannot compare Document One to Document Two" in result["answer"]
        assert "subject and purpose differ" in result["answer"]
        assert not result["trace"]
        assert not result["changes"]


def test_many_text_edits_do_not_create_line_by_line_rows(pair):
    store, docs = pair
    for doc in docs:
        text = store.read(doc["id"], "blocks")[0]["text"]
        store.write(
            doc["id"],
            doc["generation"],
            "blocks",
            [{"text": text, "location": i + 1, "label": str(i)} for i in range(65)],
        )
    result = run_comparison(store, *docs, model=ScriptedModel(), semantic=False)
    assert [r["topic"] for r in result["rows"]] == ["Cancellation", "Portability"]
    assert not result["changes"]
