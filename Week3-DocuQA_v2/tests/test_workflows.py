import pytest

from retrieval import retrieve
from workflows import answer_question, summarize

CHUNKS = [
    {
        "id": "c1",
        "text": "Mercury is the closest planet to the Sun.",
        "location": 1,
        "label": "Page 1",
    },
    {"id": "c2", "text": "Venus has a thick atmosphere.", "location": 2, "label": "Page 2"},
]


class FakeModel:
    def resolve(self, question, history):
        return "Mercury planet" if history else question

    def answer(self, question, chunks):
        return {
            "status": "answered",
            "answer": "Mercury is closest to the Sun.",
            "evidence": [{"chunk_id": "c1", "quote": "Mercury is the closest planet to the Sun."}],
        }

    def summarize(self, text):
        return "Summary: " + text[:120]


def test_bm25_returns_only_matching_active_document_chunks():
    assert retrieve(CHUNKS, "Mercury")[0]["id"] == "c1"
    assert retrieve(CHUNKS, "football") == []
    assert retrieve([CHUNKS[1]], "Mercury") == []
    assert retrieve([CHUNKS[0]], "planet")[0]["id"] == "c1"


def test_graph_returns_navigable_validated_evidence():
    result = answer_question("Which planet is closest?", CHUNKS, [], FakeModel())
    assert result["status"] == "answered"
    assert result["sources"][0]["location"] == 1
    assert result["sources"][0]["chunk_id"] == "c1"


def test_no_overlap_ends_without_generating_answer():
    class NoCalls(FakeModel):
        def answer(self, *args):
            pytest.fail("No evidence should end before generation")

    result = answer_question("football", CHUNKS, [], NoCalls())
    assert result["status"] == "not_found"
    assert result["sources"] == []


@pytest.mark.parametrize(
    "evidence",
    [
        [{"chunk_id": "unknown", "quote": "Mercury"}],
        [{"chunk_id": "c1", "quote": "Mercury is made of cheese"}],
        [],
    ],
)
def test_invalid_citations_cannot_be_presented_as_supported(evidence):
    class Invalid(FakeModel):
        def answer(self, *args):
            return {"status": "answered", "answer": "Invented", "evidence": evidence}

    result = answer_question("Mercury", CHUNKS, [], Invalid())
    assert result["status"] == "not_found"
    assert result["sources"] == []


def test_followup_uses_current_history():
    result = answer_question(
        "What about it?", CHUNKS, [{"role": "user", "content": "Mercury"}], FakeModel()
    )
    assert result["status"] == "answered"


def test_echoed_question_is_not_accepted_as_an_answer():
    class Echo(FakeModel):
        def answer(self, question, chunks):
            result = super().answer(question, chunks)
            result["answer"] = question
            return result

    result = answer_question("Mercury planet", CHUNKS, [], Echo())
    assert result["status"] != "answered"
    assert not result["sources"]


def test_model_failure_is_recoverable():
    class Offline(FakeModel):
        def answer(self, *args):
            raise ConnectionError("offline")

    result = answer_question("Mercury", CHUNKS, [], Offline())
    assert result["status"] == "error"
    assert result["sources"] == []


def test_malformed_structured_output_is_a_model_error():
    from models import Answer

    class Malformed(FakeModel):
        def answer(self, *args):
            return Answer.model_validate({"status": "invalid"})

    result = answer_question("Mercury", CHUNKS, [], Malformed())
    assert result["status"] == "error"
    assert result["sources"] == []


def test_summary_reads_entire_document_in_bounded_batches():
    seen = []

    class Summarizer(FakeModel):
        def summarize(self, text):
            seen.append(text)
            return "Short summary."

    text = "first " * 3000 + "FINAL_SECTION_UNIQUE"
    assert summarize([{"text": text}], Summarizer())
    assert any("FINAL_SECTION_UNIQUE" in batch for batch in seen)
    assert max(map(len, seen)) <= 10000
