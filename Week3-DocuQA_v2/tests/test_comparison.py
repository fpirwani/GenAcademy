import pytest

from comparison.alignment import align
from comparison.storage import ComparisonStore
from comparison.tools import research_tools
from comparison.validation import validate_claims
from storage import DocumentGone


def block(text, n=1):
    return {"text": text, "location": n, "label": f"Block {n}"}


def test_alignment_covers_insertions_and_moved_text():
    old = [block("First unchanged paragraph."), block("Second unchanged paragraph.", 2)]
    new = [
        block("Inserted completely new content."),
        block("Second unchanged paragraph.", 2),
        block("First unchanged paragraph.", 3),
    ]
    result = align(old, new)
    assert len(result) == 3
    assert sum(r["kind"] == "added" for r in result) == 1
    assert sum(r["earlier"] is not None for r in result) == 2
    assert sum(r["later"] is not None for r in result) == 3


def test_tools_scope_and_stale_generation(pair):
    store, docs = pair
    tools = research_tools(store, docs[0], "earlier", semantic=False)
    found = tools["search_passages"].invoke({"query": "cancellation"})
    assert found[0]["version"] == "earlier"
    assert found[0]["document_id"] == docs[0]["id"]
    assert "thirty" in found[0]["text"]
    assert "document_id" not in tools["search_passages"].args
    store.delete(docs[0]["id"])
    with pytest.raises(DocumentGone):
        tools["search_passages"].invoke({"query": "notice"})


def test_reject_wrong_version_and_invented_quote(pair):
    store, docs = pair
    evidence = research_tools(store, docs[0], "earlier", semantic=False)["search_passages"].invoke(
        {"query": "notice"}
    )
    ref = evidence[0]["ref"]
    valid = {
        "text": "Earlier requires thirty days.",
        "evidence": [{"ref": ref, "quote": "Cancellation requires thirty days written notice."}],
    }
    assert len(validate_claims([valid], evidence)) == 1
    valid["evidence"][0]["quote"] = "Cancellation requires fifteen days written notice."
    assert validate_claims([valid], evidence) == []


def test_ordered_pairs_clear_revision_and_delete(pair):
    store, docs = pair
    sessions = ComparisonStore(store)
    a = sessions.open(*docs)
    b = sessions.open(*reversed(docs))
    assert a["id"] != b["id"]
    sessions.clear(a["id"])
    with pytest.raises(ValueError, match="cleared"):
        sessions.save_result(a, {"answer": "stale"}, question="question")
    store.delete(docs[0]["id"])
    assert sessions.list() == []
    assert not list((store.root / "comparisons").glob("*.json"))


def test_corrupt_comparison_does_not_block_document_deletion(pair):
    store, docs = pair
    sessions = ComparisonStore(store)
    sessions.open(*docs)
    (store.root / "comparisons" / ("a" * 64 + ".json")).write_text("{broken")
    store.delete(docs[0]["id"])
    assert len(store.list()) == 1


def test_recompare_overwrites_saved_pair_in_either_order(pair):
    store, docs = pair
    sessions = ComparisonStore(store)
    first = sessions.open(*docs)
    sessions.save_result(first, {"answer": "first"})
    sessions.save_result(sessions.open(*docs), {"answer": "updated"})
    assert len(sessions.list()) == 1
    assert sessions.list()[0]["report"]["answer"] == "updated"
    reverse = sessions.open(*reversed(docs))
    sessions.save_result(reverse, {"answer": "reversed"})
    saved = sessions.list()
    assert len(saved) == 1
    assert saved[0]["report"]["answer"] == "reversed"
    assert saved[0]["documents"][0]["id"] == docs[1]["id"]
    assert len(store.list()) == 2
