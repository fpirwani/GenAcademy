"""Evaluate the manifest against currently saved documents without changing chats."""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, EMBEDDING_MODEL, MODEL  # noqa: E402
from models import LocalEmbeddings, LocalModel  # noqa: E402
from retrieval import HybridSearch, retrieve  # noqa: E402
from storage import Store  # noqa: E402
from workflows import answer_question, normalize  # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((OUT / "cases.json").read_text("utf-8"))
    store = Store(DATA_DIR)
    available = {d["filename"]: d for d in store.list()}
    docs = {k: available[n] for k, n in manifest["documents"].items() if n in available}
    embedding = LocalEmbeddings()
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model=MODEL,
        embedding_model=EMBEDDING_MODEL,
        embedding_fingerprint=embedding.fingerprint,
        corpus={},
        cases=[],
        skipped=[],
    )
    for key, d in docs.items():
        blocks, chunks = store.read(d["id"], "blocks"), store.read(d["id"], "chunks")
        HybridSearch(store, d, embedding).ensure(chunks)
        report["corpus"][key] = dict(
            filename=d["filename"],
            sha256=d["id"],
            pages=len(blocks),
            chunks=chunks,
            characters=sum(len(b["text"]) for b in blocks),
        )

    class TracedModel(LocalModel):
        last_raw = None

        def answer(self, question, chunks):
            self.last_raw = super().answer(question, chunks)
            return self.last_raw

    for case in manifest["cases"]:
        if case["doc"] not in docs:
            report["skipped"].append(
                dict(id=case["id"], reason="Source document is no longer saved")
            )
            continue
        chunks = report["corpus"][case["doc"]]["chunks"]
        gold = [
            c["id"]
            for c in chunks
            if any(normalize(a) in normalize(c["text"]) for a in case["anchors"])
        ]
        if case["anchors"] and not gold:
            report["skipped"].append(
                dict(id=case["id"], reason="Gold anchors no longer match the uploaded version")
            )
            continue
        history = []
        if case.get("followup_to"):
            previous = next((r for r in report["cases"] if r["id"] == case["followup_to"]), None)
            if previous is None:
                report["skipped"].append(
                    dict(id=case["id"], reason="Follow-up predecessor unavailable")
                )
                continue
            history = [
                dict(role="user", content=previous["question"]),
                dict(role="assistant", content=previous["result"]["answer"]),
            ]
        search = HybridSearch(store, docs[case["doc"]], embedding)
        captured = {}

        def traced(items, query):
            found = search.search(items, query)
            captured.update(
                query=query,
                retrieved=[c["id"] for c in found],
                bm25=[c["id"] for c in retrieve(items, query)],
            )
            return found

        model = TracedModel()
        started = time.perf_counter()
        result = answer_question(case["question"], chunks, history, model, traced)
        row = dict(
            **case,
            **captured,
            gold=gold,
            result=result,
            raw=model.last_raw,
            seconds=round(time.perf_counter() - started, 2),
            warning=search.warning,
        )
        for method, field in [("hybrid", "retrieved"), ("bm25", "bm25")]:
            ids = row.get(field, [])
            hits = set(gold).intersection(ids)
            row[method + "_recall_at_5"] = len(hits) / len(gold) if gold else None
            row[method + "_precision_at_5"] = len(hits) / 5 if gold else None
            row[method + "_mrr_at_5"] = (
                next((1 / i for i, c in enumerate(ids, 1) if c in gold), 0) if gold else None
            )
        report["cases"].append(row)
        (OUT / "current_results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Q{case['id']} {result['status']} hybrid recall={row['hybrid_recall_at_5']} fallback={bool(search.warning)}",
            flush=True,
        )
    scored = [r for r in report["cases"] if r["gold"]]
    report["retrieval_summary"] = {
        m + "_" + s: statistics.mean(r[m + "_" + s] for r in scored) if scored else None
        for m in ["bm25", "hybrid"]
        for s in ["recall_at_5", "precision_at_5", "mrr_at_5"]
    }
    (OUT / "current_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["retrieval_summary"], indent=2))
    print("Skipped:", report["skipped"])


if __name__ == "__main__":
    main()
