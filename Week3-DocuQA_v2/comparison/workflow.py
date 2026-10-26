"""Selected-document compatibility and orchestration of verified comparison reports."""

import logging

from pydantic import ValidationError

from comparison.agents import AgentModel
from comparison.compatibility import assess
from comparison.holistic import build_report
from storage import DocumentGone


def run_comparison(
    store,
    earlier,
    later,
    *,
    model=None,
    semantic=True,
    progress=lambda message: None,
    on_rows=lambda rows: None,
):
    trace, warnings, changes = [], [], []
    coverage = {"total_changes": 0, "analyzed_changes": 0, "unresolved_alignments": 0}
    if earlier["id"] == later["id"]:
        raise ValueError("Select two different documents.")

    def check():
        with store.lock:
            for doc in (earlier, later):
                if store.read(doc["id"], "metadata")["generation"] != doc["generation"]:
                    raise DocumentGone("A selected document was replaced.")

    try:
        check()
        model = model or AgentModel()
        progress("Checking whether Document One and Document Two are comparable...")
        compatibility = assess(
            model,
            store.read(earlier["id"], "blocks", []),
            store.read(later["id"], "blocks", []),
        )
        check()
        if not compatibility.comparable:
            return {
                "status": "incompatible",
                "claims": [],
                "changes": [],
                "trace": [],
                "warnings": [],
                "coverage": coverage,
                "compatibility": compatibility.model_dump(),
                "answer": "Cannot compare Document One to Document Two: " + compatibility.reason,
            }
        result = build_report(
            store,
            earlier,
            later,
            model=model,
            document_type=compatibility.document_one_type + " / " + compatibility.document_two_type,
            progress=progress,
            on_rows=on_rows,
        )
        result["compatibility"] = compatibility.model_dump()
        check()
        return result
    except DocumentGone:
        raise
    except Exception as exc:
        # Log type and frame locations without model outputs or document values.
        import traceback

        frames = [(frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)]
        logging.getLogger(__name__).error(
            "Comparison failed: %s; frames=%s", type(exc).__name__, frames
        )
        return {
            "status": "error",
            "claims": [],
            "changes": changes,
            "coverage": coverage,
            "warnings": warnings,
            "trace": trace,
            "answer": (
                "The comparison could not finish because structured results failed validation. "
                "This is a processing error, not necessarily an Ollama connection problem."
                if isinstance(exc, ValidationError)
                else f"The local comparison could not finish ({type(exc).__name__}). "
                "Check the application log for details and retry."
            ),
            "error_type": type(exc).__name__,
        }
