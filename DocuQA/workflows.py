"""Small explicit LangGraph workflows, with no persistent graph checkpoints."""

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from retrieval import retrieve

NOT_FOUND = "I couldn't find an answer to that question in this document. Try rephrasing it or asking about another section."
UNRELATED = (
    "Please ask a question about the selected document. I can only answer from its contents."
)
MODEL_ERROR = (
    "The local model could not complete this request. Check that Ollama is running, then try again."
)


class QAState(TypedDict, total=False):
    question: str
    chunks: list[dict]
    history: list[dict]
    query: str
    passages: list[dict]
    result: dict


def refusal(status="not_found"):
    return {
        "status": status,
        "answer": UNRELATED if status == "unrelated" else NOT_FOUND,
        "sources": [],
    }


def normalize(text: str) -> str:
    return " ".join(text.split())


def validate_answer(raw: dict, passages: list[dict]) -> dict:
    if not isinstance(raw, dict):
        return refusal()
    if raw.get("status") != "answered":
        return refusal("unrelated" if raw.get("status") == "unrelated" else "not_found")
    if not isinstance(raw.get("answer"), str) or not raw["answer"].strip():
        return refusal()
    by_id = {chunk["id"]: chunk for chunk in passages}
    sources = []
    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list):
        return refusal()
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            return refusal()
        chunk = by_id.get(item.get("chunk_id"))
        quote = normalize(item["quote"])
        if not chunk or len(quote) < 15 or quote not in normalize(chunk["text"]):
            return refusal()
        if not any(source["chunk_id"] == chunk["id"] for source in sources):
            sources.append(
                {
                    "chunk_id": chunk["id"],
                    "location": chunk["location"],
                    "label": chunk["label"],
                    "quote": item["quote"],
                }
            )
    if not sources:
        return refusal()
    return {"status": "answered", "answer": raw["answer"].strip(), "sources": sources}


def answer_question(
    question: str, chunks: list[dict], history: list[dict], model, search=None
) -> dict:
    search_fn = search or retrieve
    if not question.strip() or len(question) > 2000:
        return {
            "status": "error",
            "answer": "Enter a question between 1 and 2,000 characters.",
            "sources": [],
        }

    def resolve(state):
        query = (
            model.resolve(state["question"], state["history"])
            if state["history"]
            else state["question"]
        )
        return {"query": query}

    def search(state):
        return {"passages": search_fn(state["chunks"], state["query"])}

    def generate(state):
        raw = model.answer(state["query"], state["passages"])
        # A rewritten question with citations is still not an answer.
        if isinstance(raw, dict) and isinstance(raw.get("answer"), str):
            answer_text = normalize(raw["answer"]).casefold().rstrip(" ?.!\n")
            questions = {
                normalize(state[name]).casefold().rstrip(" ?.!\n") for name in ("query", "question")
            }
            if answer_text in questions:
                return {"result": refusal()}
        result = validate_answer(raw, state["passages"])
        return {"result": result}

    graph = StateGraph(QAState)
    graph.add_node("resolve", resolve)
    graph.add_node("retrieve", search)
    graph.add_node("answer", generate)
    graph.add_node("decline", lambda state: {"result": refusal()})
    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "retrieve")
    graph.add_conditional_edges(
        "retrieve", lambda state: "answer" if state["passages"] else "decline"
    )
    graph.add_edge("answer", END)
    graph.add_edge("decline", END)
    try:
        return graph.compile().invoke(
            {"question": question.strip(), "chunks": chunks, "history": history}
        )["result"]
    except Exception:
        return {"status": "error", "answer": MODEL_ERROR, "sources": []}


class SummaryState(TypedDict, total=False):
    texts: list[str]
    summaries: list[str]
    summary: str


def batches(texts: list[str], limit=9000) -> list[str]:
    # Fixed character slices ensure even a single very long block is bounded.
    combined = "\n\n".join(texts)
    return [combined[index : index + limit] for index in range(0, len(combined), limit)]


def summarize(blocks: list[dict], model, progress: Callable[[str], None] = lambda _: None) -> str:
    def map_sections(state):
        sections = batches(state["texts"])
        summaries = []
        for index, section in enumerate(sections, 1):
            progress(f"Summarizing section {index} of {len(sections)}…")
            summaries.append(model.summarize(section))
        return {"summaries": summaries}

    def combine(state):
        summaries = state["summaries"]
        for _ in range(8):
            if len(summaries) == 1:
                return {"summary": summaries[0]}
            progress("Combining the document summary…")
            previous_size = sum(map(len, summaries))
            summaries = [model.summarize(section) for section in batches(summaries)]
            if len(summaries) > 1 and sum(map(len, summaries)) >= previous_size:
                raise ValueError("The model could not condense the summary. Please retry.")
        raise ValueError("The summary could not be completed. Please retry.")

    graph = StateGraph(SummaryState)
    graph.add_node("summarize_sections", map_sections)
    graph.add_node("combine", combine)
    graph.add_edge(START, "summarize_sections")
    graph.add_edge("summarize_sections", "combine")
    graph.add_edge("combine", END)
    return graph.compile().invoke({"texts": [block["text"] for block in blocks]})["summary"]
