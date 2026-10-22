"""Exercise the real LangChain/Ollama JSON protocol without model downloads."""

import json

import httpx
from langchain_ollama import ChatOllama

import models


def test_orphan_thinking_end_tag_does_not_expose_process():
    assert (
        models.clean_text("I need to plan this.\n</think>\nA budget report.") == "A budget report."
    )
    assert models.clean_text("<think>Unfinished thinking") == ""


def test_followup_rewrite_uses_structured_question(monkeypatch):
    class Structured:
        def invoke(self, messages):
            return models.SearchQuestion(question="How is the employment rate calculated?")

    class Chat:
        def with_structured_output(self, schema, method):
            assert schema is models.SearchQuestion
            assert method == "json_schema"
            return Structured()

    monkeypatch.setattr(models, "ChatOllama", lambda **kwargs: Chat())
    assert (
        models.LocalModel().resolve(
            "How is it calculated?",
            [{"role": "user", "content": "Tell me about the employment rate."}],
        )
        == "How is the employment rate calculated?"
    )


def test_local_model_uses_non_thinking_structured_output(monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        answer = {
            "status": "answered",
            "answer": "The budget is $400.",
            "evidence": [{"chunk_id": "c1", "quote": "The budget is $400."}],
        }
        if "summary" in body.get("format", {}).get("properties", {}):
            content = json.dumps(
                {"summary": "A budget report.", "reasoning": "Private analysis to omit."}
            )
        else:
            content = (
                json.dumps(answer)
                if body.get("format")
                else "Hmm, I need to analyze the request first. A budget report."
            )
        response = {
            "model": "qwen3:4b",
            "created_at": "2026-10-21T12:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "total_duration": 100,
            "load_duration": 1,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 10,
            "eval_count": 10,
            "eval_duration": 20,
        }
        return httpx.Response(
            200,
            content=json.dumps(response) + "\n",
            headers={"content-type": "application/x-ndjson"},
        )

    def chat_factory(**kwargs):
        kwargs["client_kwargs"]["transport"] = httpx.MockTransport(handler)
        return ChatOllama(**kwargs)

    monkeypatch.setattr(models, "ChatOllama", chat_factory)
    local = models.LocalModel()
    assert local.summarize("The budget is $400.") == "A budget report."
    answer = local.answer("What is the budget?", [{"id": "c1", "text": "The budget is $400."}])
    assert answer["evidence"][0]["chunk_id"] == "c1"
    assert answer["status"] == "answered"
    assert all(url == "http://127.0.0.1:11434/api/chat" for url, _ in requests)
    assert all(body["think"] is False for _, body in requests)
    assert requests[-1][1]["format"]["type"] == "object"


def test_embeddings_use_loopback_retrieval_prefixes_and_model_digest(monkeypatch):
    from langchain_ollama import OllamaEmbeddings

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    def factory(**kwargs):
        assert kwargs["client_kwargs"]["trust_env"] is False
        kwargs["client_kwargs"]["transport"] = httpx.MockTransport(handler)
        return OllamaEmbeddings(**kwargs)

    monkeypatch.setattr(models, "OllamaEmbeddings", factory)
    monkeypatch.setattr(
        models.httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            200,
            json={"models": [{"name": "nomic-embed-text:latest", "digest": "version-a"}]},
            request=httpx.Request("GET", models.OLLAMA_URL + "/api/tags"),
        ),
    )
    embeddings = models.LocalEmbeddings()
    assert "version-a" in embeddings.fingerprint
    assert embeddings.embed_documents(["Contract terms"]) == [[1.0, 0.0]]
    assert embeddings.embed_query("Can I leave?") == [1.0, 0.0]
    assert all(url == "http://127.0.0.1:11434/api/embed" for url, _ in requests)
    assert requests[0][1]["input"] == ["search_document: Contract terms"]
    assert requests[1][1]["input"] == ["search_query: Can I leave?"]
