import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import providers


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def test_available_providers_in_order(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    monkeypatch.setenv("ORACLELLM_API_KEY", "b")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("LLM7IO_API_KEY", raising=False)
    assert providers.available_providers() == ["agnes", "oraclellm"]


def test_call_llm_native_tool_calls(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                }
            }
        ]
    }

    def fake_post(url, headers, json, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    assert result["type"] == "tool_calls"
    assert result["tool_calls"][0]["function"]["name"] == "search"


def test_react_fallback_when_tools_rejected(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    calls = []

    def fake_post(url, headers, json, timeout=None):
        calls.append(json)
        if "tools" in json:
            return FakeResponse({"error": "tool calling not supported"}, status=400)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"tool": "search", "arguments": {"q": "x"}}',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    assert len(calls) == 2 and "tools" in calls[0] and "tools" not in calls[1]
    assert result["type"] == "text"
    assert '"tool"' in result["content"]


def test_all_providers_dead_raises(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def fake_post(url, headers, json, timeout=None):
        raise ConnectionError("down")

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    try:
        providers.call_llm([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "No LLM provider succeeded" in str(e)


def test_no_tools_returns_plain_text(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def fake_post(url, headers, json, timeout=None):
        return FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "hello answer"}}]}
        )

    monkeypatch.setattr(providers.httpx.Client, "post", fake_post)
    result = providers.call_llm([{"role": "user", "content": "hi"}])
    assert result["type"] == "text"
    assert result["content"] == "hello answer"