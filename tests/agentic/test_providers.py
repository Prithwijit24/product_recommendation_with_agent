import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from langchain_core.messages import AIMessage

from project_folder.agentic import providers


def _provider(name="agnes", planner_model="m-plan", worker_model="m-work"):
    return {
        "name": name,
        "base_url": f"https://{name}.example/v1",
        "env_key": f"{name.upper()}_API_KEY",
        "planner_model": planner_model,
        "worker_model": worker_model,
        "planner_timeout": 10.0,
        "worker_timeout": 10.0,
    }


class FakeChat:
    """Minimal langchain-style chat double: invoke() -> AIMessage, raises on demand."""

    def __init__(self, impl):
        self.impl = impl
        self.calls = []

    def bind_tools(self, tools, tool_choice="auto"):
        return FakeBoundChat(self, tools)

    def invoke(self, messages, **kwargs):
        self.calls.append(([m.content for m in messages], kwargs))
        return self.impl(messages, tools=None)


class FakeBoundChat:
    def __init__(self, base, tools):
        self.base = base
        self.tools = tools

    def invoke(self, messages, **kwargs):
        self.base.calls.append(([m.content for m in messages], kwargs))
        return self.base.impl(messages, tools=self.tools)


def test_available_providers_in_order(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    monkeypatch.setenv("ORACLELLM_API_KEY", "b")
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("LLM7IO_API_KEY", raising=False)
    assert providers.available_providers() == ["agnes", "oraclellm"]


def test_router_fails_over_to_second_provider(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")
    monkeypatch.setenv("ORACLELLM_API_KEY", "b")
    agnes_chat = FakeChat(lambda m, tools: (_ for _ in ()).throw(ConnectionError("down")))
    oracle_chat = FakeChat(lambda m, tools: AIMessage(content="hello from oracle"))

    def fake_factory(provider, model, role):
        return agnes_chat if provider["name"] == "agnes" else oracle_chat

    monkeypatch.setattr(providers, "_chat_openai", fake_factory)
    router = providers.ProviderRouter(providers=[_provider("agnes"), _provider("oraclellm")])
    out = router.invoke([{"role": "user", "content": "hi"}])
    assert out.content == "hello from oracle"
    assert len(agnes_chat.calls) == 1 and len(oracle_chat.calls) == 1


def test_native_tool_calls_flow_through_router(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def impl(messages, tools=None):
        assert tools is not None
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "research_skin_concerns",
                    "args": {"profile": "x"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

    chat = FakeChat(impl)
    monkeypatch.setattr(providers, "_chat_openai", lambda p, m, r: chat)
    router = providers.ProviderRouter(providers=[_provider("agnes")])
    dummy_tool = object()
    out = router.bind_tools([dummy_tool]).invoke([{"role": "user", "content": "hi"}])
    assert out.tool_calls and out.tool_calls[0]["name"] == "research_skin_concerns"
    assert len(chat.calls) == 1


def test_tool_rejecting_provider_degrades_to_text_mode(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def impl(messages, tools=None):
        if tools is not None:
            raise RuntimeError("tool calling not supported")
        return AIMessage(content='{"tool": "research_skin_concerns", "arguments": {"profile": "x"}}')

    chat = FakeChat(impl)
    monkeypatch.setattr(providers, "_chat_openai", lambda p, m, r: chat)
    router = providers.ProviderRouter(providers=[_provider("agnes")])
    out = router.bind_tools([object()]).invoke([{"role": "user", "content": "hi"}])
    assert len(chat.calls) == 2  # tools attempt, then degraded text attempt
    assert '"tool"' in out.content


def test_all_providers_dead_raises(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "a")

    def dead(messages, tools=None):
        raise ConnectionError("down")

    chat = FakeChat(dead)
    monkeypatch.setattr(providers, "_chat_openai", lambda p, m, r: chat)
    router = providers.ProviderRouter(providers=[_provider("agnes")])
    try:
        router.invoke([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        assert "No LLM provider succeeded" in str(exc)


def test_langsmith_configure_requires_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert providers.configure_langsmith() is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-123")
    assert providers.configure_langsmith("unit") is True
    assert providers.os.environ["LANGSMITH_TRACING"] == "true"
    assert providers.os.environ["LANGSMITH_PROJECT"] == "unit"