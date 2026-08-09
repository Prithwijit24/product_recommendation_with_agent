"""LangChain-native LLM router: one BaseChatModel over every provider in api_config.yml.

Each provider is a ChatOpenAI endpoint (OpenAI-compatible). A provider that rejects
`tools` or fails mid-session is marked degraded and is retried in plain-text mode for the rest of
the process run. LangSmith tracing is enabled lazily when LANGSMITH_API_KEY is set.
"""

from __future__ import annotations

import os
from typing import Any, Sequence  # noqa: UP035

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger

from . import config


def _langsmith_cfg() -> dict:
    return config.get_section("langsmith")


def _providers_cfg() -> dict:
    return config.get_section("providers")


def configure_langsmith(project: str | None = None) -> bool:
    """Opt-in LangSmith tracing. No key -> graceful no-op and logged hint."""
    cfg = _langsmith_cfg()
    api_key = os.getenv(cfg.get("api_key_env", "LANGSMITH_API_KEY"), "")
    if not api_key:
        logger.warning("LANGSMITH_API_KEY not set - LangSmith tracing disabled")
        return False
    os.environ.setdefault("LANGSMITH_TRACING", cfg.get("tracing_enabled", "true"))
    os.environ.setdefault("LANGSMITH_PROJECT", f"{project or cfg.get('project', 'skincare-orchestrator')}")
    logger.info(f"LangSmith tracing enabled (project={os.environ.get('LANGSMITH_PROJECT')})")
    return True


def available_providers() -> list[str]:
    """Providers (yml order) whose env key is set."""
    return [p["name"] for p in config.get_providers() if os.getenv(p["env_key"])]


def diagnose() -> dict:
    """Return diagnostic info about provider configuration."""
    configured = config.get_providers()
    available = available_providers()
    return {
        "configured": [p["name"] for p in configured],
        "available": available,
        "missing_env": [
            {"name": p["name"], "env_key": p["env_key"]}
            for p in configured
            if p["name"] not in available
        ],
    }


def _provider_models(provider: dict, role: str) -> list[str]:
    raw = provider.get(f"{role}_model", "")
    if isinstance(raw, list):
        return [str(m) for m in raw]
    return [str(raw)] if raw else []


def _chat_openai(provider: dict, model: str, role: str) -> ChatOpenAI:
    """Concrete OpenAI-compatible client for one provider+model."""
    cfg = _providers_cfg()
    return ChatOpenAI(
        model=model,
        base_url=str(provider["base_url"]).rstrip("/"),
        api_key=os.getenv(provider["env_key"], ""),
        timeout=float(provider.get(f"{role}_timeout", config.get_default("http_timeout", 60.0))),
        temperature=float(cfg.get("temperature", 0)),
        max_retries=int(cfg.get("max_retries", 0)),
    )


class ProviderRouter(BaseChatModel):
    """Drop-in LangChain chat model that fails over across every configured provider.

    - Attempts providers in yml order, and each provider's models in order.
    - If a provider rejects tool calling (or errors), it is marked degraded and is
      retried in plain-text mode; the compiled LangGraph agent then drives it via
      ReAct-JSON tool calls instead of native tool calls.
    - Implements ``bind_tools`` so it plugs directly into LangGraph ToolNode graphs.
    """

    _role: str = "planner"
    _providers: list[dict]
    _tools: tuple[BaseTool, ...] = ()
    _tool_choice: str | None = None
    _chat_cache: dict
    _degraded: set[str]

    def __init__(
        self,
        role: str = "planner",
        providers: Sequence[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        cfg = _providers_cfg()
        self._role = role
        listed = list(providers) if providers is not None else config.get_providers()
        self._providers = [p for p in listed if os.getenv(p["env_key"])]
        self._tools = ()
        self._tool_choice = cfg.get("default_tool_choice")
        self._chat_cache = {}
        self._degraded = set()

    @property
    def _llm_type(self) -> str:
        return "provider_router"

    @property
    def model_name(self) -> str:
        for p in self._providers:
            for m in _provider_models(p, self._role):
                return f"{p['name']}:{m}"
        return self._role

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ProviderRouter:
        """Bind tools for native tool-calling; identical router, tools attached."""
        cfg = _providers_cfg()
        bound = ProviderRouter(role=self._role, providers=self._providers)
        bound._tools = tuple(tools)
        bound._tool_choice = tool_choice or cfg.get("default_tool_choice")
        bound._chat_cache = dict(self._chat_cache)
        bound._degraded = set(self._degraded)
        return bound

    # -- BaseChatModel protocol ------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self._providers:
            raise RuntimeError(
                "No LLM providers available. "
                "Check that provider API keys are set in your .env file. "
                f"Required env vars: {[p['env_key'] for p in config.get_providers()]}"
            )
        last_error: Exception | None = None
        for provider in self._providers:
            if self._tools and provider["name"] in self._degraded:
                logger.debug(f"Skipping degraded provider: {provider['name']}")
                continue
            for model in _provider_models(provider, self._role):
                logger.debug(f"Trying provider: {provider['name']}/{model}")
                chat = self._chat_for(provider, model)
                try:
                    if self._tools:
                        invocation = chat.bind_tools(
                            list(self._tools), tool_choice=self._tool_choice or "auto"
                        )
                    else:
                        invocation = chat
                    ai: AIMessage = invocation.invoke(list(messages), stop=stop)
                    logger.info(f"LLM call succeeded: {provider['name']}/{model}")
                    return ChatResult(generations=[ChatGeneration(message=ai)])
                except Exception as exc:  # noqa: BLE001 - router must not kill the session
                    last_error = exc
                    logger.warning(f"{provider['name']}/{model} failed: {exc}")
        # All tool-capable providers failed: retry everything in degraded text mode.
        return self._degraded_generate(messages, stop=stop, last_error=last_error)

    def _degraded_generate(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None,
        last_error: Exception | None,
    ) -> ChatResult:
        logger.info("All native tool-calling providers failed, retrying in degraded text mode")
        for provider in self._providers:
            for model in _provider_models(provider, self._role):
                logger.debug(f"Degraded mode trying: {provider['name']}/{model}")
                chat = self._chat_for(provider, model)
                try:
                    ai = chat.invoke(list(messages), stop=stop)
                    logger.info(f"LLM call succeeded (degraded): {provider['name']}/{model}")
                    return ChatResult(generations=[ChatGeneration(message=ai)])
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning(f"{provider['name']}/{model} failed (degraded): {exc}")
        raise RuntimeError(
            f"No LLM provider succeeded. Last error: {last_error}. "
            f"Checked providers: {[p['name'] for p in self._providers]}"
        )

    # -- internals -------------------------------------------------------------

    def _chat_for(self, provider: dict, model: str) -> ChatOpenAI:
        key = (provider["name"], model)
        if key not in self._chat_cache:
            self._chat_cache[key] = _chat_openai(provider, model, self._role)
        return self._chat_cache[key]


def chat_model(role: str = "planner") -> ProviderRouter:
    """Convenience factory: one router per role (planner/worker)."""
    return ProviderRouter(role=role)


def configure() -> None:
    """One-shot LangSmith activation (idempotent)."""
    configure_langsmith()


# Backward-compat alias used by the worker sub-agents
def call_llm(
    messages: list[BaseMessage],
    tools: Sequence[BaseTool] | None = None,
    *,
    role: str = "planner",
) -> AIMessage:
    """Route an LLM call; returns an AIMessage (tools native or degraded text)."""
    router = chat_model(role=role)
    if tools:
        return router.bind_tools(list(tools)).invoke(list(messages))
    return router.invoke(list(messages))
