"""Hybrid LLM router: native OpenAI tool-calling with ReAct-JSON fallback.

Provider order and models come from config/api_config.yml via config.py.
A provider that rejects `tools` or fails mid-session is degraded to ReAct mode
for the rest of the process run.
"""

from __future__ import annotations

import os

import httpx

from . import config

DEGRADED_SESSION: set[str] = set()


class ToolNotSupportedError(RuntimeError):
    pass


def available_providers() -> list[str]:
    """Providers (yml order) whose env key is set."""
    return [p["name"] for p in config.get_providers() if os.getenv(p["env_key"])]


def _post(p: dict, payload: dict, timeout: float) -> dict:
    """POST {base_url}/chat/completions. Raises ToolNotSupportedError on tool-reject."""
    url = f"{p['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv(p['env_key'], '')}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.post(url, json=payload)
        if resp.status_code == 400 and "tool" in (resp.text or "").lower():
            raise ToolNotSupportedError(p["name"])
        if resp.status_code != 200:
            raise RuntimeError(
                f"{p['name']}: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()


def _parse(provider: str, raw: dict) -> dict:
    msg = raw["choices"][0]["message"]
    if msg.get("tool_calls"):
        return {
            "type": "tool_calls",
            "provider": provider,
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }
    return {
        "type": "text",
        "provider": provider,
        "content": msg.get("content") or "",
        "tool_calls": [],
    }


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    role: str = "planner",
) -> dict:
    """Try providers in yml order; fall through to next on failure. Never raises on a single provider."""
    last_error: Exception | None = None
    for p in config.get_providers():
        if not os.getenv(p["env_key"]):
            continue
        model = p.get(f"{role}_model")
        timeout = float(p.get(f"{role}_timeout", 60.0))
        try:
            payload = {"model": model, "messages": messages}
            if tools and p["name"] not in DEGRADED_SESSION:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            raw = _post(p, payload, timeout)
            return _parse(p["name"], raw)
        except ToolNotSupportedError:
            DEGRADED_SESSION.add(p["name"])
            try:
                payload = {"model": model, "messages": messages}
                raw = _post(p, payload, timeout)
                return _parse(p["name"], raw)
            except Exception as e:  # noqa: BLE001 - router must not kill the session
                last_error = e
        except Exception as e:  # noqa: BLE001
            last_error = e
            DEGRADED_SESSION.add(p["name"])
    raise RuntimeError(f"No LLM provider succeeded. Last error: {last_error}")