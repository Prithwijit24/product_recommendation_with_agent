"""Research sub-agents: skin-concern inference and ingredient shortlisting.

Each is a small worker-role LLM loop over aistack search/crawl evidence, driven by a
LangChain worker chat model (ProviderRouter). Search evidence is injected as context,
and the worker is asked for strict JSON. Responses are validated against Pydantic models.
"""

from __future__ import annotations

import json
import logging

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from . import aistack, config, providers
from .models import IngredientSuggestion, SkinConcern, parse_list_and_validate

logger = logging.getLogger(__name__)


def _research_cfg() -> dict:
    return config.get_section("research")


def _gather_evidence(client, query: str, rounds: int | None = None) -> str:
    cfg = _research_cfg().get("evidence", {})
    if rounds is None:
        rounds = int(cfg.get("rounds", 2))
    max_results = int(cfg.get("max_results_per_round", 3))
    max_chunks = int(cfg.get("max_chunks", 6))
    chunk_char_limit = int(cfg.get("chunk_char_limit", 300))

    chunks = []
    seen: set[str] = set()
    try:
        for r in range(rounds):
            results = client.search(query, max_results=max_results)
            for res in results:
                url = res.get("url", "")
                if url in seen or len(chunks) >= max_chunks:
                    continue
                seen.add(url)
                chunks.append(f"- {res.get('title','')}: {res.get('content','')[:chunk_char_limit]}")
    except (httpx.HTTPError, RuntimeError):
        return "- no evidence gathered"
    return "\n".join(chunks) or "- no evidence gathered"


def research_skin_concerns(
    profile_text: str, worker=None, client: aistack.AistackClient | None = None
) -> list[SkinConcern]:
    """Infer likely skin-science considerations for the user's profile."""
    cfg = _research_cfg()
    prompts = cfg.get("prompts", {})
    system = prompts.get("skin_concerns", "").format(profile=profile_text)
    logger.info(f"[research_skin_concerns] profile={profile_text[:80]}...")
    client = client or aistack.AistackClient()
    evidence = _gather_evidence(client, f"Profile: {profile_text}")
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Profile: {profile_text}. Research and output the JSON now."),
        HumanMessage(content=f"Evidence from search:\n{evidence}\nNow respond strictly in JSON."),
    ]
    from langchain_core.messages import AIMessage

    try:
        if worker is None:
            resp: AIMessage = providers.chat_model(role="worker").invoke(messages)
        else:
            resp = worker(messages)
        content = resp.content if isinstance(resp, AIMessage) else resp.get("content", "")
        result = parse_list_and_validate(SkinConcern, content)
        logger.info(f"[research_skin_concerns] found {len(result)} concerns")
        return result
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning(f"[research_skin_concerns] failed: {e}")
        return []


def research_ingredients(
    concerns: list[str],
    worker=None,
    client: aistack.AistackClient | None = None,
) -> list[IngredientSuggestion]:
    """Shortlist cosmetic ingredients for the given concerns."""
    cfg = _research_cfg()
    prompts = cfg.get("prompts", {})
    joined = ", ".join(concerns)
    system = prompts.get("ingredients", "").format(concerns=joined)
    logger.info(f"[research_ingredients] concerns={joined}")
    client = client or aistack.AistackClient()
    evidence = _gather_evidence(client, f"Concerns: {joined}")
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Concerns: {joined}. Output the JSON now."),
        HumanMessage(content=f"Evidence from search:\n{evidence}\nNow respond strictly in JSON."),
    ]
    from langchain_core.messages import AIMessage

    try:
        if worker is None:
            resp: AIMessage = providers.chat_model(role="worker").invoke(messages)
        else:
            resp = worker(messages)
        content = resp.content if isinstance(resp, AIMessage) else resp.get("content", "")
        result = parse_list_and_validate(IngredientSuggestion, content)
        logger.info(f"[research_ingredients] found {len(result)} ingredients")
        return result
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning(f"[research_ingredients] failed: {e}")
        return []
